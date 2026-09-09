"""Integrated canonical-request decision engine for CN-Maize.

The engine coordinates the already-tested management converter, realtime
WOFOST reconstruction, observation-fusion boundary, and LagPPO inference
wrapper. It deliberately does not own HTTP, FastGPT/LLM integration, Level 2
state assimilation, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .decision_contract import build_decision_result
from .errors import (
    DecisionEngineError,
    RequestValidationError,
    DynamicCalendarInvalidError,
    ManagementEventInvalidError,
)
from .management_history import management_to_realtime_input
from .management_events import ManagementEventError, ManagementTimeline
from .operation_advice import OperationAdvice
from .forward_simulation import ForwardSimulator
from .observation_fusion import ObservationFusionEngine, not_due_audit
from .rl_inference import RLInferenceEngine
from .scenario_evaluation import ScenarioEvaluator, ScenarioGenerator
from .season_calendar import DynamicCalendarError, SeasonCalendarResolver
from .weather_providers.timeline import TimelineWeatherDataProvider
from .weather_risk import WeatherRiskEvaluator
from .weather_service import WeatherService
from .schemas import (
    SCHEMA_VERSION,
    DecisionRequest,
    SchemaValidationError,
    normalize_decision_request,
)
from .wofost_realtime import WOFOSTRealtimeEngine


_SCENARIO_MAX_TIMING_DELAY_DAYS = 1
_SCENARIO_POST_EVENT_EVALUATION_DAYS = 1


@dataclass(frozen=True)
class DecisionSlot:
    """One policy observation boundary and its following action interval."""

    observation_date: date
    action_application_date: date
    interval_start: date
    interval_end: date


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


class DecisionEngine:
    """Run one canonical CN-Maize decision without mutating WOFOST state."""

    def __init__(
        self,
        config_path: str | Path,
        model_path: str | Path,
        env_stats_path: str | Path,
        device: str = "auto",
        env_split: str = "test",
        policy_validation_status: str = "unknown",
        weather_service: WeatherService | None = None,
        season_resolver: SeasonCalendarResolver | None = None,
    ) -> None:
        self.realtime = WOFOSTRealtimeEngine(
            config_path=config_path,
            env_split=env_split,
        )
        self.observation_fusion = ObservationFusionEngine()
        self.weather_service = weather_service or WeatherService.default()
        self.season_resolver = season_resolver or SeasonCalendarResolver()
        try:
            self.rl = RLInferenceEngine(
                config_path=config_path,
                model_path=model_path,
                env_stats_path=env_stats_path,
                device=device,
                env_split=env_split,
                policy_validation_status=policy_validation_status,
            )
        except Exception:
            self.realtime.close()
            raise

        if self.realtime.timestep_days != int(
            self.rl.config["environment"].get("timestep", 7)
        ):
            self.close()
            raise DecisionEngineError(
                "TIMESTEP_MISMATCH: realtime and RL configurations use different timesteps"
            )
        if self.realtime.action_count != int(self.rl.env.action_space.n):
            self.close()
            raise DecisionEngineError(
                "ACTION_SPACE_MISMATCH: realtime and RL configurations use different action counts"
            )

    @property
    def timestep_days(self) -> int:
        return self.realtime.timestep_days

    def decision_calendar(self) -> list[dict[str, str]]:
        """Return the dynamic policy schedule derived from the WOFOST dates."""

        slots = self._decision_slots()
        return [
            {
                "observation_date": slot.observation_date.isoformat(),
                "action_application_date": slot.action_application_date.isoformat(),
                "interval_start": slot.interval_start.isoformat(),
                "interval_end": slot.interval_end.isoformat(),
            }
            for slot in slots
        ]

    def _decision_slots(self, *, crop_start: date | None = None, crop_end: date | None = None) -> list[DecisionSlot]:
        crop_start = self.realtime.crop_start_date if crop_start is None else crop_start
        crop_end = self.realtime.crop_end_date if crop_end is None else crop_end
        slots: list[DecisionSlot] = []
        observation_date = crop_start
        while observation_date < crop_end:
            action_date = min(
                observation_date + timedelta(days=self.timestep_days),
                crop_end,
            )
            slots.append(
                DecisionSlot(
                    observation_date=observation_date,
                    action_application_date=action_date,
                    interval_start=observation_date,
                    interval_end=action_date,
                )
            )
            observation_date += timedelta(days=self.timestep_days)
        return slots

    def _slot_for(self, requested: date, *, crop_start: date | None = None, crop_end: date | None = None) -> DecisionSlot | None:
        return next(
            (slot for slot in self._decision_slots(crop_start=crop_start, crop_end=crop_end) if slot.observation_date == requested),
            None,
        )

    def _previous_next_slots(
        self,
        requested: date,
        *,
        crop_start: date | None = None,
        crop_end: date | None = None,
    ) -> tuple[DecisionSlot | None, DecisionSlot | None]:
        slots = self._decision_slots(crop_start=crop_start, crop_end=crop_end)
        previous = None
        following = None
        for slot in slots:
            if slot.observation_date < requested:
                previous = slot
            elif slot.observation_date > requested:
                following = slot
                break
        return previous, following

    def decide(self, request: DecisionRequest | dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-safe state and, only at a policy boundary, a recommendation."""

        try:
            canonical = (
                normalize_decision_request(request)
                if isinstance(request, dict)
                else request
            )
        except SchemaValidationError as exc:
            raise RequestValidationError(str(exc), field="request") from exc
        if not isinstance(canonical, DecisionRequest):
            raise RequestValidationError(
                "request must be a canonical DecisionRequest or mapping",
                field="request",
            )
        if canonical.schema_version != SCHEMA_VERSION:
            raise RequestValidationError(
                f"schema_version: expected {SCHEMA_VERSION!r}, got {canonical.schema_version!r}",
                field="schema_version",
            )

        # The fixed 2025 request shape is a preserved regression path.  Its
        # historical irrigation semantics intentionally remain unchanged;
        # callers opt into the new date-based dynamic mode explicitly (or by
        # using a non-baseline season).
        if (
            canonical.crop.sowing_date == self.realtime.crop_start_date
            and canonical.crop.expected_harvest_date is None
            and canonical.decision_context.decision_mode == "auto"
            and canonical.management.irrigation_history
        ):
            management_to_realtime_input(canonical)

        is_legacy = self._is_legacy_request(canonical)
        if not is_legacy:
            return self._decide_dynamic(canonical)

        if canonical.crop.sowing_date != self.realtime.crop_start_date:
            raise DecisionEngineError(
                "CROP_START_MISMATCH: canonical crop.sowing_date="
                f"{canonical.crop.sowing_date}, configured crop start="
                f"{self.realtime.crop_start_date}"
            )

        realtime_input = management_to_realtime_input(canonical)
        reconstructed = self.realtime.reconstruct(
            canonical.query_date,
            realtime_input["action_history"],
        )
        requested = canonical.query_date
        slot = self._slot_for(requested)
        previous, following = self._previous_next_slots(requested)
        decision_due = slot is not None

        warnings: list[str] = []
        if canonical.decision_context.allow_irrigation_decision:
            warnings.append("irrigation_decision_not_supported")
        if not decision_due:
            warnings.append("query_date_is_not_a_policy_decision_date")
        if not canonical.decision_context.allow_fertilization_decision:
            warnings.append("fertilization_decision_disabled")

        if decision_due and canonical.decision_context.allow_fertilization_decision:
            fusion_result = self.observation_fusion.fuse(
                reconstructed["raw_observation"],
                canonical.observations,
                canonical.query_date,
            )
            policy_raw_observation = fusion_result["corrected_raw_observation"]
            observation_fusion = fusion_result["audit"]
        else:
            policy_raw_observation = None
            observation_fusion = not_due_audit(canonical.observations is not None)
            if decision_due and not canonical.decision_context.allow_fertilization_decision:
                observation_fusion["mode"] = "fertilization_decision_disabled"
        if observation_fusion.get("date_mismatch"):
            warnings.append("OBSERVATION_DATE_MISMATCH")

        recommendation = None
        constraint_violation = False
        if decision_due and canonical.decision_context.allow_fertilization_decision:
            prediction = self.rl.predict_from_raw_observation(
                policy_raw_observation,
                deterministic=True,
            )
            policy_action = {
                "action_index": prediction["action_index"],
                "n_rate_kg_ha": prediction["n_rate_kg_ha"],
            }
            maximum = canonical.decision_context.max_single_n_rate_kg_ha
            constraint_violation = maximum is not None and (
                prediction["n_rate_kg_ha"] > maximum
            )
            if constraint_violation:
                warnings.append("max_single_n_rate_kg_ha_exceeded")
            constraint_details = {}
            if maximum is not None:
                constraint_details = {
                    "max_single_n_rate_kg_ha": float(maximum),
                    "actual_n_rate_kg_ha": float(prediction["n_rate_kg_ha"]),
                    "exceeded_by_kg_ha": max(
                        0.0,
                        float(prediction["n_rate_kg_ha"]) - float(maximum),
                    ),
                }
            recommendation = {
                "decision_type": canonical.decision_context.decision_type,
                **policy_action,
                "constraint_violation": constraint_violation,
                "constraint_details": constraint_details,
            }

        metadata = self.rl.get_metadata()
        metadata.update(
            {
                "raw_observation_shape": reconstructed["raw_observation_shape"],
                "raw_observation_dtype": reconstructed["raw_observation_dtype"],
                "observation_feature_names": reconstructed["observation_feature_names"],
                "observation_space_dtype": reconstructed["observation_space_dtype"],
            }
        )

        result = build_decision_result(
            request_id=canonical.request.request_id,
            decision_due=decision_due,
            state_date=reconstructed["simulation_date"],
            policy_observation_date=_iso(slot.observation_date) if slot else None,
            action_application_date=(
                _iso(slot.action_application_date) if slot else None
            ),
            previous_decision_date=_iso(previous.observation_date) if previous else None,
            next_decision_date=(
                _iso(following.observation_date) if following else None
            ),
            recommendation=recommendation,
            model_state=reconstructed["crop_state"],
            management={
                "action_history": realtime_input["action_history"],
                "decision_dates": realtime_input["decision_dates"],
                "completed_steps": realtime_input["completed_steps"],
                "residual_days": realtime_input["residual_days"],
                "mapped_fertilization_events": realtime_input[
                    "mapped_fertilization_events"
                ],
            },
            observations={
                "received": canonical.observations is not None,
                "applied_to_policy": observation_fusion["applied_to_policy"],
                "applied_to_wofost_state": observation_fusion[
                    "applied_to_wofost_state"
                ],
                "fusion_audit": observation_fusion,
                # Compatibility alias for callers of the pre-contract result.
                "applied_to_model": False,
            },
            model_metadata=metadata,
            warnings=warnings,
        )
        # Compatibility aliases are retained for the already-tested local
        # pipeline; the keys above are the frozen v1 contract source of truth.
        result["constraint_violation"] = constraint_violation
        result["current_state"] = result["model_state"]
        result["observation_fusion"] = result["observations"]["fusion_audit"]
        return result

    def _is_legacy_request(self, canonical: DecisionRequest) -> bool:
        """Keep the fixed 2025 contract path for backward compatibility."""

        return (
            canonical.crop.sowing_date == self.realtime.crop_start_date
            and canonical.crop.expected_harvest_date is None
            and canonical.decision_context.decision_mode == "auto"
            and not canonical.management.irrigation_history
        )

    @staticmethod
    def _canonical_fertilizers(canonical: DecisionRequest) -> list[dict[str, Any]]:
        events = []
        for event in canonical.management.fertilization_history:
            item = event.to_dict()
            item["f_nh4n"] = event.f_nh4n if event.f_nh4n is not None else event.nh4_n_fraction
            item["f_no3n"] = event.f_no3n if event.f_no3n is not None else event.no3_n_fraction
            events.append(item)
        return events

    def _decide_dynamic(self, canonical: DecisionRequest) -> dict[str, Any]:
        """Run the weather-aware pipeline for a non-legacy season request."""

        try:
            calendar = self.season_resolver.resolve(
                canonical.crop.name,
                canonical.crop.cultivar,
                canonical.crop.sowing_date,
                canonical.crop.expected_harvest_date,
            )
            management = ManagementTimeline.from_values(
                self._canonical_fertilizers(canonical),
                [item.to_dict() for item in canonical.management.irrigation_history],
            )
        except DynamicCalendarError as exc:
            raise DynamicCalendarInvalidError(str(exc)) from exc
        except ManagementEventError as exc:
            raise ManagementEventInvalidError(str(exc)) from exc
        if not canonical.management.fertilization_history_complete:
            raise RequestValidationError(
                "INCOMPLETE_MANAGEMENT_HISTORY: fertilization history must be complete",
                field="management.fertilization_history",
            )
        if not canonical.management.irrigation_history_complete:
            raise RequestValidationError(
                "MANAGEMENT_EVENT_INVALID: irrigation history must be complete",
                field="management.irrigation_history",
            )
        horizon = canonical.weather.forecast_horizon_days
        if horizon is None:
            horizon = canonical.decision_context.forecast_horizon_days
        if horizon is None:
            horizon = self.weather_service.forecast_default_days
        horizon = int(horizon)

        slot = self._slot_for(
            canonical.query_date,
            crop_start=calendar.crop_start_date,
            crop_end=calendar.crop_end_date,
        )
        previous, following = self._previous_next_slots(
            canonical.query_date,
            crop_start=calendar.crop_start_date,
            crop_end=calendar.crop_end_date,
        )
        forecast_end_date = min(
            calendar.crop_end_date,
            canonical.query_date + timedelta(days=horizon),
        )

        # The public forecast state follows the requested/default horizon. A
        # scenario containing a future policy action needs additional weather
        # coverage through the event and at least one post-event model day.
        # ScenarioGenerator can delay an action by one day for weather risk,
        # so reserve that day as well. With the current seven-day policy this
        # remains within Open-Meteo's 16-day provider capability.
        scenario_action_date = None
        if canonical.decision_context.allow_fertilization_decision:
            if slot is not None:
                scenario_action_date = slot.action_application_date
            elif following is not None and forecast_end_date >= following.observation_date:
                scenario_action_date = following.action_application_date
        scenario_horizon_date = forecast_end_date
        if scenario_action_date is not None:
            scenario_horizon_date = min(
                calendar.crop_end_date,
                max(
                    scenario_horizon_date,
                    scenario_action_date
                    + timedelta(
                        days=(
                            _SCENARIO_MAX_TIMING_DELAY_DAYS
                            + _SCENARIO_POST_EVENT_EVALUATION_DAYS
                        )
                    ),
                ),
            )
        weather_horizon_days = max(
            0, (scenario_horizon_date - canonical.query_date).days
        )
        # The canonical lightweight WeatherObservation is not a full
        # WeatherRecord.  Full deterministic records can still be supplied by
        # an injected WeatherService/provider; incomplete client records stay
        # context-only rather than being silently promoted to model input.
        weather_context = self.weather_service.get_context(
            campaign_start=calendar.campaign_start_date,
            query_date=canonical.query_date,
            forecast_horizon_days=weather_horizon_days,
            decision_mode=canonical.decision_context.decision_mode,
            provider_name=canonical.weather.provider,
            latitude=canonical.location.latitude,
            longitude=canonical.location.longitude,
            elevation=canonical.location.elevation_m,
            timezone_name=canonical.location.timezone,
            as_of=canonical.request.as_of_datetime,
        )
        records = weather_context["timeline"]
        provider = TimelineWeatherDataProvider(records)
        dynamic_realtime = WOFOSTRealtimeEngine(
            config_path=self.realtime.config_path,
            env_split=self.realtime.env_split,
            season_calendar=calendar,
            timeline_provider=provider,
            management_timeline=management,
        )
        try:
            reconstructed = dynamic_realtime.reconstruct(canonical.query_date)
            warnings = list(weather_context.get("warnings", []))
            if not slot:
                warnings.append("query_date_is_not_a_policy_decision_date")
            fusion = self.observation_fusion.fuse(
                reconstructed["raw_observation"],
                canonical.observations,
                canonical.query_date,
            )
            if fusion["audit"].get("date_mismatch"):
                warnings.append("OBSERVATION_DATE_MISMATCH")
            recommendation = None
            if slot and canonical.decision_context.allow_fertilization_decision:
                prediction = self.rl.predict_from_raw_observation(fusion["corrected_raw_observation"], deterministic=True)
                maximum = canonical.decision_context.max_single_n_rate_kg_ha
                violation = maximum is not None and prediction["n_rate_kg_ha"] > maximum
                if violation:
                    warnings.append("max_single_n_rate_kg_ha_exceeded")
                recommendation = {
                    "decision_type": canonical.decision_context.decision_type,
                    "action_index": prediction["action_index"],
                    "n_rate_kg_ha": prediction["n_rate_kg_ha"],
                    "constraint_violation": violation,
                    "constraint_details": ({
                        "max_single_n_rate_kg_ha": float(maximum),
                        "actual_n_rate_kg_ha": float(prediction["n_rate_kg_ha"]),
                        "exceeded_by_kg_ha": max(0.0, float(prediction["n_rate_kg_ha"]) - float(maximum)),
                    } if maximum is not None else {}),
                }
            elif slot and not canonical.decision_context.allow_fertilization_decision:
                warnings.append("fertilization_decision_disabled")

            projected = None
            if slot is None and following is not None:
                if forecast_end_date >= following.observation_date:
                    next_state = dynamic_realtime.reconstruct(following.observation_date)
                    next_prediction = self.rl.predict_from_raw_observation(next_state["raw_observation"], deterministic=True)
                    projected = {
                        "projected": True,
                        "uses_forecast": True,
                        "decision_date": following.observation_date.isoformat(),
                        "action_application_date": following.action_application_date.isoformat(),
                        "action_index": next_prediction["action_index"],
                        "n_rate_kg_ha": next_prediction["n_rate_kg_ha"],
                        "policy_validation_status": self.rl.policy_validation_status,
                        "agronomic_validated": False,
                    }
                else:
                    warnings.append("FORECAST_HORIZON_INSUFFICIENT")

            horizon_state = dynamic_realtime.reconstruct(forecast_end_date)
            forecast = {
                "horizon_days": horizon,
                "horizon_end": forecast_end_date.isoformat(),
                "state_at_horizon": horizon_state["crop_state"],
                "state_estimated": (
                    forecast_end_date > canonical.query_date
                    or bool(weather_context.get("state_estimated"))
                ),
                "uses_forecast": forecast_end_date > canonical.query_date,
                "scenario_horizon_end": scenario_horizon_date.isoformat(),
            }
            risk_records = tuple(item for item in records if item.date > canonical.query_date)
            weather_risk = WeatherRiskEvaluator().evaluate(risk_records, as_of=weather_context.get("as_of"))
            scenario_source = projected or recommendation
            scenario_rate = scenario_source.get("n_rate_kg_ha") if scenario_source else None
            scenario_date = (slot.action_application_date if slot else following.action_application_date if following else None)
            scenarios = ScenarioGenerator().generate(
                current_query_date=canonical.query_date,
                action_application_date=scenario_date,
                rl_n_rate_kg_ha=scenario_rate,
                weather_risk=weather_risk,
            )
            scenario_evaluation = ScenarioEvaluator(
                ForwardSimulator(self.realtime.config_path, self.realtime.env_split)
            ).evaluate(
                scenarios=scenarios,
                calendar=calendar,
                weather_records=records,
                historical_management=management,
                query_date=canonical.query_date,
                horizon_date=scenario_horizon_date,
            )
            advice = OperationAdvice().build(
                rl_recommendation=recommendation,
                projected_recommendation=projected,
                weather_risk=weather_risk,
                next_evaluation_date=following.observation_date if following else None,
            )
            metadata = self.rl.get_metadata()
            metadata.update({
                "season_calendar": calendar.to_dict(),
                "weather_aware": True,
                "forecast_state_assimilation": False,
                "level1_fusion_only": True,
                "assumptions": calendar.assumptions,
                "raw_observation_shape": reconstructed["raw_observation_shape"],
                "raw_observation_dtype": reconstructed["raw_observation_dtype"],
                "observation_feature_names": reconstructed["observation_feature_names"],
                "observation_space_dtype": reconstructed["observation_space_dtype"],
            })
            if canonical.decision_context.allow_irrigation_decision:
                warnings.append("irrigation_policy_not_implemented")
            result = build_decision_result(
                request_id=canonical.request.request_id,
                decision_due=slot is not None,
                state_date=reconstructed["simulation_date"],
                policy_observation_date=_iso(slot.observation_date) if slot else None,
                previous_decision_date=_iso(previous.observation_date) if previous else None,
                next_decision_date=_iso(following.observation_date) if following else None,
                action_application_date=_iso(slot.action_application_date) if slot else None,
                recommendation=recommendation,
                model_state=reconstructed["crop_state"],
                management={**management.to_dict(), "calendar": calendar.to_dict()},
                observations={
                    "received": canonical.observations is not None,
                    "applied_to_policy": fusion["audit"]["applied_to_policy"],
                    "applied_to_wofost_state": False,
                    "applied_to_model": False,
                    "fusion_audit": fusion["audit"],
                },
                model_metadata=metadata,
                warnings=sorted(set(warnings)),
                weather_context={key: value for key, value in weather_context.items() if key != "timeline"},
                forecast=forecast,
                projected_next_decision=projected,
                scenario_evaluation=scenario_evaluation,
                weather_risk=weather_risk,
                operation_advice=advice,
            )
            result["constraint_violation"] = bool(recommendation and recommendation["constraint_violation"])
            result["current_state"] = result["model_state"]
            result["observation_fusion"] = result["observations"]["fusion_audit"]
            return result
        finally:
            dynamic_realtime.close()

    def close(self) -> None:
        self.realtime.close()
        self.rl.close()

    def get_metadata(self) -> dict[str, Any]:
        """Return internal model metadata for public sanitization by adapters."""

        return self.rl.get_metadata()

    def __enter__(self) -> "DecisionEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["DecisionEngine", "DecisionEngineError", "DecisionSlot"]
