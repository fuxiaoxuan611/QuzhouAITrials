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
from .errors import DecisionEngineError, RequestValidationError
from .management_history import management_to_realtime_input
from .observation_fusion import ObservationFusionEngine, not_due_audit
from .rl_inference import RLInferenceEngine
from .schemas import (
    SCHEMA_VERSION,
    DecisionRequest,
    SchemaValidationError,
    normalize_decision_request,
)
from .wofost_realtime import WOFOSTRealtimeEngine


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
    ) -> None:
        self.realtime = WOFOSTRealtimeEngine(
            config_path=config_path,
            env_split=env_split,
        )
        self.observation_fusion = ObservationFusionEngine()
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

    def _decision_slots(self) -> list[DecisionSlot]:
        crop_start = self.realtime.crop_start_date
        crop_end = self.realtime.crop_end_date
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

    def _slot_for(self, requested: date) -> DecisionSlot | None:
        return next(
            (slot for slot in self._decision_slots() if slot.observation_date == requested),
            None,
        )

    def _previous_next_slots(
        self,
        requested: date,
    ) -> tuple[DecisionSlot | None, DecisionSlot | None]:
        slots = self._decision_slots()
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

    def close(self) -> None:
        self.realtime.close()
        self.rl.close()

    def __enter__(self) -> "DecisionEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["DecisionEngine", "DecisionEngineError", "DecisionSlot"]
