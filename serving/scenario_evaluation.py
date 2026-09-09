"""Small, isolated management scenario generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

from .forward_simulation import ForwardSimulationResult, ForwardSimulator
from .management_events import FertilizerEvent, IrrigationEvent, ManagementTimeline
from .season_calendar import SeasonCalendar


@dataclass(frozen=True)
class ManagementScenario:
    scenario_id: str
    fertilizer_events: tuple[FertilizerEvent, ...] = ()
    irrigation_events: tuple[IrrigationEvent, ...] = ()
    source: str = "generated"

    def management(self, historical: ManagementTimeline) -> ManagementTimeline:
        return ManagementTimeline(
            fertilizers=tuple(sorted((*historical.fertilizers, *self.fertilizer_events), key=lambda item: item.date)),
            irrigations=tuple(sorted((*historical.irrigations, *self.irrigation_events), key=lambda item: item.date)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "source": self.source,
            "fertilizer_events": [item.to_dict() for item in self.fertilizer_events],
            "irrigation_events": [item.to_dict() for item in self.irrigation_events],
        }


class ScenarioGenerator:
    """Generate a conservative candidate set, not the full 16-action sweep."""

    def generate(
        self,
        *,
        current_query_date: date,
        action_application_date: date | None,
        rl_n_rate_kg_ha: float | None,
        weather_risk: Iterable[dict[str, Any]] = (),
        user_scenarios: Iterable[ManagementScenario] | None = None,
        include_neighbors: bool = False,
    ) -> tuple[ManagementScenario, ...]:
        if user_scenarios is not None:
            return tuple(user_scenarios)
        result = [ManagementScenario("no_additional_n")]
        if rl_n_rate_kg_ha is not None and action_application_date is not None:
            result.append(
                ManagementScenario(
                    "rl_policy_rate",
                    fertilizer_events=(FertilizerEvent(action_application_date, rl_n_rate_kg_ha),),
                    source="rl_policy",
                )
            )
            if any(item.get("severity") in {"high", "severe"} for item in weather_risk):
                delayed = action_application_date + timedelta(days=1)
                result.append(
                    ManagementScenario(
                        "rl_policy_rate_weather_timing",
                        fertilizer_events=(FertilizerEvent(delayed, rl_n_rate_kg_ha),),
                        source="weather_risk_timing",
                    )
                )
            if include_neighbors:
                for delta, suffix in ((-10.0, "lower_neighbor"), (10.0, "upper_neighbor")):
                    rate = max(0.0, rl_n_rate_kg_ha + delta)
                    result.append(
                        ManagementScenario(
                            f"rl_policy_{suffix}",
                            fertilizer_events=(FertilizerEvent(action_application_date, rate),),
                            source="experimental_neighbor",
                        )
                    )
        return tuple(result)


class ScenarioEvaluator:
    def __init__(self, simulator: ForwardSimulator | None = None) -> None:
        self.simulator = simulator or ForwardSimulator()

    def evaluate(
        self,
        *,
        scenarios: Iterable[ManagementScenario],
        calendar: SeasonCalendar,
        weather_records: Iterable[Any],
        historical_management: ManagementTimeline,
        query_date: date,
        horizon_date: date,
    ) -> list[dict[str, Any]]:
        results = []
        for scenario in scenarios:
            simulated = self.simulator.simulate(
                scenario_id=scenario.scenario_id,
                calendar=calendar,
                weather_records=weather_records,
                management=scenario.management(historical_management),
                query_date=query_date,
                horizon_date=horizon_date,
            )
            results.append({**scenario.to_dict(), **simulated.to_dict()})
        return results


__all__ = ["ManagementScenario", "ScenarioEvaluator", "ScenarioGenerator"]
