"""Independent future WOFOST forward simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .errors import ForwardSimulationError
from .management_events import ManagementTimeline
from .season_calendar import SeasonCalendar
from .weather_providers.timeline import TimelineWeatherDataProvider
from .weather_records import WeatherRecord
from .wofost_realtime import WOFOSTRealtimeEngine


@dataclass(frozen=True)
class ForwardSimulationResult:
    scenario_id: str
    query_date: date
    horizon_date: date
    state_at_query: dict[str, Any]
    state_at_horizon: dict[str, Any]
    horizon_reaches_harvest: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "query_date": self.query_date.isoformat(),
            "horizon_date": self.horizon_date.isoformat(),
            "state_at_query": dict(self.state_at_query),
            "state_at_horizon": dict(self.state_at_horizon),
            "horizon_reaches_harvest": self.horizon_reaches_harvest,
            "warnings": list(self.warnings),
        }


class ForwardSimulator:
    """Run one new PCSE engine per scenario; no mutable state is shared."""

    def __init__(self, config_path: str | Path = "configs/CN-Maize.yaml", env_split: str = "test") -> None:
        self.config_path = config_path
        self.env_split = env_split

    def simulate(
        self,
        *,
        scenario_id: str,
        calendar: SeasonCalendar,
        weather_records: Iterable[WeatherRecord],
        management: ManagementTimeline,
        query_date: date,
        horizon_date: date,
    ) -> ForwardSimulationResult:
        if horizon_date < query_date:
            raise ForwardSimulationError("horizon_date must be >= query_date")
        provider = TimelineWeatherDataProvider(tuple(weather_records))
        engine: WOFOSTRealtimeEngine | None = None
        try:
            engine = WOFOSTRealtimeEngine(
                config_path=self.config_path,
                env_split=self.env_split,
                season_calendar=calendar,
                timeline_provider=provider,
                management_timeline=management,
            )
            current = engine.reconstruct(query_date)
            horizon = engine.reconstruct(horizon_date)
            return ForwardSimulationResult(
                scenario_id=scenario_id,
                query_date=query_date,
                horizon_date=horizon_date,
                state_at_query=current["crop_state"],
                state_at_horizon=horizon["crop_state"],
                horizon_reaches_harvest=horizon_date >= calendar.crop_end_date,
            )
        except ForwardSimulationError:
            raise
        except Exception as exc:
            raise ForwardSimulationError("forward simulation failed") from exc
        finally:
            if engine is not None:
                engine.close()


__all__ = ["ForwardSimulationResult", "ForwardSimulator"]
