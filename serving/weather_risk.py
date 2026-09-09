"""Deterministic, auditable weather-risk rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from .weather_records import WeatherRecord


@dataclass(frozen=True)
class WeatherRiskThresholds:
    heavy_rain_mm: float = 25.0
    rainfall_window_mm: float = 40.0
    consecutive_rain_days: int = 3
    heat_tmax_c: float = 35.0
    dry_days: int = 5
    et0_window_mm: float = 35.0
    experimental_rule: bool = True


def _records(values: Iterable[WeatherRecord | dict[str, Any]]) -> tuple[WeatherRecord, ...]:
    result = []
    for value in values:
        result.append(value if isinstance(value, WeatherRecord) else WeatherRecord.from_mapping(value))
    return tuple(sorted(result, key=lambda item: item.date))


class WeatherRiskEvaluator:
    def __init__(self, thresholds: WeatherRiskThresholds | None = None) -> None:
        self.thresholds = thresholds or WeatherRiskThresholds()

    def evaluate(self, records: Iterable[WeatherRecord | dict[str, Any]], *, as_of: Any = None) -> list[dict[str, Any]]:
        values = _records(records)
        if not values:
            return []
        t = self.thresholds
        output: list[dict[str, Any]] = []

        def add(risk_type: str, severity: str, evidence: dict[str, Any], threshold: Any) -> None:
            output.append({
                "risk_type": risk_type,
                "severity": severity,
                "window": {"start": values[0].date.isoformat(), "end": values[-1].date.isoformat()},
                "evidence": evidence,
                "threshold": threshold,
                "source": sorted({item.source for item in values}),
                "as_of": as_of,
                "experimental_rule": t.experimental_rule,
            })

        heavy = [item for item in values if item.rain_mm >= t.heavy_rain_mm]
        if heavy:
            add("heavy_rain", "high" if any(item.rain_mm >= 2 * t.heavy_rain_mm for item in heavy) else "moderate", {"days": [item.date.isoformat() for item in heavy], "max_rain_mm": max(item.rain_mm for item in heavy)}, {"heavy_rain_mm": t.heavy_rain_mm})
        total_rain = sum(item.rain_mm for item in values)
        if total_rain >= t.rainfall_window_mm:
            add("rainfall", "moderate", {"total_rain_mm": total_rain}, {"rainfall_window_mm": t.rainfall_window_mm})
        run = 0
        max_run = 0
        run_end: date | None = None
        for item in values:
            if item.rain_mm > 0:
                run += 1
                if run > max_run:
                    max_run, run_end = run, item.date
            else:
                run = 0
        if max_run >= t.consecutive_rain_days:
            add("consecutive_rain", "moderate", {"days": max_run, "end": run_end.isoformat() if run_end else None}, {"consecutive_rain_days": t.consecutive_rain_days})
        hot = [item for item in values if item.tmax_c >= t.heat_tmax_c]
        if hot:
            add("heat", "moderate", {"days": [item.date.isoformat() for item in hot], "max_tmax_c": max(item.tmax_c for item in hot)}, {"tmax_c": t.heat_tmax_c})
        dry = 0
        max_dry = 0
        for item in values:
            dry = dry + 1 if item.rain_mm == 0 else 0
            max_dry = max(max_dry, dry)
        if max_dry >= t.dry_days:
            add("dry_period", "moderate", {"max_consecutive_dry_days": max_dry}, {"dry_days": t.dry_days})
        total_et0 = sum(item.et0_mm for item in values)
        if total_et0 >= t.et0_window_mm:
            add("et0_demand", "moderate", {"total_et0_mm": total_et0}, {"et0_window_mm": t.et0_window_mm})
        return output


__all__ = ["WeatherRiskEvaluator", "WeatherRiskThresholds"]
