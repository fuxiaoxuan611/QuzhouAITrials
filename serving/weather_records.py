"""Immutable, provider-neutral daily weather records.

All public weather code exchanges SI-like daily values in the units declared
by ``WeatherRecord``.  PCSE-specific conversion is isolated in the timeline
provider, so provider payloads never leak through the decision contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


class WeatherDataKind(str, Enum):
    HISTORICAL = "historical"
    FORECAST = "forecast"
    NOWCAST = "nowcast"
    OBSERVED = "observed"


class WeatherRecordError(ValueError):
    """Raised when a daily weather record violates the shared contract."""


def _finite(value: Any, field_name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WeatherRecordError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise WeatherRecordError(f"{field_name} must be finite and {'>= 0' if nonnegative else 'valid'}")
    return result


@dataclass(frozen=True)
class WeatherRecord:
    date: date
    latitude: float
    longitude: float
    elevation: float
    tmin_c: float
    tmax_c: float
    rain_mm: float
    irrad_mj_m2_day: float
    vap_hpa: float
    wind_m_s: float
    e0_mm: float
    es0_mm: float
    et0_mm: float
    provider: str
    source: str
    data_kind: WeatherDataKind | str
    temp_c: float | None = None
    model: str | None = None
    model_run: str | None = None
    retrieved_at: datetime | None = None
    valid_time: datetime | None = None
    as_of: datetime | None = None
    timezone: str = "Asia/Shanghai"
    quality_flags: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()
    calculation_method: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.date, date) or isinstance(self.date, datetime):
            raise WeatherRecordError("date must be a datetime.date")
        try:
            kind = self.data_kind if isinstance(self.data_kind, WeatherDataKind) else WeatherDataKind(str(self.data_kind))
        except ValueError as exc:
            raise WeatherRecordError("data_kind must be historical, forecast, nowcast, or observed") from exc
        object.__setattr__(self, "data_kind", kind)
        for field_name in ("latitude", "longitude", "elevation", "tmin_c", "tmax_c", "vap_hpa", "wind_m_s"):
            _finite(getattr(self, field_name), field_name)
        for field_name in ("rain_mm", "irrad_mj_m2_day", "e0_mm", "es0_mm", "et0_mm"):
            value = _finite(getattr(self, field_name), field_name, nonnegative=True)
            object.__setattr__(self, field_name, value)
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise WeatherRecordError("latitude/longitude out of range")
        if not self.provider or not str(self.provider).strip() or not self.source or not str(self.source).strip():
            raise WeatherRecordError("provider and source are required")
        if self.temp_c is not None:
            _finite(self.temp_c, "temp_c")
        for field_name in ("quality_flags", "missing_variables"):
            value = tuple(str(item) for item in getattr(self, field_name))
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "tmin_c": self.tmin_c,
            "tmax_c": self.tmax_c,
            "rain_mm": self.rain_mm,
            "irrad_mj_m2_day": self.irrad_mj_m2_day,
            "vap_hpa": self.vap_hpa,
            "wind_m_s": self.wind_m_s,
            "e0_mm": self.e0_mm,
            "es0_mm": self.es0_mm,
            "et0_mm": self.et0_mm,
            "provider": self.provider,
            "source": self.source,
            "data_kind": self.data_kind.value,
            "temp_c": self.temp_c,
            "model": self.model,
            "model_run": self.model_run,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "valid_time": self.valid_time.isoformat() if self.valid_time else None,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "timezone": self.timezone,
            "quality_flags": list(self.quality_flags),
            "missing_variables": list(self.missing_variables),
            "calculation_method": self.calculation_method,
            "revision": self.revision,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WeatherRecord":
        def parse_datetime(item: Any) -> datetime | None:
            if item is None or isinstance(item, datetime):
                return item
            if isinstance(item, str):
                return datetime.fromisoformat(item.replace("Z", "+00:00"))
            raise WeatherRecordError("weather datetime metadata must be ISO datetime")

        data = dict(value)
        if isinstance(data.get("date"), str):
            data["date"] = date.fromisoformat(data["date"])
        for key in ("retrieved_at", "valid_time", "as_of"):
            data[key] = parse_datetime(data.get(key))
        return cls(**data)


__all__ = ["WeatherDataKind", "WeatherRecord", "WeatherRecordError"]
