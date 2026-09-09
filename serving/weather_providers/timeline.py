"""PCSE provider backed by a validated immutable weather timeline."""

from __future__ import annotations

from datetime import date
from typing import Iterable

from pcse.base import WeatherDataContainer, WeatherDataProvider

from ..weather_records import WeatherRecord


class TimelineWeatherDataProvider(WeatherDataProvider):
    """Adapt ``WeatherRecord`` values to PCSE's daily weather contract."""

    ETmodel = "PM"

    def __init__(
        self,
        records: Iterable[WeatherRecord],
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        elevation: float | None = None,
        description: str = "WeatherService validated timeline",
    ) -> None:
        WeatherDataProvider.__init__(self)
        values = tuple(records)
        if not values:
            raise ValueError("TimelineWeatherDataProvider requires at least one record")
        if not all(isinstance(record, WeatherRecord) for record in values):
            raise TypeError("records must contain WeatherRecord values")
        days = [record.date for record in values]
        if days != sorted(days) or len(set(days)) != len(days):
            raise ValueError("timeline records must be sorted and unique")
        self.latitude = float(values[0].latitude if latitude is None else latitude)
        self.longitude = float(values[0].longitude if longitude is None else longitude)
        self.elevation = float(values[0].elevation if elevation is None else elevation)
        self.description = description
        for record in values:
            if abs(record.latitude - self.latitude) > 1e-9 or abs(record.longitude - self.longitude) > 1e-9:
                raise ValueError("timeline records have inconsistent coordinates")
            container = WeatherDataContainer(
                DAY=record.date,
                LAT=self.latitude,
                LON=self.longitude,
                ELEV=self.elevation,
                IRRAD=record.irrad_mj_m2_day * 1_000_000.0,
                TMIN=record.tmin_c,
                TMAX=record.tmax_c,
                VAP=record.vap_hpa,
                RAIN=record.rain_mm * 0.1,
                WIND=record.wind_m_s,
                E0=record.e0_mm * 0.1,
                ES0=record.es0_mm * 0.1,
                ET0=record.et0_mm * 0.1,
                TEMP=record.temp_c if record.temp_c is not None else (record.tmin_c + record.tmax_c) / 2.0,
            )
            self._store_WeatherDataContainer(container, record.date, 0)
        self._first_date = min(days)
        self._last_date = max(days)
        self.records = values


__all__ = ["TimelineWeatherDataProvider"]
