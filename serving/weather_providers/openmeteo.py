"""Documented Open-Meteo adapter for historical and forecast daily data."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping

import httpx

from ..errors import WeatherProviderError
from ..weather_records import WeatherDataKind, WeatherRecord
from .base import WeatherProvider


_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_DAILY_VARIABLES = (
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "wind_speed_10m_mean",
    "et0_fao_evapotranspiration",
)


def _vapour_pressure_hpa(temp_c: float, relative_humidity: float | None, dew_point_c: float | None) -> float:
    # Saturation vapour pressure (kPa) using the Magnus approximation.  The
    # result is hPa, which is the unit required by PCSE WeatherDataContainer.
    reference = dew_point_c
    if reference is None:
        if relative_humidity is None:
            raise WeatherProviderError("Open-Meteo response lacks humidity/dew-point data")
        rh = max(0.0, min(100.0, float(relative_humidity))) / 100.0
        saturation_kpa = 0.6108 * __import__("math").exp((17.27 * temp_c) / (temp_c + 237.3))
        return float(saturation_kpa * rh * 10.0)
    saturation_kpa = 0.6108 * __import__("math").exp((17.27 * reference) / (reference + 237.3))
    return float(saturation_kpa * 10.0)


class OpenMeteoWeatherProvider(WeatherProvider):
    """Fetch Open-Meteo daily records without using PCSE's disk cache."""

    name = "openmeteo"
    supports_historical = True
    supports_forecast = True
    supports_historical_forecast = False
    forecast_max_days = 16

    def __init__(
        self,
        *,
        model: str = "best_match",
        timezone_name: str = "Asia/Shanghai",
        timeout_seconds: float = 20.0,
        elevation: float = 0.0,
        request_json: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.timezone_name = timezone_name
        self.timeout_seconds = float(timeout_seconds)
        self.elevation = float(elevation)
        self._request_json_hook = request_json

    def _request_json(self, url: str, params: dict[str, Any]) -> Mapping[str, Any]:
        if self._request_json_hook is not None:
            return self._request_json_hook(url=url, params=params, timeout=self.timeout_seconds)
        try:
            response = httpx.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise WeatherProviderError(
                "Open-Meteo request failed.", details={"provider": self.name}
            ) from exc
        if not isinstance(payload, Mapping):
            raise WeatherProviderError("Open-Meteo returned a non-object response.")
        if payload.get("error"):
            raise WeatherProviderError("Open-Meteo returned an error response.")
        return payload

    def _get(self, url: str, start_date: date, end_date: date, latitude: float, longitude: float, *, kind: WeatherDataKind) -> tuple[WeatherRecord, ...]:
        if end_date < start_date:
            return ()
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(_DAILY_VARIABLES),
            "timezone": self.timezone_name,
            "models": self.model,
        }
        payload = self._request_json(url, params)
        daily = payload.get("daily")
        if not isinstance(daily, Mapping) or not isinstance(daily.get("time"), list):
            raise WeatherProviderError("Open-Meteo response lacks daily time series.")
        times = daily["time"]
        retrieved = datetime.now(timezone.utc)
        records: list[WeatherRecord] = []
        for index, raw_day in enumerate(times):
            try:
                day = date.fromisoformat(str(raw_day))
                def value(name: str, *, required: bool = True) -> float | None:
                    values = daily.get(name)
                    item = values[index] if isinstance(values, list) and index < len(values) else None
                    if item is None and required:
                        raise WeatherProviderError(f"Open-Meteo daily variable missing: {name}")
                    return None if item is None else float(item)

                tmin = value("temperature_2m_min")
                tmax = value("temperature_2m_max")
                tmean = value("temperature_2m_mean", required=False)
                rain = value("precipitation_sum")
                radiation = value("shortwave_radiation_sum")
                rh = value("relative_humidity_2m_mean", required=False)
                dew = value("dew_point_2m_mean", required=False)
                wind = value("wind_speed_10m_mean")
                et0 = value("et0_fao_evapotranspiration")
                temp = tmean if tmean is not None else (tmin + tmax) / 2.0
                vap = _vapour_pressure_hpa(temp, rh, dew)
                records.append(
                    WeatherRecord(
                        date=day,
                        latitude=float(latitude),
                        longitude=float(longitude),
                        elevation=self.elevation,
                        tmin_c=tmin,
                        tmax_c=tmax,
                        rain_mm=rain,
                        irrad_mj_m2_day=radiation,
                        vap_hpa=vap,
                        wind_m_s=wind,
                        e0_mm=et0,
                        es0_mm=et0,
                        et0_mm=et0,
                        provider=self.name,
                        source=url.split("/", 3)[2],
                        data_kind=kind,
                        temp_c=temp,
                        model=str(payload.get("model") or self.model),
                        model_run=str(payload.get("model_run")) if payload.get("model_run") else None,
                        retrieved_at=retrieved,
                        valid_time=datetime.combine(day, datetime.min.time()),
                        as_of=retrieved if kind != WeatherDataKind.HISTORICAL else None,
                        timezone=self.timezone_name,
                        calculation_method="Open-Meteo daily ET0; E0/ES0 use ET0 fallback",
                    )
                )
            except WeatherProviderError:
                raise
            except Exception as exc:
                raise WeatherProviderError("Invalid Open-Meteo daily record.") from exc
        return tuple(records)

    def get_historical(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        return self._get(_ARCHIVE_URL, start_date, end_date, float(kwargs["latitude"]), float(kwargs["longitude"]), kind=WeatherDataKind.HISTORICAL)

    def get_forecast(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        if (end_date - start_date).days + 1 > int(self.forecast_max_days):
            raise WeatherProviderError("Open-Meteo forecast horizon exceeds provider capability.", details={"max_days": self.forecast_max_days})
        return self._get(_FORECAST_URL, start_date, end_date, float(kwargs["latitude"]), float(kwargs["longitude"]), kind=WeatherDataKind.FORECAST)


__all__ = ["OpenMeteoWeatherProvider"]
