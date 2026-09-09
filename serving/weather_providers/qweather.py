"""QWeather adapter boundary.

The provider is deliberately not enabled by default.  API credentials are
read only from the process environment; the adapter remains useful in tests
through a request hook without embedding credentials or undocumented URLs.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Callable, Mapping

from ..errors import WeatherProviderNotConfiguredError, WeatherProviderError
from ..weather_records import WeatherRecord
from .base import WeatherProvider


class QWeatherProvider(WeatherProvider):
    name = "qweather"
    supports_historical = False
    supports_recent_history = False
    supports_forecast = True
    supports_historical_forecast = False
    forecast_max_days = 10

    def __init__(self, *, api_key: str | None = None, request_json: Callable[..., Mapping[str, Any]] | None = None) -> None:
        self.api_key = api_key or os.getenv("QWEATHER_API_KEY")
        self._request_json_hook = request_json

    def _not_configured(self) -> None:
        if not self.api_key and self._request_json_hook is None:
            raise WeatherProviderNotConfiguredError(
                "QWeather provider requires QWEATHER_API_KEY"
            )

    def get_historical(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        raise WeatherProviderNotConfiguredError("QWeather historical endpoint is not configured")

    def get_forecast(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        self._not_configured()
        if self._request_json_hook is None:
            raise WeatherProviderError("QWeather forecast adapter requires an approved endpoint configuration")
        raise WeatherProviderError("QWeather response mapping is not configured")


__all__ = ["QWeatherProvider"]
