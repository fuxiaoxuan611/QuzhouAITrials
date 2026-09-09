"""Provider abstraction with explicit capability declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Iterable

from ..weather_records import WeatherRecord


class WeatherProvider(ABC):
    name = "unknown"
    supports_historical = False
    supports_recent_history = False
    supports_forecast = False
    supports_historical_forecast = False
    forecast_max_days: int | None = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "supports_historical": bool(self.supports_historical),
            "supports_recent_history": bool(self.supports_recent_history),
            "supports_forecast": bool(self.supports_forecast),
            "supports_historical_forecast": bool(self.supports_historical_forecast),
            "forecast_max_days": self.forecast_max_days,
        }

    @abstractmethod
    def get_historical(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        raise NotImplementedError

    def get_recent_history(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        raise NotImplementedError(f"{self.name} does not support recent history")

    @abstractmethod
    def get_forecast(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        raise NotImplementedError

    def get_historical_forecast(self, start_date: date, end_date: date, **kwargs: Any) -> tuple[WeatherRecord, ...]:
        raise NotImplementedError(f"{self.name} does not support historical forecast")


__all__ = ["WeatherProvider"]
