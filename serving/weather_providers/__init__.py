"""Weather provider adapters used by :mod:`serving.weather_service`."""

from .base import WeatherProvider
from .openmeteo import OpenMeteoWeatherProvider
from .qweather import QWeatherProvider
from .timeline import TimelineWeatherDataProvider

__all__ = [
    "OpenMeteoWeatherProvider",
    "QWeatherProvider",
    "TimelineWeatherDataProvider",
    "WeatherProvider",
]
