"""Environment-driven settings for the local decision service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    """Runtime configuration without embedding machine-specific paths."""

    config_path: str | Path = "configs/CN-Maize.yaml"
    model_path: str | Path | None = None
    env_stats_path: str | Path | None = None
    device: str = "auto"
    policy_validation_status: str = "unknown"
    model_id: str | None = None
    season_pre_sowing_offset_days: int = 31
    season_crop_duration_days: int = 128
    season_max_duration: int = 400
    weather_timezone: str = "Asia/Shanghai"
    weather_provider: str = "openmeteo"
    forecast_default_days: int = 7
    weather_heavy_rain_mm: float = 25.0
    weather_heat_tmax_c: float = 35.0
    weather_dry_days: int = 5

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        def value(name: str, default: str | None = None) -> str | None:
            configured = os.getenv(name)
            return default if configured is None or configured == "" else configured

        return cls(
            config_path=value("QUZHOU_CONFIG_PATH", "configs/CN-Maize.yaml"),
            model_path=value("QUZHOU_MODEL_PATH"),
            env_stats_path=value("QUZHOU_ENV_STATS_PATH"),
            device=value("QUZHOU_DEVICE", "auto") or "auto",
            policy_validation_status=(
                value("QUZHOU_POLICY_VALIDATION_STATUS", "unknown") or "unknown"
            ),
            model_id=value("QUZHOU_MODEL_ID"),
            season_pre_sowing_offset_days=int(value("QUZHOU_PRE_SOWING_OFFSET_DAYS", "31") or 31),
            season_crop_duration_days=int(value("QUZHOU_CROP_DURATION_DAYS", "128") or 128),
            season_max_duration=int(value("QUZHOU_MAX_DURATION", "400") or 400),
            weather_timezone=value("QUZHOU_WEATHER_TIMEZONE", "Asia/Shanghai") or "Asia/Shanghai",
            weather_provider=value("QUZHOU_WEATHER_PROVIDER", "openmeteo") or "openmeteo",
            forecast_default_days=int(value("QUZHOU_FORECAST_DEFAULT_DAYS", "7") or 7),
            weather_heavy_rain_mm=float(value("QUZHOU_HEAVY_RAIN_MM", "25") or 25),
            weather_heat_tmax_c=float(value("QUZHOU_HEAT_TMAX_C", "35") or 35),
            weather_dry_days=int(value("QUZHOU_DRY_DAYS", "5") or 5),
        )


__all__ = ["ServiceSettings"]
