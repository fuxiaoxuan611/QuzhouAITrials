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
        )


__all__ = ["ServiceSettings"]
