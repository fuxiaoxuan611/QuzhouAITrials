"""Pydantic transport models that preserve canonical schema v1 names."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class CanonicalDecisionRequestModel(BaseModel):
    """HTTP parsing shell for the one canonical request representation.

    Nested scientific validation remains owned by
    ``serving.schemas.normalize_decision_request``. The transport model uses
    the same top-level names and does not introduce API-specific aliases.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    request: dict[str, Any]
    location: dict[str, Any]
    crop: dict[str, Any]
    query_date: date | str
    management: dict[str, Any]
    observations: dict[str, Any] | None = None
    weather: dict[str, Any]
    decision_context: dict[str, Any]

    def to_canonical_payload(self) -> dict[str, Any]:
        """Return the sole canonical mapping consumed by ``DecisionEngine``."""

        return self.model_dump(mode="python")


class WeatherContextRequestModel(BaseModel):
    """Transport model for the optional external weather tool endpoint."""

    model_config = ConfigDict(extra="forbid")

    location: dict[str, Any]
    query_date: date | str
    sowing_date: date | str | None = None
    forecast_horizon_days: int = 7
    decision_mode: str = "auto"
    provider: str | None = None
    as_of: datetime | str | None = None


__all__ = ["CanonicalDecisionRequestModel", "WeatherContextRequestModel"]
