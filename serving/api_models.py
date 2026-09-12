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


class CriticValidationRequestModel(BaseModel):
    """Transport schema for validating an external critic response."""

    model_config = ConfigDict(extra="forbid")

    rl_candidate_n_kg_ha: float
    critic_output: dict[str, Any] | str
    # Accepted for workflow correlation/future audit use. Validation is still
    # determined exclusively by the RL candidate and critic_output fields.
    critic_input: dict[str, Any] | None = None


class CriticValidationResponseModel(BaseModel):
    """Stable response schema for the deterministic critic validator."""

    model_config = ConfigDict(extra="forbid")

    validation_passed: bool
    critic_validation_failed: bool
    verdict: str
    rl_candidate_n_kg_ha: float
    final_n_kg_ha: float
    execution_status: str
    reason_codes: list[str]
    reasons: list[str]
    confidence: str
    validation_errors: list[str]


__all__ = [
    "CanonicalDecisionRequestModel",
    "CriticValidationRequestModel",
    "CriticValidationResponseModel",
    "WeatherContextRequestModel",
]
