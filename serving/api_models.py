"""Pydantic transport models that preserve canonical schema v1 names."""

from __future__ import annotations

from datetime import date
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


__all__ = ["CanonicalDecisionRequestModel"]
