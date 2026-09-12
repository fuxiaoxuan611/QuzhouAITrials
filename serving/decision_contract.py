"""CanonicalDecisionResult v1 shaping and JSON-safety helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path, PureWindowsPath
import re
from typing import Any

import numpy as np

from .errors import serialize_decision_error


DECISION_RESULT_SCHEMA_VERSION = "1.1"

# These keys are deliberately explicit.  A caller can rely on them being
# present even when a decision is not due or a recommendation is unavailable.
DECISION_RESULT_TOP_LEVEL_KEYS = (
    "schema_version",
    "request_id",
    "status",
    "decision_due",
    "state_date",
    "policy_observation_date",
    "previous_decision_date",
    "next_decision_date",
    "action_application_date",
    "recommendation",
    "model_state",
    "management",
    "observations",
    "model_metadata",
    "warnings",
    "weather_context",
    "forecast",
    "projected_next_decision",
    "scenario_evaluation",
    "weather_risk",
    "operation_advice",
    "critic_input",
)


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/date values to JSON-native values."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _recommendation(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "decision_type",
        "action_index",
        "n_rate_kg_ha",
        "constraint_violation",
        "constraint_details",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"recommendation missing contract fields: {sorted(missing)}")
    return json_safe(
        {key: value[key] for key in (
            "decision_type",
            "action_index",
            "n_rate_kg_ha",
            "constraint_violation",
            "constraint_details",
        )}
    )


_PATH_KEY = re.compile(r"(?:^|_)path$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _metadata_without_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep public identifiers while removing local filesystem paths."""

    result: dict[str, Any] = {}
    for key, value in metadata.items():
        key = str(key)
        if _PATH_KEY.search(key):
            if isinstance(value, (str, Path)):
                value_text = str(value)
                identifier_key = key[:-5] + "_identifier"
                identifier = (
                    PureWindowsPath(value_text).name
                    if _WINDOWS_ABSOLUTE_PATH.match(value_text)
                    else Path(value_text).name
                )
                result[identifier_key] = identifier
            continue
        if isinstance(value, str) and _WINDOWS_ABSOLUTE_PATH.match(value):
            continue
        result[key] = value
    return result


def sanitize_model_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return metadata safe for public capabilities/result responses."""

    return json_safe(_metadata_without_paths(metadata or {}))


def build_decision_result(
    *,
    request_id: str | None,
    decision_due: bool,
    state_date: Any,
    policy_observation_date: Any = None,
    previous_decision_date: Any = None,
    next_decision_date: Any = None,
    action_application_date: Any = None,
    recommendation: dict[str, Any] | None = None,
    model_state: Any = None,
    management: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
    warnings: list[Any] | None = None,
    weather_context: dict[str, Any] | None = None,
    forecast: dict[str, Any] | None = None,
    projected_next_decision: dict[str, Any] | None = None,
    scenario_evaluation: list[Any] | None = None,
    weather_risk: list[Any] | None = None,
    operation_advice: dict[str, Any] | None = None,
    critic_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a stable successful result with all v1 core keys."""

    result = {
        "schema_version": DECISION_RESULT_SCHEMA_VERSION,
        "request_id": request_id,
        "status": "ok",
        "decision_due": bool(decision_due),
        "state_date": state_date,
        "policy_observation_date": policy_observation_date,
        "previous_decision_date": previous_decision_date,
        "next_decision_date": next_decision_date,
        "action_application_date": action_application_date,
        "recommendation": _recommendation(recommendation),
        "model_state": model_state,
        "management": management or {},
        "observations": observations or {
            "received": False,
            "applied_to_policy": False,
            "applied_to_wofost_state": False,
            "fusion_audit": {},
        },
        "model_metadata": sanitize_model_metadata(model_metadata),
        "warnings": list(warnings or []),
        "weather_context": weather_context,
        "forecast": forecast,
        "projected_next_decision": projected_next_decision,
        "scenario_evaluation": list(scenario_evaluation or []),
        "weather_risk": list(weather_risk or []),
        "operation_advice": operation_advice,
        # Additive, JSON-safe input prepared for an external LLM critic. It
        # does not invoke the LLM or alter the current recommendation.
        "critic_input": critic_input,
    }
    return json_safe(result)


__all__ = [
    "DECISION_RESULT_SCHEMA_VERSION",
    "DECISION_RESULT_TOP_LEVEL_KEYS",
    "build_decision_result",
    "json_safe",
    "sanitize_model_metadata",
    "serialize_decision_error",
]
