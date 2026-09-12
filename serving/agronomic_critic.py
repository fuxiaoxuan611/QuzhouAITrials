"""Contracts and deterministic validation for an external agronomic critic.

The critic is intentionally an external decision-review step (for example a
FastGPT LLM node).  This module does not call an LLM, run WOFOST, or change
the RL action space.  It prepares a JSON-safe input object and validates the
small, safety-constrained output contract returned by the critic.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
import numbers
from typing import Any, Mapping

from .decision_contract import json_safe


CRITIC_VERDICTS = frozenset({"ACCEPT", "REDUCE", "DEFER", "REJECT"})
CRITIC_EXECUTION_STATUSES = frozenset({"proceed", "defer", "reject"})
CRITIC_CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
RL_N_RATES_KG_HA = frozenset(range(0, 151, 10))
CRITIC_OUTPUT_FIELDS = frozenset(
    {
        "verdict",
        "rl_candidate_n_kg_ha",
        "final_n_kg_ha",
        "execution_status",
        "reason_codes",
        "reasons",
        "confidence",
    }
)


class CriticValidationError(ValueError):
    """Raised when an agronomic critic output violates its safety contract."""


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _state_value(state: Mapping[str, Any], name: str) -> float | None:
    """Read a numeric WOFOST feature without inventing missing values."""

    aliases = {
        "Nsurp": ("Nsurp", "N_SURPLUS", "n_surplus"),
        "NLOSSCUM": ("NLOSSCUM", "NLOSS", "n_loss"),
        "NuptakeTotal": ("NuptakeTotal", "NUPTAKET", "n_uptake"),
    }
    for key in aliases.get(name, (name,)):
        if key in state:
            return _finite_or_none(state[key])
    return None


def prepare_critic_input(
    *,
    query_date: date | datetime | str,
    decision_date: date | datetime | str | None,
    projected_application_date: date | datetime | str | None,
    simulated_state: Mapping[str, Any] | None,
    observation_fusion: Mapping[str, Any] | None,
    user_observations: Any = None,
    rl_candidate_action: int | None = None,
    rl_candidate_n_rate_kg_ha: float | None = None,
    fertilization_history: Any = None,
    irrigation_history: Any = None,
    weather_risk: Any = None,
    forecast: Mapping[str, Any] | None = None,
    decision_due: bool = False,
) -> dict[str, Any]:
    """Build the stable JSON object supplied to an external LLM critic.

    ``simulated_state`` is the state produced by WOFOST.  Observation fusion
    is passed separately so a critic can see simulated, observed, and policy
    input values without being told that WOFOST itself was corrected.
    """

    state = dict(simulated_state or {})
    candidate = None
    if rl_candidate_action is not None or rl_candidate_n_rate_kg_ha is not None:
        candidate = {
            "action_index": None if rl_candidate_action is None else int(rl_candidate_action),
            "n_rate_kg_ha": _finite_or_none(rl_candidate_n_rate_kg_ha),
        }
    state_features = {
        name: _state_value(state, name)
        for name in (
            "DVS",
            "LAI",
            "TAGP",
            "NuptakeTotal",
            "NO3",
            "NH4",
            "WC",
            "NLOSSCUM",
            "NUE",
            "Nsurp",
        )
    }
    return json_safe(
        {
            "query_date": _iso(query_date),
            "decision_date": _iso(decision_date),
            "projected_application_date": _iso(projected_application_date),
            "decision_due": bool(decision_due),
            "wofost_simulated_state": state,
            "state_features": state_features,
            "observation_fusion": dict(observation_fusion or {}),
            "user_observations": user_observations,
            "rl_candidate_action": candidate,
            "rl_candidate_n_rate_kg_ha": (
                None
                if rl_candidate_n_rate_kg_ha is None
                else float(rl_candidate_n_rate_kg_ha)
            ),
            "fertilization_history": list(fertilization_history or []),
            "irrigation_history": list(irrigation_history or []),
            "weather_risk": list(weather_risk or []),
            "forecast": forecast,
        }
    )


def _rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CriticValidationError(f"{field}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CriticValidationError(f"{field}: expected a finite number")
    if result not in RL_N_RATES_KG_HA:
        raise CriticValidationError(
            f"{field}: must be one of 0, 10, ..., 150 kg N/ha"
        )
    return result


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CriticValidationError(f"{field}: expected an array of strings")
    return list(value)


def _parse_output(raw_output: Any) -> Mapping[str, Any]:
    if isinstance(raw_output, str):
        try:
            raw_output = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise CriticValidationError("critic output is not valid JSON") from exc
    if not isinstance(raw_output, Mapping):
        raise CriticValidationError("critic output must be a JSON object")
    return raw_output


def validate_critic_output(
    raw_output: Mapping[str, Any] | str,
    *,
    rl_candidate_n_rate_kg_ha: float | None = None,
) -> dict[str, Any]:
    """Validate and normalize one critic response.

    The supplied candidate rate is authoritative.  If present in the LLM
    output it must match it exactly, preventing the critic from silently
    changing the policy candidate before applying the monotonic safety rules.
    """

    data = _parse_output(raw_output)
    unknown = sorted(set(data) - CRITIC_OUTPUT_FIELDS)
    if unknown:
        raise CriticValidationError(
            f"critic output: unsupported field(s): {', '.join(unknown)}"
        )
    missing = sorted(CRITIC_OUTPUT_FIELDS - set(data))
    if missing:
        raise CriticValidationError(
            f"critic output: missing field(s): {', '.join(missing)}"
        )

    verdict = data["verdict"]
    if verdict not in CRITIC_VERDICTS:
        raise CriticValidationError(
            "verdict: expected ACCEPT, REDUCE, DEFER, or REJECT"
        )
    candidate = _rate(data["rl_candidate_n_kg_ha"], "rl_candidate_n_kg_ha")
    if rl_candidate_n_rate_kg_ha is not None:
        expected = _rate(rl_candidate_n_rate_kg_ha, "rl_candidate_n_rate_kg_ha")
        if not math.isclose(candidate, expected, abs_tol=1e-9):
            raise CriticValidationError(
                "rl_candidate_n_kg_ha: does not match the policy candidate"
            )
    final_rate = _rate(data["final_n_kg_ha"], "final_n_kg_ha")
    expected_status = {
        "ACCEPT": "proceed",
        "REDUCE": "proceed",
        "DEFER": "defer",
        "REJECT": "reject",
    }[verdict]
    if data["execution_status"] != expected_status:
        raise CriticValidationError(
            f"execution_status: {verdict} requires {expected_status!r}"
        )
    if verdict == "ACCEPT" and not math.isclose(final_rate, candidate, abs_tol=1e-9):
        raise CriticValidationError("ACCEPT requires final_n_kg_ha == candidate")
    if verdict == "REDUCE" and not final_rate < candidate:
        raise CriticValidationError("REDUCE requires final_n_kg_ha < candidate")
    if verdict == "DEFER" and not math.isclose(final_rate, candidate, abs_tol=1e-9):
        raise CriticValidationError("DEFER requires final_n_kg_ha == candidate")
    if verdict == "REJECT" and not math.isclose(final_rate, 0.0, abs_tol=1e-9):
        raise CriticValidationError("REJECT requires final_n_kg_ha == 0")

    confidence = data["confidence"]
    if confidence not in CRITIC_CONFIDENCE_LEVELS:
        raise CriticValidationError("confidence: expected low, medium, or high")
    reason_codes = _string_list(data["reason_codes"], "reason_codes")
    reasons = _string_list(data["reasons"], "reasons")
    # Deferring a zero-N candidate cannot change the amount or represent a
    # meaningful nitrogen operation. Normalize it to an explicit no-action
    # acceptance while keeping the action at zero.
    if verdict == "DEFER" and candidate == 0.0:
        verdict = "ACCEPT"
        execution_status = "proceed"
        reason_codes.append("NO_ACTION_REQUIRED")
        reasons.append("The RL candidate is already zero; no nitrogen action is required.")
    else:
        execution_status = data["execution_status"]
    return {
        "verdict": verdict,
        "rl_candidate_n_kg_ha": candidate,
        "final_n_kg_ha": final_rate,
        "execution_status": execution_status,
        "reason_codes": reason_codes,
        "reasons": reasons,
        "confidence": confidence,
        "critic_validation_failed": False,
    }


def critic_validation_fallback(
    rl_candidate_n_rate_kg_ha: float,
    *,
    reason: str = "Critic output failed deterministic validation.",
) -> dict[str, Any]:
    """Return the fail-safe policy-preserving fallback for invalid output."""

    candidate = _rate(rl_candidate_n_rate_kg_ha, "rl_candidate_n_rate_kg_ha")
    return {
        "verdict": "ACCEPT",
        "rl_candidate_n_kg_ha": candidate,
        "final_n_kg_ha": candidate,
        "execution_status": "proceed",
        "reason_codes": ["CRITIC_VALIDATION_FAILED"],
        "reasons": [str(reason)],
        "confidence": "low",
        "critic_validation_failed": True,
    }


def validate_rl_candidate_n_rate(value: Any) -> float:
    """Validate the policy candidate using the same action-space rule."""

    return _rate(value, "rl_candidate_n_rate_kg_ha")


def validate_or_fallback(
    raw_output: Mapping[str, Any] | str,
    *,
    rl_candidate_n_rate_kg_ha: float,
) -> dict[str, Any]:
    """Validate critic output, preserving the RL candidate on any failure."""

    try:
        return validate_critic_output(
            raw_output,
            rl_candidate_n_rate_kg_ha=rl_candidate_n_rate_kg_ha,
        )
    except (CriticValidationError, TypeError, ValueError) as exc:
        return critic_validation_fallback(
            rl_candidate_n_rate_kg_ha,
            reason=str(exc),
        )


__all__ = [
    "CRITIC_CONFIDENCE_LEVELS",
    "CRITIC_EXECUTION_STATUSES",
    "CRITIC_OUTPUT_FIELDS",
    "CRITIC_VERDICTS",
    "CriticValidationError",
    "RL_N_RATES_KG_HA",
    "critic_validation_fallback",
    "prepare_critic_input",
    "validate_rl_candidate_n_rate",
    "validate_critic_output",
    "validate_or_fallback",
]
