"""Convert canonical fertilizer history to the current CN-Maize actions."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .rl_inference import N_KG_HA_PER_ACTION_UNIT
from .schemas import (
    DecisionRequest,
    FertilizationEvent,
    SchemaValidationError,
    normalize_decision_request,
)


class ManagementHistoryError(ValueError):
    """Raised when canonical management cannot be represented by CN-Maize."""


def _event_from_value(value: Any, index: int) -> FertilizationEvent:
    if isinstance(value, FertilizationEvent):
        return value
    try:
        return FertilizationEvent.from_mapping(value, index)
    except SchemaValidationError as exc:
        raise ManagementHistoryError(str(exc)) from exc


def _exact_action_index(n_rate_kg_ha: float, action_count: int, multiplier: float) -> int:
    try:
        units = (
            Decimal(str(n_rate_kg_ha))
            / Decimal(str(multiplier))
            / Decimal(str(N_KG_HA_PER_ACTION_UNIT))
        )
    except (InvalidOperation, ValueError) as exc:
        raise ManagementHistoryError(
            f"UNREPRESENTABLE_N_RATE: {n_rate_kg_ha!r}"
        ) from exc
    if units != units.to_integral_value():
        raise ManagementHistoryError(
            f"UNREPRESENTABLE_N_RATE: {n_rate_kg_ha} is not an exact 10 kg N/ha action"
        )
    action_index = int(units)
    if not 0 <= action_index < action_count:
        raise ManagementHistoryError(
            f"UNREPRESENTABLE_N_RATE: {n_rate_kg_ha} maps outside action indices "
            f"[0, {action_count - 1}]"
        )
    return action_index


def build_action_history(
    sowing_date: date,
    query_date: date,
    fertilization_history: Iterable[FertilizationEvent | dict[str, Any]],
    *,
    fertilization_history_complete: bool,
    timestep_days: int = 7,
    action_count: int = 16,
    action_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build every complete decision action required before ``query_date``.

    The current environment applies an action on the final PCSE day of each
    interval, while its info dictionary labels that interval by its start
    date.  Therefore this converter uses application dates
    ``sowing_date + timestep_days * (step + 1)``.  Multiple N events on one
    exact application date are safely combined because the environment emits a
    single total N signal on that day; the combined rate must still be exactly
    representable by the discrete action space.
    """

    if not isinstance(sowing_date, date) or not isinstance(query_date, date):
        raise ManagementHistoryError("INVALID_DATE: sowing_date and query_date must be dates")
    if query_date < sowing_date:
        raise ManagementHistoryError("QUERY_DATE_BEFORE_SOWING")
    if timestep_days <= 0:
        raise ManagementHistoryError("INVALID_TIMESTEP: must be > 0")
    if action_count <= 0:
        raise ManagementHistoryError("INVALID_ACTION_SPACE: action_count must be > 0")
    if not fertilization_history_complete:
        raise ManagementHistoryError(
            "INCOMPLETE_MANAGEMENT_HISTORY: fertilization history must be explicitly complete"
        )

    events = [_event_from_value(value, index) for index, value in enumerate(fertilization_history)]
    completed_steps, residual_days = divmod(
        (query_date - sowing_date).days,
        timestep_days,
    )
    decision_dates = [
        sowing_date + timedelta(days=timestep_days * (index + 1))
        for index in range(completed_steps)
    ]
    decision_date_to_index = {value: index for index, value in enumerate(decision_dates)}
    action_history = [0 for _ in range(completed_steps)]

    combined_by_date: dict[date, float] = {}
    for event in events:
        if event.date > query_date:
            raise ManagementHistoryError(
                f"FERTILIZATION_AFTER_QUERY: event={event.date}, query={query_date}"
            )
        if event.date not in decision_date_to_index:
            if decision_dates and decision_dates[-1] < event.date <= query_date:
                raise ManagementHistoryError(
                    "UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT: "
                    f"event={event.date}, last_decision_date={decision_dates[-1]}, "
                    f"query={query_date}"
                )
            raise ManagementHistoryError(
                f"UNREPRESENTABLE_FERTILIZATION_DATE: {event.date} is not a decision date"
            )
        combined_by_date[event.date] = combined_by_date.get(event.date, 0.0) + event.n_rate_kg_ha

    mapped_events = []
    for event_date, total_rate in sorted(combined_by_date.items()):
        action_index = _exact_action_index(total_rate, action_count, action_multiplier)
        step_index = decision_date_to_index[event_date]
        action_history[step_index] = action_index
        mapped_events.append(
            {
                "date": event_date.isoformat(),
                "n_rate_kg_ha": total_rate,
                "action_index": action_index,
                "completed_step": step_index,
            }
        )

    return {
        "action_history": action_history,
        "decision_dates": [value.isoformat() for value in decision_dates],
        "completed_steps": completed_steps,
        "residual_days": residual_days,
        "mapped_fertilization_events": mapped_events,
        "unsupported_fields": [],
        "warnings": [],
    }


def management_to_realtime_input(
    request: DecisionRequest | dict[str, Any],
    *,
    timestep_days: int = 7,
    action_count: int = 16,
    action_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Convert a canonical request into ``WOFOSTRealtimeEngine`` input.

    Observations are preserved as context only.  They are not assimilated
    into WOFOST in this version.  Non-empty custom irrigation is rejected
    because the current model still obtains irrigation from agromanagement.
    """

    if isinstance(request, dict):
        request = normalize_decision_request(request)
    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest or canonical mapping")
    if request.management.irrigation_history:
        raise ManagementHistoryError(
            "CUSTOM_IRRIGATION_NOT_YET_SUPPORTED: current WOFOST uses agromanagement irrigation"
        )

    result = build_action_history(
        request.crop.sowing_date,
        request.query_date,
        request.management.fertilization_history,
        fertilization_history_complete=request.management.fertilization_history_complete,
        timestep_days=timestep_days,
        action_count=action_count,
        action_multiplier=action_multiplier,
    )
    result.update(
        {
            "query_date": request.query_date.isoformat(),
            "sowing_date": request.crop.sowing_date.isoformat(),
            "irrigation_history": [],
            "irrigation_applied_to_model": False,
            "observations": (
                request.observations.to_dict()
                if request.observations is not None
                else None
            ),
            "observations_applied_to_model": False,
        }
    )
    return result


__all__ = [
    "ManagementHistoryError",
    "build_action_history",
    "management_to_realtime_input",
]
