"""Calendar-date management events used by dynamic WOFOST replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping


DEFAULT_N_RECOVERY = 0.7
DEFAULT_F_NH4N = 0.5
DEFAULT_F_NO3N = 0.5
DEFAULT_APPLICATION_DEPTH_CM = 10.0
DEFAULT_IRRIGATION_EFFICIENCY = 0.8


class ManagementEventError(ValueError):
    """Raised when a date-based management event is invalid."""

    code = "MANAGEMENT_EVENT_INVALID"


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManagementEventError(f"{field} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ManagementEventError(f"{field} must be finite and {'>= 0' if nonnegative else 'valid'}")
    return result


@dataclass(frozen=True)
class FertilizerEvent:
    date: date
    n_rate_kg_ha: float
    n_recovery: float = DEFAULT_N_RECOVERY
    f_nh4n: float = DEFAULT_F_NH4N
    f_no3n: float = DEFAULT_F_NO3N
    application_depth_cm: float = DEFAULT_APPLICATION_DEPTH_CM
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.date, date):
            raise ManagementEventError("fertilizer date must be a date")
        for field in ("n_rate_kg_ha", "n_recovery", "f_nh4n", "f_no3n", "application_depth_cm"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ManagementEventError(f"fertilizer {field} must be finite")
        if self.n_rate_kg_ha < 0 or self.application_depth_cm <= 0:
            raise ManagementEventError("fertilizer rate must be >= 0 and depth must be > 0")
        if not 0 <= self.n_recovery <= 1 or not 0 <= self.f_nh4n <= 1 or not 0 <= self.f_no3n <= 1:
            raise ManagementEventError("fertilizer fractions must be in [0, 1]")
        if not math.isclose(self.f_nh4n + self.f_no3n, 1.0, abs_tol=1e-6):
            raise ManagementEventError("f_nh4n + f_no3n must equal 1")

    @classmethod
    def from_value(cls, value: Any, index: int = 0) -> "FertilizerEvent":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ManagementEventError(f"fertilization_history[{index}] must be an object")
        try:
            event_date = value["date"]
            if isinstance(event_date, str):
                event_date = date.fromisoformat(event_date)
            provided = []
            mapping = {
                "n_recovery": "n_recovery",
                "f_nh4n": "f_nh4n",
                "f_no3n": "f_no3n",
                "application_depth_cm": "application_depth_cm",
            }
            kwargs: dict[str, Any] = {"date": event_date, "n_rate_kg_ha": value["n_rate_kg_ha"]}
            for key, dest in mapping.items():
                if key in value and value[key] is not None:
                    kwargs[dest] = value[key]
                else:
                    provided.append(key)
            assumptions = tuple(f"{key}=default" for key in provided)
            kwargs["assumptions"] = assumptions
            return cls(**kwargs)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ManagementEventError):
                raise
            raise ManagementEventError(f"invalid fertilizer event {index}: {exc}") from exc

    def to_pcse_timed_event(self) -> dict[str, Any]:
        # The active CN-Maize soil wrapper is SNOMIN.  Its
        # ``apply_n_snomin`` signal consumes ``amount`` and the SNOMIN
        # composition/depth fields below; it does not consume the legacy
        # ``apply_n`` fields ``N_amount``/``N_recovery``.  Keep
        # ``n_recovery`` in the canonical/audit representation, but do not
        # send an inert parameter that could be mistaken for active physics.
        params = {
            "amount": float(self.n_rate_kg_ha),
            "application_depth": float(self.application_depth_cm),
            "cnratio": 0.0,
            "f_orgmat": 0.0,
            "f_NH4N": float(self.f_nh4n),
            "f_NO3N": float(self.f_no3n),
            "initial_age": 0.0,
        }
        return {
            "event_signal": "apply_n_snomin",
            "name": "dynamic fertilizer application",
            "comment": "Canonical date-based fertilizer event; model defaults are explicit.",
            "events_table": [{self.date: params}],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "n_rate_kg_ha": float(self.n_rate_kg_ha),
            "n_recovery": float(self.n_recovery),
            "f_nh4n": float(self.f_nh4n),
            "f_no3n": float(self.f_no3n),
            "application_depth_cm": float(self.application_depth_cm),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class IrrigationEvent:
    date: date
    amount_mm: float
    efficiency: float | None = None
    amount_basis: str = "gross"
    method: str | None = None
    duration_min: float | None = None
    notes: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.date, date):
            raise ManagementEventError("irrigation date must be a date")
        if self.amount_basis not in {"gross", "effective"}:
            raise ManagementEventError("irrigation amount_basis must be gross or effective")
        if self.amount_mm < 0 or not math.isfinite(float(self.amount_mm)):
            raise ManagementEventError("irrigation amount_mm must be finite and >= 0")
        if self.efficiency is not None and not 0 < self.efficiency <= 1:
            raise ManagementEventError("irrigation efficiency must be in (0, 1]")
        if self.duration_min is not None and self.duration_min < 0:
            raise ManagementEventError("irrigation duration_min must be >= 0")

    @classmethod
    def from_value(cls, value: Any, index: int = 0) -> "IrrigationEvent":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ManagementEventError(f"irrigation_history[{index}] must be an object")
        event_date = value.get("date")
        if isinstance(event_date, str):
            try:
                event_date = date.fromisoformat(event_date)
            except ValueError as exc:
                raise ManagementEventError(f"invalid irrigation date {event_date!r}") from exc
        efficiency = value.get("efficiency")
        amount_basis = value.get("amount_basis", "gross")
        assumptions = []
        if efficiency is None:
            efficiency = DEFAULT_IRRIGATION_EFFICIENCY
            assumptions.append("IRRIGATION_EFFICIENCY_ASSUMED")
        if "amount_basis" not in value:
            assumptions.append("IRRIGATION_AMOUNT_BASIS_ASSUMED_GROSS")
        try:
            return cls(
                date=event_date,
                amount_mm=_number(value.get("amount_mm"), f"irrigation_history[{index}].amount_mm", nonnegative=True),
                efficiency=_number(efficiency, f"irrigation_history[{index}].efficiency"),
                amount_basis=amount_basis,
                method=value.get("method"),
                duration_min=(None if value.get("duration_min") is None else _number(value["duration_min"], "duration_min", nonnegative=True)),
                notes=value.get("notes"),
                assumptions=tuple(assumptions),
            )
        except ManagementEventError:
            raise
        except (TypeError, ValueError) as exc:
            raise ManagementEventError(f"invalid irrigation event {index}: {exc}") from exc

    @property
    def pcse_amount_cm(self) -> float:
        """Return the gross PCSE amount in cm water depth."""

        amount_cm = float(self.amount_mm) / 10.0
        if self.amount_basis == "effective":
            assert self.efficiency is not None
            return amount_cm / float(self.efficiency)
        return amount_cm

    def to_pcse_timed_event(self) -> dict[str, Any]:
        return {
            "event_signal": "irrigate",
            "name": "dynamic irrigation application",
            "comment": "Canonical date-based irrigation event.",
            "events_table": [
                {
                    self.date: {
                        "amount": self.pcse_amount_cm,
                        "efficiency": float(self.efficiency or DEFAULT_IRRIGATION_EFFICIENCY),
                    }
                }
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "amount_mm": float(self.amount_mm),
            "efficiency": float(self.efficiency) if self.efficiency is not None else None,
            "amount_basis": self.amount_basis,
            "method": self.method,
            "duration_min": self.duration_min,
            "notes": self.notes,
            "assumptions": list(self.assumptions),
            "pcse_amount_cm_gross": self.pcse_amount_cm,
        }


@dataclass(frozen=True)
class ManagementTimeline:
    fertilizers: tuple[FertilizerEvent, ...] = ()
    irrigations: tuple[IrrigationEvent, ...] = ()

    @classmethod
    def from_values(
        cls,
        fertilizers: Iterable[Any] = (),
        irrigations: Iterable[Any] = (),
    ) -> "ManagementTimeline":
        fert = tuple(FertilizerEvent.from_value(value, i) for i, value in enumerate(fertilizers))
        irr = tuple(IrrigationEvent.from_value(value, i) for i, value in enumerate(irrigations))
        if len({item.date for item in fert}) != len(fert):
            raise ManagementEventError("duplicate fertilizer dates are not supported")
        if len({item.date for item in irr}) != len(irr):
            raise ManagementEventError("duplicate irrigation dates are not supported")
        return cls(tuple(sorted(fert, key=lambda item: item.date)), tuple(sorted(irr, key=lambda item: item.date)))

    def to_pcse_timed_events(self) -> list[dict[str, Any]]:
        return [item.to_pcse_timed_event() for item in (*self.fertilizers, *self.irrigations)]

    def events_on(self, event_date: date) -> tuple[Any, ...]:
        return tuple(item for item in (*self.fertilizers, *self.irrigations) if item.date == event_date)

    def to_dict(self) -> dict[str, Any]:
        assumptions = sorted({assumption for item in (*self.fertilizers, *self.irrigations) for assumption in item.assumptions})
        return {
            "fertilization_history": [item.to_dict() for item in self.fertilizers],
            "irrigation_history": [item.to_dict() for item in self.irrigations],
            "assumptions": assumptions,
        }


__all__ = [
    "DEFAULT_APPLICATION_DEPTH_CM",
    "DEFAULT_F_NH4N",
    "DEFAULT_F_NO3N",
    "DEFAULT_IRRIGATION_EFFICIENCY",
    "DEFAULT_N_RECOVERY",
    "FertilizerEvent",
    "IrrigationEvent",
    "ManagementEventError",
    "ManagementTimeline",
]
