"""Canonical, dependency-free input schema for future serving layers."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    """Raised when a canonical decision request is invalid."""


def _parse_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise SchemaValidationError(
                f"{field}: expected ISO date YYYY-MM-DD, got {value!r}"
            ) from exc
    raise SchemaValidationError(f"{field}: expected ISO date YYYY-MM-DD")


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SchemaValidationError(f"{field}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{field}: expected a finite number")
    return result


def _nonnegative(value: Any, field: str) -> float:
    result = _number(value, field)
    if result < 0:
        raise SchemaValidationError(f"{field}: must be >= 0")
    return result


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field}: expected an object")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SchemaValidationError(
            f"{field}: unsupported field(s): {', '.join(unknown)}"
        )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field}: expected a non-empty string")
    return value


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float

    @classmethod
    def from_mapping(cls, value: Any) -> "Location":
        data = _required_mapping(value, "location")
        _reject_unknown(data, {"latitude", "longitude"}, "location")
        latitude = _number(data.get("latitude"), "location.latitude")
        longitude = _number(data.get("longitude"), "location.longitude")
        if not -90 <= latitude <= 90:
            raise SchemaValidationError("location.latitude: must be in [-90, 90]")
        if not -180 <= longitude <= 180:
            raise SchemaValidationError("location.longitude: must be in [-180, 180]")
        return cls(latitude=latitude, longitude=longitude)

    def to_dict(self) -> dict[str, Any]:
        return {"latitude": self.latitude, "longitude": self.longitude}


@dataclass(frozen=True)
class Crop:
    name: str
    cultivar: str
    sowing_date: date

    @classmethod
    def from_mapping(cls, value: Any) -> "Crop":
        data = _required_mapping(value, "crop")
        _reject_unknown(data, {"name", "cultivar", "sowing_date"}, "crop")
        return cls(
            name=_required_string(data.get("name"), "crop.name"),
            cultivar=_required_string(data.get("cultivar"), "crop.cultivar"),
            sowing_date=_parse_date(data.get("sowing_date"), "crop.sowing_date"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cultivar": self.cultivar,
            "sowing_date": self.sowing_date.isoformat(),
        }


@dataclass(frozen=True)
class FertilizationEvent:
    date: date
    n_rate_kg_ha: float

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "FertilizationEvent":
        field = f"management.fertilization_history[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(data, {"date", "n_rate_kg_ha"}, field)
        return cls(
            date=_parse_date(
                data.get("date"),
                f"management.fertilization_history[{index}].date",
            ),
            n_rate_kg_ha=_nonnegative(
                data.get("n_rate_kg_ha"),
                f"management.fertilization_history[{index}].n_rate_kg_ha",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "n_rate_kg_ha": self.n_rate_kg_ha,
        }


@dataclass(frozen=True)
class IrrigationEvent:
    date: date
    amount_mm: float

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "IrrigationEvent":
        field = f"management.irrigation_history[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(data, {"date", "amount_mm"}, field)
        return cls(
            date=_parse_date(
                data.get("date"),
                f"management.irrigation_history[{index}].date",
            ),
            amount_mm=_nonnegative(
                data.get("amount_mm"),
                f"management.irrigation_history[{index}].amount_mm",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"date": self.date.isoformat(), "amount_mm": self.amount_mm}


@dataclass(frozen=True)
class ManagementHistory:
    fertilization_history: tuple[FertilizationEvent, ...]
    irrigation_history: tuple[IrrigationEvent, ...]
    fertilization_history_complete: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "ManagementHistory":
        data = _required_mapping(value, "management")
        _reject_unknown(
            data,
            {
                "fertilization_history",
                "irrigation_history",
                "fertilization_history_complete",
            },
            "management",
        )
        fertilization = data.get("fertilization_history", [])
        irrigation = data.get("irrigation_history", [])
        if not isinstance(fertilization, (list, tuple)):
            raise SchemaValidationError(
                "management.fertilization_history: expected an array"
            )
        if not isinstance(irrigation, (list, tuple)):
            raise SchemaValidationError(
                "management.irrigation_history: expected an array"
            )
        complete = data.get("fertilization_history_complete")
        if not isinstance(complete, bool):
            raise SchemaValidationError(
                "management.fertilization_history_complete: expected boolean"
            )
        return cls(
            fertilization_history=tuple(
                FertilizationEvent.from_mapping(item, index)
                for index, item in enumerate(fertilization)
            ),
            irrigation_history=tuple(
                IrrigationEvent.from_mapping(item, index)
                for index, item in enumerate(irrigation)
            ),
            fertilization_history_complete=complete,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fertilization_history": [item.to_dict() for item in self.fertilization_history],
            "irrigation_history": [item.to_dict() for item in self.irrigation_history],
            "fertilization_history_complete": self.fertilization_history_complete,
        }


@dataclass(frozen=True)
class ObservationSnapshot:
    date: date
    lai: float | None = None
    soil_water: float | None = None
    no3: float | None = None
    nh4: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "ObservationSnapshot":
        data = _required_mapping(value, "observations")
        _reject_unknown(
            data,
            {"date", "lai", "soil_water", "no3", "nh4"},
            "observations",
        )
        lai = None if data.get("lai") is None else _nonnegative(data["lai"], "observations.lai")
        soil_water = None if data.get("soil_water") is None else _number(data["soil_water"], "observations.soil_water")
        no3 = None if data.get("no3") is None else _number(data["no3"], "observations.no3")
        nh4 = None if data.get("nh4") is None else _number(data["nh4"], "observations.nh4")
        return cls(
            date=_parse_date(data.get("date"), "observations.date"),
            lai=lai,
            soil_water=soil_water,
            no3=no3,
            nh4=nh4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "lai": self.lai,
            "soil_water": self.soil_water,
            "no3": self.no3,
            "nh4": self.nh4,
        }


@dataclass(frozen=True)
class DecisionRequest:
    location: Location
    crop: Crop
    query_date: date
    management: ManagementHistory
    observations: ObservationSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "location": self.location.to_dict(),
            "crop": self.crop.to_dict(),
            "query_date": self.query_date.isoformat(),
            "management": self.management.to_dict(),
        }
        if self.observations is not None:
            result["observations"] = self.observations.to_dict()
        return result


def normalize_decision_request(payload: Mapping[str, Any]) -> DecisionRequest:
    """Parse and strictly validate the canonical request shape."""

    data = _required_mapping(payload, "request")
    _reject_unknown(
        data,
        {"location", "crop", "query_date", "management", "observations"},
        "request",
    )
    location = Location.from_mapping(data.get("location"))
    crop = Crop.from_mapping(data.get("crop"))
    query_date = _parse_date(data.get("query_date"), "query_date")
    if query_date < crop.sowing_date:
        raise SchemaValidationError(
            "query_date: must be >= crop.sowing_date"
        )
    management = ManagementHistory.from_mapping(data.get("management"))
    observations = None
    if data.get("observations") is not None:
        observations = ObservationSnapshot.from_mapping(data["observations"])
    return DecisionRequest(
        location=location,
        crop=crop,
        query_date=query_date,
        management=management,
        observations=observations,
    )


__all__ = [
    "Crop",
    "DecisionRequest",
    "FertilizationEvent",
    "IrrigationEvent",
    "Location",
    "ManagementHistory",
    "ObservationSnapshot",
    "SchemaValidationError",
    "normalize_decision_request",
]
