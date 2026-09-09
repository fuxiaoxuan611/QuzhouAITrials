"""Canonical schema v1 for future FastGPT and HTTP integrations.

The dataclasses intentionally have no Pydantic dependency.  A future
FastAPI layer can map the same canonical field names onto Pydantic models.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"


class SchemaValidationError(ValueError):
    """Raised when a canonical decision request is invalid."""


class ObservationSource(str, Enum):
    FIELD_MEASUREMENT = "field_measurement"
    SENSOR = "sensor"
    LABORATORY = "laboratory"
    REMOTE_SENSING = "remote_sensing"
    USER_REPORTED = "user_reported"
    MODEL_ESTIMATE = "model_estimate"
    UNKNOWN = "unknown"


def _parse_datetime(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{field}: expected ISO datetime") from exc
    raise SchemaValidationError(f"{field}: expected ISO datetime")


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


def _optional_nonnegative(data: Mapping[str, Any], key: str, field: str) -> float | None:
    return None if data.get(key) is None else _nonnegative(data[key], field)


def _optional_number(data: Mapping[str, Any], key: str, field: str) -> float | None:
    return None if data.get(key) is None else _number(data[key], field)


def _optional_string(data: Mapping[str, Any], key: str, field: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field}: expected a string")
    return value


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


def _optional_bool(data: Mapping[str, Any], key: str, default: bool, field: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{field}: expected boolean")
    return value


@dataclass(frozen=True)
class RequestContext:
    request_id: str | None = None
    user_query: str | None = None
    as_of_datetime: datetime | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "RequestContext":
        data = _required_mapping(value, "request")
        _reject_unknown(data, {"request_id", "user_query", "as_of_datetime"}, "request")
        return cls(
            request_id=_optional_string(data, "request_id", "request.request_id"),
            user_query=_optional_string(data, "user_query", "request.user_query"),
            as_of_datetime=_parse_datetime(data.get("as_of_datetime"), "request.as_of_datetime"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_query": self.user_query,
            "as_of_datetime": self.as_of_datetime.isoformat() if self.as_of_datetime else None,
        }


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    field_id: str | None = None
    field_name: str | None = None
    elevation_m: float | None = None
    timezone: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "Location":
        data = _required_mapping(value, "location")
        _reject_unknown(
            data,
            {"field_id", "field_name", "latitude", "longitude", "elevation_m", "timezone"},
            "location",
        )
        latitude = _number(data.get("latitude"), "location.latitude")
        longitude = _number(data.get("longitude"), "location.longitude")
        if not -90 <= latitude <= 90:
            raise SchemaValidationError("location.latitude: must be in [-90, 90]")
        if not -180 <= longitude <= 180:
            raise SchemaValidationError("location.longitude: must be in [-180, 180]")
        return cls(
            latitude=latitude,
            longitude=longitude,
            field_id=_optional_string(data, "field_id", "location.field_id"),
            field_name=_optional_string(data, "field_name", "location.field_name"),
            elevation_m=_optional_number(data, "elevation_m", "location.elevation_m"),
            timezone=_optional_string(data, "timezone", "location.timezone"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation_m": self.elevation_m,
            "timezone": self.timezone,
        }


@dataclass(frozen=True)
class Crop:
    name: str
    cultivar: str
    sowing_date: date
    planting_density_plants_ha: float | None = None
    row_spacing_m: float | None = None
    expected_harvest_date: date | None = None
    phenology_stage_observed: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "Crop":
        data = _required_mapping(value, "crop")
        _reject_unknown(
            data,
            {
                "name", "cultivar", "sowing_date", "planting_density_plants_ha",
                "row_spacing_m", "expected_harvest_date", "phenology_stage_observed",
            },
            "crop",
        )
        return cls(
            name=_required_string(data.get("name"), "crop.name"),
            cultivar=_required_string(data.get("cultivar"), "crop.cultivar"),
            sowing_date=_parse_date(data.get("sowing_date"), "crop.sowing_date"),
            planting_density_plants_ha=_optional_nonnegative(
                data, "planting_density_plants_ha", "crop.planting_density_plants_ha"
            ),
            row_spacing_m=_optional_nonnegative(data, "row_spacing_m", "crop.row_spacing_m"),
            expected_harvest_date=(
                None
                if data.get("expected_harvest_date") is None
                else _parse_date(data["expected_harvest_date"], "crop.expected_harvest_date")
            ),
            phenology_stage_observed=_optional_number(
                data, "phenology_stage_observed", "crop.phenology_stage_observed"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cultivar": self.cultivar,
            "sowing_date": self.sowing_date.isoformat(),
            "planting_density_plants_ha": self.planting_density_plants_ha,
            "row_spacing_m": self.row_spacing_m,
            "expected_harvest_date": (
                None if self.expected_harvest_date is None else self.expected_harvest_date.isoformat()
            ),
            "phenology_stage_observed": self.phenology_stage_observed,
        }


@dataclass(frozen=True)
class FertilizationEvent:
    date: date
    n_rate_kg_ha: float
    fertilizer_name: str | None = None
    fertilizer_type: str | None = None
    n_form: str | None = None
    urea_n_fraction: float | None = None
    nh4_n_fraction: float | None = None
    no3_n_fraction: float | None = None
    application_method: str | None = None
    notes: str | None = None
    n_recovery: float | None = None
    f_nh4n: float | None = None
    f_no3n: float | None = None
    application_depth_cm: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "FertilizationEvent":
        field = f"management.fertilization_history[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(
            data,
            {
                "date", "n_rate_kg_ha", "fertilizer_name", "fertilizer_type", "n_form",
                "urea_n_fraction", "nh4_n_fraction", "no3_n_fraction",
                "application_method", "notes",
                "n_recovery", "f_nh4n", "f_no3n", "application_depth_cm",
            },
            field,
        )
        fractions = {
            key: _optional_number(data, key, f"{field}.{key}")
            for key in ("urea_n_fraction", "nh4_n_fraction", "no3_n_fraction")
        }
        for key, fraction in fractions.items():
            if fraction is not None and not 0 <= fraction <= 1:
                raise SchemaValidationError(f"{field}.{key}: must be in [0, 1]")
        provided = [fraction for fraction in fractions.values() if fraction is not None]
        if len(provided) == 3 and not math.isclose(sum(provided), 1.0, abs_tol=1e-6):
            raise SchemaValidationError(
                f"{field}: urea_n_fraction + nh4_n_fraction + no3_n_fraction must sum to 1"
            )
        for key in ("n_recovery", "f_nh4n", "f_no3n"):
            value_number = _optional_number(data, key, f"{field}.{key}")
            if value_number is not None and not 0 <= value_number <= 1:
                raise SchemaValidationError(f"{field}.{key}: must be in [0, 1]")
        depth = _optional_nonnegative(data, "application_depth_cm", f"{field}.application_depth_cm")
        if depth is not None and depth <= 0:
            raise SchemaValidationError(f"{field}.application_depth_cm: must be > 0")
        return cls(
            date=_parse_date(data.get("date"), f"{field}.date"),
            n_rate_kg_ha=_nonnegative(data.get("n_rate_kg_ha"), f"{field}.n_rate_kg_ha"),
            fertilizer_name=_optional_string(data, "fertilizer_name", f"{field}.fertilizer_name"),
            fertilizer_type=_optional_string(data, "fertilizer_type", f"{field}.fertilizer_type"),
            n_form=_optional_string(data, "n_form", f"{field}.n_form"),
            urea_n_fraction=fractions["urea_n_fraction"],
            nh4_n_fraction=fractions["nh4_n_fraction"],
            no3_n_fraction=fractions["no3_n_fraction"],
            application_method=_optional_string(data, "application_method", f"{field}.application_method"),
            notes=_optional_string(data, "notes", f"{field}.notes"),
            n_recovery=_optional_number(data, "n_recovery", f"{field}.n_recovery"),
            f_nh4n=_optional_number(data, "f_nh4n", f"{field}.f_nh4n"),
            f_no3n=_optional_number(data, "f_no3n", f"{field}.f_no3n"),
            application_depth_cm=depth,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "n_rate_kg_ha": self.n_rate_kg_ha,
            "fertilizer_name": self.fertilizer_name,
            "fertilizer_type": self.fertilizer_type,
            "n_form": self.n_form,
            "urea_n_fraction": self.urea_n_fraction,
            "nh4_n_fraction": self.nh4_n_fraction,
            "no3_n_fraction": self.no3_n_fraction,
            "application_method": self.application_method,
            "notes": self.notes,
            "n_recovery": self.n_recovery,
            "f_nh4n": self.f_nh4n,
            "f_no3n": self.f_no3n,
            "application_depth_cm": self.application_depth_cm,
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

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "IrrigationEvent":
        field = f"management.irrigation_history[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(
            data,
            {"date", "amount_mm", "efficiency", "amount_basis", "method", "duration_min", "notes"},
            field,
        )
        efficiency = _optional_number(data, "efficiency", f"{field}.efficiency")
        if efficiency is not None and not 0 < efficiency <= 1:
            raise SchemaValidationError(f"{field}.efficiency: must be in (0, 1]")
        amount_basis = data.get("amount_basis", "gross")
        if amount_basis not in {"gross", "effective"}:
            raise SchemaValidationError(f"{field}.amount_basis: expected gross or effective")
        return cls(
            date=_parse_date(data.get("date"), f"{field}.date"),
            amount_mm=_nonnegative(data.get("amount_mm"), f"{field}.amount_mm"),
            efficiency=efficiency,
            amount_basis=amount_basis,
            method=_optional_string(data, "method", f"{field}.method"),
            duration_min=_optional_nonnegative(data, "duration_min", f"{field}.duration_min"),
            notes=_optional_string(data, "notes", f"{field}.notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "amount_mm": self.amount_mm,
            "efficiency": self.efficiency,
            "amount_basis": self.amount_basis,
            "method": self.method,
            "duration_min": self.duration_min,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ManagementHistory:
    fertilization_history_complete: bool
    fertilization_history: tuple[FertilizationEvent, ...]
    irrigation_history_complete: bool
    irrigation_history: tuple[IrrigationEvent, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "ManagementHistory":
        data = _required_mapping(value, "management")
        _reject_unknown(
            data,
            {
                "fertilization_history_complete", "fertilization_history",
                "irrigation_history_complete", "irrigation_history",
            },
            "management",
        )
        fertilization = data.get("fertilization_history", [])
        irrigation = data.get("irrigation_history", [])
        if not isinstance(fertilization, (list, tuple)):
            raise SchemaValidationError("management.fertilization_history: expected an array")
        if not isinstance(irrigation, (list, tuple)):
            raise SchemaValidationError("management.irrigation_history: expected an array")
        fertilization_complete = data.get("fertilization_history_complete")
        irrigation_complete = data.get("irrigation_history_complete")
        if not isinstance(fertilization_complete, bool):
            raise SchemaValidationError(
                "management.fertilization_history_complete: expected boolean"
            )
        if not isinstance(irrigation_complete, bool):
            raise SchemaValidationError(
                "management.irrigation_history_complete: expected boolean"
            )
        return cls(
            fertilization_history_complete=fertilization_complete,
            fertilization_history=tuple(
                FertilizationEvent.from_mapping(item, index)
                for index, item in enumerate(fertilization)
            ),
            irrigation_history_complete=irrigation_complete,
            irrigation_history=tuple(
                IrrigationEvent.from_mapping(item, index)
                for index, item in enumerate(irrigation)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fertilization_history_complete": self.fertilization_history_complete,
            "fertilization_history": [item.to_dict() for item in self.fertilization_history],
            "irrigation_history_complete": self.irrigation_history_complete,
            "irrigation_history": [item.to_dict() for item in self.irrigation_history],
        }


@dataclass(frozen=True)
class CropObservation:
    lai: float | None = None
    spad: float | None = None
    canopy_cover: float | None = None
    plant_height_m: float | None = None
    aboveground_biomass_kg_ha: float | None = None
    yield_kg_ha: float | None = None
    leaf_n_concentration_g_kg: float | None = None
    phenology_stage: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "CropObservation":
        data = _required_mapping(value, "observations.crop")
        _reject_unknown(
            data,
            {
                "lai", "spad", "canopy_cover", "plant_height_m",
                "aboveground_biomass_kg_ha", "yield_kg_ha",
                "leaf_n_concentration_g_kg", "phenology_stage",
            },
            "observations.crop",
        )
        canopy_cover = _optional_number(data, "canopy_cover", "observations.crop.canopy_cover")
        if canopy_cover is not None and not 0 <= canopy_cover <= 1:
            raise SchemaValidationError("observations.crop.canopy_cover: must be in [0, 1]")
        return cls(
            lai=_optional_nonnegative(data, "lai", "observations.crop.lai"),
            spad=_optional_nonnegative(data, "spad", "observations.crop.spad"),
            canopy_cover=canopy_cover,
            plant_height_m=_optional_nonnegative(data, "plant_height_m", "observations.crop.plant_height_m"),
            aboveground_biomass_kg_ha=_optional_nonnegative(
                data, "aboveground_biomass_kg_ha", "observations.crop.aboveground_biomass_kg_ha"
            ),
            yield_kg_ha=_optional_nonnegative(data, "yield_kg_ha", "observations.crop.yield_kg_ha"),
            leaf_n_concentration_g_kg=_optional_nonnegative(
                data, "leaf_n_concentration_g_kg", "observations.crop.leaf_n_concentration_g_kg"
            ),
            phenology_stage=_optional_number(data, "phenology_stage", "observations.crop.phenology_stage"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lai": self.lai,
            "spad": self.spad,
            "canopy_cover": self.canopy_cover,
            "plant_height_m": self.plant_height_m,
            "aboveground_biomass_kg_ha": self.aboveground_biomass_kg_ha,
            "yield_kg_ha": self.yield_kg_ha,
            "leaf_n_concentration_g_kg": self.leaf_n_concentration_g_kg,
            "phenology_stage": self.phenology_stage,
        }


@dataclass(frozen=True)
class SoilLayerObservation:
    depth_top_cm: float
    depth_bottom_cm: float
    soil_water: float | None = None
    volumetric_water_content: float | None = None
    no3_n_mg_kg: float | None = None
    nh4_n_mg_kg: float | None = None
    ec_ds_m: float | None = None
    ph: float | None = None
    soil_temperature_c: float | None = None
    # Appended to preserve the positional order of the frozen v1 fields.
    bulk_density_g_cm3: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "SoilLayerObservation":
        field = f"observations.soil[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(
            data,
            {
                "depth_top_cm", "depth_bottom_cm", "bulk_density_g_cm3", "soil_water",
                "volumetric_water_content", "no3_n_mg_kg", "nh4_n_mg_kg",
                "ec_ds_m", "ph", "soil_temperature_c",
            },
            field,
        )
        top = _nonnegative(data.get("depth_top_cm"), f"{field}.depth_top_cm")
        bottom = _number(data.get("depth_bottom_cm"), f"{field}.depth_bottom_cm")
        if bottom <= top:
            raise SchemaValidationError(f"{field}.depth_bottom_cm: must be > depth_top_cm")
        ph = _optional_number(data, "ph", f"{field}.ph")
        if ph is not None and not 0 <= ph <= 14:
            raise SchemaValidationError(f"{field}.ph: must be in [0, 14]")
        bulk_density = _optional_number(data, "bulk_density_g_cm3", f"{field}.bulk_density_g_cm3")
        if bulk_density is not None and bulk_density <= 0:
            raise SchemaValidationError(f"{field}.bulk_density_g_cm3: must be > 0")
        return cls(
            depth_top_cm=top,
            depth_bottom_cm=bottom,
            bulk_density_g_cm3=bulk_density,
            soil_water=_optional_nonnegative(data, "soil_water", f"{field}.soil_water"),
            volumetric_water_content=_optional_nonnegative(
                data, "volumetric_water_content", f"{field}.volumetric_water_content"
            ),
            no3_n_mg_kg=_optional_nonnegative(data, "no3_n_mg_kg", f"{field}.no3_n_mg_kg"),
            nh4_n_mg_kg=_optional_nonnegative(data, "nh4_n_mg_kg", f"{field}.nh4_n_mg_kg"),
            ec_ds_m=_optional_nonnegative(data, "ec_ds_m", f"{field}.ec_ds_m"),
            ph=ph,
            soil_temperature_c=_optional_number(
                data, "soil_temperature_c", f"{field}.soil_temperature_c"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth_top_cm": self.depth_top_cm,
            "depth_bottom_cm": self.depth_bottom_cm,
            "bulk_density_g_cm3": self.bulk_density_g_cm3,
            "soil_water": self.soil_water,
            "volumetric_water_content": self.volumetric_water_content,
            "no3_n_mg_kg": self.no3_n_mg_kg,
            "nh4_n_mg_kg": self.nh4_n_mg_kg,
            "ec_ds_m": self.ec_ds_m,
            "ph": self.ph,
            "soil_temperature_c": self.soil_temperature_c,
        }


@dataclass(frozen=True)
class ObservationSnapshot:
    observation_date: date
    source: str
    crop: CropObservation | None = None
    soil: tuple[SoilLayerObservation, ...] = ()
    notes: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "ObservationSnapshot":
        data = _required_mapping(value, "observations")
        _reject_unknown(data, {"observation_date", "source", "crop", "soil", "notes"}, "observations")
        source = data.get("source", ObservationSource.UNKNOWN.value)
        if not isinstance(source, str) or source not in {item.value for item in ObservationSource}:
            raise SchemaValidationError(
                "observations.source: unsupported source; expected a controlled value"
            )
        soil = data.get("soil", [])
        if not isinstance(soil, (list, tuple)):
            raise SchemaValidationError("observations.soil: expected an array")
        return cls(
            observation_date=_parse_date(data.get("observation_date"), "observations.observation_date"),
            source=source,
            crop=(None if data.get("crop") is None else CropObservation.from_mapping(data["crop"])),
            soil=tuple(SoilLayerObservation.from_mapping(item, index) for index, item in enumerate(soil)),
            notes=_optional_string(data, "notes", "observations.notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_date": self.observation_date.isoformat(),
            "source": self.source,
            "crop": None if self.crop is None else self.crop.to_dict(),
            "soil": [item.to_dict() for item in self.soil],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class WeatherObservation:
    date: date
    tmin_c: float | None = None
    tmax_c: float | None = None
    precipitation_mm: float | None = None
    solar_radiation: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, index: int, field_root: str) -> "WeatherObservation":
        field = f"{field_root}[{index}]"
        data = _required_mapping(value, field)
        _reject_unknown(data, {"date", "tmin_c", "tmax_c", "precipitation_mm", "solar_radiation"}, field)
        return cls(
            date=_parse_date(data.get("date"), f"{field}.date"),
            tmin_c=_optional_number(data, "tmin_c", f"{field}.tmin_c"),
            tmax_c=_optional_number(data, "tmax_c", f"{field}.tmax_c"),
            precipitation_mm=_optional_nonnegative(
                data, "precipitation_mm", f"{field}.precipitation_mm"
            ),
            solar_radiation=_optional_nonnegative(
                data, "solar_radiation", f"{field}.solar_radiation"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "tmin_c": self.tmin_c,
            "tmax_c": self.tmax_c,
            "precipitation_mm": self.precipitation_mm,
            "solar_radiation": self.solar_radiation,
        }


@dataclass(frozen=True)
class Weather:
    provider: str = "openmeteo"
    use_external_provider: bool = True
    forecast_horizon_days: int | None = None
    history: tuple[WeatherObservation, ...] | None = None
    forecast: tuple[WeatherObservation, ...] | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "Weather":
        data = _required_mapping(value, "weather")
        _reject_unknown(
            data,
            {"provider", "use_external_provider", "forecast_horizon_days", "history", "forecast"},
            "weather",
        )
        provider = data.get("provider", "openmeteo")
        if not isinstance(provider, str) or not provider.strip():
            raise SchemaValidationError("weather.provider: expected a non-empty string")
        use_external = _optional_bool(
            data, "use_external_provider", True, "weather.use_external_provider"
        )
        horizon = data.get("forecast_horizon_days")
        if horizon is not None:
            if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 0:
                raise SchemaValidationError("weather.forecast_horizon_days: expected an integer >= 0")
            horizon = int(horizon)
        parsed = {}
        for kind in ("history", "forecast"):
            records = data.get(kind)
            if records is None:
                parsed[kind] = None
                continue
            if not isinstance(records, (list, tuple)):
                raise SchemaValidationError(f"weather.{kind}: expected an array")
            parsed[kind] = tuple(
                WeatherObservation.from_mapping(item, index, f"weather.{kind}")
                for index, item in enumerate(records)
            )
        return cls(
            provider=provider,
            use_external_provider=use_external,
            forecast_horizon_days=horizon,
            history=parsed["history"],
            forecast=parsed["forecast"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "use_external_provider": self.use_external_provider,
            "forecast_horizon_days": self.forecast_horizon_days,
            "history": None if self.history is None else [item.to_dict() for item in self.history],
            "forecast": None if self.forecast is None else [item.to_dict() for item in self.forecast],
        }


@dataclass(frozen=True)
class DecisionContext:
    decision_type: str = "nitrogen"
    forecast_horizon_days: int | None = None
    allow_fertilization_decision: bool = True
    allow_irrigation_decision: bool = False
    max_single_n_rate_kg_ha: float | None = None
    notes: str | None = None
    decision_mode: str = "auto"

    @classmethod
    def from_mapping(cls, value: Any) -> "DecisionContext":
        data = _required_mapping(value, "decision_context")
        _reject_unknown(
            data,
            {
                "decision_type", "forecast_horizon_days", "allow_fertilization_decision",
                "allow_irrigation_decision", "max_single_n_rate_kg_ha", "notes", "decision_mode",
            },
            "decision_context",
        )
        horizon = data.get("forecast_horizon_days")
        if horizon is not None:
            if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 0:
                raise SchemaValidationError(
                    "decision_context.forecast_horizon_days: expected an integer >= 0"
                )
            horizon = int(horizon)
        decision_mode = data.get("decision_mode", "auto")
        if decision_mode not in {"auto", "historical_replay", "live", "simulation"}:
            raise SchemaValidationError(
                "decision_context.decision_mode: expected auto, historical_replay, live, or simulation"
            )
        return cls(
            decision_type=_required_string(
                data.get("decision_type", "nitrogen"), "decision_context.decision_type"
            ),
            forecast_horizon_days=horizon,
            allow_fertilization_decision=_optional_bool(
                data, "allow_fertilization_decision", True,
                "decision_context.allow_fertilization_decision",
            ),
            allow_irrigation_decision=_optional_bool(
                data, "allow_irrigation_decision", False,
                "decision_context.allow_irrigation_decision",
            ),
            max_single_n_rate_kg_ha=_optional_nonnegative(
                data, "max_single_n_rate_kg_ha", "decision_context.max_single_n_rate_kg_ha"
            ),
            notes=_optional_string(data, "notes", "decision_context.notes"),
            decision_mode=decision_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "forecast_horizon_days": self.forecast_horizon_days,
            "allow_fertilization_decision": self.allow_fertilization_decision,
            "allow_irrigation_decision": self.allow_irrigation_decision,
            "max_single_n_rate_kg_ha": self.max_single_n_rate_kg_ha,
            "notes": self.notes,
            "decision_mode": self.decision_mode,
        }


@dataclass(frozen=True)
class DecisionRequest:
    schema_version: str
    request: RequestContext
    location: Location
    crop: Crop
    query_date: date
    management: ManagementHistory
    observations: ObservationSnapshot | None
    weather: Weather
    decision_context: DecisionContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "location": self.location.to_dict(),
            "crop": self.crop.to_dict(),
            "query_date": self.query_date.isoformat(),
            "management": self.management.to_dict(),
            "observations": None if self.observations is None else self.observations.to_dict(),
            "weather": self.weather.to_dict(),
            "decision_context": self.decision_context.to_dict(),
        }


def normalize_decision_request(payload: Mapping[str, Any]) -> DecisionRequest:
    """Parse and strictly validate canonical schema version 1.0."""

    data = _required_mapping(payload, "request")
    _reject_unknown(
        data,
        {
            "schema_version", "request", "location", "crop", "query_date",
            "management", "observations", "weather", "decision_context",
        },
        "request",
    )
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"schema_version: expected {SCHEMA_VERSION!r}, got {version!r}"
        )
    location = Location.from_mapping(data.get("location"))
    crop = Crop.from_mapping(data.get("crop"))
    query_date = _parse_date(data.get("query_date"), "query_date")
    if query_date < crop.sowing_date:
        raise SchemaValidationError("query_date: must be >= crop.sowing_date")
    management = ManagementHistory.from_mapping(data.get("management"))
    observations = (
        None
        if data.get("observations") is None
        else ObservationSnapshot.from_mapping(data["observations"])
    )
    return DecisionRequest(
        schema_version=SCHEMA_VERSION,
        request=RequestContext.from_mapping(data.get("request", {})),
        location=location,
        crop=crop,
        query_date=query_date,
        management=management,
        observations=observations,
        weather=Weather.from_mapping(data.get("weather", {})),
        decision_context=DecisionContext.from_mapping(data.get("decision_context", {})),
    )


def get_schema_capabilities() -> dict[str, Any]:
    """Return model-usage status for every important canonical area."""

    return {
        "schema_version": SCHEMA_VERSION,
        "capabilities": {
            "request": {"status": "RESERVED", "model_usage": "context only"},
            "location": {
                "status": "RESERVED",
                "model_usage": "validated/preserved; CN-Maize config remains authoritative",
            },
            "crop.name": {"status": "RESERVED", "model_usage": "validated/preserved"},
            "crop.cultivar": {"status": "RESERVED", "model_usage": "validated/preserved"},
            "crop.sowing_date": {
                "status": "ACTIVE",
                "model_usage": "used by management-to-action timing; does not override YAML",
            },
            "crop.additional_parameters": {"status": "RESERVED", "model_usage": "preserved only"},
            "fertilization_history": {
                "status": "ACTIVE",
                "model_usage": "converted to current discrete RL action_history",
            },
            "nitrogen_form": {"status": "RESERVED", "model_usage": "preserved; not used by RL"},
            "irrigation_history": {
                "status": "UNSUPPORTED",
                "model_usage": "current model uses fixed agromanagement irrigation",
            },
            "observations.crop": {
                "status": "RESERVED",
                "model_usage": "preserved; only same-date LAI has a decision-time override",
            },
            "observations.crop.lai": {
                "status": "ACTIVE_DECISION_OVERRIDE",
                "model_usage": "same-date replacement of RL raw feature index 2; WOFOST state unchanged",
            },
            "observations.crop.spad": {
                "status": "RESERVED",
                "model_usage": "preserved; no direct feature in the current 22-dimensional policy observation",
            },
            "observations.soil": {
                "status": "ACTIVE_DECISION_OVERRIDE",
                "model_usage": "same-date complete soil profiles can override supported raw RL features; WOFOST state unchanged",
            },
            "observations.soil.no3_n_mg_kg": {
                "status": "ACTIVE_DECISION_OVERRIDE",
                "model_usage": "same-date complete 0-120 cm profile maps to raw RL feature index 4 in kg N/ha; model RHOD is an explicit fallback",
            },
            "observations.soil.nh4_n_mg_kg": {
                "status": "ACTIVE_DECISION_OVERRIDE",
                "model_usage": "same-date complete 0-120 cm profile maps to raw RL feature index 5 in kg N/ha; model RHOD is an explicit fallback",
            },
            "observations.soil.soil_water": {
                "status": "RESERVED",
                "model_usage": "canonical water semantics do not directly match the current scalar WC feature",
            },
            "observations.soil.volumetric_water_content": {
                "status": "ACTIVE_DECISION_OVERRIDE",
                "model_usage": "same-date complete 0-120 cm profile maps to raw RL feature index 6 as mean cm water across model layers",
            },
            "weather.openmeteo": {
                "status": "ACTIVE",
                "model_usage": "current CN-Maize runtime provider",
            },
            "weather.history": {"status": "RESERVED", "model_usage": "preserved only"},
            "weather.forecast": {"status": "RESERVED", "model_usage": "preserved only"},
            "decision_context": {"status": "RESERVED", "model_usage": "request metadata only"},
        },
    }


__all__ = [
    "SCHEMA_VERSION",
    "Crop",
    "CropObservation",
    "DecisionContext",
    "DecisionRequest",
    "FertilizationEvent",
    "IrrigationEvent",
    "Location",
    "ManagementHistory",
    "ObservationSnapshot",
    "ObservationSource",
    "RequestContext",
    "SchemaValidationError",
    "SoilLayerObservation",
    "Weather",
    "WeatherObservation",
    "get_schema_capabilities",
    "normalize_decision_request",
]
