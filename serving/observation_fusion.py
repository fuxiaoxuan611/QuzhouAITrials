"""Conservative decision-time fusion of canonical observations.

This module implements Level 1 observation handling only: a validated,
same-date observation may replace the corresponding raw policy feature before
VecNormalize and inference. It never mutates the WOFOST/PCSE state.
"""

from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any, Mapping

import numpy as np

from .schemas import ObservationSnapshot


LAI_FEATURE_INDEX = 2
LAI_FEATURE_NAME = "LAI"
EXPECTED_RAW_OBSERVATION_DIMENSION = 22

_CROP_RESERVED_FIELDS = (
    ("spad", "NO_DIRECT_RL_FEATURE"),
    ("canopy_cover", "NO_DIRECT_RL_FEATURE"),
    ("plant_height_m", "NO_DIRECT_RL_FEATURE"),
    ("aboveground_biomass_kg_ha", "NO_DIRECT_RL_FEATURE"),
    ("yield_kg_ha", "NO_DIRECT_RL_FEATURE"),
    ("leaf_n_concentration_g_kg", "NO_DIRECT_RL_FEATURE"),
    ("phenology_stage", "NO_DIRECT_RL_FEATURE"),
)
_SOIL_UNIT_RESERVED_FIELDS = {
    "soil_water",
    "volumetric_water_content",
    "no3_n_mg_kg",
    "nh4_n_mg_kg",
}
_SOIL_RESERVED_FIELDS = (
    "ec_ds_m",
    "ph",
    "soil_temperature_c",
)


class ObservationFusionError(ValueError):
    """Raised when a raw observation cannot be safely fused."""


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ObservationFusionError(
                f"query_date: expected ISO date YYYY-MM-DD, got {value!r}"
            ) from exc
    raise ObservationFusionError("query_date: expected a date or ISO date string")


def _empty_audit(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "wofost_state_mutated": False,
        "applied": [],
        "not_applied": [],
        "applied_to_policy": False,
        "applied_to_wofost_state": False,
    }


def _non_null_fields(observations: ObservationSnapshot) -> list[str]:
    """Return canonical fields carrying a value, for mismatch auditing."""

    fields: list[str] = []
    crop = observations.crop
    if crop is not None:
        if crop.lai is not None:
            fields.append("observations.crop.lai")
        for field, _reason in _CROP_RESERVED_FIELDS:
            if getattr(crop, field) is not None:
                fields.append(f"observations.crop.{field}")
    for index, layer in enumerate(observations.soil):
        for field in (
            "soil_water",
            "volumetric_water_content",
            "no3_n_mg_kg",
            "nh4_n_mg_kg",
            *_SOIL_RESERVED_FIELDS,
        ):
            if getattr(layer, field) is not None:
                fields.append(f"observations.soil[{index}].{field}")
    return fields


class ObservationFusionEngine:
    """Apply only safe canonical fields to a decision-time raw observation."""

    def fuse(
        self,
        raw_observation: Any,
        observations: ObservationSnapshot | Mapping[str, Any] | None,
        query_date: date | datetime | str,
    ) -> dict[str, Any]:
        """Return a copied raw vector and a JSON-safe fusion audit.

        A canonical observation is eligible only when its observation date is
        exactly the requested policy date. The first implementation has one
        active mapping: ``observations.crop.lai`` to raw feature index 2.
        """

        raw = np.asarray(raw_observation)
        if raw.ndim != 1:
            raise ObservationFusionError(
                f"raw_observation: expected a one-dimensional vector, got {raw.shape}"
            )
        if not np.issubdtype(raw.dtype, np.number):
            raise ObservationFusionError(
                f"raw_observation: expected numeric values, got {raw.dtype}"
            )
        if not np.isfinite(raw).all():
            raise ObservationFusionError("raw_observation: contains NaN or Inf")
        if raw.shape != (EXPECTED_RAW_OBSERVATION_DIMENSION,):
            raise ObservationFusionError(
                "raw_observation: expected current CN-Maize shape "
                f"({EXPECTED_RAW_OBSERVATION_DIMENSION},), got {raw.shape}"
            )

        requested = _as_date(query_date)
        if observations is None:
            return {
                "corrected_raw_observation": raw.copy(),
                "audit": _empty_audit("no_observation"),
            }
        if isinstance(observations, Mapping):
            observations = ObservationSnapshot.from_mapping(observations)
        if not isinstance(observations, ObservationSnapshot):
            raise TypeError("observations must be an ObservationSnapshot, mapping, or None")

        audit = _empty_audit("decision_observation_override")
        audit["query_date"] = requested.isoformat()
        audit["observation_date"] = observations.observation_date.isoformat()
        corrected = raw.copy()

        if observations.observation_date != requested:
            audit["mode"] = "observation_date_mismatch"
            audit["date_mismatch"] = "OBSERVATION_DATE_MISMATCH"
            for canonical_field in _non_null_fields(observations):
                audit["not_applied"].append(
                    {
                        "canonical_field": canonical_field,
                        "reason": "OBSERVATION_DATE_MISMATCH",
                    }
                )
            return {"corrected_raw_observation": corrected, "audit": audit}

        crop = observations.crop
        if crop is not None and crop.lai is not None:
            if not math.isfinite(crop.lai) or crop.lai < 0:
                raise ObservationFusionError(
                    "observations.crop.lai: expected a finite non-negative value"
                )
            model_value = float(corrected[LAI_FEATURE_INDEX])
            observed_value = float(crop.lai)
            corrected[LAI_FEATURE_INDEX] = observed_value
            audit["applied"].append(
                {
                    "canonical_field": "observations.crop.lai",
                    "rl_feature": LAI_FEATURE_NAME,
                    "feature_index": LAI_FEATURE_INDEX,
                    "model_value": model_value,
                    "observed_value": observed_value,
                }
            )

        if crop is not None:
            for field, reason in _CROP_RESERVED_FIELDS:
                if getattr(crop, field) is not None:
                    audit["not_applied"].append(
                        {
                            "canonical_field": f"observations.crop.{field}",
                            "reason": reason,
                        }
                    )

        for index, layer in enumerate(observations.soil):
            for field in _SOIL_UNIT_RESERVED_FIELDS:
                if getattr(layer, field) is not None:
                    audit["not_applied"].append(
                        {
                            "canonical_field": f"observations.soil[{index}].{field}",
                            "reason": "RESERVED_UNIT_OR_LAYER_CONVERSION_UNCONFIRMED",
                        }
                    )
            for field in _SOIL_RESERVED_FIELDS:
                if getattr(layer, field) is not None:
                    audit["not_applied"].append(
                        {
                            "canonical_field": f"observations.soil[{index}].{field}",
                            "reason": "NO_DIRECT_RL_FEATURE",
                        }
                    )

        audit["applied_to_policy"] = bool(audit["applied"])
        return {"corrected_raw_observation": corrected, "audit": audit}


def not_due_audit(observations_received: bool) -> dict[str, Any]:
    """Return an audit when no policy observation is generated."""

    audit = _empty_audit("not_a_decision_boundary")
    audit["observations_received"] = bool(observations_received)
    return audit


__all__ = [
    "LAI_FEATURE_INDEX",
    "LAI_FEATURE_NAME",
    "EXPECTED_RAW_OBSERVATION_DIMENSION",
    "ObservationFusionEngine",
    "ObservationFusionError",
    "not_due_audit",
]
