"""Convert canonical soil observations to current WOFOST/SB3 soil features.

This module is an observation adapter only. It never writes to a PCSE object
and it does not activate NO3, NH4, or WC in the policy fusion path. The
conversion is depth based: observed intervals are intersected with the
authoritative model intervals instead of being paired by list position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .schemas import SoilLayerObservation


READY_FOR_DECISION_OVERRIDE = "READY_FOR_DECISION_OVERRIDE"
INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"
_EPSILON_CM = 1e-9


class SoilObservationError(ValueError):
    """Raised when soil layers cannot be converted unambiguously."""


@dataclass(frozen=True)
class ModelSoilLayer:
    """A model soil layer with explicit depth boundaries and density."""

    index: int
    depth_top_cm: float
    depth_bottom_cm: float
    thickness_cm: float | None = None
    bulk_density_g_cm3: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise SoilObservationError("model layer index must be a non-negative integer")
        for name in ("depth_top_cm", "depth_bottom_cm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SoilObservationError(f"model layer {name} must be finite")
        top = float(self.depth_top_cm)
        bottom = float(self.depth_bottom_cm)
        if top < 0 or bottom <= top:
            raise SoilObservationError("model layer must satisfy 0 <= top < bottom")
        derived_thickness = bottom - top
        thickness = derived_thickness if self.thickness_cm is None else float(self.thickness_cm)
        if not math.isfinite(thickness) or thickness <= 0:
            raise SoilObservationError("model layer thickness must be positive and finite")
        if not math.isclose(thickness, derived_thickness, rel_tol=0.0, abs_tol=_EPSILON_CM):
            raise SoilObservationError("model layer thickness does not match depth boundaries")
        object.__setattr__(self, "depth_top_cm", top)
        object.__setattr__(self, "depth_bottom_cm", bottom)
        object.__setattr__(self, "thickness_cm", thickness)
        if self.bulk_density_g_cm3 is not None:
            density = float(self.bulk_density_g_cm3)
            if not math.isfinite(density) or density <= 0:
                raise SoilObservationError("model layer bulk density must be positive and finite")
            object.__setattr__(self, "bulk_density_g_cm3", density)


def _finite_nonnegative(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SoilObservationError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise SoilObservationError(f"{field} must be finite and >= 0")
    return number


def _positive(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SoilObservationError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise SoilObservationError(f"{field} must be finite and > 0")
    return number


def mg_n_kg_to_kg_ha(
    concentration_mg_kg: float,
    bulk_density_g_cm3: float,
    thickness_cm: float,
) -> float:
    """Convert soil N concentration to an amount over one hectare.

    ``mg N/kg soil * g soil/cm3 * cm * 0.1`` gives ``kg N/ha``. The
    factor follows from one hectare being 1e8 cm2: a layer of one cm at
    1 g/cm3 contains 1e5 kg soil, and 1 mg/kg therefore equals 0.1 kg N/ha.
    The thickness passed here is the depth overlap, not necessarily a whole
    source or model layer.
    """

    concentration = _finite_nonnegative(concentration_mg_kg, "concentration_mg_kg")
    density = _positive(bulk_density_g_cm3, "bulk_density_g_cm3")
    thickness = _positive(thickness_cm, "thickness_cm")
    return concentration * density * thickness * 0.1


def volumetric_water_to_cm_water(
    volumetric_water_content: float,
    thickness_cm: float,
) -> float:
    """Convert volumetric water fraction over a depth to cm water."""

    fraction = _finite_nonnegative(volumetric_water_content, "volumetric_water_content")
    thickness = _positive(thickness_cm, "thickness_cm")
    return fraction * thickness


class SoilObservationAdapter:
    """Map canonical soil intervals to WOFOST/SB3-compatible values."""

    def __init__(self, model_layers: Iterable[ModelSoilLayer]) -> None:
        layers = tuple(model_layers)
        if not layers:
            raise SoilObservationError("at least one model soil layer is required")
        previous_bottom = None
        for expected_index, layer in enumerate(layers):
            if not isinstance(layer, ModelSoilLayer):
                raise SoilObservationError("model_layers must contain ModelSoilLayer values")
            if layer.index != expected_index:
                raise SoilObservationError("model layers must have contiguous indexes from zero")
            if previous_bottom is not None and layer.depth_top_cm < previous_bottom - _EPSILON_CM:
                raise SoilObservationError("model soil layers must not overlap")
            previous_bottom = layer.depth_bottom_cm
        self.model_layers = layers

    @classmethod
    def from_soil_yaml(cls, path: str | Path) -> "SoilObservationAdapter":
        """Build model intervals from the explicit ``SoilLayers`` YAML list."""

        source_path = Path(path)
        with source_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        try:
            raw_layers = data["SoilProfileDescription"]["SoilLayers"]
        except (TypeError, KeyError) as exc:
            raise SoilObservationError("soil YAML lacks SoilProfileDescription.SoilLayers") from exc
        layers = []
        depth = 0.0
        for index, raw_layer in enumerate(raw_layers):
            try:
                thickness = float(raw_layer["Thickness"])
                density = float(raw_layer["RHOD"])
            except (TypeError, KeyError, ValueError) as exc:
                raise SoilObservationError(f"invalid soil YAML layer {index}") from exc
            layers.append(
                ModelSoilLayer(
                    index=index,
                    depth_top_cm=depth,
                    depth_bottom_cm=depth + thickness,
                    thickness_cm=thickness,
                    bulk_density_g_cm3=density,
                )
            )
            depth += thickness
        return cls(layers)

    @staticmethod
    def _coerce_observations(
        observations: Iterable[SoilLayerObservation | Mapping[str, Any]],
    ) -> tuple[SoilLayerObservation, ...]:
        try:
            values = tuple(observations)
        except TypeError as exc:
            raise SoilObservationError("observations must be an iterable of soil layers") from exc
        result = []
        for index, value in enumerate(values):
            if isinstance(value, SoilLayerObservation):
                result.append(value)
            elif isinstance(value, Mapping):
                try:
                    result.append(SoilLayerObservation.from_mapping(value, index))
                except ValueError as exc:
                    raise SoilObservationError(str(exc)) from exc
            else:
                raise SoilObservationError(
                    f"observations[{index}] must be SoilLayerObservation or a mapping"
                )
        previous_bottom = None
        for index, value in enumerate(result):
            if previous_bottom is not None and value.depth_top_cm < previous_bottom - _EPSILON_CM:
                raise SoilObservationError(
                    f"observations[{index}] overlaps the previous observed layer"
                )
            previous_bottom = value.depth_bottom_cm
        return tuple(result)

    def _overlaps(
        self,
        model_layer: ModelSoilLayer,
        observations: tuple[SoilLayerObservation, ...],
    ) -> tuple[dict[str, Any], ...]:
        overlaps = []
        for source_index, observation in enumerate(observations):
            top = max(model_layer.depth_top_cm, observation.depth_top_cm)
            bottom = min(model_layer.depth_bottom_cm, observation.depth_bottom_cm)
            thickness = bottom - top
            if thickness <= _EPSILON_CM:
                continue
            overlaps.append(
                {
                    "source_observation_layer": source_index,
                    "overlap_top_cm": float(top),
                    "overlap_bottom_cm": float(bottom),
                    "overlap_thickness_cm": float(thickness),
                    "overlap_fraction_of_model_layer": float(thickness / model_layer.thickness_cm),
                }
            )
        return tuple(overlaps)

    @staticmethod
    def _feature_value(
        overlaps: tuple[dict[str, Any], ...],
        observations: tuple[SoilLayerObservation, ...],
        feature: str,
        model_thickness_cm: float,
    ) -> tuple[float | None, str | None]:
        overlap_thickness = sum(item["overlap_thickness_cm"] for item in overlaps)
        if not math.isclose(overlap_thickness, model_thickness_cm, rel_tol=0.0, abs_tol=_EPSILON_CM):
            return None, "INSUFFICIENT_DEPTH_COVERAGE"

        total = 0.0
        for item in overlaps:
            observation = observations[item["source_observation_layer"]]
            value = getattr(observation, feature)
            if value is None:
                return None, f"MISSING_{feature.upper()}"
            if feature in ("no3_n_mg_kg", "nh4_n_mg_kg"):
                if observation.bulk_density_g_cm3 is None:
                    return None, "MISSING_BULK_DENSITY"
                total += mg_n_kg_to_kg_ha(
                    value,
                    observation.bulk_density_g_cm3,
                    item["overlap_thickness_cm"],
                )
            elif feature == "volumetric_water_content":
                total += volumetric_water_to_cm_water(
                    value,
                    item["overlap_thickness_cm"],
                )
            else:  # pragma: no cover - private method guard
                raise SoilObservationError(f"unsupported soil feature: {feature}")
        return float(total), None

    def adapt(
        self,
        observations: Iterable[SoilLayerObservation | Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return JSON-safe mapped layers and derived RL feature candidates."""

        values = self._coerce_observations(observations)
        model_layers = []
        per_feature_values: dict[str, list[float]] = {"NO3": [], "NH4": [], "WC": []}
        per_feature_reasons: dict[str, list[str]] = {"NO3": [], "NH4": [], "WC": []}
        feature_names = {
            "NO3": "no3_n_mg_kg",
            "NH4": "nh4_n_mg_kg",
            "WC": "volumetric_water_content",
        }
        output_names = {"NO3": "NO3_kg_ha", "NH4": "NH4_kg_ha", "WC": "WC_cm"}

        for model_layer in self.model_layers:
            overlaps = self._overlaps(model_layer, values)
            derived: dict[str, float | None] = {}
            for output_name, input_name in feature_names.items():
                value, reason = self._feature_value(
                    overlaps, values, input_name, model_layer.thickness_cm
                )
                derived[output_names[output_name]] = value
                if value is None:
                    per_feature_reasons[output_name].append(reason or "INSUFFICIENT_INPUT")
                else:
                    per_feature_values[output_name].append(value)
            model_layers.append(
                {
                    "index": model_layer.index,
                    "depth_top_cm": model_layer.depth_top_cm,
                    "depth_bottom_cm": model_layer.depth_bottom_cm,
                    "thickness_cm": model_layer.thickness_cm,
                    "bulk_density_g_cm3": model_layer.bulk_density_g_cm3,
                    "overlap_fraction": float(
                        sum(item["overlap_thickness_cm"] for item in overlaps)
                        / model_layer.thickness_cm
                    ),
                    "source_observation_layers": [
                        item["source_observation_layer"] for item in overlaps
                    ],
                    "overlaps": list(overlaps),
                    "derived": derived,
                }
            )

        feature_status: dict[str, str] = {}
        rl_features: dict[str, float | None] = {}
        for feature in ("NO3", "NH4", "WC"):
            reasons = per_feature_reasons[feature]
            if reasons:
                rl_features[feature] = None
                feature_status[feature] = INSUFFICIENT_INPUT
            elif feature == "WC":
                rl_features[feature] = float(sum(per_feature_values[feature]) / len(self.model_layers))
                feature_status[feature] = READY_FOR_DECISION_OVERRIDE
            else:
                rl_features[feature] = float(sum(per_feature_values[feature]))
                feature_status[feature] = READY_FOR_DECISION_OVERRIDE

        ignored_fields = []
        if any(item.soil_water is not None for item in values):
            ignored_fields.append(
                {
                    "canonical_field": "observations.soil[].soil_water",
                    "reason": "AMBIGUOUS_PCSE_WATER_SEMANTICS",
                }
            )
        return {
            "rl_features": rl_features,
            "feature_status": feature_status,
            "model_layers": model_layers,
            "audit": {
                "mode": "soil_observation_conversion_only",
                "wofost_state_mutated": False,
                "source_observation_layer_count": len(values),
                "model_layer_count": len(self.model_layers),
                "feature_units": {
                    "NO3": "kg N/ha",
                    "NH4": "kg N/ha",
                    "WC": "cm water per model layer, then mean across model layers",
                },
                "feature_reasons": {
                    feature: sorted(set(reasons))
                    for feature, reasons in per_feature_reasons.items()
                    if reasons
                },
                "ignored_fields": ignored_fields,
            },
        }


__all__ = [
    "INSUFFICIENT_INPUT",
    "READY_FOR_DECISION_OVERRIDE",
    "ModelSoilLayer",
    "SoilObservationAdapter",
    "SoilObservationError",
    "mg_n_kg_to_kg_ha",
    "volumetric_water_to_cm_water",
]
