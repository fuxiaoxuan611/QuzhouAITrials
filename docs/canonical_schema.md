# Canonical decision schema v1

This document freezes the application-facing decision input accepted by
`serving.schemas.normalize_decision_request`.  The schema is implemented with
dependency-free Python dataclasses so a future FastAPI/FastGPT adapter can map
the same contract onto transport models without changing the WOFOST or RL
code.

## Contract

Every request declares `schema_version: "1.0"` and contains these top-level
objects:

| Field | Required | Purpose |
| --- | --- | --- |
| `schema_version` | yes | Canonical contract version; currently `1.0`. |
| `request` | no | Optional `request_id` and original `user_query`. |
| `location` | yes | Latitude/longitude plus optional field identity and timezone. |
| `crop` | yes | Crop, cultivar, sowing date, and optional agronomic metadata. |
| `query_date` | yes | State date requested by the caller; cannot precede sowing. |
| `management` | yes | Fertilization and irrigation history with explicit completeness flags. |
| `observations` | no | A dated, sourced snapshot of crop and soil observations. |
| `weather` | no | Provider metadata and optional historical/forecast records. |
| `decision_context` | no | Decision type, horizon, permissions, and N-rate limit. |

The parser rejects unknown fields and converts dates to ISO `date` values in
the typed representation.  `to_dict()` converts the result back to a JSON-safe
mapping.

## Management

Fertilizer events use `n_rate_kg_ha` as the canonical nitrogen rate.  Optional
metadata includes fertilizer name/type, N form, the urea/NH4/NO3 fractions,
application method, and notes.  If all three fractions are supplied, they must
be in `[0, 1]` and sum to one (within a small floating-point tolerance).

Both `fertilization_history_complete` and `irrigation_history_complete` are
required booleans.  This distinguishes a verified empty history from an
unknown or incomplete history.  The current converter uses complete
fertilization history to reconstruct the discrete CN-Maize action history.  A
non-empty custom irrigation history is rejected because current CN-Maize
irrigation remains defined by its agromanagement configuration.

## Observations

`observations` has the following shape:

```json
{
  "observation_date": "2025-07-14",
  "source": "field_measurement",
  "crop": {
    "lai": 1.48,
    "spad": 47.0,
    "canopy_cover": 0.62,
    "plant_height_m": 0.73,
    "aboveground_biomass_kg_ha": 3200.0,
    "yield_kg_ha": 8500.0,
    "leaf_n_concentration_g_kg": 28.0,
    "phenology_stage": 1.6
  },
  "soil": [
    {
      "depth_top_cm": 0,
      "depth_bottom_cm": 20,
      "soil_water": 7.4,
      "volumetric_water_content": 0.24,
      "no3_n_mg_kg": 45.8,
      "nh4_n_mg_kg": 2.6,
      "ec_ds_m": 0.41,
      "ph": 7.8,
      "soil_temperature_c": 24.1
    }
  ],
  "notes": "人工测量"
}
```

Allowed `source` values are `field_measurement`, `sensor`, `laboratory`,
`remote_sensing`, `user_reported`, `model_estimate`, and `unknown`.  Soil
layers require increasing `depth_top_cm`/`depth_bottom_cm`; concentration,
water, and EC values are non-negative and pH is constrained to `[0, 14]`.

Observations are parsed and preserved in the management conversion output but
are not assimilated into the current WOFOST state.

## Weather and decision context

Weather defaults to the current runtime contract: provider `openmeteo` and
`use_external_provider: true`.  Optional `history` and `forecast` arrays are
metadata/transport records; the current Open-Meteo provider pipeline remains
unchanged.

Decision context defaults to a nitrogen decision with fertilization decisions
allowed and irrigation decisions disallowed.  These fields are validated and
preserved, but do not yet alter the current CN-Maize environment.

## Capability status

`serving.schemas.get_schema_capabilities()` reports the current implementation
boundary:

| Area | Status | Current behavior |
| --- | --- | --- |
| Sowing date and fertilizer history | `ACTIVE` | Used for action-history reconstruction. |
| Open-Meteo provider metadata | `ACTIVE` | Matches the current runtime provider. |
| Request/location/crop metadata | `RESERVED` | Validated and preserved; YAML remains authoritative. |
| Fertilizer N-form metadata | `RESERVED` | Preserved but not used by the discrete RL action. |
| Crop/soil observations | `RESERVED` | Preserved; no state assimilation. |
| Weather history/forecast | `RESERVED` | Preserved; no provider override. |
| Decision context | `RESERVED` | Preserved request metadata. |
| Custom irrigation history | `UNSUPPORTED` | Current irrigation is fixed by agromanagement. |

No FastAPI endpoint, FastGPT adapter, observation assimilation, or new RL
training behavior is part of this schema freeze.
