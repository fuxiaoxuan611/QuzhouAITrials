# CanonicalDecisionResult v1.1

This document is the contract boundary between the canonical request and the
local HTTP adapter. The scientific pipeline remains in the serving
modules: [`canonical_schema.md`](canonical_schema.md) → management conversion →
WOFOST reconstruction → observation fusion → VecNormalize → LagrangianPPO.
The request schema remains v1.0. This result contract is an additive v1.1
extension; it does not change the frozen 22-dimensional policy observation,
training artifacts, or WOFOST state assimilation boundary.

## Successful result

`DecisionEngine.decide()` returns a JSON-safe object with these v1 core keys on every
successful request:

```text
schema_version
request_id
status
decision_due
state_date
policy_observation_date
previous_decision_date
next_decision_date
action_application_date
recommendation
model_state
management
observations
model_metadata
warnings
weather_context
forecast
projected_next_decision
scenario_evaluation
weather_risk
operation_advice
```

Dates have distinct meanings. `state_date` is the reconstructed WOFOST state;
`policy_observation_date` is populated only at a policy boundary;
`action_application_date` is the following management date. For the current
seven-day CN-Maize schedule, a policy observation on `2025-07-14` applies its
action on `2025-07-21`.

At a decision boundary, `recommendation` contains:

```json
{
  "decision_type": "nitrogen",
  "action_index": 3,
  "n_rate_kg_ha": 30.0,
  "constraint_violation": false,
  "constraint_details": {}
}
```

The policy action is not clipped when `max_single_n_rate_kg_ha` is exceeded;
the original action is returned with `constraint_violation: true` and details.
On a non-boundary date, or when fertilization decisions are disabled,
`recommendation` is `null` and the result remains `status: "ok"`.

Dynamic fertilizer management uses the active SNOMIN `apply_n_snomin` signal:
the N rate, fertilizer-form fractions, and application depth affect the model
through that signal. The legacy `N_recovery` value remains auditable metadata;
it is not treated as an active SNOMIN input because the configured SNOMIN
handler does not consume the legacy `apply_n` recovery fields.

The additive weather-aware fields contain provider coverage/provenance,
forecast-assisted WOFOST state, a projected next policy decision when the
query is between boundaries, isolated management scenarios, deterministic
weather-risk evidence, and timing advice. `operation_advice` may delay or
monitor an RL rate, but never invents a new nitrogen rate.

The top-level `forecast` is a future WOFOST state projection, not a duplicate
query-date state summary. `horizon_end` is `query_date` plus the explicitly
requested horizon, or the configured seven-day default when the request omits
one, capped at crop end. `uses_forecast` is true whenever that horizon is after
the query date. Scenario evaluation can extend weather coverage beyond this
public forecast horizon: its `horizon_date` is at least one model day after the
latest generated management event (including the generator's possible
one-day weather delay), capped at crop end. The evaluator rejects scenario
events outside its horizon. This extension never changes the current policy
observation or the projected-decision timing.

## Observations

The `observations` object distinguishes received data from model mutation:

```json
{
  "received": true,
  "applied_to_policy": true,
  "applied_to_wofost_state": false,
  "fusion_audit": {
    "mode": "decision_observation_override",
    "wofost_state_mutated": false,
    "applied": [
      {
        "canonical_field": "observations.crop.lai",
        "rl_feature": "LAI",
        "feature_index": 2,
        "model_value": 1.483216,
        "observed_value": 1.6
      }
    ],
    "not_applied": []
  }
}
```

This is Level 1 decision-observation override. There is no `assimilated: true`
shortcut, and Level 2 WOFOST/PCSE state assimilation is not implemented.
Same-date eligibility and all field capability rules are enforced by
`ObservationFusionEngine`.

## Management and model metadata

`management` contains JSON-safe `completed_steps`, `decision_dates`, and
`action_history` (plus audited conversion details). It does not expose Python
objects or the original request.

`model_metadata` includes the algorithm, observation/action dimensions,
timestep, identifiers, validation status, and whether the policy has been
validated for agronomic recommendation. The status is explicit:

```json
{
  "algorithm": "LagrangianPPO",
  "observation_dim": 22,
  "action_n": 16,
  "timestep_days": 7,
  "validation_status": "engineering_only",
  "validated_for_agronomic_recommendation": false
}
```

The default validation status is `unknown`; it never silently becomes
agronomically validated. Public results contain filenames/identifiers only,
not local absolute paths such as `E:\\QuzhouAITrials\\...`.

## Error envelope

Expected domain failures can be serialized with
`serving.errors.serialize_decision_error`:

```json
{
  "schema_version": "1.1",
  "request_id": "req-001",
  "status": "error",
  "error": {
    "code": "INCOMPLETE_FERTILIZATION_HISTORY",
    "message": "...",
    "field": "management.fertilization_history",
    "details": {}
  },
  "decision_due": null,
  "recommendation": null,
  "warnings": []
}
```

Stable codes are:

| Code | Meaning |
| --- | --- |
| `REQUEST_VALIDATION_ERROR` | Canonical request is invalid. |
| `QUERY_DATE_BEFORE_CROP_START` / `QUERY_DATE_AFTER_CROP_END` | Date is outside the configured crop window. |
| `INCOMPLETE_FERTILIZATION_HISTORY` | History was not declared complete. |
| `UNREPRESENTABLE_N_RATE` | Rate cannot be represented by the discrete action space. |
| `UNREPRESENTABLE_FERTILIZATION_DATE` | Event date is not a policy application date. |
| `UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT` | Event falls in an unsupported partial interval. |
| `CUSTOM_IRRIGATION_NOT_SUPPORTED` | Custom irrigation is not represented by the current model. |
| `OBSERVATION_DATE_MISMATCH` | Observation is not same-date and is not silently applied. |
| `MODEL_ARTIFACT_MISSING` | Required model or environment artifact is absent. |
| `MODEL_ENV_INCOMPATIBLE` | Model and restored environment are incompatible. |
| `WEATHER_PROVIDER_ERROR` | An external weather provider failed. |
| `WEATHER_PROVIDER_NOT_CONFIGURED` | Requested weather provider is unavailable in this service. |
| `WEATHER_DATA_GAP` | The selected timeline has a missing daily record. |
| `WEATHER_TIMELINE_INVALID` | Weather records are duplicated, malformed, or inconsistent. |
| `WEATHER_LIVE_QUERY_DAY_UNAVAILABLE` | Safe live query-day data could not be obtained. |
| `FORECAST_HORIZON_INSUFFICIENT` | Provider coverage is shorter than requested. |
| `DYNAMIC_CALENDAR_INVALID` | Dynamic season dates are invalid. |
| `MANAGEMENT_EVENT_INVALID` | A date-based management event is invalid. |
| `FORWARD_SIMULATION_ERROR` / `SCENARIO_EVALUATION_ERROR` | Isolated future simulation failed. |
| `DECISION_ENGINE_ERROR` | Other decision-engine-owned failure. |

Unknown programmer/system exceptions are not caught by `DecisionEngine.decide`
and should be mapped by a future API layer to an internal server error rather
than presented as a successful result.

## HTTP mapping

The local adapter implements these mappings and also provides a weather-only
tool endpoint at `POST /v1/weather/context`.

| Domain result | Suggested HTTP status |
| --- | ---: |
| `REQUEST_VALIDATION_ERROR` | 400 or 422 |
| Unsupported management/capability | 422 |
| Query outside crop period | 422 |
| Missing model artifact | 503 |
| Model/environment incompatibility | 503 |
| External weather provider error | 503 |
| Unexpected programmer/system error | 500 |
