# CanonicalDecisionResult v1

This document is the contract boundary between the canonical request and a
future HTTP/FastGPT adapter. The scientific pipeline remains in the serving
modules: [`canonical_schema.md`](canonical_schema.md) → management conversion →
WOFOST reconstruction → observation fusion → VecNormalize → LagrangianPPO.
This contract does not add HTTP, training, or WOFOST state assimilation.

## Successful result

`DecisionEngine.decide()` returns a JSON-safe object with these v1 keys on every
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
  "schema_version": "1.0",
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
| `DECISION_ENGINE_ERROR` | Other decision-engine-owned failure. |

Unknown programmer/system exceptions are not caught by `DecisionEngine.decide`
and should be mapped by a future API layer to an internal server error rather
than presented as a successful result.

## Suggested future HTTP mapping

This is guidance only; no HTTP route is implemented here.

| Domain result | Suggested HTTP status |
| --- | ---: |
| `REQUEST_VALIDATION_ERROR` | 400 or 422 |
| Unsupported management/capability | 422 |
| Query outside crop period | 422 |
| Missing model artifact | 503 (or 500 for deployment misconfiguration) |
| Model/environment incompatibility | 500 |
| External weather provider error | 503 |
| Unexpected programmer/system error | 500 |
