# LLM Agronomic Critic Contract

The first Agent workflow v2 critic is an external review step. The local
service prepares `critic_input`; it does not call an LLM, run WOFOST again, or
mutate WOFOST/PCSE state. The existing Level 1 observation behavior remains:

```text
observed LAI
  -> raw RL observation[2]
  -> VecNormalize
  -> LagrangianPPO
```

`observation_fusion` keeps `simulated`, `observed`, and `policy_input` values
separate. In particular, `policy_input.LAI` must not be described as a
corrected WOFOST state. The API's additive `critic_input` object contains:

```json
{
  "query_date": "2025-07-14",
  "decision_date": "2025-07-14",
  "projected_application_date": "2025-07-21",
  "decision_due": true,
  "wofost_simulated_state": {},
  "state_features": {
    "DVS": null,
    "LAI": null,
    "TAGP": null,
    "NuptakeTotal": null,
    "NO3": null,
    "NH4": null,
    "WC": null,
    "NLOSSCUM": null,
    "NUE": null,
    "Nsurp": null
  },
  "observation_fusion": {},
  "user_observations": null,
  "rl_candidate_action": {
    "action_index": 5,
    "n_rate_kg_ha": 50
  },
  "rl_candidate_n_rate_kg_ha": 50,
  "fertilization_history": [],
  "irrigation_history": [],
  "weather_risk": []
}
```

The critic may return only `ACCEPT`, `REDUCE`, `DEFER`, or `REJECT`:

```json
{
  "verdict": "ACCEPT",
  "rl_candidate_n_kg_ha": 50,
  "final_n_kg_ha": 50,
  "execution_status": "proceed",
  "reason_codes": [],
  "reasons": [],
  "confidence": "medium"
}
```

The deterministic validator accepts only rates in `{0, 10, ..., 150}`.
`ACCEPT` and `DEFER` preserve the candidate amount; `REDUCE` must be strictly
lower; `REJECT` must be zero. There is no `INCREASE` verdict. Invalid or
malformed output falls back to `ACCEPT` of the already validated RL candidate,
with `critic_validation_failed: true`. This fallback is not an LLM approval;
it is a fail-safe that prevents invalid critic output from entering fertilizer
conversion.

Weather risk should normally affect `execution_status` through `DEFER`, not
invent a larger amount. The critic must not recalculate WOFOST, invent a
continuous N rate, alter the action space, or fabricate missing observations.

Soil-moisture and all Level 2 WOFOST state assimilation remain outside this
contract.
