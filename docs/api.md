# Local Decision HTTP API

`serving.api` is a thin FastAPI transport adapter around the frozen
`CanonicalDecisionRequest v1.0` and additive `DecisionResult v1.1` contracts. It does not
implement WOFOST, soil conversion, observation fusion, normalization, RL
prediction, or decision-calendar logic.

## Routes

| Method | Route | Meaning |
| --- | --- | --- |
| GET | `/healthz` | Process is alive; does not load or query the model. |
| GET | `/readyz` | Decision engine was loaded successfully. |
| GET | `/v1/capabilities` | Schema capabilities and safe service/model metadata. |
| POST | `/v1/decision` | Canonical schema v1 request to `DecisionEngine.decide()`. |
| POST | `/v1/weather/context` | Weather-only historical/forecast context for a FastGPT weather tool. |

`POST /v1/decision` accepts the same top-level canonical field names as
[`canonical_schema.md`](canonical_schema.md). Pydantic performs only HTTP
transport parsing; canonical validation remains in `serving.schemas` and the
decision engine. The response is the documented
[`DecisionResult v1.1`](decision_contract.md).

In a weather-aware result, top-level `forecast.horizon_end` is the future
WOFOST forecast endpoint (seven days by default), while each
`scenario_evaluation[].horizon_date` may be later so every generated future
management event is included and has at least one post-event simulation day.
Future weather is never inserted into the current RL policy observation.

`POST /v1/weather/context` accepts `location`, `query_date`, optional
`sowing_date`, `forecast_horizon_days`, `decision_mode`, `provider`, and
`as_of`. It returns provider-neutral daily coverage and provenance. In live
mode the service refuses to use archive data through the query date; an unsafe
or unavailable query day returns `WEATHER_LIVE_QUERY_DAY_UNAVAILABLE`.

## Configuration

The model is loaded once during FastAPI lifespan startup and closed during
shutdown. Configure it with environment variables:

```text
QUZHOU_CONFIG_PATH
QUZHOU_MODEL_PATH
QUZHOU_ENV_STATS_PATH
QUZHOU_DEVICE=auto
QUZHOU_POLICY_VALIDATION_STATUS=unknown
QUZHOU_MODEL_ID
```

`QUZHOU_MODEL_PATH` and `QUZHOU_ENV_STATS_PATH` are required for a ready
default service. Paths are resolved internally and are never returned in a
public response. `HOME`/`USERPROFILE` must still point to a writable PCSE
runtime home when the real engine is launched.

The current local smoke model must be started with
`QUZHOU_POLICY_VALIDATION_STATUS=engineering_only`. It is not validated for
agronomic recommendation.

## Health and errors

`/healthz` returns HTTP 200 while the process is alive. `/readyz` returns 200
only after the engine is loaded, otherwise 503. `/v1/decision` returns 503
when the service is not ready.

Expected domain errors use the error envelope from
[`decision_contract.md`](decision_contract.md). Request, management, invalid
calendar/timeline, and insufficient-horizon errors map to 422 where they are
request-domain errors; missing artifacts, incompatible model/env, and
weather-provider availability errors map to 503; forward simulation and
unexpected decision-engine errors map to 500. Tracebacks and local paths are
never returned to callers.
Unexpected exceptions are logged server-side and return only the transport
error `INTERNAL_SERVER_ERROR` without tracebacks or local paths.

Non-boundary requests, disabled fertilization, and constraint violations are
normal HTTP 200 domain results.

## Request IDs and concurrency

An existing non-empty `request.request_id` is preserved. If it is absent, the
service generates a UUID4. Every response includes `X-Request-ID`.

The first implementation uses one in-process engine instance and a lock around
each `DecisionEngine.decide()` call. This is serialized because the underlying
WOFOST, VecNormalize, and RL objects are mutable and thread-safety is not
proven. Higher concurrency should use an engine pool or worker strategy later.

## Local startup

From the project root, after setting the environment variables and writable
PCSE `HOME`/`USERPROFILE`, use one local worker:

```powershell
uvicorn serving.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep this development server bound to `127.0.0.1`; do not expose it publicly.
No CORS, authentication, FastGPT integration, or cloud deployment is included
in this stage.
