"""Thin local FastAPI adapter for the frozen decision contract.

The module owns HTTP transport, lifecycle, request IDs, serialization, and
concurrency. All scientific behavior remains in ``DecisionEngine``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse

from .api_models import CanonicalDecisionRequestModel, WeatherContextRequestModel
from .decision_contract import (
    DECISION_RESULT_SCHEMA_VERSION,
    json_safe,
    sanitize_model_metadata,
    serialize_decision_error,
)
from .errors import (
    DecisionDomainError,
    ModelArtifactMissingError,
    ModelEnvironmentIncompatibleError,
    WeatherProviderError,
    WeatherProviderNotConfiguredError,
    WeatherDataGapError,
    WeatherTimelineInvalidError,
    ForecastHorizonInsufficientError,
)
from .management_history import ManagementHistoryError
from .rl_inference import InferenceCompatibilityError, RLInferenceEngine
from .schemas import SCHEMA_VERSION, SchemaValidationError, get_schema_capabilities
from .settings import ServiceSettings
from .wofost_realtime import QueryDateError, WOFOSTRealtimeError
from .decision_engine import DecisionEngine
from .season_calendar import SeasonCalendarResolver
from .weather_service import WeatherService


logger = logging.getLogger(__name__)

_HTTP_STATUS_BY_CODE = {
    "REQUEST_VALIDATION_ERROR": 422,
    "QUERY_DATE_BEFORE_CROP_START": 422,
    "QUERY_DATE_AFTER_CROP_END": 422,
    "INCOMPLETE_FERTILIZATION_HISTORY": 422,
    "UNREPRESENTABLE_N_RATE": 422,
    "UNREPRESENTABLE_FERTILIZATION_DATE": 422,
    "UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT": 422,
    "CUSTOM_IRRIGATION_NOT_SUPPORTED": 422,
    "OBSERVATION_DATE_MISMATCH": 422,
    "MODEL_ARTIFACT_MISSING": 503,
    "MODEL_ENV_INCOMPATIBLE": 503,
    "WEATHER_PROVIDER_ERROR": 503,
    "WEATHER_PROVIDER_NOT_CONFIGURED": 503,
    "WEATHER_DATA_GAP": 503,
    "WEATHER_TIMELINE_INVALID": 422,
    "FORECAST_HORIZON_INSUFFICIENT": 422,
    "DYNAMIC_CALENDAR_INVALID": 422,
    "MANAGEMENT_EVENT_INVALID": 422,
    "FORWARD_SIMULATION_ERROR": 500,
    "SCENARIO_EVALUATION_ERROR": 500,
    "DECISION_ENGINE_ERROR": 500,
}

_EXPECTED_LEGACY_ERRORS = (
    FileNotFoundError,
    ManagementHistoryError,
    QueryDateError,
    WOFOSTRealtimeError,
    InferenceCompatibilityError,
    SchemaValidationError,
)


def _request_id_from_body(body: Any) -> str:
    if isinstance(body, dict):
        request_context = body.get("request")
        if isinstance(request_context, dict):
            request_id = request_context.get("request_id")
            if isinstance(request_id, str) and request_id:
                return request_id
    return str(uuid.uuid4())


def _header(request_id: str) -> dict[str, str]:
    return {"X-Request-ID": request_id}


def _safe_not_ready(request_id: str) -> JSONResponse:
    # Startup diagnostics stay server-side. Do not serialize their exception
    # text because it may contain local paths or configuration details.
    envelope = {
        "schema_version": DECISION_RESULT_SCHEMA_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": "MODEL_ARTIFACT_MISSING",
            "message": "Decision service is not ready.",
            "field": None,
            "details": {},
        },
        "decision_due": None,
        "recommendation": None,
        "warnings": [],
    }
    return JSONResponse(status_code=503, content=envelope, headers=_header(request_id))


def _safe_domain_error(error: BaseException, request_id: str) -> JSONResponse:
    envelope = serialize_decision_error(error, request_id=request_id)
    code = envelope["error"]["code"]
    if code in {
        "MODEL_ARTIFACT_MISSING",
        "MODEL_ENV_INCOMPATIBLE",
        "WEATHER_PROVIDER_ERROR",
        "WEATHER_PROVIDER_NOT_CONFIGURED",
        "WEATHER_DATA_GAP",
        "WEATHER_TIMELINE_INVALID",
        "WEATHER_LIVE_QUERY_DAY_UNAVAILABLE",
        "FORECAST_HORIZON_INSUFFICIENT",
        "DECISION_ENGINE_ERROR",
    }:
        safe_messages = {
            "MODEL_ARTIFACT_MISSING": "Decision model artifacts are unavailable.",
            "MODEL_ENV_INCOMPATIBLE": "Decision model and environment are incompatible.",
            "WEATHER_PROVIDER_ERROR": "Weather provider is unavailable.",
            "WEATHER_PROVIDER_NOT_CONFIGURED": "Weather provider is not configured.",
            "WEATHER_DATA_GAP": "Weather data coverage is incomplete.",
            "WEATHER_TIMELINE_INVALID": "Weather timeline is invalid.",
            "WEATHER_LIVE_QUERY_DAY_UNAVAILABLE": "Safe live query-day weather is unavailable.",
            "FORECAST_HORIZON_INSUFFICIENT": "Requested forecast horizon is unavailable.",
            "DECISION_ENGINE_ERROR": "Decision service could not complete the request.",
        }
        envelope["error"]["message"] = safe_messages[code]
        envelope["error"]["details"] = {}
    return JSONResponse(
        status_code=_HTTP_STATUS_BY_CODE.get(code, 500),
        content=json_safe(envelope),
        headers=_header(request_id),
    )


def _validation_response(exc: FastAPIRequestValidationError, request_id: str) -> JSONResponse:
    details = {
        "errors": [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "message": "invalid request value",
            }
            for error in exc.errors()
        ]
    }
    envelope = {
        "schema_version": DECISION_RESULT_SCHEMA_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": "REQUEST_VALIDATION_ERROR",
            "message": "Request validation failed.",
            "field": "request",
            "details": details,
        },
        "decision_due": None,
        "recommendation": None,
        "warnings": [],
    }
    return JSONResponse(status_code=422, content=envelope, headers=_header(request_id))


def _internal_error(request_id: str) -> JSONResponse:
    envelope = {
        "schema_version": DECISION_RESULT_SCHEMA_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Internal decision service error.",
            "field": None,
            "details": {},
        },
        "decision_due": None,
        "recommendation": None,
        "warnings": [],
    }
    return JSONResponse(status_code=500, content=envelope, headers=_header(request_id))


def _default_engine_factory(settings: ServiceSettings) -> DecisionEngine:
    if settings.model_path is None or settings.env_stats_path is None:
        raise ModelArtifactMissingError(
            "QUZHOU_MODEL_PATH and QUZHOU_ENV_STATS_PATH must be configured"
        )
    weather_service = WeatherService.default(timezone_name=settings.weather_timezone)
    weather_service = replace(
        weather_service,
        default_provider=settings.weather_provider,
        forecast_default_days=settings.forecast_default_days,
    )
    return DecisionEngine(
        config_path=settings.config_path,
        model_path=settings.model_path,
        env_stats_path=settings.env_stats_path,
        device=settings.device,
        policy_validation_status=settings.policy_validation_status,
        weather_service=weather_service,
        season_resolver=SeasonCalendarResolver(
            pre_sowing_offset_days=settings.season_pre_sowing_offset_days,
            crop_duration_days=settings.season_crop_duration_days,
            max_duration=settings.season_max_duration,
        ),
    )


def create_app(
    *,
    engine: Any | None = None,
    engine_factory: Callable[[], Any] | None = None,
    settings: ServiceSettings | None = None,
) -> FastAPI:
    """Create the service app; dependency injection keeps API tests offline."""

    configured_settings = settings or ServiceSettings.from_env()
    app = FastAPI(title="QuzhouAITrials Decision Service", version=SCHEMA_VERSION)
    app.state.engine = None
    app.state.startup_error = None
    app.state.decision_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if engine is not None:
            _app.state.engine = engine
        else:
            factory = engine_factory or (
                lambda: _default_engine_factory(configured_settings)
            )
            try:
                _app.state.engine = factory()
            except Exception as exc:
                _app.state.startup_error = exc
                logger.exception("Decision engine startup failed")
        try:
            yield
        finally:
            with _app.state.decision_lock:
                current_engine = _app.state.engine
                _app.state.engine = None
                if current_engine is not None:
                    current_engine.close()

    app.router.lifespan_context = lifespan

    @app.exception_handler(FastAPIRequestValidationError)
    async def handle_validation_error(request: Request, exc: FastAPIRequestValidationError):
        try:
            body = await request.json()
        except Exception:
            body = None
        return _validation_response(exc, _request_id_from_body(body))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        if app.state.engine is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        with app.state.decision_lock:
            current_engine = app.state.engine
            metadata = None
            if current_engine is not None:
                metadata = current_engine.get_metadata()
                if configured_settings.model_id:
                    metadata = {
                        **metadata,
                        "model_id": configured_settings.model_id,
                    }
            return {
                "schema_version": SCHEMA_VERSION,
                "capabilities": get_schema_capabilities(),
                "ready": current_engine is not None,
                "service_concurrency_mode": "serialized",
                "model_metadata": sanitize_model_metadata(metadata),
            }

    @app.post("/v1/decision")
    def decision(payload: CanonicalDecisionRequestModel, request: Request):
        canonical_payload = payload.to_canonical_payload()
        request_id = _request_id_from_body(canonical_payload)
        canonical_payload.setdefault("request", {})["request_id"] = request_id

        with app.state.decision_lock:
            current_engine = app.state.engine
            if current_engine is None:
                return _safe_not_ready(request_id)
            try:
                result = current_engine.decide(canonical_payload)
            except DecisionDomainError as exc:
                return _safe_domain_error(exc, request_id)
            except _EXPECTED_LEGACY_ERRORS as exc:
                return _safe_domain_error(exc, request_id)
            except Exception:
                logger.exception("Unexpected decision service failure")
                return _internal_error(request_id)
        return JSONResponse(status_code=200, content=json_safe(result), headers=_header(request_id))

    @app.post("/v1/weather/context")
    def weather_context(payload: WeatherContextRequestModel, request: Request):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        with app.state.decision_lock:
            current_engine = app.state.engine
            if current_engine is None or not hasattr(current_engine, "weather_service"):
                return _safe_not_ready(request_id)
            try:
                location = payload.location
                latitude = float(location["latitude"])
                longitude = float(location["longitude"])
                query = payload.query_date
                if isinstance(query, str):
                    query = __import__("datetime").date.fromisoformat(query)
                sowing = payload.sowing_date
                if isinstance(sowing, str):
                    sowing = __import__("datetime").date.fromisoformat(sowing)
                if sowing is None:
                    sowing = query
                as_of = payload.as_of
                if isinstance(as_of, str):
                    as_of = __import__("datetime").datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                calendar = current_engine.season_resolver.resolve(
                    "maize", "Quzhou_maize_2025_Opt", sowing
                )
                result = current_engine.weather_service.get_context(
                    campaign_start=calendar.campaign_start_date,
                    query_date=query,
                    forecast_horizon_days=payload.forecast_horizon_days,
                    decision_mode=payload.decision_mode,
                    provider_name=payload.provider,
                    latitude=latitude,
                    longitude=longitude,
                    as_of=as_of,
                )
                result.pop("timeline", None)
                return JSONResponse(status_code=200, content=json_safe(result), headers=_header(request_id))
            except DecisionDomainError as exc:
                return _safe_domain_error(exc, request_id)
            except Exception:
                logger.exception("Weather context request failed")
                return _internal_error(request_id)

    return app


app = create_app()


__all__ = ["app", "create_app"]
