"""Thin local FastAPI adapter for the frozen decision contract.

The module owns HTTP transport, lifecycle, request IDs, serialization, and
concurrency. All scientific behavior remains in ``DecisionEngine``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse

from .api_models import CanonicalDecisionRequestModel
from .decision_contract import json_safe, sanitize_model_metadata, serialize_decision_error
from .errors import (
    DecisionDomainError,
    ModelArtifactMissingError,
    ModelEnvironmentIncompatibleError,
    WeatherProviderError,
)
from .management_history import ManagementHistoryError
from .rl_inference import InferenceCompatibilityError, RLInferenceEngine
from .schemas import SCHEMA_VERSION, SchemaValidationError, get_schema_capabilities
from .settings import ServiceSettings
from .wofost_realtime import QueryDateError, WOFOSTRealtimeError
from .decision_engine import DecisionEngine


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
        "schema_version": SCHEMA_VERSION,
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
        "DECISION_ENGINE_ERROR",
    }:
        safe_messages = {
            "MODEL_ARTIFACT_MISSING": "Decision model artifacts are unavailable.",
            "MODEL_ENV_INCOMPATIBLE": "Decision model and environment are incompatible.",
            "WEATHER_PROVIDER_ERROR": "Weather provider is unavailable.",
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
        "schema_version": SCHEMA_VERSION,
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
        "schema_version": SCHEMA_VERSION,
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
    return DecisionEngine(
        config_path=settings.config_path,
        model_path=settings.model_path,
        env_stats_path=settings.env_stats_path,
        device=settings.device,
        policy_validation_status=settings.policy_validation_status,
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

    return app


app = create_app()


__all__ = ["app", "create_app"]
