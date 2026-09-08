"""Stable domain errors for the decision boundary.

The decision engine itself deliberately does not catch arbitrary exceptions.
This module gives a future HTTP adapter a typed, JSON-safe representation for
expected request, model, weather, and capability failures.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .schemas import SchemaValidationError


class DecisionDomainError(ValueError):
    """Base class for expected errors at the canonical decision boundary."""

    code: ClassVar[str] = "DECISION_ENGINE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.field = field
        self.details = {} if details is None else dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "details": self.details,
        }


class DecisionEngineError(DecisionDomainError):
    """Unexpected-but-domain-owned decision-engine failure."""


class RequestValidationError(SchemaValidationError, DecisionDomainError):
    code = "REQUEST_VALIDATION_ERROR"


class QueryDateBeforeCropStartError(DecisionDomainError):
    code = "QUERY_DATE_BEFORE_CROP_START"


class QueryDateAfterCropEndError(DecisionDomainError):
    code = "QUERY_DATE_AFTER_CROP_END"


class IncompleteFertilizationHistoryError(DecisionDomainError):
    code = "INCOMPLETE_FERTILIZATION_HISTORY"


class UnrepresentableNRateError(DecisionDomainError):
    code = "UNREPRESENTABLE_N_RATE"


class UnrepresentableFertilizationDateError(DecisionDomainError):
    code = "UNREPRESENTABLE_FERTILIZATION_DATE"


class UnsupportedResidualPeriodManagementError(DecisionDomainError):
    code = "UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT"


class CustomIrrigationNotSupportedError(DecisionDomainError):
    code = "CUSTOM_IRRIGATION_NOT_SUPPORTED"


class ObservationDateMismatchError(DecisionDomainError):
    code = "OBSERVATION_DATE_MISMATCH"


class ModelArtifactMissingError(DecisionDomainError):
    code = "MODEL_ARTIFACT_MISSING"


class ModelEnvironmentIncompatibleError(DecisionDomainError):
    code = "MODEL_ENV_INCOMPATIBLE"


class WeatherProviderError(DecisionDomainError):
    code = "WEATHER_PROVIDER_ERROR"


_LEGACY_PREFIXES: tuple[tuple[str, type[DecisionDomainError], str | None], ...] = (
    (
        "INCOMPLETE_MANAGEMENT_HISTORY",
        IncompleteFertilizationHistoryError,
        "management.fertilization_history",
    ),
    (
        "UNREPRESENTABLE_N_RATE",
        UnrepresentableNRateError,
        "management.fertilization_history[].n_rate_kg_ha",
    ),
    (
        "UNREPRESENTABLE_FERTILIZATION_DATE",
        UnrepresentableFertilizationDateError,
        "management.fertilization_history[].date",
    ),
    (
        "UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT",
        UnsupportedResidualPeriodManagementError,
        "management.fertilization_history",
    ),
    (
        "CUSTOM_IRRIGATION_NOT_YET_SUPPORTED",
        CustomIrrigationNotSupportedError,
        "management.irrigation_history",
    ),
    (
        "QUERY_DATE_BEFORE_SOWING",
        QueryDateBeforeCropStartError,
        "query_date",
    ),
    (
        "QUERY_DATE_AFTER_CROP_END",
        QueryDateAfterCropEndError,
        "query_date",
    ),
    (
        "OBSERVATION_DATE_MISMATCH",
        ObservationDateMismatchError,
        "observations.observation_date",
    ),
)


def _legacy_error(error: BaseException) -> DecisionDomainError | None:
    """Convert known legacy project exceptions without handling unknown ones."""

    message = str(error)
    for prefix, error_type, field in _LEGACY_PREFIXES:
        if message.startswith(prefix) or prefix in message:
            return error_type(message, field=field)

    # Imports stay local so this module remains safe to import from all serving
    # modules, including during package initialization.
    from .rl_inference import InferenceCompatibilityError
    from .wofost_realtime import QueryDateError

    if isinstance(error, SchemaValidationError):
        return RequestValidationError(message, field="request")
    if isinstance(error, InferenceCompatibilityError):
        return ModelEnvironmentIncompatibleError(message)
    if isinstance(error, FileNotFoundError):
        return ModelArtifactMissingError(message)
    if isinstance(error, QueryDateError):
        if "QUERY_DATE_BEFORE_CROP_START" in message:
            return QueryDateBeforeCropStartError(message, field="query_date")
        if "QUERY_DATE_AFTER_CROP_END" in message:
            return QueryDateAfterCropEndError(message, field="query_date")
        return DecisionEngineError(message)
    return None


def as_domain_error(error: BaseException) -> DecisionDomainError:
    """Return a stable domain error for expected failures.

    Unknown exceptions are represented as ``DECISION_ENGINE_ERROR`` only when
    an adapter explicitly asks for serialization.  ``DecisionEngine.decide``
    does not call this fallback and therefore does not hide programmer errors.
    """

    if isinstance(error, DecisionDomainError):
        return error
    converted = _legacy_error(error)
    if converted is not None:
        return converted
    return DecisionEngineError(
        "Unexpected decision engine failure.",
        details={"exception_type": type(error).__name__},
    )


def serialize_decision_error(
    error: BaseException,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a future-HTTP-ready, JSON-safe error envelope."""

    domain_error = as_domain_error(error)
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "status": "error",
        "error": domain_error.to_dict(),
        "decision_due": None,
        "recommendation": None,
        "warnings": [],
    }


__all__ = [
    "DecisionDomainError",
    "DecisionEngineError",
    "RequestValidationError",
    "QueryDateBeforeCropStartError",
    "QueryDateAfterCropEndError",
    "IncompleteFertilizationHistoryError",
    "UnrepresentableNRateError",
    "UnrepresentableFertilizationDateError",
    "UnsupportedResidualPeriodManagementError",
    "CustomIrrigationNotSupportedError",
    "ObservationDateMismatchError",
    "ModelArtifactMissingError",
    "ModelEnvironmentIncompatibleError",
    "WeatherProviderError",
    "as_domain_error",
    "serialize_decision_error",
]
