"""Reusable inference components for QuzhouAITrials."""

from .decision_engine import DecisionEngine, DecisionEngineError, DecisionSlot
from .observation_fusion import ObservationFusionEngine, ObservationFusionError
from .rl_inference import (
    RLInferenceEngine,
    action_index_to_n_rate,
)
from .management_history import (
    ManagementHistoryError,
    build_action_history,
    management_to_realtime_input,
)
from .schemas import (
    SCHEMA_VERSION,
    Crop,
    CropObservation,
    DecisionContext,
    DecisionRequest,
    FertilizationEvent,
    IrrigationEvent,
    Location,
    ManagementHistory,
    ObservationSnapshot,
    ObservationSource,
    RequestContext,
    SchemaValidationError,
    SoilLayerObservation,
    Weather,
    WeatherObservation,
    get_schema_capabilities,
    normalize_decision_request,
)
from .wofost_realtime import WOFOSTRealtimeEngine
from .soil_observation import (
    ModelSoilLayer,
    SoilObservationAdapter,
    SoilObservationError,
    mg_n_kg_to_kg_ha,
    volumetric_water_to_cm_water,
)

__all__ = [
    "SCHEMA_VERSION",
    "Crop",
    "CropObservation",
    "DecisionEngine",
    "DecisionEngineError",
    "DecisionContext",
    "DecisionRequest",
    "DecisionSlot",
    "ObservationFusionEngine",
    "ObservationFusionError",
    "FertilizationEvent",
    "IrrigationEvent",
    "Location",
    "ManagementHistory",
    "ManagementHistoryError",
    "ObservationSnapshot",
    "ObservationSource",
    "RLInferenceEngine",
    "RequestContext",
    "SchemaValidationError",
    "SoilLayerObservation",
    "Weather",
    "WeatherObservation",
    "WOFOSTRealtimeEngine",
    "action_index_to_n_rate",
    "build_action_history",
    "get_schema_capabilities",
    "management_to_realtime_input",
    "normalize_decision_request",
    "ModelSoilLayer",
    "SoilObservationAdapter",
    "SoilObservationError",
    "mg_n_kg_to_kg_ha",
    "volumetric_water_to_cm_water",
]
