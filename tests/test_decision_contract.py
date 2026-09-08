import json
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from serving.decision_contract import (
    DECISION_RESULT_TOP_LEVEL_KEYS,
    build_decision_result,
    serialize_decision_error,
)
from serving.decision_engine import DecisionEngine
from serving.errors import (
    CustomIrrigationNotSupportedError,
    DecisionEngineError,
    ModelArtifactMissingError,
    RequestValidationError,
    WeatherProviderError,
)
from serving.management_history import ManagementHistoryError
from serving.schemas import SchemaValidationError


def canonical_payload(query_date="2025-07-14", *, observations=None, context=None):
    return {
        "schema_version": "1.0",
        "request": {"request_id": "contract-001", "user_query": "是否需要追氮？"},
        "location": {"latitude": 36.77, "longitude": 114.96},
        "crop": {
            "name": "maize",
            "cultivar": "Quzhou_maize_2025_Opt",
            "sowing_date": "2025-06-09",
        },
        "query_date": query_date,
        "management": {
            "fertilization_history": [],
            "irrigation_history": [],
            "fertilization_history_complete": True,
            "irrigation_history_complete": True,
        },
        "observations": observations,
        "weather": {"provider": "openmeteo", "use_external_provider": True},
        "decision_context": context or {},
    }


class FakeRealtime:
    crop_start_date = date(2025, 6, 9)
    crop_end_date = date(2025, 10, 15)
    timestep_days = 7
    action_count = 16

    def reconstruct(self, query_date, action_history):
        return {
            "simulation_date": query_date.isoformat(),
            "crop_state": {"LAI": np.float64(1.483216), "DVS": np.float64(0.8)},
            "raw_observation": np.arange(22, dtype=np.float64),
            "raw_observation_shape": [22],
            "raw_observation_dtype": "float64",
            "observation_feature_names": [f"feature-{i}" for i in range(22)],
            "observation_space_dtype": "float32",
        }

    def close(self):
        pass


class FakeFusion:
    def fuse(self, raw_observation, observations, query_date):
        corrected = np.asarray(raw_observation).copy()
        audit = {
            "mode": "decision_observation_override" if observations else "no_observation",
            "wofost_state_mutated": False,
            "applied": [],
            "not_applied": [],
            "applied_to_policy": bool(observations),
            "applied_to_wofost_state": False,
        }
        if observations and observations.crop and observations.crop.lai is not None:
            corrected[2] = observations.crop.lai
            audit["applied"].append({
                "canonical_field": "observations.crop.lai",
                "feature_index": 2,
                "observed_value": observations.crop.lai,
            })
        return {"corrected_raw_observation": corrected, "audit": audit}


class FakeRL:
    def __init__(self, *, validation_status="unknown", failure=None):
        self.config = {"environment": {"timestep": 7}}
        self.env = SimpleNamespace(action_space=gym.spaces.Discrete(16))
        self.validation_status = validation_status
        self.failure = failure

    def predict_from_raw_observation(self, raw_observation, deterministic=True):
        if self.failure is not None:
            raise self.failure
        return {"action_index": 3, "n_rate_kg_ha": 30.0}

    def get_metadata(self):
        return {
            "algorithm": "LagrangianPPO",
            "observation_dim": np.int64(22),
            "action_n": 16,
            "timestep_days": 7,
            "validation_status": self.validation_status,
            "validated_for_agronomic_recommendation": False,
            "model_path": r"E:\private\latest-model.zip",
            "env_stats_path": r"E:\private\latest-env.pkl",
        }

    def close(self):
        pass


def fake_engine(*, validation_status="unknown", rl_failure=None):
    engine = object.__new__(DecisionEngine)
    engine.realtime = FakeRealtime()
    engine.observation_fusion = FakeFusion()
    engine.rl = FakeRL(validation_status=validation_status, failure=rl_failure)
    return engine


class TestDecisionContract(unittest.TestCase):
    def test_boundary_has_fixed_contract_keys_and_recommendation(self):
        result = fake_engine(validation_status="engineering_only").decide(canonical_payload())
        self.assertTrue(set(DECISION_RESULT_TOP_LEVEL_KEYS).issubset(result))
        self.assertEqual(result["request_id"], "contract-001")
        self.assertEqual(result["recommendation"]["decision_type"], "nitrogen")
        self.assertEqual(result["recommendation"]["action_index"], 3)
        self.assertEqual(result["action_application_date"], "2025-07-21")
        self.assertEqual(result["model_metadata"]["validation_status"], "engineering_only")
        self.assertFalse(result["model_metadata"]["validated_for_agronomic_recommendation"])

    def test_non_boundary_has_fixed_keys_and_no_recommendation(self):
        result = fake_engine().decide(canonical_payload("2025-07-17"))
        self.assertTrue(set(DECISION_RESULT_TOP_LEVEL_KEYS).issubset(result))
        self.assertFalse(result["decision_due"])
        self.assertIsNone(result["recommendation"])
        self.assertIsNone(result["action_application_date"])

    def test_observation_audit_and_metadata_are_json_safe_without_paths(self):
        payload = canonical_payload(
            observations={
                "observation_date": "2025-07-14",
                "source": "field_measurement",
                "crop": {"lai": 1.6},
                "soil": [],
            }
        )
        result = fake_engine().decide(payload)
        encoded = json.dumps(result)
        self.assertNotIn(r"E:\private", encoded)
        self.assertEqual(result["observations"]["fusion_audit"]["applied"][0]["feature_index"], 2)
        self.assertEqual(result["model_metadata"]["model_identifier"], "latest-model.zip")
        self.assertEqual(json.loads(encoded)["state_date"], "2025-07-14")

    def test_disabled_fertilization_is_normal_state(self):
        result = fake_engine().decide(
            canonical_payload(context={"allow_fertilization_decision": False})
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["decision_due"])
        self.assertIsNone(result["recommendation"])
        self.assertIn("fertilization_decision_disabled", result["warnings"])

    def test_constraint_violation_is_returned_without_clipping(self):
        result = fake_engine().decide(
            canonical_payload(context={"max_single_n_rate_kg_ha": 20})
        )
        recommendation = result["recommendation"]
        self.assertTrue(recommendation["constraint_violation"])
        self.assertEqual(recommendation["n_rate_kg_ha"], 30.0)
        self.assertEqual(recommendation["constraint_details"]["exceeded_by_kg_ha"], 10.0)

    def test_default_validation_is_conservative(self):
        result = fake_engine().decide(canonical_payload())
        self.assertEqual(result["model_metadata"]["validation_status"], "unknown")
        self.assertFalse(result["model_metadata"]["validated_for_agronomic_recommendation"])

    def test_error_serialization_is_stable_and_json_safe(self):
        cases = [
            (RequestValidationError("bad request", field="query_date"), "REQUEST_VALIDATION_ERROR"),
            (ManagementHistoryError("INCOMPLETE_MANAGEMENT_HISTORY: missing"), "INCOMPLETE_FERTILIZATION_HISTORY"),
            (ManagementHistoryError("CUSTOM_IRRIGATION_NOT_YET_SUPPORTED: irrigation"), "CUSTOM_IRRIGATION_NOT_SUPPORTED"),
            (ModelArtifactMissingError("missing artifact"), "MODEL_ARTIFACT_MISSING"),
            (WeatherProviderError("provider unavailable"), "WEATHER_PROVIDER_ERROR"),
        ]
        for error, code in cases:
            envelope = serialize_decision_error(error, request_id="req-err")
            self.assertEqual(envelope["status"], "error")
            self.assertEqual(envelope["error"]["code"], code)
            self.assertIsNone(envelope["recommendation"])
            json.loads(json.dumps(envelope))

    def test_build_result_round_trips_numpy_and_dates(self):
        result = build_decision_result(
            request_id="round-trip",
            decision_due=False,
            state_date=date(2025, 7, 14),
            model_state={"LAI": np.float64(1.5)},
            model_metadata={"observation_dim": np.int64(22)},
        )
        self.assertEqual(json.loads(json.dumps(result))["state_date"], "2025-07-14")

    def test_unknown_programmer_exception_is_not_success(self):
        engine = fake_engine(rl_failure=RuntimeError("programmer failure"))
        with self.assertRaises(RuntimeError):
            engine.decide(canonical_payload())
        self.assertNotIsInstance(DecisionEngineError("x"), RuntimeError)


if __name__ == "__main__":
    unittest.main()
