import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from serving.decision_engine import DecisionEngine
from serving.rl_inference import InferenceCompatibilityError, RLInferenceEngine
from serving.schemas import SchemaValidationError
from serving.wofost_realtime import QueryDateError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "CN-Maize.yaml"
RUN_DIR = (
    PROJECT_ROOT
    / "tensorboard_logs"
    / "WOFOST_maize_experiments"
    / "CN-Maize-Seed-107-nsteps-2208-LagPPO-POT-run_1"
)
MODEL_PATH = RUN_DIR / "latest-model.zip"
ENV_STATS_PATH = RUN_DIR / "latest-env.pkl"


def canonical_payload(query_date="2025-07-14"):
    return {
        "schema_version": "1.0",
        "request": {"request_id": "decision-test-001", "user_query": "是否需要追氮？"},
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
        "observations": None,
        "weather": {"provider": "openmeteo", "use_external_provider": True},
        "decision_context": {},
    }


class TestDecisionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DecisionEngine(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            env_stats_path=ENV_STATS_PATH,
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_decision_calendar_is_dynamic_and_seven_day(self):
        calendar = self.engine.decision_calendar()
        self.assertEqual(calendar[0], {
            "observation_date": "2025-06-09",
            "action_application_date": "2025-06-16",
            "interval_start": "2025-06-09",
            "interval_end": "2025-06-16",
        })
        self.assertEqual(calendar[1]["observation_date"], "2025-06-16")
        self.assertEqual(calendar[1]["action_application_date"], "2025-06-23")
        self.assertLessEqual(len(calendar), 30)

    def test_boundary_decision_reconstructs_and_predicts(self):
        result = self.engine.decide(canonical_payload())
        self.assertTrue(result["decision_due"])
        self.assertEqual(result["state_date"], "2025-07-14")
        self.assertEqual(result["policy_observation_date"], "2025-07-14")
        self.assertEqual(result["action_application_date"], "2025-07-21")
        self.assertIsNotNone(result["recommendation"])
        self.assertIn(result["recommendation"]["action_index"], range(16))
        self.assertEqual(result["model_metadata"]["raw_observation_shape"], [22])

    def test_non_boundary_query_does_not_predict(self):
        result = self.engine.decide(canonical_payload("2025-07-17"))
        self.assertFalse(result["decision_due"])
        self.assertIsNone(result["recommendation"])
        self.assertEqual(result["state_date"], "2025-07-17")
        self.assertEqual(result["previous_decision_date"], "2025-07-14")
        self.assertEqual(result["next_decision_date"], "2025-07-21")
        self.assertEqual(result["action_application_date"], None)

    def test_exact_feature_order_and_raw_observation_dimension(self):
        result = self.engine.decide(canonical_payload("2025-06-09"))
        self.assertEqual(result["model_metadata"]["raw_observation_shape"], [22])
        self.assertEqual(
            result["model_metadata"]["observation_feature_names"],
            [
                "DVS", "TAGP", "LAI", "NuptakeTotal", "NO3", "NH4", "WC",
                "RFTRA", "WSO", "NLOSSCUM", "RNO3DEPOSTT", "RNH4DEPOSTT",
                "NamountSO", "week", "Naction", "NUE", "Nsurp",
                "action_history", "IRRAD", "TMIN", "TMAX", "RAIN",
            ],
        )
        self.assertEqual(result["model_metadata"]["raw_observation_dtype"], "float64")

    def test_realtime_raw_normalization_matches_rl_environment(self):
        realtime = self.engine.realtime.reconstruct("2025-07-14", [0, 0, 0, 0, 0])
        self.engine.rl.reset(seed=107)
        for _ in range(5):
            self.engine.rl.step(0)
        expected = self.engine.rl._last_observation.copy()
        actual = self.engine.rl.normalize_raw_observation(realtime["raw_observation"])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(actual.dtype, np.float32)

    def test_result_is_json_safe_and_observations_are_not_assimilated(self):
        payload = canonical_payload()
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "field_measurement",
            "crop": {"lai": 1.48},
            "soil": [],
        }
        result = self.engine.decide(payload)
        json.dumps(result, ensure_ascii=False)
        self.assertTrue(result["observations"]["received"])
        self.assertFalse(result["observations"]["applied_to_model"])

    def test_incomplete_fertilization_history_fails_explicitly(self):
        payload = canonical_payload()
        payload["management"]["fertilization_history_complete"] = False
        with self.assertRaisesRegex(ValueError, "INCOMPLETE_MANAGEMENT_HISTORY"):
            self.engine.decide(payload)

    def test_custom_irrigation_fails_explicitly(self):
        payload = canonical_payload()
        payload["management"]["irrigation_history"] = [
            {"date": "2025-06-16", "amount_mm": 10}
        ]
        with self.assertRaisesRegex(ValueError, "CUSTOM_IRRIGATION_NOT_YET_SUPPORTED"):
            self.engine.decide(payload)

    def test_fertilization_disabled_skips_policy(self):
        payload = canonical_payload()
        payload["decision_context"] = {"allow_fertilization_decision": False}
        with patch.object(self.engine.rl, "predict_from_raw_observation") as predict:
            result = self.engine.decide(payload)
        self.assertTrue(result["decision_due"])
        self.assertIsNone(result["recommendation"])
        self.assertIn("fertilization_decision_disabled", result["warnings"])
        predict.assert_not_called()

    def test_max_rate_violation_is_reported_without_clipping(self):
        payload = canonical_payload()
        payload["decision_context"] = {"max_single_n_rate_kg_ha": 20}
        fake_prediction = {
            "action_index": 3,
            "n_rate_kg_ha": 30.0,
        }
        with patch.object(
            self.engine.rl,
            "predict_from_raw_observation",
            return_value=fake_prediction,
        ):
            result = self.engine.decide(payload)
        self.assertTrue(result["constraint_violation"])
        self.assertTrue(result["recommendation"]["constraint_violation"])
        self.assertEqual(result["recommendation"]["action_index"], 3)
        self.assertEqual(result["recommendation"]["n_rate_kg_ha"], 30.0)

    def test_query_date_bounds_fail_explicitly(self):
        with self.assertRaisesRegex(SchemaValidationError, "query_date"):
            self.engine.decide(canonical_payload("2025-06-08"))
        with self.assertRaisesRegex(QueryDateError, "QUERY_DATE_AFTER_CROP_END"):
            self.engine.decide(canonical_payload("2025-10-16"))

    def test_wrong_model_environment_compatibility_fails_explicitly(self):
        with patch.object(
            RLInferenceEngine,
            "_validate_compatibility",
            side_effect=InferenceCompatibilityError("Observation space mismatch"),
        ):
            with self.assertRaisesRegex(InferenceCompatibilityError, "Observation space mismatch"):
                DecisionEngine(
                    config_path=CONFIG_PATH,
                    model_path=MODEL_PATH,
                    env_stats_path=ENV_STATS_PATH,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
