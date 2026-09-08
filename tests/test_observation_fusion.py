import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from serving.decision_engine import DecisionEngine
from serving.observation_fusion import ObservationFusionEngine
from serving.schemas import SchemaValidationError, get_schema_capabilities


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


def canonical_payload(query_date="2025-07-14", observations=None):
    return {
        "schema_version": "1.0",
        "request": {"request_id": "fusion-test-001", "user_query": "是否需要追氮？"},
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
        "decision_context": {},
    }


def observation_payload(observation_date="2025-07-14", crop=None, soil=None):
    return {
        "observation_date": observation_date,
        "source": "field_measurement",
        "crop": crop,
        "soil": [] if soil is None else soil,
    }


class TestObservationFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DecisionEngine(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            env_stats_path=ENV_STATS_PATH,
            device="cpu",
        )
        cls.fusion = ObservationFusionEngine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    @staticmethod
    def raw_observation():
        raw = np.arange(22, dtype=np.float64)
        raw[2] = 1.483216
        return raw

    def test_no_observations_leave_raw_observation_unchanged(self):
        raw = self.raw_observation()
        result = self.fusion.fuse(raw, None, "2025-07-14")
        np.testing.assert_array_equal(result["corrected_raw_observation"], raw)
        self.assertEqual(result["audit"]["applied"], [])

    def test_same_date_lai_replaces_feature_index_two(self):
        raw = self.raw_observation()
        result = self.fusion.fuse(
            raw,
            observation_payload(crop={"lai": 1.60}),
            "2025-07-14",
        )
        self.assertEqual(result["corrected_raw_observation"][2], 1.60)
        self.assertEqual(result["audit"]["applied"][0]["feature_index"], 2)
        self.assertTrue(result["audit"]["applied_to_policy"])

    def test_decision_engine_does_not_mutate_wofost_state(self):
        baseline = self.engine.decide(canonical_payload())
        observed = self.engine.decide(
            canonical_payload(
                observations=observation_payload(crop={"lai": 1.60})
            )
        )
        self.assertEqual(observed["model_state"], baseline["model_state"])
        self.assertFalse(observed["observations"]["applied_to_wofost_state"])
        self.assertFalse(observed["observation_fusion"]["wofost_state_mutated"])

    def test_corrected_raw_normalizes_to_finite_float32(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(crop={"lai": 1.60}),
            "2025-07-14",
        )
        normalized = self.engine.rl.normalize_raw_observation(
            result["corrected_raw_observation"]
        )
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.isfinite(normalized).all())

    def test_decision_engine_predicts_from_corrected_lai(self):
        fake_prediction = {"action_index": 0, "n_rate_kg_ha": 0.0}
        with patch.object(
            self.engine.rl,
            "predict_from_raw_observation",
            return_value=fake_prediction,
        ) as predict:
            result = self.engine.decide(
                canonical_payload(
                    observations=observation_payload(crop={"lai": 1.60})
                )
            )
        passed_raw = predict.call_args.args[0]
        self.assertEqual(float(passed_raw[2]), 1.60)
        self.assertEqual(result["recommendation"]["action_index"], 0)
        self.assertTrue(result["observations"]["applied_to_policy"])

    def test_observation_date_mismatch_is_explicit_and_unapplied(self):
        raw = self.raw_observation()
        result = self.fusion.fuse(
            raw,
            observation_payload("2025-07-13", crop={"lai": 1.60}),
            "2025-07-14",
        )
        np.testing.assert_array_equal(result["corrected_raw_observation"], raw)
        self.assertEqual(result["audit"]["date_mismatch"], "OBSERVATION_DATE_MISMATCH")
        self.assertFalse(result["audit"]["applied_to_policy"])

    def test_negative_lai_fails_schema_validation(self):
        with self.assertRaisesRegex(SchemaValidationError, "observations.crop.lai"):
            self.engine.decide(
                canonical_payload(observations=observation_payload(crop={"lai": -0.1}))
            )

    def test_spad_is_preserved_but_not_applied(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(crop={"spad": 47.0}),
            "2025-07-14",
        )
        self.assertFalse(result["audit"]["applied_to_policy"])
        self.assertEqual(
            result["audit"]["not_applied"][0]["canonical_field"],
            "observations.crop.spad",
        )
        self.assertEqual(result["audit"]["not_applied"][0]["reason"], "NO_DIRECT_RL_FEATURE")

    def test_soil_no3_nh4_wc_remain_reserved(self):
        soil = [{
            "depth_top_cm": 0,
            "depth_bottom_cm": 30,
            "no3_n_mg_kg": 45.8,
            "nh4_n_mg_kg": 2.6,
            "volumetric_water_content": 0.24,
        }]
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=soil),
            "2025-07-14",
        )
        self.assertFalse(result["audit"]["applied_to_policy"])
        self.assertEqual(len(result["audit"]["not_applied"]), 3)
        self.assertTrue(all(
            item["reason"] == "RESERVED_UNIT_OR_LAYER_CONVERSION_UNCONFIRMED"
            for item in result["audit"]["not_applied"]
        ))

    def test_audit_is_json_safe(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(crop={"lai": 1.60, "spad": 47.0}),
            "2025-07-14",
        )
        json.dumps(result["audit"], ensure_ascii=False)

    def test_non_decision_query_does_not_trigger_policy(self):
        payload = canonical_payload(
            query_date="2025-07-17",
            observations=observation_payload("2025-07-17", crop={"lai": 1.60}),
        )
        with patch.object(self.engine.rl, "predict_from_raw_observation") as predict:
            result = self.engine.decide(payload)
        predict.assert_not_called()
        self.assertIsNone(result["recommendation"])
        self.assertEqual(result["observation_fusion"]["mode"], "not_a_decision_boundary")
        self.assertFalse(result["observations"]["applied_to_policy"])

    def test_no_observation_decision_is_stable(self):
        first = self.engine.decide(canonical_payload())
        second = self.engine.decide(canonical_payload())
        self.assertEqual(first, second)

    def test_schema_capability_marks_only_lai_active(self):
        capabilities = get_schema_capabilities()["capabilities"]
        self.assertEqual(
            capabilities["observations.crop.lai"]["status"],
            "ACTIVE_DECISION_OVERRIDE",
        )
        for field in (
            "observations.crop.spad",
            "observations.soil.no3_n_mg_kg",
            "observations.soil.nh4_n_mg_kg",
            "observations.soil.volumetric_water_content",
        ):
            self.assertEqual(capabilities[field]["status"], "RESERVED")


if __name__ == "__main__":
    unittest.main()
