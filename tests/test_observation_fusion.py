import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from serving.decision_engine import DecisionEngine
from serving.observation_fusion import ObservationFusionEngine, ObservationFusionError
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


def complete_soil_profile(*, no3=None, nh4=None, vwc=None, density=None):
    profile = []
    for top, bottom in ((0, 30), (30, 60), (60, 90), (90, 120)):
        layer = {"depth_top_cm": top, "depth_bottom_cm": bottom}
        if no3 is not None:
            layer["no3_n_mg_kg"] = no3
        if nh4 is not None:
            layer["nh4_n_mg_kg"] = nh4
        if vwc is not None:
            layer["volumetric_water_content"] = vwc
        if density is not None:
            layer["bulk_density_g_cm3"] = density
        profile.append(layer)
    return profile


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

    def test_partial_soil_profile_is_not_applied(self):
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
            item["reason"] == "INSUFFICIENT_DEPTH_COVERAGE"
            for item in result["audit"]["not_applied"]
        ))

    def test_complete_no3_only_overrides_index_four(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(no3=5.0)),
            "2025-07-14",
        )
        corrected = result["corrected_raw_observation"]
        self.assertAlmostEqual(corrected[4], 85.2)
        self.assertEqual(corrected[5], 5.0)
        self.assertEqual(corrected[6], 6.0)
        applied = result["audit"]["applied"]
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["feature_index"], 4)
        self.assertEqual(applied[0]["bulk_density_source"], "model_config")

    def test_complete_nh4_only_overrides_index_five(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(nh4=1.0)),
            "2025-07-14",
        )
        corrected = result["corrected_raw_observation"]
        self.assertAlmostEqual(corrected[5], 17.04)
        self.assertEqual(corrected[4], 4.0)
        self.assertEqual(corrected[6], 6.0)
        self.assertEqual(result["audit"]["applied"][0]["feature_index"], 5)

    def test_complete_wc_only_overrides_index_six(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(vwc=0.2)),
            "2025-07-14",
        )
        corrected = result["corrected_raw_observation"]
        self.assertAlmostEqual(corrected[6], 6.0)
        self.assertEqual(corrected[4], 4.0)
        self.assertEqual(corrected[5], 5.0)
        self.assertEqual(result["audit"]["applied"][0]["feature_index"], 6)

    def test_complete_soil_and_lai_override_all_supported_indexes(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(
                crop={"lai": 1.60},
                soil=complete_soil_profile(no3=5.0, nh4=1.0, vwc=0.2),
            ),
            "2025-07-14",
        )
        corrected = result["corrected_raw_observation"]
        self.assertEqual([corrected[i] for i in (2, 4, 5, 6)], [1.6, 85.2, 17.04, 6.0])
        self.assertEqual(
            {item["feature_index"] for item in result["audit"]["applied"]},
            {2, 4, 5, 6},
        )
        self.assertFalse(result["audit"]["applied_to_wofost_state"])

    def test_partial_no3_profile_is_not_applied(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(no3=5.0)[:2]),
            "2025-07-14",
        )
        self.assertEqual(result["corrected_raw_observation"][4], 4.0)
        self.assertEqual(result["audit"]["not_applied"][0]["reason"], "INSUFFICIENT_DEPTH_COVERAGE")

    def test_nh4_profile_gap_is_not_applied(self):
        profile = complete_soil_profile(nh4=1.0)
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=profile[:1] + profile[2:]),
            "2025-07-14",
        )
        self.assertEqual(result["corrected_raw_observation"][5], 5.0)
        self.assertEqual(result["audit"]["not_applied"][0]["reason"], "INSUFFICIENT_DEPTH_COVERAGE")

    def test_wc_incomplete_profile_is_not_applied(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(vwc=0.2)[:3]),
            "2025-07-14",
        )
        self.assertEqual(result["corrected_raw_observation"][6], 6.0)
        self.assertEqual(result["audit"]["not_applied"][0]["reason"], "INSUFFICIENT_DEPTH_COVERAGE")

    def test_overlapping_observed_layers_are_rejected(self):
        with self.assertRaisesRegex(ObservationFusionError, "OVERLAPPING_OBSERVATION_LAYERS"):
            self.fusion.fuse(
                self.raw_observation(),
                observation_payload(
                    soil=[
                        {"depth_top_cm": 0, "depth_bottom_cm": 30, "no3_n_mg_kg": 5.0},
                        {"depth_top_cm": 20, "depth_bottom_cm": 40, "no3_n_mg_kg": 5.0},
                    ]
                ),
                "2025-07-14",
            )

    def test_observation_bulk_density_overrides_model_density(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(soil=complete_soil_profile(no3=5.0, density=1.0)),
            "2025-07-14",
        )
        self.assertAlmostEqual(result["corrected_raw_observation"][4], 60.0)
        self.assertEqual(result["audit"]["applied"][0]["bulk_density_source"], "observation")

    def test_stale_soil_observation_is_not_applied(self):
        result = self.fusion.fuse(
            self.raw_observation(),
            observation_payload(
                observation_date="2025-07-13",
                soil=complete_soil_profile(no3=5.0, nh4=1.0, vwc=0.2),
            ),
            "2025-07-14",
        )
        np.testing.assert_array_equal(result["corrected_raw_observation"], self.raw_observation())
        self.assertEqual(result["audit"]["date_mismatch"], "OBSERVATION_DATE_MISMATCH")

    def test_complete_observation_reaches_vecnormalize_and_lagrangianppo(self):
        payload = canonical_payload(
            observations=observation_payload(
                crop={"lai": 1.60},
                soil=complete_soil_profile(no3=5.0, nh4=1.0, vwc=0.2),
            )
        )
        result = self.engine.decide(payload)
        self.assertIsNotNone(result["recommendation"])
        self.assertEqual(
            {item["feature_index"] for item in result["observation_fusion"]["applied"]},
            {2, 4, 5, 6},
        )
        self.assertTrue(result["observations"]["applied_to_policy"])
        self.assertFalse(result["observations"]["applied_to_wofost_state"])
        reconstructed = self.engine.realtime.reconstruct(
            "2025-07-14", [0, 0, 0, 0, 0]
        )
        fused = self.engine.observation_fusion.fuse(
            reconstructed["raw_observation"], payload["observations"], "2025-07-14"
        )
        normalized = self.engine.rl.normalize_raw_observation(
            fused["corrected_raw_observation"]
        )
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.isfinite(normalized).all())

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

    def test_schema_capability_marks_supported_soil_features_active(self):
        capabilities = get_schema_capabilities()["capabilities"]
        self.assertEqual(
            capabilities["observations.crop.lai"]["status"],
            "ACTIVE_DECISION_OVERRIDE",
        )
        self.assertEqual(capabilities["observations.crop.spad"]["status"], "RESERVED")
        for field in (
            "observations.soil.no3_n_mg_kg",
            "observations.soil.nh4_n_mg_kg",
            "observations.soil.volumetric_water_content",
        ):
            self.assertEqual(capabilities[field]["status"], "ACTIVE_DECISION_OVERRIDE")


if __name__ == "__main__":
    unittest.main()
