import json
import unittest
from pathlib import Path

import numpy as np

from serving.schemas import SchemaValidationError, SoilLayerObservation
from serving.soil_observation import (
    INSUFFICIENT_INPUT,
    READY_FOR_DECISION_OVERRIDE,
    ModelSoilLayer,
    SoilObservationAdapter,
    SoilObservationError,
    mg_n_kg_to_kg_ha,
    volumetric_water_to_cm_water,
)
from serving.wofost_realtime import WOFOSTRealtimeEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "CN-Maize.yaml"
SOIL_PATH = PROJECT_ROOT / "pcse_gym" / "envs" / "configs" / "soil" / "quzhou_maize_4layer_whcns_2025.yaml"


def soil_layer(top, bottom, *, no3=None, nh4=None, vwc=None, density=None, soil_water=None):
    return SoilLayerObservation(
        depth_top_cm=top,
        depth_bottom_cm=bottom,
        bulk_density_g_cm3=density,
        no3_n_mg_kg=no3,
        nh4_n_mg_kg=nh4,
        volumetric_water_content=vwc,
        soil_water=soil_water,
    )


class TestSoilObservationAdapter(unittest.TestCase):
    def test_bulk_density_validation(self):
        with self.assertRaisesRegex(SchemaValidationError, "bulk_density_g_cm3"):
            SoilLayerObservation.from_mapping(
                {"depth_top_cm": 0, "depth_bottom_cm": 20, "bulk_density_g_cm3": 0},
                0,
            )
        with self.assertRaisesRegex(SchemaValidationError, "bulk_density_g_cm3"):
            SoilLayerObservation.from_mapping(
                {"depth_top_cm": 0, "depth_bottom_cm": 20, "bulk_density_g_cm3": -1},
                0,
            )

    def test_layer_top_bottom_validation(self):
        with self.assertRaisesRegex(SchemaValidationError, "depth_bottom_cm"):
            SoilLayerObservation.from_mapping(
                {"depth_top_cm": 20, "depth_bottom_cm": 20},
                0,
            )
        with self.assertRaises(SoilObservationError):
            ModelSoilLayer(0, 0, 30, thickness_cm=20)

    def test_mg_per_kg_conversion_hand_calculation(self):
        # 10 mg/kg * 1.5 g/cm3 * 20 cm * 0.1 = 30 kg N/ha.
        self.assertAlmostEqual(mg_n_kg_to_kg_ha(10, 1.5, 20), 30.0)

    def test_volumetric_water_conversion(self):
        self.assertAlmostEqual(volumetric_water_to_cm_water(0.24, 30), 7.2)

    def test_exact_layer_mapping(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30, bulk_density_g_cm3=1.4)])
        result = adapter.adapt([soil_layer(0, 30, no3=10, nh4=2, vwc=0.2, density=1.4)])
        self.assertEqual(result["feature_status"], {
            "NO3": READY_FOR_DECISION_OVERRIDE,
            "NH4": READY_FOR_DECISION_OVERRIDE,
            "WC": READY_FOR_DECISION_OVERRIDE,
        })
        self.assertAlmostEqual(result["rl_features"]["NO3"], 42.0)
        self.assertAlmostEqual(result["rl_features"]["NH4"], 8.4)
        self.assertAlmostEqual(result["rl_features"]["WC"], 6.0)
        self.assertEqual(result["model_layers"][0]["source_observation_layers"], [0])
        self.assertAlmostEqual(result["model_layers"][0]["overlap_fraction"], 1.0)

    def test_partial_overlap_is_audited(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30)])
        result = adapter.adapt([soil_layer(0, 15, no3=10, density=1.4)])
        layer = result["model_layers"][0]
        self.assertAlmostEqual(layer["overlap_fraction"], 0.5)
        self.assertEqual(layer["source_observation_layers"], [0])
        self.assertEqual(result["rl_features"]["NO3"], None)
        self.assertEqual(result["feature_status"]["NO3"], INSUFFICIENT_INPUT)

    def test_multiple_observed_layers_to_one_model_layer(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30)])
        result = adapter.adapt([
            soil_layer(0, 10, no3=1, nh4=1, vwc=0.2, density=1.0),
            soil_layer(10, 30, no3=2, nh4=2, vwc=0.3, density=1.0),
        ])
        self.assertAlmostEqual(result["rl_features"]["NO3"], 5.0)
        self.assertAlmostEqual(result["rl_features"]["NH4"], 5.0)
        self.assertAlmostEqual(result["rl_features"]["WC"], 8.0)
        self.assertEqual(result["model_layers"][0]["source_observation_layers"], [0, 1])

    def test_one_observed_layer_to_multiple_model_layers(self):
        adapter = SoilObservationAdapter([
            ModelSoilLayer(0, 0, 30),
            ModelSoilLayer(1, 30, 60),
        ])
        result = adapter.adapt([
            soil_layer(0, 60, no3=1, nh4=2, vwc=0.2, density=1.0),
        ])
        self.assertAlmostEqual(result["rl_features"]["NO3"], 6.0)
        self.assertAlmostEqual(result["rl_features"]["NH4"], 12.0)
        self.assertAlmostEqual(result["rl_features"]["WC"], 6.0)
        self.assertEqual(result["model_layers"][0]["source_observation_layers"], [0])
        self.assertEqual(result["model_layers"][1]["source_observation_layers"], [0])

    def test_insufficient_depth_coverage(self):
        adapter = SoilObservationAdapter([
            ModelSoilLayer(0, 0, 30),
            ModelSoilLayer(1, 30, 60),
        ])
        result = adapter.adapt([soil_layer(0, 30, no3=10, nh4=2, vwc=0.2, density=1.0)])
        self.assertEqual(result["feature_status"]["NO3"], INSUFFICIENT_INPUT)
        self.assertEqual(result["feature_status"]["NH4"], INSUFFICIENT_INPUT)
        self.assertEqual(result["feature_status"]["WC"], INSUFFICIENT_INPUT)

    def test_missing_bulk_density_only_blocks_nitrogen_conversion(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30)])
        result = adapter.adapt([soil_layer(0, 30, no3=10, nh4=2, vwc=0.2)])
        self.assertEqual(result["feature_status"]["NO3"], INSUFFICIENT_INPUT)
        self.assertEqual(result["feature_status"]["NH4"], INSUFFICIENT_INPUT)
        self.assertEqual(result["feature_status"]["WC"], READY_FOR_DECISION_OVERRIDE)
        self.assertEqual(result["rl_features"]["WC"], 6.0)

    def test_soil_water_is_preserved_but_not_used(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30)])
        result = adapter.adapt([soil_layer(0, 30, soil_water=7.2)])
        self.assertEqual(result["feature_status"]["WC"], INSUFFICIENT_INPUT)
        self.assertEqual(
            result["audit"]["ignored_fields"][0]["reason"],
            "AMBIGUOUS_PCSE_WATER_SEMANTICS",
        )

    def test_json_serialization(self):
        adapter = SoilObservationAdapter([ModelSoilLayer(0, 0, 30)])
        result = adapter.adapt([soil_layer(0, 30, no3=10, nh4=2, vwc=0.2, density=1.0)])
        json.dumps(result, allow_nan=False)

    @classmethod
    def setUpClass(cls):
        cls.adapter = SoilObservationAdapter.from_soil_yaml(SOIL_PATH)
        cls.engine = WOFOSTRealtimeEngine(CONFIG_PATH, env_split="test", seed=107)
        cls.engine.reconstruct("2025-07-14", [0, 0, 0, 0, 0])
        output = cls.engine.env.model.get_output()[-1]
        cls.raw_no3 = np.asarray(output["NO3"], dtype=float).copy()
        cls.raw_nh4 = np.asarray(output["NH4"], dtype=float).copy()
        cls.raw_wc = np.asarray(output["WC"], dtype=float).copy()
        cls.roundtrip_observations = []
        for index, layer in enumerate(cls.adapter.model_layers):
            no3_kg_ha = cls.raw_no3[index] / 1e-4
            nh4_kg_ha = cls.raw_nh4[index] / 1e-4
            density = layer.bulk_density_g_cm3
            cls.roundtrip_observations.append(
                soil_layer(
                    layer.depth_top_cm,
                    layer.depth_bottom_cm,
                    no3=no3_kg_ha / (density * layer.thickness_cm * 0.1),
                    nh4=nh4_kg_ha / (density * layer.thickness_cm * 0.1),
                    vwc=cls.raw_wc[index] / layer.thickness_cm,
                    density=density,
                )
            )

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def _roundtrip(self):
        return self.adapter.adapt(self.roundtrip_observations)

    def test_no3_round_trip_matches_sb3_scalar(self):
        result = self._roundtrip()
        expected = float(np.sum(self.raw_no3) / 1e-4)
        self.assertAlmostEqual(result["rl_features"]["NO3"], expected, places=10)
        self.assertEqual(result["feature_status"]["NO3"], READY_FOR_DECISION_OVERRIDE)

    def test_nh4_round_trip_matches_sb3_scalar(self):
        result = self._roundtrip()
        expected = float(np.sum(self.raw_nh4) / 1e-4)
        self.assertAlmostEqual(result["rl_features"]["NH4"], expected, places=10)
        self.assertEqual(result["feature_status"]["NH4"], READY_FOR_DECISION_OVERRIDE)

    def test_wc_round_trip_matches_sb3_scalar(self):
        result = self._roundtrip()
        expected = float(np.mean(self.raw_wc))
        self.assertAlmostEqual(result["rl_features"]["WC"], expected, places=10)
        self.assertEqual(result["feature_status"]["WC"], READY_FOR_DECISION_OVERRIDE)

    def test_adapter_does_not_mutate_wofost_state(self):
        before_no3 = self.raw_no3.copy()
        before_nh4 = self.raw_nh4.copy()
        before_wc = self.raw_wc.copy()
        self._roundtrip()
        output = self.engine.env.model.get_output()[-1]
        np.testing.assert_array_equal(np.asarray(output["NO3"], dtype=float), before_no3)
        np.testing.assert_array_equal(np.asarray(output["NH4"], dtype=float), before_nh4)
        np.testing.assert_array_equal(np.asarray(output["WC"], dtype=float), before_wc)


if __name__ == "__main__":
    unittest.main()
