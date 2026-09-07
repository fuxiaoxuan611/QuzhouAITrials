import math
import unittest
from pathlib import Path

from gymnasium.spaces import Discrete

from serving.rl_inference import RLInferenceEngine, action_index_to_n_rate


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


class TestRLInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RLInferenceEngine(
            config_path=CONFIG_PATH,
            model_path=MODEL_PATH,
            env_stats_path=ENV_STATS_PATH,
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def setUp(self):
        self.engine.reset(seed=107)

    def test_engine_loads_smoke_assets(self):
        metadata = self.engine.get_metadata()
        self.assertEqual(metadata["algorithm"], "LagrangianPPO")
        self.assertTrue(Path(metadata["model_path"]).is_file())
        self.assertTrue(Path(metadata["env_stats_path"]).is_file())

    def test_observation_dimension(self):
        self.assertEqual(self.engine.env.observation_space.shape, (22,))

    def test_action_space(self):
        self.assertIsInstance(self.engine.env.action_space, Discrete)
        self.assertEqual(self.engine.env.action_space.n, 16)

    def test_deterministic_predict_returns_legal_action(self):
        result = self.engine.predict(deterministic=True)
        self.assertGreaterEqual(result["action_index"], 0)
        self.assertLess(result["action_index"], 16)
        self.assertTrue(result["deterministic"])

    def test_n_rate_mapping(self):
        rates = [action_index_to_n_rate(index) for index in range(16)]
        self.assertEqual(rates, list(range(0, 151, 10)))

    def test_one_step_is_finite_and_advances_date(self):
        result = self.engine.step(0)
        self.assertEqual(result["days_advanced"], 7)
        self.assertEqual(result["n_applied_kg_ha"], 0.0)
        self.assertTrue(math.isfinite(result["reward"]))
        self.assertIsInstance(result["terminated"], bool)
        self.assertIsInstance(result["truncated"], bool)

    def test_missing_model_path_is_explicit(self):
        with self.assertRaisesRegex(FileNotFoundError, "model file does not exist"):
            RLInferenceEngine(CONFIG_PATH, RUN_DIR / "missing.zip", ENV_STATS_PATH)


if __name__ == "__main__":
    unittest.main()
