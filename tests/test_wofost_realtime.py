import json
import math
import unittest
from datetime import date
from pathlib import Path

from serving.wofost_realtime import (
    ActionHistoryError,
    QueryDateError,
    WOFOSTRealtimeEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "CN-Maize.yaml"


class TestWOFOSTRealtimeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = WOFOSTRealtimeEngine(CONFIG_PATH, seed=107)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_crop_start_requires_no_action(self):
        result = self.engine.reconstruct("2025-06-09", [])
        self.assertEqual(result["steps_executed"], 0)
        self.assertEqual(result["simulation_date"], "2025-06-09")

    def test_one_decision_step(self):
        result = self.engine.reconstruct(date(2025, 6, 16), [0])
        self.assertEqual(result["steps_executed"], 1)
        self.assertEqual(result["simulation_date"], "2025-06-16")
        self.assertEqual(result["residual_days"], 0)

    def test_five_steps_reconstruct_july_14(self):
        result = self.engine.reconstruct("2025-07-14", [0, 0, 0, 0, 0])
        self.assertEqual(result["steps_executed"], 5)
        self.assertEqual(result["simulation_date"], "2025-07-14")
        self.assertTrue(result["exact_date_supported"])
        self.assertTrue(result["crop_state"])

    def test_non_aligned_date_uses_exact_residual(self):
        result = self.engine.reconstruct("2025-07-17", [0, 0, 0, 0, 0])
        self.assertEqual(result["steps_executed"], 5)
        self.assertEqual(result["residual_days"], 3)
        self.assertEqual(result["simulation_date"], "2025-07-17")

    def test_invalid_action_indices(self):
        with self.assertRaisesRegex(ActionHistoryError, "INVALID_ACTION_INDEX"):
            self.engine.reconstruct("2025-06-16", [-1])
        with self.assertRaisesRegex(ActionHistoryError, "INVALID_ACTION_INDEX"):
            self.engine.reconstruct("2025-06-16", [16])

    def test_insufficient_history_is_explicit(self):
        with self.assertRaisesRegex(
            ActionHistoryError, "INSUFFICIENT_ACTION_HISTORY"
        ):
            self.engine.reconstruct("2025-07-14", [0, 0])

    def test_query_date_bounds_are_explicit(self):
        with self.assertRaisesRegex(QueryDateError, "QUERY_DATE_BEFORE_CROP_START"):
            self.engine.reconstruct("2025-06-08", [])
        with self.assertRaisesRegex(QueryDateError, "QUERY_DATE_AFTER_CROP_END"):
            self.engine.reconstruct("2025-10-16", [])

    def test_result_is_json_serializable_and_finite(self):
        result = self.engine.reconstruct("2025-07-14", [0, 0, 0, 0, 0])
        json.dumps(result)
        for value in result["crop_state"].values():
            if isinstance(value, list):
                self.assertTrue(all(math.isfinite(float(item)) for item in value))
            else:
                self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
