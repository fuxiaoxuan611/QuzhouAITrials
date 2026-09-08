import json
import math
import unittest
from datetime import date
from pathlib import Path

from serving.management_history import (
    ManagementHistoryError,
    build_action_history,
    management_to_realtime_input,
)
from serving.schemas import normalize_decision_request
from serving.wofost_realtime import WOFOSTRealtimeEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "CN-Maize.yaml"


def request_payload(**management_overrides):
    management = {
        "fertilization_history": [],
        "irrigation_history": [],
        "fertilization_history_complete": True,
        "irrigation_history_complete": True,
    }
    management.update(management_overrides)
    return {
        "schema_version": "1.0",
        "request": {},
        "location": {"latitude": 36.77, "longitude": 114.96},
        "crop": {
            "name": "maize",
            "cultivar": "Quzhou_maize_2025_Opt",
            "sowing_date": "2025-06-09",
        },
        "query_date": "2025-07-14",
        "management": management,
        "weather": {"provider": "openmeteo", "use_external_provider": True},
        "decision_context": {},
    }


class TestManagementHistory(unittest.TestCase):
    def test_30_kg_maps_exactly_to_action_3(self):
        result = build_action_history(
            date(2025, 6, 9),
            date(2025, 6, 23),
            [{"date": "2025-06-23", "n_rate_kg_ha": 30}],
            fertilization_history_complete=True,
        )
        self.assertEqual(result["action_history"], [0, 3])

    def test_unrepresentable_rates_fail_without_rounding(self):
        with self.assertRaisesRegex(ManagementHistoryError, "UNREPRESENTABLE_N_RATE"):
            build_action_history(
                date(2025, 6, 9), date(2025, 6, 23),
                [{"date": "2025-06-23", "n_rate_kg_ha": 33}],
                fertilization_history_complete=True,
            )
        with self.assertRaisesRegex(ManagementHistoryError, "UNREPRESENTABLE_N_RATE"):
            build_action_history(
                date(2025, 6, 9), date(2025, 7, 14),
                [{"date": "2025-07-14", "n_rate_kg_ha": 160}],
                fertilization_history_complete=True,
            )

    def test_non_decision_date_fails(self):
        with self.assertRaisesRegex(
            ManagementHistoryError, "UNREPRESENTABLE_FERTILIZATION_DATE"
        ):
            build_action_history(
                date(2025, 6, 9), date(2025, 7, 14),
                [{"date": "2025-06-24", "n_rate_kg_ha": 30}],
                fertilization_history_complete=True,
            )

    def test_incomplete_history_fails_even_when_empty(self):
        with self.assertRaisesRegex(
            ManagementHistoryError, "INCOMPLETE_MANAGEMENT_HISTORY"
        ):
            build_action_history(
                date(2025, 6, 9), date(2025, 7, 14), [],
                fertilization_history_complete=False,
            )

    def test_complete_no_event_steps_are_zero_actions(self):
        result = build_action_history(
            date(2025, 6, 9), date(2025, 7, 14), [],
            fertilization_history_complete=True,
        )
        self.assertEqual(result["action_history"], [0, 0, 0, 0, 0])
        self.assertEqual(result["completed_steps"], 5)

    def test_residual_period_event_fails(self):
        with self.assertRaisesRegex(
            ManagementHistoryError, "UNSUPPORTED_RESIDUAL_PERIOD_MANAGEMENT"
        ):
            build_action_history(
                date(2025, 6, 9), date(2025, 7, 17),
                [{"date": "2025-07-15", "n_rate_kg_ha": 30}],
                fertilization_history_complete=True,
            )

    def test_custom_irrigation_is_explicitly_rejected(self):
        request = normalize_decision_request(
            request_payload(
                irrigation_history=[{"date": "2025-06-10", "amount_mm": 12.9}]
            )
        )
        with self.assertRaisesRegex(
            ManagementHistoryError, "CUSTOM_IRRIGATION_NOT_YET_SUPPORTED"
        ):
            management_to_realtime_input(request)

    def test_observations_are_preserved_but_not_applied(self):
        payload = request_payload()
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "field_measurement",
            "crop": {"lai": 1.48},
            "soil": [
                {
                    "depth_top_cm": 0,
                    "depth_bottom_cm": 20,
                    "soil_water": 7.4,
                    "no3_n_mg_kg": 45.8,
                    "nh4_n_mg_kg": 2.6,
                }
            ],
        }
        result = management_to_realtime_input(normalize_decision_request(payload))
        self.assertEqual(result["observations"]["crop"]["lai"], 1.48)
        self.assertFalse(result["observations_applied_to_model"])

    def test_query_before_sowing_fails(self):
        with self.assertRaisesRegex(ManagementHistoryError, "QUERY_DATE_BEFORE_SOWING"):
            build_action_history(
                date(2025, 6, 9), date(2025, 6, 8), [],
                fertilization_history_complete=True,
            )

    def test_converted_input_reconstructs_wofost_state(self):
        request = normalize_decision_request(request_payload())
        realtime_input = management_to_realtime_input(request)
        self.assertEqual(realtime_input["action_history"], [0, 0, 0, 0, 0])
        json.dumps(realtime_input)

        engine = WOFOSTRealtimeEngine(CONFIG_PATH, seed=107)
        try:
            state = engine.reconstruct(
                realtime_input["query_date"],
                realtime_input["action_history"],
            )
        finally:
            engine.close()
        self.assertEqual(state["simulation_date"], "2025-07-14")
        self.assertTrue(all(math.isfinite(float(value)) for value in state["crop_state"].values() if not isinstance(value, list)))


if __name__ == "__main__":
    unittest.main()
