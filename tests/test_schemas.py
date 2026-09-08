import json
import unittest

from serving.schemas import SchemaValidationError, normalize_decision_request


def valid_payload():
    return {
        "location": {"latitude": 36.77, "longitude": 114.96},
        "crop": {
            "name": "maize",
            "cultivar": "Quzhou_maize_2025_Opt",
            "sowing_date": "2025-06-09",
        },
        "query_date": "2025-07-14",
        "management": {
            "fertilization_history": [
                {"date": "2025-06-23", "n_rate_kg_ha": 30}
            ],
            "irrigation_history": [],
            "fertilization_history_complete": True,
        },
        "observations": {
            "date": "2025-07-14",
            "lai": 1.48,
            "soil_water": 7.4,
            "no3": 45.8,
            "nh4": 2.6,
        },
    }


class TestSchemas(unittest.TestCase):
    def test_valid_canonical_payload(self):
        request = normalize_decision_request(valid_payload())
        self.assertEqual(request.crop.sowing_date.isoformat(), "2025-06-09")
        self.assertEqual(request.management.fertilization_history[0].n_rate_kg_ha, 30.0)

    def test_invalid_coordinates(self):
        payload = valid_payload()
        payload["location"]["latitude"] = 91
        with self.assertRaisesRegex(SchemaValidationError, "location.latitude"):
            normalize_decision_request(payload)

        payload = valid_payload()
        payload["location"]["longitude"] = -181
        with self.assertRaisesRegex(SchemaValidationError, "location.longitude"):
            normalize_decision_request(payload)

    def test_query_before_sowing(self):
        payload = valid_payload()
        payload["query_date"] = "2025-06-08"
        with self.assertRaisesRegex(SchemaValidationError, "query_date"):
            normalize_decision_request(payload)

    def test_negative_values_are_rejected(self):
        payload = valid_payload()
        payload["management"]["fertilization_history"][0]["n_rate_kg_ha"] = -1
        with self.assertRaisesRegex(SchemaValidationError, "n_rate_kg_ha"):
            normalize_decision_request(payload)

        payload = valid_payload()
        payload["observations"]["lai"] = -0.1
        with self.assertRaisesRegex(SchemaValidationError, "observations.lai"):
            normalize_decision_request(payload)

    def test_noncanonical_fields_are_rejected(self):
        payload = valid_payload()
        payload["management"]["fertilization_history"][0]["N"] = 30
        with self.assertRaisesRegex(SchemaValidationError, "unsupported field"):
            normalize_decision_request(payload)

    def test_normalized_request_is_json_serializable(self):
        request = normalize_decision_request(valid_payload())
        json.dumps(request.to_dict())


if __name__ == "__main__":
    unittest.main()
