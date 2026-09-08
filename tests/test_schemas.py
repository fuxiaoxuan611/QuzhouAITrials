import json
import unittest

from serving.schemas import (
    SCHEMA_VERSION,
    SchemaValidationError,
    get_schema_capabilities,
    normalize_decision_request,
)


def valid_payload():
    return {
        "schema_version": "1.0",
        "request": {
            "request_id": "req-001",
            "user_query": "曲周玉米现在需要追氮吗？",
        },
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
            "irrigation_history_complete": True,
        },
        "observations": None,
    }


class TestSchemas(unittest.TestCase):
    def test_valid_canonical_payload(self):
        request = normalize_decision_request(valid_payload())
        self.assertEqual(request.schema_version, SCHEMA_VERSION)
        self.assertEqual(request.request.request_id, "req-001")
        self.assertEqual(request.crop.sowing_date.isoformat(), "2025-06-09")
        self.assertEqual(request.management.fertilization_history[0].n_rate_kg_ha, 30.0)
        self.assertTrue(request.management.irrigation_history_complete)

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
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "field_measurement",
            "crop": {"lai": -0.1},
        }
        with self.assertRaisesRegex(SchemaValidationError, "observations.crop.lai"):
            normalize_decision_request(payload)

    def test_noncanonical_fields_are_rejected(self):
        payload = valid_payload()
        payload["management"]["fertilization_history"][0]["N"] = 30
        with self.assertRaisesRegex(SchemaValidationError, "unsupported field"):
            normalize_decision_request(payload)

    def test_extended_schema_round_trips_all_sections(self):
        payload = valid_payload()
        payload["location"].update(
            {
                "field_id": "QZ-001",
                "field_name": "曲周试验田",
                "elevation_m": 42.5,
                "timezone": "Asia/Shanghai",
            }
        )
        payload["crop"].update(
            {
                "planting_density_plants_ha": 67500,
                "row_spacing_m": 0.6,
                "expected_harvest_date": "2025-10-10",
                "phenology_stage_observed": 1.8,
            }
        )
        payload["management"]["fertilization_history"][0].update(
            {
                "fertilizer_name": "尿素",
                "fertilizer_type": "compound",
                "n_form": "mixed",
                "urea_n_fraction": 0.5,
                "nh4_n_fraction": 0.3,
                "no3_n_fraction": 0.2,
                "application_method": "broadcast",
                "notes": "基于田间记录",
            }
        )
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "field_measurement",
            "crop": {
                "lai": 1.48,
                "spad": 47,
                "canopy_cover": 0.62,
                "plant_height_m": 0.73,
                "aboveground_biomass_kg_ha": 3200,
                "yield_kg_ha": 8500,
                "leaf_n_concentration_g_kg": 28,
                "phenology_stage": 1.6,
            },
            "soil": [
                {
                    "depth_top_cm": 0,
                    "depth_bottom_cm": 20,
                    "soil_water": 7.4,
                    "volumetric_water_content": 0.24,
                    "no3_n_mg_kg": 45.8,
                    "nh4_n_mg_kg": 2.6,
                    "ec_ds_m": 0.41,
                    "ph": 7.8,
                    "soil_temperature_c": 24.1,
                }
            ],
            "notes": "人工测量",
        }
        payload["weather"] = {
            "provider": "openmeteo",
            "use_external_provider": True,
            "forecast_horizon_days": 7,
            "history": [
                {
                    "date": "2025-07-13",
                    "tmin_c": 21.0,
                    "tmax_c": 33.2,
                    "precipitation_mm": 0,
                    "solar_radiation": 21.5,
                }
            ],
            "forecast": [],
        }
        payload["decision_context"] = {
            "decision_type": "nitrogen",
            "forecast_horizon_days": 7,
            "allow_fertilization_decision": True,
            "allow_irrigation_decision": False,
            "max_single_n_rate_kg_ha": 80,
            "notes": "只允许氮肥决策",
        }

        request = normalize_decision_request(payload)
        self.assertEqual(request.location.field_id, "QZ-001")
        self.assertEqual(request.crop.expected_harvest_date.isoformat(), "2025-10-10")
        self.assertEqual(request.observations.crop.lai, 1.48)
        self.assertEqual(request.observations.soil[0].no3_n_mg_kg, 45.8)
        self.assertEqual(request.weather.history[0].precipitation_mm, 0.0)
        self.assertEqual(request.decision_context.max_single_n_rate_kg_ha, 80.0)
        json.dumps(request.to_dict(), ensure_ascii=False)

    def test_management_completion_flags_are_required(self):
        payload = valid_payload()
        del payload["management"]["irrigation_history_complete"]
        with self.assertRaisesRegex(SchemaValidationError, "irrigation_history_complete"):
            normalize_decision_request(payload)

    def test_invalid_fertilizer_fractions_are_rejected(self):
        payload = valid_payload()
        event = payload["management"]["fertilization_history"][0]
        event.update(
            {"urea_n_fraction": 0.5, "nh4_n_fraction": 0.5, "no3_n_fraction": 0.5}
        )
        with self.assertRaisesRegex(SchemaValidationError, "must sum to 1"):
            normalize_decision_request(payload)

        event["no3_n_fraction"] = 0.0
        event["urea_n_fraction"] = 1.2
        with self.assertRaisesRegex(SchemaValidationError, "must be in \[0, 1\]"):
            normalize_decision_request(payload)

    def test_invalid_soil_observation_is_rejected(self):
        payload = valid_payload()
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "sensor",
            "soil": [{"depth_top_cm": 20, "depth_bottom_cm": 10}],
        }
        with self.assertRaisesRegex(SchemaValidationError, "depth_bottom_cm"):
            normalize_decision_request(payload)

        payload["observations"]["soil"] = [
            {"depth_top_cm": 0, "depth_bottom_cm": 20, "no3_n_mg_kg": -1}
        ]
        with self.assertRaisesRegex(SchemaValidationError, "no3_n_mg_kg"):
            normalize_decision_request(payload)

        payload["observations"]["soil"] = [
            {"depth_top_cm": 0, "depth_bottom_cm": 20, "ph": 14.1}
        ]
        with self.assertRaisesRegex(SchemaValidationError, "ph"):
            normalize_decision_request(payload)

    def test_controlled_observation_source_and_weather_values(self):
        payload = valid_payload()
        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "telephone",
        }
        with self.assertRaisesRegex(SchemaValidationError, "unsupported source"):
            normalize_decision_request(payload)

        payload["observations"] = {
            "observation_date": "2025-07-14",
            "source": "unknown",
        }
        payload["weather"] = {
            "history": [{"date": "2025-07-13", "precipitation_mm": -0.1}]
        }
        with self.assertRaisesRegex(SchemaValidationError, "precipitation_mm"):
            normalize_decision_request(payload)

    def test_defaults_and_capability_report_are_json_safe(self):
        request = normalize_decision_request(valid_payload())
        self.assertEqual(request.weather.provider, "openmeteo")
        self.assertTrue(request.weather.use_external_provider)
        self.assertTrue(request.decision_context.allow_fertilization_decision)
        self.assertFalse(request.decision_context.allow_irrigation_decision)
        capabilities = get_schema_capabilities()
        self.assertEqual(capabilities["schema_version"], "1.0")
        self.assertEqual(capabilities["capabilities"]["fertilization_history"]["status"], "ACTIVE")
        self.assertEqual(capabilities["capabilities"]["observations.crop"]["status"], "RESERVED")
        self.assertEqual(capabilities["capabilities"]["irrigation_history"]["status"], "UNSUPPORTED")
        json.dumps(capabilities)


if __name__ == "__main__":
    unittest.main()
