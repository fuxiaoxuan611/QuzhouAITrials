import concurrent.futures
import json
import time
import unittest
import uuid
from datetime import date

from fastapi.testclient import TestClient

from serving.api import create_app
from serving.decision_contract import DECISION_RESULT_TOP_LEVEL_KEYS, build_decision_result
from serving.errors import (
    CustomIrrigationNotSupportedError,
    ModelArtifactMissingError,
    ModelEnvironmentIncompatibleError,
    QueryDateAfterCropEndError,
    RequestValidationError,
    WeatherProviderError,
)
from serving.settings import ServiceSettings


def request_payload(query_date="2025-07-14", *, request_id="api-001", context=None):
    request = {}
    if request_id is not None:
        request["request_id"] = request_id
    return {
        "schema_version": "1.0",
        "request": request,
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
        "decision_context": context or {},
    }


class FakeDecisionEngine:
    def __init__(self, *, mode=None):
        self.mode = mode
        self.active_calls = 0
        self.max_active_calls = 0
        self._counter_lock = __import__("threading").Lock()

    def get_metadata(self):
        return {
            "algorithm": "LagrangianPPO",
            "observation_dim": 22,
            "action_n": 16,
            "timestep_days": 7,
            "validation_status": "engineering_only",
            "validated_for_agronomic_recommendation": False,
            "model_path": r"E:\private\latest-model.zip",
            "env_stats_path": r"E:\private\latest-env.pkl",
        }

    def decide(self, payload):
        with self._counter_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.mode is not None:
                raise self.mode
            time.sleep(0.03)
            query_date = str(payload["query_date"])
            due = query_date == "2025-07-14"
            disabled = not payload["decision_context"].get(
                "allow_fertilization_decision", True
            )
            violation = payload["decision_context"].get("max_single_n_rate_kg_ha") == 20
            recommendation = None
            if due and not disabled:
                recommendation = {
                    "decision_type": "nitrogen",
                    "action_index": 3,
                    "n_rate_kg_ha": 30.0,
                    "constraint_violation": violation,
                    "constraint_details": (
                        {
                            "max_single_n_rate_kg_ha": 20.0,
                            "actual_n_rate_kg_ha": 30.0,
                            "exceeded_by_kg_ha": 10.0,
                        }
                        if violation
                        else {}
                    ),
                }
            return build_decision_result(
                request_id=payload["request"].get("request_id"),
                decision_due=due,
                state_date=query_date,
                policy_observation_date=query_date if due else None,
                previous_decision_date="2025-07-07" if not due else "2025-07-07",
                next_decision_date="2025-07-21",
                action_application_date="2025-07-21" if due else None,
                recommendation=recommendation,
                model_state={"LAI": 1.48},
                management={"completed_steps": 5, "decision_dates": [], "action_history": []},
                observations={
                    "received": False,
                    "applied_to_policy": False,
                    "applied_to_wofost_state": False,
                    "fusion_audit": {},
                },
                model_metadata=self.get_metadata(),
                warnings=["fertilization_decision_disabled"] if disabled else [],
            )
        finally:
            with self._counter_lock:
                self.active_calls -= 1

    def close(self):
        pass


def ready_app(engine):
    return create_app(engine=engine)


class TestAPI(unittest.TestCase):
    def test_import_does_not_load_real_model(self):
        import serving.api as api

        self.assertIsNone(api.app.state.engine)

    def test_healthz_is_process_health(self):
        with TestClient(create_app()) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_readyz_not_ready_is_503(self):
        def failed_factory():
            raise FileNotFoundError(r"E:\private\missing-model.zip")

        with TestClient(create_app(engine_factory=failed_factory)) as client:
            response = client.get("/readyz")
            decision = client.post("/v1/decision", json=request_payload())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "not_ready"})
        self.assertEqual(decision.status_code, 503)
        self.assertNotIn(r"E:\private", decision.text)

    def test_readyz_ready_and_capabilities_are_safe(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            ready = client.get("/readyz")
            capabilities = client.get("/v1/capabilities")
        self.assertEqual(ready.status_code, 200)
        body = capabilities.json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["service_concurrency_mode"], "serialized")
        self.assertNotIn(r"E:\private", capabilities.text)
        self.assertEqual(body["model_metadata"]["model_identifier"], "latest-model.zip")

    def test_valid_boundary_preserves_result_and_request_id(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            response = client.post("/v1/decision", json=request_payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "api-001")
        body = response.json()
        self.assertTrue(set(DECISION_RESULT_TOP_LEVEL_KEYS).issubset(body))
        self.assertTrue(body["decision_due"])
        self.assertEqual(body["recommendation"]["action_index"], 3)

    def test_non_boundary_is_200_with_null_recommendation(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            response = client.post(
                "/v1/decision", json=request_payload("2025-07-17")
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["decision_due"])
        self.assertIsNone(response.json()["recommendation"])

    def test_missing_request_id_generates_uuid_and_header(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            response = client.post("/v1/decision", json=request_payload(request_id=None))
        request_id = response.headers["X-Request-ID"]
        self.assertEqual(str(uuid.UUID(request_id)), request_id)
        self.assertEqual(response.json()["request_id"], request_id)

    def test_fastapi_validation_uses_error_envelope(self):
        invalid = request_payload()
        invalid.pop("crop")
        invalid["request"]["request_id"] = "validation-001"
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            response = client.post("/v1/decision", json=invalid)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers["X-Request-ID"], "validation-001")
        self.assertEqual(response.json()["error"]["code"], "REQUEST_VALIDATION_ERROR")
        self.assertIsNone(response.json()["recommendation"])

    def test_domain_error_http_mapping(self):
        cases = [
            (RequestValidationError("bad"), 422, "REQUEST_VALIDATION_ERROR"),
            (QueryDateAfterCropEndError("outside"), 422, "QUERY_DATE_AFTER_CROP_END"),
            (CustomIrrigationNotSupportedError("unsupported"), 422, "CUSTOM_IRRIGATION_NOT_SUPPORTED"),
            (ModelArtifactMissingError("missing"), 503, "MODEL_ARTIFACT_MISSING"),
            (ModelEnvironmentIncompatibleError("incompatible"), 503, "MODEL_ENV_INCOMPATIBLE"),
            (WeatherProviderError("weather"), 503, "WEATHER_PROVIDER_ERROR"),
        ]
        for error, status, code in cases:
            with self.subTest(code=code):
                with TestClient(ready_app(FakeDecisionEngine(mode=error))) as client:
                    response = client.post("/v1/decision", json=request_payload())
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], code)
                self.assertEqual(response.headers["X-Request-ID"], "api-001")

    def test_unexpected_error_is_500_without_traceback_or_path(self):
        with TestClient(ready_app(FakeDecisionEngine(mode=RuntimeError(r"E:\secret")))) as client:
            response = client.post("/v1/decision", json=request_payload())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn(r"E:\secret", response.text)

    def test_constraint_and_disabled_states_remain_200(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            constrained = client.post(
                "/v1/decision",
                json=request_payload(context={"max_single_n_rate_kg_ha": 20}),
            )
            disabled = client.post(
                "/v1/decision",
                json=request_payload(context={"allow_fertilization_decision": False}),
            )
        self.assertEqual(constrained.status_code, 200)
        self.assertTrue(constrained.json()["recommendation"]["constraint_violation"])
        self.assertEqual(disabled.status_code, 200)
        self.assertIsNone(disabled.json()["recommendation"])

    def test_response_is_json_safe_and_openapi_builds(self):
        with TestClient(ready_app(FakeDecisionEngine())) as client:
            response = client.post("/v1/decision", json=request_payload())
            openapi = client.get("/openapi.json")
        json.loads(response.text)
        self.assertEqual(openapi.status_code, 200)
        self.assertIn("/v1/decision", openapi.json()["paths"])

    def test_decision_calls_are_serialized_per_engine(self):
        fake = FakeDecisionEngine()
        with TestClient(ready_app(fake)) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(
                    pool.map(
                        lambda _index: client.post(
                            "/v1/decision", json=request_payload()
                        ),
                        range(2),
                    )
                )
        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(fake.max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()
