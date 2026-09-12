import unittest

from fastapi.testclient import TestClient

from serving.api import create_app


class FakeEngine:
    def close(self):
        pass


def output(verdict="ACCEPT", candidate=50, final=50, execution="proceed"):
    return {
        "verdict": verdict,
        "rl_candidate_n_kg_ha": candidate,
        "final_n_kg_ha": final,
        "execution_status": execution,
        "reason_codes": [],
        "reasons": [],
        "confidence": "medium",
    }


class TestCriticValidationAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(engine=FakeEngine()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, candidate, critic_output):
        response = self.client.post(
            "/v1/critic/validate",
            json={
                "rl_candidate_n_kg_ha": candidate,
                "critic_output": critic_output,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_accept_50_passes(self):
        result = self.post(50, output())
        self.assertTrue(result["validation_passed"])
        self.assertFalse(result["critic_validation_failed"])
        self.assertEqual(result["final_n_kg_ha"], 50)

    def test_reduce_30_passes(self):
        result = self.post(50, output("REDUCE", final=30))
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 30)

    def test_reduce_60_falls_back(self):
        result = self.post(50, output("REDUCE", final=60))
        self.assertFalse(result["validation_passed"])
        self.assertTrue(result["critic_validation_failed"])
        self.assertEqual(result["final_n_kg_ha"], 50)
        self.assertTrue(result["validation_errors"])

    def test_reduce_35_falls_back(self):
        result = self.post(50, output("REDUCE", final=35))
        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 50)

    def test_defer_50_passes(self):
        result = self.post(50, output("DEFER", final=50, execution="defer"))
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["execution_status"], "defer")

    def test_reject_50_passes(self):
        result = self.post(50, output("REJECT", final=0, execution="reject"))
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 0)

    def test_zero_accept_passes(self):
        result = self.post(0, output(candidate=0, final=0))
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 0)

    def test_zero_reduce_falls_back(self):
        result = self.post(0, output("REDUCE", candidate=0, final=0))
        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 0)

    def test_zero_increase_falls_back(self):
        result = self.post(0, output("INCREASE", candidate=0, final=10))
        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["verdict"], "ACCEPT")
        self.assertEqual(result["final_n_kg_ha"], 0)

    def test_zero_defer_is_normalized_to_no_action_accept(self):
        result = self.post(
            0, output("DEFER", candidate=0, final=0, execution="defer")
        )
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["verdict"], "ACCEPT")
        self.assertEqual(result["execution_status"], "proceed")
        self.assertIn("NO_ACTION_REQUIRED", result["reason_codes"])

    def test_malformed_json_falls_back(self):
        result = self.post(50, "{not valid json")
        self.assertFalse(result["validation_passed"])
        self.assertTrue(result["critic_validation_failed"])
        self.assertEqual(result["final_n_kg_ha"], 50)

    def test_out_of_range_and_negative_final_rates_fall_back(self):
        for final in (160, -10):
            with self.subTest(final=final):
                result = self.post(50, output(final=final))
                self.assertFalse(result["validation_passed"])
                self.assertEqual(result["final_n_kg_ha"], 50)

    def test_invalid_verdict_falls_back(self):
        result = self.post(50, output("UNKNOWN", final=50))
        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["final_n_kg_ha"], 50)


if __name__ == "__main__":
    unittest.main()
