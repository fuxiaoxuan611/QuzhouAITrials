import json
import unittest

from serving.agronomic_critic import (
    CriticValidationError,
    prepare_critic_input,
    validate_critic_output,
    validate_or_fallback,
)


def critic_output(verdict="ACCEPT", candidate=50, final=50, execution="proceed"):
    return {
        "verdict": verdict,
        "rl_candidate_n_kg_ha": candidate,
        "final_n_kg_ha": final,
        "execution_status": execution,
        "reason_codes": [],
        "reasons": [],
        "confidence": "medium",
    }


class TestAgronomicCritic(unittest.TestCase):
    def test_accept_preserves_candidate(self):
        result = validate_critic_output(
            critic_output(), rl_candidate_n_rate_kg_ha=50
        )
        self.assertEqual(result["verdict"], "ACCEPT")
        self.assertEqual(result["final_n_kg_ha"], 50.0)
        self.assertFalse(result["critic_validation_failed"])

    def test_reduce_allows_only_lower_action_rate(self):
        result = validate_critic_output(
            critic_output("REDUCE", final=30), rl_candidate_n_rate_kg_ha=50
        )
        self.assertEqual(result["final_n_kg_ha"], 30.0)

    def test_reduce_to_higher_rate_is_rejected(self):
        with self.assertRaises(CriticValidationError):
            validate_critic_output(
                critic_output("REDUCE", final=60), rl_candidate_n_rate_kg_ha=50
            )

    def test_reduce_to_non_action_rate_is_rejected(self):
        with self.assertRaises(CriticValidationError):
            validate_critic_output(
                critic_output("REDUCE", final=35), rl_candidate_n_rate_kg_ha=50
            )

    def test_defer_preserves_candidate_and_changes_execution(self):
        result = validate_critic_output(
            critic_output("DEFER", final=50, execution="defer"),
            rl_candidate_n_rate_kg_ha=50,
        )
        self.assertEqual(result["execution_status"], "defer")
        self.assertEqual(result["final_n_kg_ha"], 50.0)

    def test_reject_forces_zero(self):
        result = validate_critic_output(
            critic_output("REJECT", final=0, execution="reject"),
            rl_candidate_n_rate_kg_ha=50,
        )
        self.assertEqual(result["final_n_kg_ha"], 0.0)

    def test_out_of_range_and_negative_rates_are_rejected(self):
        for rate in (160, -10):
            with self.subTest(rate=rate):
                with self.assertRaises(CriticValidationError):
                    validate_critic_output(
                        critic_output(final=rate), rl_candidate_n_rate_kg_ha=50
                    )

    def test_invalid_verdict_is_rejected(self):
        with self.assertRaises(CriticValidationError):
            validate_critic_output(
                critic_output("INCREASE", final=60),
                rl_candidate_n_rate_kg_ha=50,
            )

    def test_malformed_json_uses_policy_preserving_fallback(self):
        result = validate_or_fallback(
            "not-json", rl_candidate_n_rate_kg_ha=50
        )
        self.assertEqual(result["verdict"], "ACCEPT")
        self.assertEqual(result["final_n_kg_ha"], 50.0)
        self.assertEqual(result["execution_status"], "proceed")
        self.assertTrue(result["critic_validation_failed"])

    def test_invalid_output_cannot_pass_an_invalid_rate_to_conversion(self):
        result = validate_or_fallback(
            critic_output("REDUCE", final=35),
            rl_candidate_n_rate_kg_ha=50,
        )
        self.assertTrue(result["critic_validation_failed"])
        self.assertIn(result["final_n_kg_ha"], range(0, 151, 10))
        self.assertEqual(result["final_n_kg_ha"], 50.0)

    def test_critic_input_preserves_simulated_and_policy_lai(self):
        result = prepare_critic_input(
            query_date="2025-07-14",
            decision_date="2025-07-14",
            projected_application_date="2025-07-21",
            simulated_state={"LAI": 1.4801, "DVS": 0.8},
            observation_fusion={
                "wofost_state_mutated": False,
                "simulated": {"LAI": 1.4801},
                "observed": {"LAI": 1.6},
                "policy_input": {"LAI": 1.6},
            },
            user_observations={"observation_snapshot": {"crop": {"lai": 1.6}}},
            rl_candidate_action=5,
            rl_candidate_n_rate_kg_ha=50,
            decision_due=True,
        )
        self.assertEqual(result["wofost_simulated_state"]["LAI"], 1.4801)
        self.assertEqual(result["observation_fusion"]["observed"]["LAI"], 1.6)
        self.assertEqual(result["observation_fusion"]["policy_input"]["LAI"], 1.6)
        self.assertFalse(result["observation_fusion"]["wofost_state_mutated"])

    def test_no_observation_input_is_json_safe(self):
        result = prepare_critic_input(
            query_date="2025-07-14",
            decision_date=None,
            projected_application_date=None,
            simulated_state={"LAI": 1.48},
            observation_fusion={"mode": "no_observation"},
        )
        json.dumps(result)
        self.assertIsNone(result["user_observations"])
        self.assertIsNone(result["rl_candidate_action"])


if __name__ == "__main__":
    unittest.main()
