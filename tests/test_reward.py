import datetime
import unittest
import numpy as np
from math import isclose

import tests.initialize_env as init_env
from pcse_gym.envs.rewards import Rewards as RewardContainer
from tests.network_utils import network_test


@network_test
class Rewards(unittest.TestCase):
    def setUp(self):
        self.dep = init_env.initialize_env_reward_dep()
        self.eny = init_env.initialize_env_eny_reward()
        self.nue = init_env.initialize_env_nue_reward()
        self.nup = init_env.initialize_env_nup_reward()
        self.har = init_env.initialize_env_har_reward()
        self.dnu = init_env.initialize_env_dnu_reward()
        self.fin = init_env.initialize_env_fin_reward()
        self.def1 = init_env.initialize_env(reward='DEF', pcse_env=2)
        self.dne = init_env.initialize_env(reward='DNE', pcse_env=2)

    @staticmethod
    def run_steps_sp(env, year, terminated):
        env.overwrite_year(year)
        env.reset()

        week = 0
        n = 8
        rewards = 0
        while not terminated:
            if week == n or week == n + 4:
                action = np.array([6])
            else:
                action = np.array([0])
            _, reward, terminated, _, _ = env.step(action)
            rewards += reward
            week += 1

        return rewards

    def test_reward_functions(self):
        rfs = [self.dep, self.nue, self.eny, self.nup,
               self.har, self.dnu, self.fin, self.def1, self.dne]
        expected_rs = [8638.21, -20, 8658.21, 137.42, 1266.20, -47.95, 1359.65, 2242.42, -120.0,]
        for rf_env, expected_r in zip(rfs, expected_rs):
            r = self.run_steps_sp(rf_env, 2002, False)
            print(f"reward {r} and rf {rf_env.reward_function}")
            check_if_close = isclose(r, expected_r, abs_tol=5)
            self.assertTrue(check_if_close)


class ContainerNUERewardTest(unittest.TestCase):
    def setUp(self):
        self.container = RewardContainer.ContainerNUE(timestep=7)

    @staticmethod
    def _inside_spec_n_inputs():
        total_n_input = 20 / 0.3
        n_fertilized = total_n_input - 0.4
        n_output = total_n_input * 0.7
        return n_fertilized, n_output

    @staticmethod
    def _episode_outputs(agent_final_yield, baseline_final_yield, agent_initial_yield=0.0, baseline_initial_yield=0.0):
        start = datetime.date(2002, 1, 1)
        end = datetime.date(2002, 8, 1)
        output = [
            {"day": start, "WSO": agent_initial_yield},
            {"day": end, "WSO": agent_final_yield},
        ]
        output_baseline = [
            {"day": start, "WSO": baseline_initial_yield},
            {"day": end, "WSO": baseline_final_yield},
        ]
        return output, output_baseline

    def _calculate_inside_spec_reward(self, yield_improvement):
        n_fertilized, n_output = self._inside_spec_n_inputs()
        output, output_baseline = self._episode_outputs(
            agent_final_yield=1000 + yield_improvement,
            baseline_final_yield=1000,
        )
        return self.container.calculate_reward_nue(
            n_fertilized=n_fertilized,
            n_output=n_output,
            no3_depo=0,
            nh4_depo=0,
            output=output,
            output_baseline=output_baseline,
        )

    def test_calculate_reward_nue_adds_zero_bonus_at_zero_baseline_improvement(self):
        self.assertAlmostEqual(
            self._calculate_inside_spec_reward(yield_improvement=0),
            1.0
        )

    def test_calculate_reward_nue_adds_scaled_bonus_from_zero_n_baseline_improvement(self):
        self.assertAlmostEqual(
            self._calculate_inside_spec_reward(yield_improvement=1000),
            2.0
        )
        self.assertAlmostEqual(
            self._calculate_inside_spec_reward(yield_improvement=2500),
            3.5
        )

    def test_calculate_reward_nue_floors_negative_baseline_improvement(self):
        self.assertAlmostEqual(
            self._calculate_inside_spec_reward(yield_improvement=-100),
            1.0
        )

    def test_calculate_reward_nue_does_not_add_yield_bonus_outside_specs(self):
        n_fertilized, _ = self._inside_spec_n_inputs()
        n_output = 10
        total_n_input = n_fertilized + 0.4
        nue = n_output / total_n_input
        n_surplus = total_n_input - n_output
        output, output_baseline = self._episode_outputs(
            agent_final_yield=3500,
            baseline_final_yield=1000,
        )
        expected = self.container.n_surplus_formula(n_surplus=n_surplus, nue=nue)
        self.assertAlmostEqual(
            self.container.calculate_reward_nue(
                n_fertilized=n_fertilized,
                n_output=n_output,
                no3_depo=0,
                nh4_depo=0,
                output=output,
                output_baseline=output_baseline,
            ),
            expected
        )
        self.assertLess(expected, 1.0)

    @network_test
    def test_nue_environment_passes_zero_nitrogen_baseline_to_terminal_reward(self):
        env = init_env.initialize_env_nue_reward()
        captured = {}
        original_calculate_reward_nue = env.reward_container.calculate_reward_nue

        def wrapped_calculate_reward_nue(*args, **kwargs):
            captured["output"] = kwargs.get("output")
            captured["output_baseline"] = kwargs.get("output_baseline")
            captured["multiplier"] = kwargs.get("multiplier", 1)
            return original_calculate_reward_nue(*args, **kwargs)

        env.reward_container.calculate_reward_nue = wrapped_calculate_reward_nue

        try:
            env.overwrite_year(2002)
            env.reset()
            self.assertIsNotNone(env.baseline_env)
            baseline_output = env.zero_nitrogen_env_storage.get_episode_output(env.baseline_env)
            self.assertTrue(baseline_output)
            self.assertIn("WSO", baseline_output)

            terminated, truncated = False, False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(np.array([0]))

            self.assertTrue(captured["output_baseline"])
            expected_improvement = (
                self.container._episode_storage_organ_growth(captured["output"], captured["multiplier"])
                - self.container._episode_storage_organ_growth(captured["output_baseline"], captured["multiplier"])
            )
            actual_improvement = self.container.calculate_yield_improvement(
                captured["output"],
                captured["output_baseline"],
                captured["multiplier"],
            )
            self.assertAlmostEqual(actual_improvement, expected_improvement)
        finally:
            if hasattr(env, "close"):
                env.close()
