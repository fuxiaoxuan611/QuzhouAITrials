import unittest
import gymnasium as gym
import yaml
import os
import numpy as np

import tests.initialize_env as init_env
from pcse_gym.envs.constraints import ActionConstrainer
from tests.network_utils import network_test


class _ActionLimitDummyEnv(gym.Env):
    """Minimal local env for testing ActionConstrainer without weather I/O."""

    metadata = {"render_modes": []}

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


# class TestRecoveryRate(unittest.TestCase):
#     def setUp(self):
#         self.env = init_env.initialize_env_rr()
#
#     def test_rr(self):
#         self.env.reset()
#         action = np.array([0])
#         _, _, _, _, info = self.env.step(action)
#         initial_n = list(info['NAVAIL'].values())[0]
#         action = np.array([2])
#         actual = int(action) * 10 * self.env.sb3_env.recovery_penalty()
#         _, _, _, _, info = self.env.step(action)
#         # key = info['NAVAIL'].keys()[-1]
#         end_n = list(info['NAVAIL'].values())[-1]
#
#         expected = end_n - initial_n
#
#         # assert almost equal due to possibly the crop taking up some nitrogen
#         self.assertAlmostEqual(expected, actual, 1)


class ActionLimit(unittest.TestCase):
    def setUp(self):
        self.env_meas = ActionConstrainer(
            _ActionLimitDummyEnv(gym.spaces.MultiDiscrete([7, 2, 2, 2, 2, 2])),
            action_limit=4,
        )
        self.env_no_meas = ActionConstrainer(
            _ActionLimitDummyEnv(gym.spaces.Discrete(7)), action_limit=4
        )
        self.env_budget = ActionConstrainer(
            _ActionLimitDummyEnv(gym.spaces.Discrete(7)),
            action_limit=5,
            n_budget=180,
        )
        self.env_budget_meas = ActionConstrainer(
            _ActionLimitDummyEnv(gym.spaces.MultiDiscrete([7, 2, 2, 2, 2, 2])),
            action_limit=6,
            n_budget=180,
        )

    def test_limit_measure(self):
        self.env_meas.reset()

        loop = 16
        hist = []
        # for action
        for i in range(loop):
            action = np.array([1, 1, 0, 0, 0, 1])
            check = self.env_meas.action(action)
            hist.append(check)

        actions_hist = [item[0] for item in hist]

        actions_expected = np.zeros(loop, int)
        for a, b in enumerate(actions_expected[:4]):
            actions_expected[a] = 1

        self.assertListEqual(actions_hist, list(actions_expected))

    def test_limit_no_measure(self):
        self.env_no_meas.reset()

        loop = 16
        hist = []
        # for action
        for i in range(loop):
            action = np.array(1)
            check = self.env_no_meas.action(action)
            hist.append(check)

        actions_expected = np.zeros(loop, int)
        for a, b in enumerate(actions_expected[:4]):
            actions_expected[a] = 1

        self.assertListEqual(hist, list(actions_expected))

    def test_limit_budget_no_measure(self):
        self.env_budget.reset()

        actions = [2, 4, 6, 4, 5, 5, 5]
        loop = 7
        hist = []
        for act, i in zip(actions, range(loop)):
            action = np.array([act])
            check = self.env_budget.action(action)
            hist.append(check)

        actions_expected = [2, 4, 6, 4, 2, 0, 0]

        self.assertListEqual(hist, actions_expected)

    def test_limit_budget_measure(self):
        self.env_budget_meas.reset()

        loop = 6
        hist = []
        for i in range(loop):
            action = np.array([5, 1, 0, 0, 0, 1])
            check = self.env_budget_meas.action(action)
            hist.append(check)

        actions_hist = [item[0] for item in hist]

        actions_expected = [5, 5, 5, 3, 0, 0]

        self.assertListEqual(actions_hist, actions_expected)


@network_test
class TestStartType(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env(pcse_env=2, start_type='sowing')
        self.env2 = init_env.initialize_env(pcse_env=2, start_type='emergence')

    def test_sow_start(self):
        self.env.reset()

        year = [2012]
        self.env.overwrite_year(year)
        self.env.reset()
        self.assertEqual(year[0] - 1, int(self.env.date.year))

        year = [2016]
        self.env.overwrite_year(year)
        self.env.reset()
        self.assertEqual(year[0] - 1, int(self.env.date.year))

    def test_emergence_start(self):
        self.env2.reset()

        year = [1990]
        self.env2.overwrite_year(year)
        self.env2.reset()
        self.assertEqual(year[0], int(self.env2.date.year))

        year = [2000]
        self.env2.overwrite_year(year)
        self.env2.reset()
        self.assertEqual(year[0], int(self.env2.date.year))


@network_test
class TestEnvFeatures(unittest.TestCase):
    def setUp(self):
        self.env = init_env.initialize_env_random_init()

    def test_different_initial_conditions(self):
        dir_root = os.path.dirname(os.path.realpath(__file__))[:-5]
        site_params = yaml.safe_load(open(os.path.join(dir_root,
                                                       'pcse_gym', 'envs', 'configs', 'site', 'arminda_site.yaml')))
        self.assertEqual(site_params, self.env.sb3_env.model.parameterprovider._sitedata)
        self.env.reset()
        self.assertNotEqual(site_params, self.env.sb3_env.model.parameterprovider._sitedata)






