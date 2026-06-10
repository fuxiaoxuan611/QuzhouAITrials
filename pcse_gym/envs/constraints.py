from datetime import timedelta

import gymnasium as gym
from gymnasium import ActionWrapper, Wrapper
import numpy as np


class ConstraintCostWrapper(Wrapper):
    """
    Soft agronomic constraint wrapper.

    The wrapper never changes actions. It computes scalar Lagrangian costs from
    the attempted action and environment info, then exposes them through
    ``info["cost"]`` and ``info["cost_components"]``.
    """

    def __init__(self, env, constraint_config=None):
        super().__init__(env)
        config = constraint_config or {}
        self.cost_key = config.get("cost_key", "cost")
        self.components_key = config.get("components_key", "cost_components")
        self.max_non_zero_actions = config.get("max_non_zero_actions", 4)
        self.fertilize_until_dvs = config.get("fertilize_until_dvs")
        self.nue_threshold = config.get("nue_threshold", (0.5, 0.9))
        self.n_surplus_threshold = config.get("n_surplus_threshold", (0.0, 40.0))
        self.weights = {
            "too_many_nonzero_actions": config.get("n_weight", 5.0),
            "nue_violation": config.get("nue_weight", 1.0),
            "n_surplus_violation": config.get("n_surplus_weight", 1.0),
        }
        if self.fertilize_until_dvs is not None:
            self.fertilize_until_dvs = float(self.fertilize_until_dvs)
            self.weights["fertilize_after_dvs"] = config.get("dvs_weight", 1.0)
        self.reset_constraint_state()

    def reset_constraint_state(self):
        self.non_zero_action_count = 0

    @staticmethod
    def _fertilizer_action(action):
        if isinstance(action, np.ndarray):
            return float(np.asarray(action).flatten()[0])
        return float(action)

    @staticmethod
    def _latest_info_value(info, key, default=0.0):
        value = info.get(key, default)
        if isinstance(value, dict):
            if not value:
                return default
            return list(value.values())[-1]
        return value

    def _terminal_n_component(self, info, key, threshold, terminated, truncated):
        if not (terminated or truncated) or key not in info:
            return 0.0
        value = self._latest_info_value(info, key, None)
        if value is None:
            return 0.0
        lower, upper = threshold
        return float(value < lower or value > upper)

    def _dvs_timing_component(self, fertilizer_action, info):
        if self.fertilize_until_dvs is None or fertilizer_action <= 0:
            return 0.0
        dvs = self._latest_info_value(info, "DVS", None)
        if dvs is None:
            return 0.0
        return float(float(dvs) > self.fertilize_until_dvs) * self.weights["fertilize_after_dvs"]

    def _cost_components(self, action, info, terminated, truncated):
        fertilizer_action = self._fertilizer_action(action)
        has_non_zero_action = fertilizer_action > 0
        next_non_zero_count = self.non_zero_action_count + int(has_non_zero_action)
        too_many = max(0, next_non_zero_count - self.max_non_zero_actions)

        components = {
            "too_many_nonzero_actions": too_many * self.weights["too_many_nonzero_actions"],
            "nue_violation": self._terminal_n_component(info, "NUE", self.nue_threshold, terminated, truncated)
            * self.weights["nue_violation"],
            "n_surplus_violation": self._terminal_n_component(
                info, "Nsurplus", self.n_surplus_threshold, terminated, truncated
            )
            * self.weights["n_surplus_violation"],
        }
        if self.fertilize_until_dvs is not None:
            components["fertilize_after_dvs"] = self._dvs_timing_component(fertilizer_action, info)
        return components

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        components = self._cost_components(action, info, terminated, truncated)
        info[self.components_key] = components
        info[self.cost_key] = float(sum(components.values()))

        if self._fertilizer_action(action) > 0:
            self.non_zero_action_count += 1
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.reset_constraint_state()
        return self.env.reset(**kwargs)


class ActionConstrainer(ActionWrapper):
    """
    Action Wrapper to limit fertilization actions
    """
    def __init__(self, env, action_limit=0, n_budget=0, temporal=False, fertilize_until_dvs=1.0):
        super(ActionConstrainer, self).__init__(env)
        self.counter = 0
        self.action_limit = action_limit
        self.n_counter = 0
        self.temporal = temporal
        self.n_budget = n_budget
        self.fertilize_until_dvs = float(fertilize_until_dvs)

    def action(self, action):
        if self.action_limit > 0:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.freq_limiter_discrete(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.freq_limiter_multi_discrete(action)
        if self.n_budget > 0:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.discrete_n_budget(action)
            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.multi_discrete_n_budget(action)
        if self.temporal is not False:
            if isinstance(self.action_space, gym.spaces.Discrete):
                action = self.discrete_temporal_constraint(action)
            if isinstance(self.action_space, gym.spaces.MultiDiscrete):
                action = self.multi_discrete_temporal_constraint(action)
        return action

    def discrete_temporal_constraint(self, action):
        if self._current_dvs() > self.fertilize_until_dvs:
            action = 0
            return action
        return action

    def multi_discrete_temporal_constraint(self, action):
        if self._current_dvs() > self.fertilize_until_dvs:
            action[0] = 0
            return action
        return action

    def _current_dvs(self):
        env = self.env
        while env is not None:
            if hasattr(env, "dvs"):
                return float(env.dvs)
            env = getattr(env, "env", None)
        return 0.0

    def discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action = 0
            return action
        self.n_counter += action * 10
        if self.n_counter > self.n_budget:
            action = ((self.n_budget + action * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def multi_discrete_n_budget(self, action):
        if self.n_counter == self.n_budget:
            action = action.copy()  # Needed for RLlib VecEnvs
            action[0] = 0
            return action
        self.n_counter += action[0] * 10
        if self.n_counter > self.n_budget:
            action = action.copy()
            action[0] = ((self.n_budget + action[0] * 10) - self.n_counter) / 10
            self.n_counter = self.n_budget
        return action

    def freq_limiter_discrete(self, action):
        if action != 0:  # if there's an action, increase the counter
            self.counter += 1
        if self.counter > self.action_limit:  # return 0 if the action exceeds limit
            action = 0
        return action

    def freq_limiter_multi_discrete(self, action):
        if action[0] != 0:
            self.counter += 1
        if self.counter > self.action_limit:
            action = action.copy()
            action[0] = 0
        return action

    def reset(self, **kwargs):
        self.counter = 0
        self.n_counter = 0
        return self.env.reset(**kwargs)


def ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    new_value = (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_value


def non_linear_ratio_rescale(value, old_max=None, old_min=None, new_max=None, new_min=None):
    # Normalize the input value to [0, 1]
    normalized_value = (value - old_min) / (old_max - old_min)

    # Determine which segment we are in based on the normalized value
    if normalized_value <= 0.5:
        # Scale sine to [0, 1] and then rescale to [0.3, 0.55]
        return 0.3 + 0.25 * np.sin(np.pi * normalized_value)
    else:
        # Scale sine to [0, 1] and then rescale to [0.55, 0.8]
        return 0.55 + 0.25 * np.sin(np.pi * (normalized_value - 0.5))


class VariableRecoveryRate(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(env.__dict__)

    def _apply_action(self, action) -> None:
        import pcse

        amount = action * self.action_multiplier
        recovery_rate = self.recovery_penalty()
        self._model._send_signal(signal=pcse.signals.apply_n, N_amount=amount * 10, N_recovery=recovery_rate,
                                 amount=amount, recovery=recovery_rate)

    def recovery_penalty(self):
        """
        estimation function due to static recovery rate of WOFOST/LINTUL
        Potentially enforcing the agent not to dump everything at the start
        Not to be used with CERES 'start-dump'
        Adapted, based on the findings of Raun, W.R. and Johnson, G.V. (1999)
        """
        date_now = self.date - self.start_date
        date_end = self.end_date - self.start_date  # TODO end on flowering?
        recovery = ratio_rescale(date_now / timedelta(days=1),
                                 old_max=date_end / timedelta(days=1), old_min=0.0, new_max=0.8, new_min=0.3)
        return recovery


def generate_combinations(elements):
    from itertools import combinations
    all_combinations = []

    for x in range(1, len(elements) + 1):
        all_combinations.extend(combinations(elements, x))

    return all_combinations
