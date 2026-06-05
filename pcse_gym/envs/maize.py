import gymnasium as gym

import pcse_gym.envs.common_env as common_env
import pcse_gym.utils.defaults as defaults
from pcse_gym.envs.rewards import reward_functions_with_baseline
from pcse_gym.envs.sb3 import get_quzhou_maize_kwargs
from pcse_gym.envs.winterwheat import WinterWheat


QUZHOU_LOCATION = (36.87, 115.02)


class Maize(WinterWheat):
    """
    Quzhou maize RL environment.

    This reuses the Stable-Baselines3 WOFOST/SNOMIN wrapper and the reward/info
    handling from WinterWheat, but points PCSE at the calibrated Quzhou maize
    crop, site, soil, agromanagement, and weather inputs.
    """

    def __init__(self, crop_features=defaults.get_default_crop_features(pcse_env=2),
                 action_features=defaults.get_default_action_features(True),
                 weather_features=defaults.get_default_weather_features(),
                 seed=0, costs_nitrogen=10.0, timestep=7, years=None, locations=None,
                 action_space=gym.spaces.Discrete(9), action_multiplier=1.0, reward='NUE',
                 remove_timed_n=True, preserve_site_n=True, *args, **kwargs):
        if years is None:
            years = [2025]
        if locations is None:
            locations = [QUZHOU_LOCATION]

        self.preserve_site_n = preserve_site_n
        model_kwargs = get_quzhou_maize_kwargs(remove_timed_n=remove_timed_n)
        if kwargs.get('weather_provider') is not None and 'weather_data_file' not in kwargs \
                and 'weather_data_provider' not in kwargs:
            model_kwargs.pop('weather_data_file', None)
        for key, value in model_kwargs.items():
            kwargs.setdefault(key, value)

        super().__init__(crop_features=crop_features,
                         action_features=action_features,
                         weather_features=weather_features,
                         seed=seed,
                         costs_nitrogen=costs_nitrogen,
                         timestep=timestep,
                         years=years,
                         locations=locations,
                         action_space=action_space,
                         action_multiplier=action_multiplier,
                         reward=reward,
                         *args, **kwargs)

    def special_init_conditions(self):
        if self.preserve_site_n:
            return {}
        return super().special_init_conditions() or {}

    def set_location(self, location):
        if self.reward_function in reward_functions_with_baseline():
            self._set_location_on_sb3_env(self.baseline_env, location)
        self._set_location_on_sb3_env(self.sb3_env, location)

    def _set_location_on_sb3_env(self, env, location):
        env.loc = location
        if not env.fixed_weather_data_provider:
            env.weather_data_provider = common_env.get_weather_data_provider(
                location,
                random_weather=self.random_weather,
                weather_provider=self.weather_provider,
                openmeteo_kwargs=self.openmeteo_kwargs,
            )
