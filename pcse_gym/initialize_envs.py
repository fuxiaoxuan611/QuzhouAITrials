import gymnasium as gym

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.maize import Maize
import pcse_gym.utils.defaults as defaults
from pcse_gym.envs.sb3 import get_model_kwargs

def get_action_space(nitrogen_levels=7, po_features=[]):
    if po_features:
        a_shape = [nitrogen_levels] + [2] * len(po_features)
        space_return = gym.spaces.MultiDiscrete(a_shape)
    else:
        space_return = gym.spaces.Discrete(nitrogen_levels)
    return space_return


def initialize_env(pcse_env=1, po_features=[],
                   crop_features=defaults.get_default_crop_features(pcse_env=1, vision='None'),
                   action_features=defaults.get_default_action_features(True), costs_nitrogen=10, reward='NUE', nitrogen_levels=9, action_multiplier=1.0, add_random=False,
                   years=defaults.get_default_train_years(), locations=defaults.get_default_location(), args_vrr=False,
                   action_limit=0, noisy_measure=False, n_budget=0, no_weather=False, framework='sb3',
                   mask_binary=False, random_weather=False, weather_features=defaults.get_default_weather_features(),
                   placeholder_val=-1.11, normalize=False, loc_code='NL', cost_measure='real', start_type='sowing',
                   random_init=False, m_multiplier=1, measure_all=False, seed=None, soil=None,
                   weather_provider=None, openmeteo_kwargs=None):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=nitrogen_levels, po_features=po_features)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None, args_vrr=args_vrr,
                  action_limit=action_limit, noisy_measure=noisy_measure, n_budget=n_budget, no_weather=no_weather,
                  mask_binary=mask_binary, placeholder_val=placeholder_val, normalize=normalize, loc_code=loc_code,
                  cost_measure=cost_measure, start_type=start_type, random_init=random_init, m_multiplier=m_multiplier,
                  measure_all=measure_all, random_weather=random_weather, weather_provider=weather_provider,
                  openmeteo_kwargs=openmeteo_kwargs or {},)
    if framework == 'sb3':
        env_return = WinterWheat(crop_features=crop_features,
                                 action_features=action_features,
                                 weather_features=weather_features,
                                 costs_nitrogen=costs_nitrogen,
                                 years=years,
                                 locations=locations,
                                 action_space=action_space,
                                 action_multiplier=action_multiplier,
                                 reward=reward,
                                 **get_model_kwargs(pcse_env, locations, soil=soil, start_type=kwargs.get('start_type', 'sowing')),
                                 **kwargs, seed=seed)
    elif framework == 'rllib':
        from pcse_gym.utils.rllib_helpers import ww_lim, winterwheat_config_maker
        config = winterwheat_config_maker(crop_features=crop_features,
                                          costs_nitrogen=costs_nitrogen, years=years,
                                          locations=locations,
                                          action_space=action_space,
                                          action_multiplier=1.0,
                                          reward=reward, pcse_model=1,
                                          **get_model_kwargs(1, locations),
                                          **kwargs)
        env_return = ww_lim(config)
    else:
        raise Exception("Invalid framework!")
    return env_return


def initialize_maize_env(po_features=[],
                         crop_features=defaults.get_default_crop_features(pcse_env=2),
                         action_features=defaults.get_default_action_features(True),
                         costs_nitrogen=10, reward='NUE', nitrogen_levels=9,
                         action_multiplier=1.0, add_random=False, years=None, locations=None,
                         action_limit=0, noisy_measure=False, n_budget=0, no_weather=False,
                         mask_binary=False, weather_features=defaults.get_default_weather_features(),
                         placeholder_val=-1.11, normalize=False, cost_measure='real',
                         random_init=False, m_multiplier=1, measure_all=False, seed=None,
                         remove_timed_n=True, preserve_site_n=True, weather_provider=None,
                         openmeteo_kwargs=None):
    if add_random:
        po_features.append('random'), crop_features.append('random')
    action_space = get_action_space(nitrogen_levels=nitrogen_levels, po_features=po_features)
    kwargs = dict(po_features=po_features, args_measure=po_features is not None, noisy_measure=noisy_measure,
                  action_limit=action_limit, n_budget=n_budget, no_weather=no_weather,
                  mask_binary=mask_binary, placeholder_val=placeholder_val, normalize=normalize,
                  loc_code='CN-Quzhou', cost_measure=cost_measure, start_type='sowing',
                  random_init=random_init, m_multiplier=m_multiplier, measure_all=measure_all,
                  random_weather=False, weather_provider=weather_provider,
                  openmeteo_kwargs=openmeteo_kwargs or {})
    return Maize(crop_features=crop_features,
                 action_features=action_features,
                 weather_features=weather_features,
                 costs_nitrogen=costs_nitrogen,
                 years=years,
                 locations=locations,
                 action_space=action_space,
                 action_multiplier=action_multiplier,
                 reward=reward,
                 remove_timed_n=remove_timed_n,
                 preserve_site_n=preserve_site_n,
                 **kwargs,
                 seed=seed)
