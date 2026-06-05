import argparse
import os
import os.path
import sys
from datetime import datetime

import gymnasium as gym
import lib_programname
import torch
import torch.nn as nn
from comet_ml import Experiment

from pcse_gym.agent.masked_actorcriticpolicy import MaskedActorCriticPolicy, MaskedRecurrentActorCriticPolicy
from pcse_gym.envs.constraints import ActionConstrainer, ConstraintCostWrapper
from pcse_gym.envs.maize import Maize, QUZHOU_LOCATION
from pcse_gym.envs.sb3 import get_policy_kwargs
from pcse_gym.utils.eval import EvalCallback
import pcse_gym.utils.defaults as defaults


path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

if os.path.join(rootdir, "pcse_gym") not in sys.path:
    sys.path.insert(0, os.path.join(rootdir, "pcse_gym"))


def args_func(parser):
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of training steps")
    parser.add_argument("-c", "--costs-nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-p", "--multiprocess", action='store_true', dest='multiproc',
                        help="Use Stable-Baselines3 multiprocessing")
    parser.add_argument("--nenvs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--eval-freq", type=int, default=20_000, dest='eval_freq')
    parser.add_argument("--no-comet", action='store_false', dest='comet')
    parser.add_argument('-d', "--device", type=str, default="cpu")
    parser.add_argument("-a", "--agent", type=str, default="PPO",
                        help="RL agent. PPO, RPPO, RecurrentLagPPO, A2C, DQN, MaskedPPO, or LagPPO")
    parser.add_argument("-r", "--reward", type=str, default="NUE",
                        help="Reward function. DEF, DEP, GRO, END, NUE, ANE, etc.")
    parser.add_argument("-b", "--n-budget", type=int, default=0, help="Nitrogen budget. kg/ha")
    parser.add_argument("--action-limit", type=int, default=0, dest="action_limit",
                        help="Limit fertilization frequency. Recommended 4 times")
    parser.add_argument("--discrete-space", type=int, default=None, dest='discrete_space',
                        help="Number of discrete fertilizer levels. Defaults to 9")
    parser.add_argument("--action-multiplier", type=float, default=1.0, dest="action_multiplier",
                        help="Multiplier applied to discrete action amounts")
    parser.add_argument("--no-weather", action='store_true', dest='no_weather')
    parser.add_argument("--normalize", action='store_true', dest='normalize')
    parser.add_argument("--mask-binary", action='store_true', dest='obs_mask')
    parser.add_argument("--placeholder-0", action='store_const', const=0.0, dest='placeholder_val')
    parser.add_argument("--random-init", action='store_true', dest='random_init')
    parser.add_argument("--cost-measure", type=str, default='real', dest='cost_measure', help='real, no, or same')
    parser.add_argument("--measure-cost-multiplier", type=int, default=1, dest='m_multiplier')
    parser.add_argument("--measure-all", action='store_true', dest='measure_all')
    parser.add_argument("--masked-ac", type=int, default=0, dest='masked_ac')
    parser.add_argument("--decay-entropy", action='store_true', default=False, dest='decay_entropy')
    parser.add_argument("--mask-later", action='store_true', default=False, dest='mask_later')
    parser.add_argument("--regl2", type=float, default=0.0, dest='regl2')
    parser.add_argument("--regl1", type=float, default=0.0, dest='regl1')
    parser.add_argument("--irs", type=str, default=None, dest='irs')
    parser.add_argument("--temporal-constraint", type=bool, default=False, dest='temporal_constraint')
    parser.add_argument("--lag-cost-limit", type=float, default=0.0, dest="lag_cost_limit")
    parser.add_argument("--lag-lambda-init", type=float, default=1.0, dest="lag_lambda_init")
    parser.add_argument("--lag-lambda-lr", type=float, default=0.05, dest="lag_lambda_lr")
    parser.add_argument("--lag-lambda-max", type=float, default=None, dest="lag_lambda_max")
    parser.add_argument("--lag-cost-vf-coef", type=float, default=0.7, dest="lag_cost_vf_coef")
    parser.add_argument("--lag-cost-key", type=str, default="cost", dest="lag_cost_key")
    parser.add_argument("--year", type=int, default=2025, dest='year',
                        help="Quzhou maize year. Defaults to the calibrated 2025 season")
    parser.add_argument("--weather-provider", type=str, default=None, dest="weather_provider",
                        help="Optional dynamic weather provider, e.g. openmeteo. Defaults to calibrated Excel weather")
    parser.add_argument("--openmeteo-model", type=str, default="best_match", dest="openmeteo_model")
    parser.add_argument("--openmeteo-start-date", type=str, default=None, dest="openmeteo_start_date")
    parser.add_argument("--openmeteo-timezone", type=str, default="Asia/Shanghai", dest="openmeteo_timezone")
    parser.add_argument("--openmeteo-forecast", action='store_true', default=False, dest="openmeteo_forecast")
    parser.add_argument("--openmeteo-force-update", action='store_true', default=False,
                        dest="openmeteo_force_update")
    parser.add_argument("--keep-timed-n", action='store_false', default=True, dest='remove_timed_n',
                        help="Keep scheduled calibration fertilizer events instead of letting RL fully control N")
    parser.add_argument("--reset-site-n", action='store_false', default=True, dest='preserve_site_n',
                        help="Use generated initial N instead of the calibrated Quzhou site N profile")
    parser.add_argument("--dry-run", action='store_true', default=False, dest='dry_run',
                        help="Build env/model and exit without learning")
    parser.set_defaults(no_weather=False, normalize=False, random_init=False, m_multiplier=1, measure_all=False,
                        multiproc=False, comet=True)
    return parser.parse_args()


def wrapper_vectorized_env(env_pcse_train, flag_po, flag_eval=False, multiproc=False, n_envs=4, normalize=False):
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv

    if normalize:
        return DummyVecEnv([lambda: env_pcse_train])
    if multiproc and not flag_eval:
        vec_env = SubprocVecEnv([lambda: env_pcse_train for _ in range(n_envs)])
        return VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10000000., clip_reward=100000., gamma=1)
    return VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                        clip_obs=10000000., clip_reward=100000., gamma=1)


def get_hyperparams(agent, no_weather, flag_po, mask_binary, actor_critic_masked, decay_entropy, mask_later,
                    crop_features, weather_features, action_features, po_features, timestep=7):
    policy_kwargs = {}
    if not no_weather:
        policy_kwargs = get_policy_kwargs(n_crop_features=len(crop_features),
                                          n_weather_features=len(weather_features),
                                          n_action_features=len(action_features),
                                          n_po_features=len(po_features) if flag_po else 0,
                                          mask_binary=mask_binary,
                                          n_timesteps=timestep)

    if agent in ['PPO', 'MaskedPPO', 'LagPPO']:
        hyperparams = {'batch_size': 276, 'n_steps': 2208, 'learning_rate': 0.001,
                       'ent_coef': 1.0 if decay_entropy else 0.01, 'clip_range': 0.2,
                       'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.5,
                       'policy_kwargs': policy_kwargs}
    elif agent in ['RPPO', 'RecurrentLagPPO']:
        hyperparams = {'batch_size': 256, 'n_steps': 2048, 'learning_rate': 0.00001,
                       'ent_coef': 1.0 if decay_entropy else 0.0, 'clip_range': 0.15,
                       'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.6,
                       'policy_kwargs': policy_kwargs}
    elif agent == 'A2C':
        hyperparams = {'n_steps': 2048, 'learning_rate': 0.0002, 'ent_coef': 1.0 if decay_entropy else 0.0,
                       'gae_lambda': 0.9, 'vf_coef': 0.4, 'policy_kwargs': policy_kwargs}
    elif agent == 'DQN':
        hyperparams = {'exploration_fraction': 0.3, 'exploration_initial_eps': 1.0,
                       'exploration_final_eps': 0.001, 'policy_kwargs': policy_kwargs}
    else:
        raise ValueError(f"Unsupported agent: {agent}")

    if agent == 'DQN':
        hyperparams['policy_kwargs']['net_arch'] = [256, 256]
    else:
        hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs']['ortho_init'] = False
    hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh

    if actor_critic_masked > 0 and agent in ['PPO', 'RPPO']:
        hyperparams['policy_kwargs']['max_non_zero_actions'] = actor_critic_masked
        hyperparams['policy_kwargs']['apply_masking'] = not (decay_entropy or mask_later)

    return hyperparams


def get_actor_critic_policy(masked, agent):
    if masked == 0 and agent == 'RPPO':
        return 'MlpLstmPolicy'
    if masked == 0 and agent == 'PPO':
        return 'MlpPolicy'
    if masked > 0 and agent == 'RPPO':
        return MaskedRecurrentActorCriticPolicy
    if masked > 0 and agent == 'PPO':
        return MaskedActorCriticPolicy
    return 'MlpPolicy'


def build_maize_env(crop_features, weather_features, action_features, years, locations, action_space,
                    action_multiplier, costs_nitrogen, reward, seed, kwargs):
    env_kwargs = {k: v for k, v in kwargs.items() if k not in ['remove_timed_n', 'preserve_site_n']}
    return Maize(crop_features=crop_features,
                 action_features=action_features,
                 weather_features=weather_features,
                 costs_nitrogen=costs_nitrogen,
                 years=years,
                 locations=locations,
                 action_space=action_space,
                 action_multiplier=action_multiplier,
                 reward=reward,
                 remove_timed_n=kwargs.get('remove_timed_n', True),
                 preserve_site_n=kwargs.get('preserve_site_n', True),
                 seed=seed,
                 **env_kwargs)


def train(log_dir, n_steps,
          crop_features=defaults.get_default_crop_features(pcse_env=2),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(True),
          train_years=None, test_years=None,
          train_locations=None, test_locations=None,
          action_space=gym.spaces.Discrete(9),
          action_multiplier=1.0, agent='PPO', reward='NUE',
          seed=0, tag="Maize", costs_nitrogen=10.0,
          multiprocess=False, eval_freq=20_000,
          dry_run=False, **kwargs):
    if train_years is None:
        train_years = [2025]
    if test_years is None:
        test_years = [2025]
    if train_locations is None:
        train_locations = [QUZHOU_LOCATION]
    if test_locations is None:
        test_locations = [QUZHOU_LOCATION]

    action_limit = kwargs.get('action_limit', 0)
    flag_po = kwargs.get('po_features', [])
    n_budget = kwargs.get('n_budget', 0)
    no_weather = kwargs.get('no_weather', False)
    normalize = kwargs.get('normalize', False)
    mask_binary = kwargs.get('mask_binary', False)
    loc_code = kwargs.get('loc_code', 'CN-Quzhou')
    cost_measure = kwargs.get('cost_measure', None)
    measure_all = kwargs.get('measure_all', None)
    n_envs = kwargs.get('n_envs', 4)
    masked_ac = kwargs.get('masked_ac')
    decay_entropy = kwargs.get('decay_entropy')
    mask_later = kwargs.get('mask_later')
    irs = kwargs.get('irs')
    temporal_constraint = kwargs.get('temporal_constraint')

    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.monitor import Monitor

    print('Using the StableBaselines3 framework')
    print(f'Train Quzhou maize with {agent} algorithm and seed {seed}. Logdir: {log_dir}')

    hyperparams = get_hyperparams(agent, no_weather, flag_po, mask_binary, actor_critic_masked=masked_ac,
                                  decay_entropy=decay_entropy, mask_later=mask_later,
                                  crop_features=crop_features, weather_features=weather_features,
                                  action_features=action_features, po_features=flag_po)

    env_pcse_train = build_maize_env(crop_features, weather_features, action_features, train_years,
                                     train_locations, action_space, action_multiplier, costs_nitrogen,
                                     reward, seed, kwargs)
    env_pcse_train = Monitor(env_pcse_train)
    if agent in ['LagPPO', 'RecurrentLagPPO']:
        env_pcse_train = ConstraintCostWrapper(env_pcse_train, {
            "cost_key": kwargs.get("lag_cost_key", "cost"),
            "max_non_zero_actions": action_limit or 4,
        })
    else:
        env_pcse_train = ActionConstrainer(env_pcse_train, action_limit=action_limit, n_budget=n_budget,
                                           temporal=temporal_constraint)

    device = kwargs.get('device')
    if device == 'cuda':
        print('CUDA not available... Using CPU!') if not torch.cuda.is_available() else print('using CUDA!')
    else:
        print('Using CPU!')

    env_pcse_train = wrapper_vectorized_env(env_pcse_train, flag_po,
                                            multiproc=multiprocess, normalize=normalize, n_envs=n_envs)
    if agent == 'PPO':
        model = PPO(get_actor_critic_policy(masked_ac, agent), env_pcse_train, gamma=1, seed=seed, verbose=0,
                    **hyperparams, tensorboard_log=log_dir, device=device)
    elif agent == 'DQN':
        model = DQN('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'A2C':
        model = A2C('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'RPPO':
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(get_actor_critic_policy(masked_ac, agent), env_pcse_train, gamma=1, seed=seed,
                             verbose=0, **hyperparams, tensorboard_log=log_dir, device=device)
    elif agent == 'RecurrentLagPPO':
        from pcse_gym.agent.ppo_mod import RecurrentLagrangianPPO
        print('Using RecurrentLagrangianPPO!')
        model = RecurrentLagrangianPPO("MlpLstmPolicy", env_pcse_train, gamma=1, seed=seed, verbose=0,
                                       cost_key=kwargs.get("lag_cost_key", "cost"),
                                       cost_limit=kwargs.get("lag_cost_limit", 0.0),
                                       lambda_init=kwargs.get("lag_lambda_init", 1.0),
                                       lambda_lr=kwargs.get("lag_lambda_lr", 0.05),
                                       lambda_max=kwargs.get("lag_lambda_max"),
                                       cost_vf_coef=kwargs.get("lag_cost_vf_coef", 0.7),
                                       **hyperparams, tensorboard_log=log_dir, device=device)
    elif agent == 'MaskedPPO':
        from sb3_contrib import MaskablePPO
        print('Using MaskedPPO!')
        model = MaskablePPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                            tensorboard_log=log_dir, device=device)
    elif agent == 'LagPPO':
        from pcse_gym.agent.ppo_mod import LagrangianPPO, CostActorCriticPolicy
        print('Using LagrangianPPO!')
        model = LagrangianPPO(CostActorCriticPolicy, env_pcse_train, gamma=1, seed=seed, verbose=0,
                              cost_key=kwargs.get("lag_cost_key", "cost"),
                              cost_limit=kwargs.get("lag_cost_limit", 0.0),
                              lambda_init=kwargs.get("lag_lambda_init", 1.0),
                              lambda_lr=kwargs.get("lag_lambda_lr", 0.05),
                              lambda_max=kwargs.get("lag_lambda_max"),
                              cost_vf_coef=kwargs.get("lag_cost_vf_coef", 0.7),
                              **hyperparams,
                              tensorboard_log=log_dir, device=device)
    else:
        raise ValueError(f"Unsupported agent: {agent}")

    irs_method = None
    if irs is not None:
        from rllte.xplore.reward import E3B, ICM, RIDE

        if irs == 'E3B':
            irs_method = E3B(envs=env_pcse_train, device=device, latent_dim=128)
        elif irs == 'ICM':
            irs_method = ICM(envs=env_pcse_train, device=device, latent_dim=256)
        elif irs == 'RIDE':
            irs_method = RIDE(envs=env_pcse_train, device=device)
        print(f"Using {irs} for intrinsic rewards!")

    comet_log = None
    use_comet = kwargs.get('comet', True)
    if use_comet:
        with open(os.path.join(rootdir, 'comet', 'comet_key'), 'r') as f:
            api_key = f.readline()
        comet_log = Experiment(
            api_key=api_key,
            project_name="cropGym_quzhou_maize_experiments",
            workspace="pcse-gym",
            log_code=True,
            log_graph=True,
            auto_metric_logging=True,
            auto_histogram_tensorboard_logging=True,
        )
        comet_log.log_code(folder=os.path.join(rootdir, 'pcse_gym'))
        comet_log.log_parameters(hyperparams)
        comet_log.add_tags(['sb3', 'maize', agent, seed, loc_code, reward, 'WOFOST SNOMIN'])
        print('Using Comet!')

    env_pcse_eval = build_maize_env(crop_features, weather_features, action_features, test_years,
                                    test_locations, action_space, action_multiplier, costs_nitrogen,
                                    reward, seed, kwargs)
    if action_limit or n_budget > 0 or temporal_constraint:
        env_pcse_eval = ActionConstrainer(env_pcse_eval, action_limit=action_limit, n_budget=n_budget,
                                          temporal=temporal_constraint)
    env_pcse_eval = wrapper_vectorized_env(env_pcse_eval, flag_po,
                                           multiproc=multiprocess, normalize=normalize, flag_eval=True)

    if measure_all:
        cost_measure = 'all'
    tb_log_name = f'{tag}-nsteps-{n_steps}-{agent}-{reward}'
    if use_comet:
        comet_log.set_name(f'{tag}-{agent}-{reward}-maize-experiment')
        comet_log.add_tag(cost_measure)
    tb_log_name = tb_log_name + '-run'

    if dry_run:
        print(f"Dry run complete. Built {agent} model and Quzhou maize train/eval envs.")
        return model

    start_time = datetime.now()
    print(f"Time started training: {start_time}")

    model.learn(total_timesteps=n_steps,
                callback=EvalCallback(env_eval=env_pcse_eval, test_years=test_years,
                                      train_years=train_years, train_locations=train_locations,
                                      test_locations=test_locations, seed=seed, pcse_model=2,
                                      comet_experiment=comet_log, multiprocess=multiprocess, eval_freq=eval_freq,
                                      irs_method=irs_method, **kwargs),
                tb_log_name=tb_log_name)

    print(f'Time taken to train {tb_log_name}: {datetime.now() - start_time}')
    return model


if __name__ == '__main__':
    if any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        from train import main as config_main
        raise SystemExit(config_main(sys.argv[1:]))

    parser = argparse.ArgumentParser()
    args = args_func(parser)

    if args.agent not in ['PPO', 'A2C', 'RPPO', 'RecurrentLagPPO', 'DQN', 'MaskedPPO', 'LagPPO']:
        parser.error("Invalid agent argument. Please choose PPO, A2C, RPPO, RecurrentLagPPO, MaskedPPO, LagPPO, or DQN")

    print(rootdir)
    log_dir = os.path.join(rootdir, 'tensorboard_logs', 'WOFOST_maize_experiments')
    print(f'train maize for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')

    train_years = [args.year]
    test_years = [args.year]
    train_locations = [QUZHOU_LOCATION]
    test_locations = [QUZHOU_LOCATION]

    crop_features = defaults.get_default_crop_features(pcse_env=2, vision=None)
    weather_features = defaults.get_default_weather_features()
    action_features = defaults.get_default_action_features(True)

    if args.discrete_space is None:
        action_spaces = gym.spaces.Discrete(9)
    else:
        action_spaces = gym.spaces.Discrete(args.discrete_space)

    openmeteo_kwargs = {}
    if args.weather_provider in ('openmeteo', 'open-meteo', 'open_meteo'):
        openmeteo_kwargs = {
            'timezone': args.openmeteo_timezone,
            'openmeteo_model': args.openmeteo_model,
            'start_date': args.openmeteo_start_date or f'{args.year}-01-01',
            'forecast': args.openmeteo_forecast,
            'force_update': args.openmeteo_force_update,
        }

    tag = f'Maize-Seed-{args.seed}'
    kwargs = {'action_limit': args.action_limit, 'n_budget': args.n_budget, 'framework': 'sb3',
              'no_weather': args.no_weather, 'mask_binary': args.obs_mask,
              'placeholder_val': args.placeholder_val, 'normalize': args.normalize,
              'loc_code': 'CN-Quzhou', 'cost_measure': args.cost_measure, 'start_type': 'sowing',
              'random_init': args.random_init, 'm_multiplier': args.m_multiplier,
              'measure_all': args.measure_all, 'random_weather': False, 'comet': args.comet,
              'n_envs': args.nenvs, 'masked_ac': args.masked_ac, 'decay_entropy': args.decay_entropy,
              'nsteps': args.nsteps, 'mask_later': args.mask_later, 'regl2': args.regl2,
              'regl1': args.regl1, 'irs': args.irs, 'discrete_space': args.discrete_space,
              'temporal_constraint': args.temporal_constraint, 'remove_timed_n': args.remove_timed_n,
              'preserve_site_n': args.preserve_site_n, 'weather_provider': args.weather_provider,
              'openmeteo_kwargs': openmeteo_kwargs, 'device': args.device,
              'lag_cost_limit': args.lag_cost_limit, 'lag_lambda_init': args.lag_lambda_init,
              'lag_lambda_lr': args.lag_lambda_lr, 'lag_lambda_max': args.lag_lambda_max,
              'lag_cost_vf_coef': args.lag_cost_vf_coef, 'lag_cost_key': args.lag_cost_key}

    if args.decay_entropy:
        print('Training with entropy decay')

    print(f"Train and test years are {train_years} and {test_years},"
          f" with train location of {train_locations} and test location of {test_locations}")

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations, test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed, tag=tag,
          costs_nitrogen=args.costs_nitrogen, crop_features=crop_features,
          weather_features=weather_features, action_features=action_features,
          action_space=action_spaces, action_multiplier=args.action_multiplier,
          agent=args.agent, reward=args.reward, multiprocess=args.multiproc,
          eval_freq=args.eval_freq, dry_run=args.dry_run, **kwargs)
