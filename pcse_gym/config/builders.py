import copy
from pathlib import Path

import gymnasium as gym
import torch.nn as nn
import yaml

from pcse_gym.envs.constraints import ActionConstrainer, ConstraintCostWrapper


ACTIVATIONS = {
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "ELU": nn.ELU,
}

PCSE_MODEL_CODES = {
    "lintul3": 0,
    "lintul": 0,
    0: 0,
    "wofost_cnb": 1,
    "wofost_classic": 1,
    "wofost_classic_n": 1,
    1: 1,
    "wofost_snomin": 2,
    "wofost_snominn": 2,
    "snomin": 2,
    2: 2,
}

def build_site_from_config(config):
    pcse = _pcse()
    site = config["site"]
    provider = site.get("provider", "yaml")
    if provider == "yaml":
        return _read_yaml(site["path"])
    if provider == "cabo":
        return pcse.input.PCSEFileReader(site["path"])
    if provider == "wofost80_site":
        params = copy.deepcopy(site.get("parameters", {}))
        return pcse.util.WOFOST80SiteDataProvider(**params)
    raise ValueError(f"Unsupported site provider: {provider}")


def build_soil_from_config(config):
    pcse = _pcse()
    soil = config["soil"]
    provider = soil.get("provider", "yaml")
    if provider == "yaml":
        return _read_yaml(soil["path"])
    if provider == "cabo":
        return pcse.input.CABOFileReader(soil["path"])
    if provider == "pcse_file":
        return pcse.input.PCSEFileReader(soil["path"])
    raise ValueError(f"Unsupported soil provider: {provider}")


def build_crop_provider_from_config(config):
    pcse = _pcse()
    crop = config["crop"]
    provider = crop.get("provider", "yaml_crop_data")
    if provider == "yaml_crop_data":
        return pcse.input.YAMLCropDataProvider(
            fpath=crop["fpath"],
            force_reload=bool(crop.get("force_reload", False)),
        )
    if provider in ("pcse_file", "file"):
        return pcse.input.PCSEFileReader(crop["path"])
    raise ValueError(f"Unsupported crop provider: {provider}")


def build_crop_model_from_config(config, crop_model_overrides=None):
    agro = config["crop_model"].get("agro", {})
    kwargs = {
        "model_config": config["crop_model"]["model_config"],
        "agro_config": agro["path"],
        "crop_parameters": build_crop_provider_from_config(config),
        "site_parameters": build_site_from_config(config),
        "soil_parameters": build_soil_from_config(config),
        "start_type": agro.get("start_type"),
        "campaign_start": agro.get("campaign_start", "previous_october"),
        "preserve_agro_dates": bool(agro.get("preserve_dates", False)),
        "remove_timed_n": bool(agro.get("remove_timed_n", False)),
    }
    if crop_model_overrides:
        kwargs.update(copy.deepcopy(crop_model_overrides))
    kwargs.update(build_weather_from_config(config))
    return kwargs


def build_weather_from_config(config):
    weather = config.get("weather", {})
    source = weather.get("source", "provider")
    if source == "file":
        return {"weather_data_file": weather["file"], "random_weather": False}
    if source == "provider":
        provider = weather.get("provider")
        result = {
            "random_weather": bool(weather.get("random_weather", False)),
            "weather_provider": provider,
        }
        if provider in ("openmeteo", "open-meteo", "open_meteo"):
            openmeteo = copy.deepcopy(weather.get("openmeteo", {}))
            result["openmeteo_kwargs"] = {
                "timezone": openmeteo.get("timezone", "UTC"),
                "openmeteo_model": openmeteo.get("model", openmeteo.get("openmeteo_model", "best_match")),
                "start_date": openmeteo.get("start_date"),
                "forecast": bool(openmeteo.get("forecast", False)),
                "force_update": bool(openmeteo.get("force_update", False)),
            }
        else:
            result["openmeteo_kwargs"] = copy.deepcopy(weather.get("openmeteo", {}))
        return result
    if source == "auto":
        return {"random_weather": bool(weather.get("random_weather", False)), "weather_provider": "auto"}
    raise ValueError(f"Unsupported weather source: {source}")


def build_action_space_from_config(config):
    action = config["action_space"]
    partial = config["observation"].get("partial", {})
    if partial.get("enabled", False):
        m_shape = 3 if partial.get("noisy_measure", False) else 2
        if partial.get("measure_all", False):
            return gym.spaces.MultiDiscrete([action["n"], m_shape])
        return gym.spaces.MultiDiscrete([action["n"]] + [m_shape] * len(partial.get("features", [])))
    if action.get("type", "discrete") == "discrete":
        return gym.spaces.Discrete(action["n"])
    if action.get("type") == "box":
        return gym.spaces.Box(action.get("low", 0), action.get("high", float("inf")), shape=(action.get("shape", 1),))
    raise ValueError(f"Unsupported action space type: {action.get('type')}")


def build_env_from_config(config, split="train", crop_model_overrides=None):
    from pcse_gym.envs.maize import Maize
    from pcse_gym.envs.winterwheat import WinterWheat

    env_class_name = config["environment"].get("class", "WinterWheat")
    env_class = {"WinterWheat": WinterWheat, "Maize": Maize}[env_class_name]
    years = config["experiment"][f"{split}_years"]
    locations = _locations_to_tuples(config["experiment"][f"{split}_locations"])
    obs = config["observation"]
    partial = obs.get("partial", {})
    env_cfg = config["environment"]
    reward = config["reward"]
    action = config["action_space"]

    kwargs = {
        "crop_features": obs.get("crop_features", []),
        "action_features": obs.get("action_features", []),
        "weather_features": obs.get("weather_features", []),
        "costs_nitrogen": reward.get("costs_nitrogen", 10.0),
        "potential_fertilizer_action": reward.get("potential_fertilizer_action"),
        "potential_unbounded_box_action": reward.get("potential_unbounded_box_action", 100.0),
        "timestep": env_cfg.get("timestep", 7),
        "years": years,
        "locations": locations,
        "action_space": build_action_space_from_config(config),
        "action_multiplier": action.get("multiplier", 1.0),
        "reward": reward.get("name"),
        "seed": _seed(config),
        "po_features": partial.get("features", []) if partial.get("enabled", False) else [],
        "args_measure": bool(partial.get("enabled", False)),
        "noisy_measure": bool(partial.get("noisy_measure", False)),
        "mask_binary": bool(partial.get("mask_binary", False)),
        "placeholder_val": partial.get("placeholder", -1.11),
        "cost_measure": partial.get("cost_measure", "real"),
        "m_multiplier": partial.get("measure_cost_multiplier", 1),
        "measure_all": bool(partial.get("measure_all", False)),
        "no_weather": bool(env_cfg.get("no_weather", False)),
        "normalize": bool(env_cfg.get("normalize", False)),
        "random_init": bool(env_cfg.get("random_init", False)),
        "reward_var": reward.get("reward_var"),
        "discrete_space": action.get("discrete_amounts"),
        "loc_code": _first_location_code(config["experiment"][f"{split}_locations"]),
        "masked_ac": config["agent"].get("policy_kwargs", {}).get("masked_actor_critic_max_actions", 0),
        "nsteps": config["training"].get("total_timesteps"),
        **build_crop_model_from_config(config, crop_model_overrides=crop_model_overrides),
    }

    if env_class is Maize:
        kwargs["preserve_site_n"] = bool(env_cfg.get("preserve_site_n", True))
        return env_class(**kwargs)
    return env_class(**kwargs)


def build_wrapped_env_from_config(config, env, split="train"):
    if config.get("wrappers", {}).get("monitor", True):
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)

    env = apply_constraints_from_config(config, env)
    return _vectorize_env(config, env, split)


def apply_constraints_from_config(config, env):
    agent_name = config["agent"]["name"]
    soft = config.get("constraints", {}).get("soft", {})
    hard = config.get("constraints", {}).get("hard", {})
    soft_agents = soft.get("enabled_for_agents", [])
    if agent_name in soft_agents:
        env = ConstraintCostWrapper(env, _soft_constraint_kwargs(soft))
    elif hard.get("enabled", True):
        env = ActionConstrainer(
            env,
            action_limit=hard.get("action_limit", 0),
            n_budget=hard.get("n_budget", 0),
            temporal=hard.get("temporal", False),
            fertilize_until_dvs=hard.get("fertilize_until_dvs", 1.0),
        )

    return env


def build_agent_from_config(config, env):
    agent_name = config["agent"]["name"]
    device = config["agent"].get("device", "cpu")
    gamma = config["agent"].get("gamma", 1.0)
    seed = _seed(config)
    kwargs = copy.deepcopy(config["agent"].get("hyperparameters", {}))
    kwargs["policy_kwargs"] = _policy_kwargs(config)

    if agent_name == "PPO":
        from stable_baselines3 import PPO
        return PPO(_policy(config, agent_name), env, gamma=gamma, seed=seed, verbose=0, device=device,
                   tensorboard_log=config.get("run", {}).get("log_dir"), **kwargs)
    if agent_name == "A2C":
        from stable_baselines3 import A2C
        return A2C("MlpPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                   tensorboard_log=config.get("run", {}).get("log_dir"), **kwargs)
    if agent_name == "DQN":
        from stable_baselines3 import DQN
        return DQN("MlpPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                   tensorboard_log=config.get("run", {}).get("log_dir"), **kwargs)
    if agent_name in ("RPPO", "RecurrentPPO"):
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO(_policy(config, agent_name), env, gamma=gamma, seed=seed, verbose=0, device=device,
                            tensorboard_log=config.get("run", {}).get("log_dir"), **kwargs)
    if agent_name == "MaskedPPO":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                           tensorboard_log=config.get("run", {}).get("log_dir"), **kwargs)
    if agent_name == "LagPPO":
        from pcse_gym.agent.ppo_mod import CostActorCriticPolicy, LagrangianPPO
        return LagrangianPPO(CostActorCriticPolicy, env, gamma=gamma, seed=seed, verbose=0, device=device,
                             tensorboard_log=config.get("run", {}).get("log_dir"),
                             **_lagrangian_kwargs(config), **kwargs)
    if agent_name == "RecurrentLagPPO":
        from pcse_gym.agent.ppo_mod import RecurrentLagrangianPPO
        return RecurrentLagrangianPPO("MlpLstmPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                                      tensorboard_log=config.get("run", {}).get("log_dir"),
                                      **_lagrangian_kwargs(config), **kwargs)
    if agent_name == "RecurrentFOCOPS":
        from pcse_gym.agent.ppo_mod import RecurrentFOCOPS
        focops_kwargs = _focops_kwargs(config)
        for key in focops_kwargs:
            kwargs.pop(key, None)
        return RecurrentFOCOPS("MlpLstmPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                               tensorboard_log=config.get("run", {}).get("log_dir"),
                               **_lagrangian_kwargs(config), **focops_kwargs, **kwargs)
    if agent_name == "RecurrentCUP":
        from pcse_gym.agent.ppo_mod import RecurrentCUP
        return RecurrentCUP("MlpLstmPolicy", env, gamma=gamma, seed=seed, verbose=0, device=device,
                            tensorboard_log=config.get("run", {}).get("log_dir"),
                            **_lagrangian_kwargs(config), **kwargs)
    raise ValueError(f"Unsupported agent: {agent_name}")


def build_callbacks_from_config(config, env_eval, comet_experiment=None, irs_method=None):
    from stable_baselines3.common.callbacks import CallbackList

    from pcse_gym.utils.eval import EvalCallback

    eval_callback = EvalCallback(
        env_eval=env_eval,
        train_years=config["experiment"]["train_years"],
        test_years=config["experiment"]["test_years"],
        train_locations=_locations_to_tuples(config["experiment"]["train_locations"]),
        test_locations=_locations_to_tuples(config["experiment"]["test_locations"]),
        seed=_seed(config),
        pcse_model=pcse_model_code(config),
        comet_experiment=comet_experiment,
        multiprocess=config["wrappers"]["vec_env"].get("multiprocess", False),
        eval_freq=config["training"].get("eval_freq", 20_000),
        irs_method=irs_method,
        po_features=config["observation"].get("partial", {}).get("features", []),
        random_weather=config["weather"].get("random_weather", False),
        masked_ac=config["agent"].get("policy_kwargs", {}).get("masked_actor_critic_max_actions", 0),
        decay_entropy=config["agent"].get("policy_kwargs", {}).get("decay_entropy", False),
        mask_later=config["agent"].get("policy_kwargs", {}).get("mask_later", False),
        nsteps=config["training"].get("total_timesteps"),
        n_envs=config["wrappers"]["vec_env"].get("n_envs", 1),
    )
    if comet_experiment is None:
        return eval_callback

    from pcse_gym.config.comet import CometLoggerCallback

    return CallbackList([CometLoggerCallback(comet_experiment), eval_callback])


def build_comet_from_config(config, hyperparams=None):
    comet_cfg = config.get("logging", {}).get("comet", {})
    if not comet_cfg.get("enabled", False):
        return None
    from comet_ml import Experiment

    with open(comet_cfg["api_key_file"], "r") as handle:
        api_key = handle.readline().strip()
    experiment = Experiment(
        api_key=api_key,
        project_name=comet_cfg.get("project_name"),
        workspace=comet_cfg.get("workspace"),
        log_code=comet_cfg.get("log_code", True),
        log_graph=comet_cfg.get("log_graph", True),
        auto_metric_logging=comet_cfg.get("auto_metric_logging", True),
        auto_histogram_tensorboard_logging=comet_cfg.get("auto_histogram_tensorboard_logging", True),
    )
    if comet_cfg.get("log_code", True):
        code_folder = comet_cfg.get("code_folder")
        if code_folder:
            experiment.log_code(folder=code_folder)
        else:
            repo_root = Path(__file__).resolve().parents[2]
            experiment.log_code(file_name=str(repo_root / "train.py"))
            experiment.log_code(folder=str(repo_root / "pcse_gym"))
    if hyperparams:
        experiment.log_parameters(hyperparams)
    experiment.log_parameters(config, prefix="config")
    experiment.log_asset_data(
        yaml.safe_dump(config, sort_keys=False),
        file_name="resolved_config.yaml",
    )
    tags = [str(tag) for tag in comet_cfg.get("tags", [])]
    tags.extend([config["agent"]["name"], str(_seed(config)), config["reward"].get("name", "")])
    experiment.add_tags(tags)
    experiment.set_name(f"{config['experiment']['name']}-{config['agent']['name']}-{config['reward'].get('name')}")
    return experiment


def build_intrinsic_reward_from_config(config, env):
    irs = config.get("training", {}).get("intrinsic_reward", {})
    if not irs.get("enabled", False):
        return None

    name = irs.get("name")
    device = config["agent"].get("device", "cpu")
    if name == "E3B":
        from pcse_gym.intrinsic import E3BIntrinsicReward

        return E3BIntrinsicReward(
            envs=env,
            device=device,
            latent_dim=irs.get("latent_dim") or 128,
            beta=irs.get("beta", 1.0),
            kappa=irs.get("kappa", 0.0),
            gamma=irs.get("gamma"),
            ridge=irs.get("ridge", 0.1),
            lr=irs.get("lr", 0.001),
            batch_size=irs.get("batch_size", 256),
            update_proportion=irs.get("update_proportion", 1.0),
            obs_norm_type=irs.get("obs_norm_type", "none"),
            rwd_norm_type=irs.get("rwd_norm_type", "rms"),
            hidden_dim=irs.get("hidden_dim", 256),
        )
    if name == "ICM":
        from rllte.xplore.reward import ICM

        return ICM(envs=env, device=device, latent_dim=irs.get("latent_dim", 256))
    if name == "RIDE":
        from rllte.xplore.reward import RIDE

        return RIDE(envs=env, device=device)
    raise ValueError(f"Unsupported intrinsic reward: {name}")


def pcse_model_code(config):
    value = config["crop_model"].get("pcse_model")
    if value not in PCSE_MODEL_CODES:
        raise ValueError(f"Unsupported pcse_model: {value}")
    return PCSE_MODEL_CODES[value]


def _read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _pcse():
    import pcse

    return pcse


def _locations_to_tuples(locations):
    result = []
    for location in locations:
        if isinstance(location, dict):
            result.append((location["lat"], location["lon"]))
        else:
            result.append(tuple(location))
    return result


def _first_location_code(locations):
    if not locations:
        return None
    first = locations[0]
    if isinstance(first, dict):
        return first.get("code")
    return None


def _seed(config):
    return config["agent"].get("seed", config["experiment"].get("seed", 0))


def _policy_kwargs(config):
    from pcse_gym.envs.sb3 import get_policy_kwargs

    agent_name = config["agent"]["name"]
    policy_cfg = copy.deepcopy(config["agent"].get("policy_kwargs", {}))
    masked = policy_cfg.pop("masked_actor_critic_max_actions", 0)
    apply_masking = policy_cfg.pop("apply_masking", True)
    activation = policy_cfg.get("activation_fn")
    if isinstance(activation, str):
        policy_cfg["activation_fn"] = ACTIVATIONS[activation]

    if not config["environment"].get("no_weather", False):
        obs = config["observation"]
        partial = obs.get("partial", {})
        extractor_kwargs = get_policy_kwargs(
            n_crop_features=len(obs.get("crop_features", [])),
            n_weather_features=len(obs.get("weather_features", [])),
            n_action_features=len(obs.get("action_features", [])),
            n_po_features=len(partial.get("features", [])) if partial.get("enabled", False) else 0,
            mask_binary=bool(partial.get("mask_binary", False)),
            n_timesteps=config["environment"].get("timestep", 7),
        )
        extractor_kwargs.update(policy_cfg)
        policy_cfg = extractor_kwargs

    if masked and agent_name in ["PPO", "RPPO", "RecurrentPPO"]:
        policy_cfg["max_non_zero_actions"] = masked
        policy_cfg["apply_masking"] = bool(apply_masking)
    return policy_cfg


def _policy(config, agent_name):
    from pcse_gym.agent.masked_actorcriticpolicy import MaskedActorCriticPolicy, MaskedRecurrentActorCriticPolicy

    explicit = config["agent"].get("policy")
    if explicit and explicit != "auto":
        return explicit
    masked = config["agent"].get("policy_kwargs", {}).get("masked_actor_critic_max_actions", 0)
    if agent_name in ("RPPO", "RecurrentPPO"):
        return MaskedRecurrentActorCriticPolicy if masked else "MlpLstmPolicy"
    if agent_name == "PPO":
        return MaskedActorCriticPolicy if masked else "MlpPolicy"
    return "MlpPolicy"


def _soft_constraint_kwargs(soft):
    weights = soft.get("weights", {})
    return {
        "cost_key": soft.get("cost_key", "cost"),
        "components_key": soft.get("components_key", "cost_components"),
        "max_non_zero_actions": soft.get("max_non_zero_actions", 4),
        "fertilize_until_dvs": soft.get("fertilize_until_dvs"),
        "nue_threshold": tuple(soft.get("nue_threshold", (0.5, 0.9))),
        "n_surplus_threshold": tuple(soft.get("n_surplus_threshold", (0.0, 40.0))),
        "n_weight": weights.get("n", 5.0),
        "dvs_weight": weights.get("dvs", 1.0),
        "nue_weight": weights.get("nue", 1.0),
        "n_surplus_weight": weights.get("n_surplus", 1.0),
    }


def _lagrangian_kwargs(config):
    lag = config["agent"].get("lagrangian", {})
    soft = config.get("constraints", {}).get("soft", {})
    return {
        "cost_key": soft.get("cost_key", "cost"),
        "cost_limit": lag.get("cost_limit", 0.0),
        "lambda_init": lag.get("lambda_init", 1.0),
        "lambda_lr": lag.get("lambda_lr", 0.05),
        "lambda_max": lag.get("lambda_max"),
        "cost_vf_coef": lag.get("cost_vf_coef", 0.7),
    }


def _focops_kwargs(config):
    lag = config["agent"].get("lagrangian", {})
    hyperparameters = config["agent"].get("hyperparameters", {})
    result = {}
    for key, default in (("focops_eta", 0.02), ("focops_lam", 1.5)):
        if key in lag:
            result[key] = lag[key]
        elif key in hyperparameters:
            result[key] = hyperparameters[key]
        else:
            result[key] = default
    return result


def _vectorize_env(config, env, split):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    vec_cfg = config["wrappers"].get("vec_env", {})
    if vec_cfg.get("type", "vec_normalize") == "dummy":
        return DummyVecEnv([lambda: env])

    if vec_cfg.get("multiprocess", False) and split == "train":
        vec_env = SubprocVecEnv([lambda: env for _ in range(vec_cfg.get("n_envs", 1))])
    else:
        vec_env = DummyVecEnv([lambda: env])

    if vec_cfg.get("type", "vec_normalize") == "vec_normalize":
        return VecNormalize(
            vec_env,
            norm_obs=vec_cfg.get("norm_obs", True),
            norm_reward=vec_cfg.get("norm_reward", True),
            clip_obs=vec_cfg.get("clip_obs", 10000000.0),
            clip_reward=vec_cfg.get("clip_reward", 100000.0),
            gamma=vec_cfg.get("gamma", 1.0),
        )
    return vec_env
