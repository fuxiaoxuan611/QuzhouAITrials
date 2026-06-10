from datetime import datetime

from pcse_gym.config.builders import (
    build_agent_from_config,
    build_callbacks_from_config,
    build_comet_from_config,
    build_env_from_config,
    build_intrinsic_reward_from_config,
    build_wrapped_env_from_config,
)


def train_from_config(config):
    comet_experiment = build_comet_from_config(
        config,
        hyperparams=config["agent"].get("hyperparameters", {}),
    )

    try:
        train_env = build_env_from_config(config, split="train")
        train_env = build_wrapped_env_from_config(config, train_env, split="train")
        model = build_agent_from_config(config, train_env)

        eval_env = build_env_from_config(config, split="test")
        eval_env = build_wrapped_env_from_config(config, eval_env, split="test")
        irs_method = build_intrinsic_reward_from_config(config, train_env)
        callback = build_callbacks_from_config(config, eval_env, comet_experiment, irs_method)

        if config.get("run", {}).get("dry_run", False):
            print(f"Dry run complete. Built {config['agent']['name']} for {config['experiment']['name']}.")
            return model

        tb_log_name = _tb_log_name(config)
        start_time = datetime.now()
        print(f"Time started training: {start_time}")
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callback,
            tb_log_name=tb_log_name,
        )
        print(f"Time taken to train {tb_log_name}: {datetime.now() - start_time}")
        return model
    finally:
        if comet_experiment is not None:
            comet_experiment.end()


def _tb_log_name(config):
    suffix = config["training"].get("tb_log_name_suffix", "run")
    return (
        f"{config['experiment']['name']}-Seed-{config['experiment'].get('seed', 0)}"
        f"-nsteps-{config['training']['total_timesteps']}"
        f"-{config['agent']['name']}-{config['reward'].get('name')}-{suffix}"
    )
