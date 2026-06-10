from pcse_gym.config.loader import ConfigError, load_config


def train_from_config(config):
    from pcse_gym.config.training import train_from_config as _train_from_config

    return _train_from_config(config)

__all__ = ["ConfigError", "load_config", "train_from_config"]
