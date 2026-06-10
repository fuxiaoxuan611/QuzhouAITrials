import copy
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "run",
    "environment",
    "observation",
    "action_space",
    "reward",
    "constraints",
    "crop_model",
    "crop",
    "soil",
    "site",
    "weather",
    "agent",
    "wrappers",
    "training",
    "logging",
}

PATH_KEYS = {
    ("run", "log_dir"),
    ("crop_model", "model_config"),
    ("crop_model", "agro", "path"),
    ("crop", "fpath"),
    ("soil", "path"),
    ("site", "path"),
    ("weather", "file"),
    ("logging", "comet", "api_key_file"),
    ("logging", "comet", "code_folder"),
}


def load_config(path, overrides=None):
    config_path = Path(path)
    if not config_path.is_absolute() and "configs" not in config_path.parts:
        config_path = Path("configs") / config_path
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ConfigError("Experiment config must be a YAML mapping.")
    _validate_top_level_keys(config)
    if config.get("schema_version") != 1:
        raise ConfigError("Only schema_version: 1 is supported.")

    config = copy.deepcopy(config)
    for override in overrides or []:
        apply_override(config, override)
    _resolve_references(config)
    _resolve_paths(config)
    _validate_required_sections(config)
    return config


def apply_override(config, override):
    if "=" not in override:
        raise ConfigError(f"Override must use key=value syntax: {override}")
    key, raw_value = override.split("=", 1)
    path = key.split(".")
    if not all(path):
        raise ConfigError(f"Invalid override key: {key}")
    value = yaml.safe_load(raw_value)

    node = config
    for index, part in enumerate(path[:-1]):
        if not isinstance(node, dict):
            raise ConfigError(f"Unknown override key: {key}")
        if part not in node:
            raise ConfigError(f"Unknown override key: {key}")
        node = node[part]
    final_key = path[-1]
    if not isinstance(node, dict) or final_key not in node:
        raise ConfigError(f"Unknown override key: {key}")
    node[final_key] = value


def _validate_top_level_keys(config):
    unknown = sorted(set(config) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"Unknown top-level config keys: {', '.join(unknown)}")


def _validate_required_sections(config):
    required = [
        "experiment",
        "environment",
        "observation",
        "action_space",
        "reward",
        "crop_model",
        "crop",
        "soil",
        "site",
        "weather",
        "agent",
        "wrappers",
        "training",
        "logging",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required config sections: {', '.join(missing)}")


def _resolve_references(config):
    def resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return _get_by_path(config, value[2:-1].split("."))
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = resolve(item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = resolve(item)
        return value

    resolve(config)


def _get_by_path(config, path):
    node = config
    for part in path:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"Unknown interpolation reference: {'.'.join(path)}")
        node = node[part]
    return copy.deepcopy(node)


def _resolve_paths(config):
    for path in PATH_KEYS:
        parent = config
        for part in path[:-1]:
            parent = parent.get(part, {}) if isinstance(parent, dict) else {}
        key = path[-1]
        value = parent.get(key) if isinstance(parent, dict) else None
        if value in (None, ""):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        parent[key] = str(candidate)
