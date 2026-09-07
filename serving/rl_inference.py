"""Inference wrapper for the current Stable-Baselines3 CN-Maize pipeline.

The module deliberately delegates crop, soil, site, agro-management, weather,
wrapper, and policy construction to the project's config loader and builders.
It only owns artifact validation, VecNormalize/model restoration, inference
mode, compatibility checks, and JSON-safe result shaping.
"""

from __future__ import annotations

import copy
import numbers
from datetime import date, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pcse_gym.agent.ppo_mod import LagrangianPPO
from pcse_gym.config import load_config
from pcse_gym.config.builders import (
    build_env_from_config,
    build_wrapped_env_from_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_KG_HA_PER_ACTION_UNIT = 10.0


class InferenceCompatibilityError(ValueError):
    """Raised when a model and restored environment cannot be paired."""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def action_index_to_n_rate(action_index: int, multiplier: float = 1.0) -> float:
    """Map a discrete action to kg N/ha using the current SB3 environment.

    The source mapping is ``StableBaselinesWrapper._apply_action`` and
    ``StableBaselinesWrapper.step`` in ``pcse_gym/envs/sb3.py``: a discrete
    action unit is multiplied by the configured action multiplier and then by
    10 kg N/ha.
    """

    if isinstance(action_index, bool) or not isinstance(action_index, numbers.Integral):
        raise TypeError("action_index must be an integer")
    return float(action_index) * float(multiplier) * N_KG_HA_PER_ACTION_UNIT


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _latest_fertilizer_value(info: dict[str, Any]) -> float | None:
    value = info.get("fertilizer")
    if isinstance(value, dict):
        if not value:
            return None
        return float(list(value.values())[-1])
    if value is None:
        return None
    return float(value)


class RLInferenceEngine:
    """Load and run a current CN-Maize ``LagrangianPPO`` artifact pair."""

    def __init__(
        self,
        config_path: str | Path,
        model_path: str | Path,
        env_stats_path: str | Path,
        device: str = "auto",
        env_split: str = "test",
    ) -> None:
        self.config_path = _resolve_path(config_path)
        self.model_path = _resolve_path(model_path)
        self.env_stats_path = _resolve_path(env_stats_path)
        self.device = device
        self.env_split = env_split
        self._last_observation: np.ndarray | None = None

        for name, path in (
            ("config", self.config_path),
            ("model", self.model_path),
            ("environment statistics", self.env_stats_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{name} file does not exist: {path}")

        self.config = load_config(
            str(self.config_path),
            ["logging.comet.enabled=false"],
        )
        if self.config["agent"]["name"] != "LagPPO":
            raise InferenceCompatibilityError(
                "RLInferenceEngine requires agent.name=LagPPO; "
                f"got {self.config['agent']['name']!r}"
            )

        self.env = self._build_inference_environment()
        try:
            self.model = LagrangianPPO.load(
                str(self.model_path),
                env=self.env,
                device=self.device,
            )
            if not isinstance(self.model, LagrangianPPO):
                raise InferenceCompatibilityError(
                    "Loaded artifact is not a LagrangianPPO model: "
                    f"{type(self.model)!r}"
                )
            self._validate_compatibility()
        except Exception:
            self.env.close()
            raise

        self.env.training = False
        self.env.norm_reward = False

    def _build_inference_environment(self) -> VecNormalize:
        """Build the current pre-VecNormalize chain through project builders."""

        builder_config = copy.deepcopy(self.config)
        vec_config = builder_config["wrappers"]["vec_env"]
        # The builder normally returns VecNormalize directly.  Build its
        # underlying DummyVecEnv first so the saved normalization state can be
        # attached with VecNormalize.load().  This changes only in-memory
        # config, never the YAML file.
        vec_config["type"] = "dummy"
        vec_config["multiprocess"] = False

        raw_env = build_env_from_config(builder_config, split=self.env_split)
        base_vec_env: DummyVecEnv | None = None
        try:
            base_vec_env = build_wrapped_env_from_config(
                builder_config,
                raw_env,
                split=self.env_split,
            )
            if not isinstance(base_vec_env, DummyVecEnv):
                raise InferenceCompatibilityError(
                    "Expected current builder to produce DummyVecEnv before "
                    f"VecNormalize.load(); got {type(base_vec_env)!r}"
                )
            env = VecNormalize.load(str(self.env_stats_path), base_vec_env)
            env.training = False
            env.norm_reward = False
            return env
        except Exception:
            if base_vec_env is not None:
                base_vec_env.close()
            else:
                raw_env.close()
            raise

    def _validate_compatibility(self) -> None:
        model_shape = self.model.observation_space.shape
        env_shape = self.env.observation_space.shape
        if model_shape != env_shape:
            raise InferenceCompatibilityError(
                f"Observation space mismatch: model={model_shape}, env={env_shape}"
            )

        model_action_space = self.model.action_space
        env_action_space = self.env.action_space
        if not isinstance(model_action_space, gym.spaces.Discrete) or not isinstance(
            env_action_space, gym.spaces.Discrete
        ):
            raise InferenceCompatibilityError(
                "CN-Maize inference requires Discrete action spaces: "
                f"model={model_action_space}, env={env_action_space}"
            )
        if model_action_space.n != env_action_space.n:
            raise InferenceCompatibilityError(
                f"Action space mismatch: model.n={model_action_space.n}, "
                f"env.n={env_action_space.n}"
            )

    def _simulation_date(self) -> date:
        return self.env.get_attr("date")[0]

    def _validate_observation(self, observation: np.ndarray) -> None:
        expected_shape = (self.env.num_envs,) + tuple(self.env.observation_space.shape)
        if observation.shape != expected_shape:
            raise InferenceCompatibilityError(
                f"Observation shape mismatch: got {observation.shape}, "
                f"expected {expected_shape}"
            )
        if not np.issubdtype(observation.dtype, np.number):
            raise InferenceCompatibilityError(
                f"Observation dtype is not numeric: {observation.dtype}"
            )
        if not np.isfinite(observation).all():
            raise InferenceCompatibilityError("Observation contains NaN or Inf")

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the vectorized inference environment and return JSON-safe data."""

        if seed is not None:
            # The installed SB3 VecNormalize API exposes reset() without a
            # seed keyword. VecEnv.seed() schedules the seed for the next
            # reset, which is the supported Gymnasium/SB3 path here.
            self.env.seed(seed)
        observation = self.env.reset()
        observation = np.asarray(observation)
        self._validate_observation(observation)
        self._last_observation = observation
        return {
            "observation": _json_safe(observation),
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "simulation_date": self._simulation_date().isoformat(),
        }

    def predict(self, deterministic: bool = True) -> dict[str, Any]:
        """Predict one action from the most recently reset observation."""

        if self._last_observation is None:
            raise RuntimeError("Call reset() before predict().")
        action, _state = self.model.predict(
            self._last_observation,
            deterministic=deterministic,
        )
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise InferenceCompatibilityError(
                f"Expected one discrete action, got shape {action_array.shape}"
            )
        action_index = int(action_array.reshape(-1)[0])
        if not self.env.action_space.contains(action_index):
            raise InferenceCompatibilityError(
                f"Predicted action {action_index} is outside {self.env.action_space}"
            )
        n_rate = action_index_to_n_rate(
            action_index,
            self.config["action_space"].get("multiplier", 1.0),
        )
        return {
            "action_index": action_index,
            "action_dtype": str(action_array.dtype),
            "n_rate_kg_ha": n_rate,
            "deterministic": bool(deterministic),
        }

    def step(self, action_index: int) -> dict[str, Any]:
        """Apply exactly one discrete action and return JSON-safe transition data."""

        if isinstance(action_index, bool) or not isinstance(action_index, numbers.Integral):
            raise TypeError("action_index must be an integer")
        action_index = int(action_index)
        if not self.env.action_space.contains(action_index):
            raise ValueError(
                f"action_index={action_index} is outside {self.env.action_space}"
            )
        if self._last_observation is None:
            raise RuntimeError("Call reset() before step().")

        date_before = self._simulation_date()
        action = np.asarray([action_index], dtype=np.int64)
        observation, rewards, dones, infos = self.env.step(action)
        observation = np.asarray(observation)
        self._validate_observation(observation)
        reward = float(np.asarray(rewards).reshape(-1)[0])
        if not np.isfinite(reward):
            raise InferenceCompatibilityError("Reward contains NaN or Inf")

        info = infos[0]
        done = bool(np.asarray(dones).reshape(-1)[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done and not truncated)
        date_after = self._simulation_date()
        n_rate = action_index_to_n_rate(
            action_index,
            self.config["action_space"].get("multiplier", 1.0),
        )
        n_applied = _latest_fertilizer_value(info)
        if n_applied is None:
            n_applied = n_rate

        self._last_observation = observation
        return {
            "observation": _json_safe(observation),
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "date_before": date_before.isoformat(),
            "date_after": date_after.isoformat(),
            "days_advanced": int((date_after - date_before).days),
            "action_index": action_index,
            "n_rate_kg_ha": n_rate,
            "n_applied_kg_ha": float(n_applied),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return model/environment metadata without embedding model arrays."""

        return {
            "algorithm": "LagrangianPPO",
            "config_path": str(self.config_path),
            "model_path": str(self.model_path),
            "env_stats_path": str(self.env_stats_path),
            "observation_dim": int(self.env.observation_space.shape[0]),
            "action_n": int(self.env.action_space.n),
            "timestep_days": int(self.config["environment"].get("timestep", 7)),
            "observation_dtype": str(self.env.observation_space.dtype),
            "device": str(self.model.device),
        }

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "RLInferenceEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
