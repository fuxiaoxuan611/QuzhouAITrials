"""Reconstruct a CN-Maize WOFOST state at a requested date.

This module deliberately builds the current Maize environment through the
project config loader and builders.  It does not load an RL checkpoint or
make policy decisions.  Historical decision actions are replayed through the
same environment step method; a final non-aligned residual is advanced
directly through the project's PCSE Engine with no new fertilizer action.
"""

from __future__ import annotations

import numbers
import copy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from pcse_gym.config import load_config
from pcse_gym.config.builders import build_env_from_config

from .management_events import ManagementTimeline
from .season_calendar import SeasonCalendar
from .weather_providers.timeline import TimelineWeatherDataProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FIELDS = (
    "DVS",
    "LAI",
    "TAGP",
    "NuptakeTotal",
    "NO3",
    "NH4",
    "WC",
    "RFTRA",
    "WSO",
    "NLOSSCUM",
    "NUE",
    "Nsurp",
)


class WOFOSTRealtimeError(ValueError):
    """Base error for invalid realtime reconstruction requests or states."""


class QueryDateError(WOFOSTRealtimeError):
    """Raised when a requested date is outside the configured crop window."""


class ActionHistoryError(WOFOSTRealtimeError):
    """Raised when historical actions cannot be replayed unambiguously."""


class StateValidationError(WOFOSTRealtimeError):
    """Raised when the reconstructed WOFOST state is not finite."""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise QueryDateError(
                f"QUERY_DATE_INVALID: expected ISO date, got {value!r}"
            ) from exc
    raise TypeError("query_date must be an ISO date string or datetime.date")


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


def _validate_finite(value: Any, field: str) -> None:
    if value is None:
        return
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise StateValidationError(
            f"STATE_VALUE_INVALID: {field} is not numeric"
        ) from exc
    if not np.isfinite(values).all():
        raise StateValidationError(f"STATE_VALUE_NONFINITE: {field} contains NaN or Inf")


class WOFOSTRealtimeEngine:
    """Replay CN-Maize actions and return the WOFOST state at a date."""

    def __init__(
        self,
        config_path: str | Path = "configs/CN-Maize.yaml",
        env_split: str = "test",
        seed: int | None = None,
        season_calendar: SeasonCalendar | None = None,
        timeline_provider: TimelineWeatherDataProvider | None = None,
        management_timeline: ManagementTimeline | None = None,
    ) -> None:
        self.config_path = _resolve_path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"config file does not exist: {self.config_path}")

        self.config = load_config(
            str(self.config_path),
            ["logging.comet.enabled=false"],
        )
        self.dynamic = season_calendar is not None or timeline_provider is not None
        self.season_calendar = season_calendar
        self.timeline_provider = timeline_provider
        self.management_timeline = management_timeline or ManagementTimeline()
        self.env_split = env_split
        self.seed = seed if seed is not None else self.config["experiment"].get("seed")
        self.timestep_days = int(self.config["environment"].get("timestep", 7))
        if self.dynamic:
            if season_calendar is None or timeline_provider is None:
                raise WOFOSTRealtimeError(
                    "dynamic reconstruction requires season_calendar and timeline_provider"
                )
            dynamic_config = copy.deepcopy(self.config)
            dynamic_config.setdefault("crop_model", {}).setdefault("agro", {})[
                "preserve_dates"
            ] = True
            overrides = {
                "agro_config": season_calendar.agromanagement_structure(self.management_timeline),
                "weather_data_provider": timeline_provider,
                "weather_provider": "timeline",
                "remove_timed_n": False,
                "preserve_agro_dates": True,
            }
            self.env = build_env_from_config(
                dynamic_config,
                split=env_split,
                crop_model_overrides=overrides,
            )
        else:
            self.env = build_env_from_config(self.config, split=env_split)

        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            self.close()
            raise WOFOSTRealtimeError(
                "CN-Maize realtime reconstruction requires a Discrete action space"
            )
        if not hasattr(self.env, "sb3_env") or not hasattr(self.env, "model"):
            self.close()
            raise WOFOSTRealtimeError(
                "Current Maize builder did not return the expected WOFOST environment"
            )

    @property
    def crop_start_date(self) -> date:
        if self.season_calendar is not None:
            return self.season_calendar.crop_start_date
        return self.env.sb3_env.agmt.crop_start_date

    @property
    def crop_end_date(self) -> date:
        if self.season_calendar is not None:
            return self.season_calendar.crop_end_date
        return self.env.sb3_env.agmt.crop_end_date

    @property
    def action_count(self) -> int:
        return int(self.env.action_space.n)

    def _normalise_actions(self, action_history: Iterable[int] | None) -> list[int]:
        if action_history is None:
            return []
        if isinstance(action_history, (str, bytes)):
            raise ActionHistoryError("ACTION_HISTORY_INVALID: expected an iterable of integers")

        try:
            actions = list(action_history)
        except TypeError as exc:
            raise ActionHistoryError(
                "ACTION_HISTORY_INVALID: expected an iterable of integers"
            ) from exc

        result = []
        for position, action in enumerate(actions):
            if isinstance(action, bool) or not isinstance(action, numbers.Integral):
                raise ActionHistoryError(
                    f"INVALID_ACTION_INDEX: position {position} is not an integer"
                )
            action_index = int(action)
            if not self.env.action_space.contains(action_index):
                raise ActionHistoryError(
                    f"INVALID_ACTION_INDEX: {action_index} is outside "
                    f"[0, {self.action_count - 1}]"
                )
            result.append(action_index)
        return result

    def _reset(self) -> Any:
        observation = self.env.reset(seed=self.seed)
        if isinstance(observation, tuple):
            observation = observation[0]
        if self.env.date != self.crop_start_date:
            raise WOFOSTRealtimeError(
                "RESET_DATE_MISMATCH: "
                f"environment={self.env.date}, crop_start={self.crop_start_date}"
            )
        return observation

    def _current_state(self) -> dict[str, Any]:
        output = self.env.model.get_output()
        if not output:
            raise StateValidationError("STATE_EMPTY: WOFOST returned no output")
        latest = output[-1]
        state = {}
        for field in STATE_FIELDS:
            if field not in latest or latest[field] is None:
                continue
            _validate_finite(latest[field], field)
            state[field] = _json_safe(latest[field])
        return state

    @property
    def observation_feature_names(self) -> list[str]:
        """Return the feature order used by the current SB3 observation code."""

        sb3_env = self.env.sb3_env
        names = list(sb3_env.crop_features) + list(sb3_env.action_features)
        if not sb3_env.no_weather:
            if self.timestep_days == 7:
                names.extend(sb3_env.weather_features)
            else:
                for day_index in range(self.timestep_days):
                    names.extend(
                        f"{feature}[day_{day_index}]"
                        for feature in sb3_env.weather_features
                    )
        if sb3_env.mask_binary:
            names.extend(sb3_env.po_features)
        return names

    def _format_training_observation(self, observation: Any) -> dict[str, Any]:
        """Validate an observation returned by the real training wrapper."""

        sb3_env = self.env.sb3_env
        vector = np.asarray(observation)
        expected_shape = tuple(sb3_env.observation_space.shape)
        if vector.shape != expected_shape:
            raise StateValidationError(
                "OBSERVATION_SHAPE_MISMATCH: got "
                f"{vector.shape}, expected {expected_shape}"
            )
        _validate_finite(vector, "raw_observation")
        return {
            "raw_observation": _json_safe(vector),
            "raw_observation_shape": list(vector.shape),
            "raw_observation_dtype": str(vector.dtype),
            "observation_feature_names": self.observation_feature_names,
            "observation_space_dtype": str(sb3_env.observation_space.dtype),
        }

    def reconstruct(
        self,
        query_date: str | date | datetime,
        action_history: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Reconstruct the WOFOST state at ``query_date``.

        ``action_history`` contains one discrete action for each complete
        decision interval before the query date.  No missing actions are
        filled automatically.  If the query is between decision boundaries,
        the remaining days are advanced through the PCSE engine with
        ``action=0``; this is a residual simulation interval, not a new RL
        decision.
        """

        requested = _as_date(query_date)
        crop_start = self.crop_start_date
        crop_end = self.crop_end_date
        if requested < crop_start:
            raise QueryDateError(
                "QUERY_DATE_BEFORE_CROP_START: "
                f"requested={requested}, crop_start={crop_start}"
            )
        if requested > crop_end:
            raise QueryDateError(
                "QUERY_DATE_AFTER_CROP_END: "
                f"requested={requested}, crop_end={crop_end}"
            )

        if self.dynamic:
            return self._reconstruct_dynamic(requested)

        actions = self._normalise_actions(action_history)
        days_from_start = (requested - crop_start).days
        required_steps, residual_days = divmod(days_from_start, self.timestep_days)
        if len(actions) < required_steps:
            raise ActionHistoryError(
                "INSUFFICIENT_ACTION_HISTORY: "
                f"required={required_steps}, received={len(actions)}"
            )
        if len(actions) > required_steps:
            raise ActionHistoryError(
                "ACTION_HISTORY_AFTER_QUERY: "
                f"expected={required_steps}, received={len(actions)}"
            )

        last_observation = self._reset()
        steps_executed = 0
        terminated = False
        truncated = False
        for action_index in actions:
            last_observation, _, terminated, truncated, _ = self.env.step(action_index)
            steps_executed += 1
            if self.env.date > requested:
                raise WOFOSTRealtimeError(
                    "RECONSTRUCTION_OVERSHOT_QUERY: "
                    f"simulation={self.env.date}, requested={requested}"
                )
            if terminated or truncated:
                if self.env.date != requested:
                    raise WOFOSTRealtimeError(
                        "TERMINATED_BEFORE_QUERY: "
                        f"simulation={self.env.date}, requested={requested}"
                    )
                break

        if self.env.date < requested:
            if terminated or truncated:
                raise WOFOSTRealtimeError(
                    "TERMINATED_BEFORE_QUERY: "
                    f"simulation={self.env.date}, requested={requested}"
                )
            if residual_days:
                model_run = getattr(self.env.model, "run", None)
                if not callable(model_run):
                    raise WOFOSTRealtimeError(
                        "PARTIAL_STEP_UNSUPPORTED: current PCSE engine has no run()"
                    )
                model_run(days=residual_days, action=0)

        simulation_date = self.env.date
        if simulation_date != requested:
            raise WOFOSTRealtimeError(
                "RECONSTRUCTION_DATE_MISMATCH: "
                f"simulation={simulation_date}, requested={requested}"
            )

        terminated = bool(getattr(self.env.model, "terminated", terminated))
        if residual_days == 0:
            # This is the exact flat vector returned by StableBaselinesWrapper
            # at reset/step.  Reusing the returned vector also preserves the
            # wrapper's counter timing for ``week`` and ``Naction``.
            training_observation = self._format_training_observation(last_observation)
        else:
            # A residual date is a valid WOFOST state but not a policy
            # observation boundary.  Do not label a newly assembled vector as
            # an exact training observation for that date.
            training_observation = {
                "raw_observation": None,
                "raw_observation_shape": None,
                "raw_observation_dtype": None,
                "observation_feature_names": self.observation_feature_names,
                "observation_space_dtype": str(self.env.sb3_env.observation_space.dtype),
            }
        return {
            "requested_query_date": requested.isoformat(),
            "simulation_date": simulation_date.isoformat(),
            "crop_start_date": crop_start.isoformat(),
            "crop_end_date": crop_end.isoformat(),
            "timestep_days": self.timestep_days,
            "steps_executed": steps_executed,
            "residual_days": residual_days,
            "exact_date_supported": True,
            "terminated": terminated,
            "truncated": bool(truncated),
            "action_history": actions,
            "crop_state": self._current_state(),
            **training_observation,
        }

    def _reconstruct_dynamic(self, requested: date) -> dict[str, Any]:
        """Replay a dynamic calendar by date, including timed management events."""

        if self.timeline_provider is None or self.season_calendar is None:
            raise WOFOSTRealtimeError("dynamic engine is missing calendar or weather timeline")
        if requested < self.crop_start_date:
            raise QueryDateError(
                f"QUERY_DATE_BEFORE_CROP_START: requested={requested}, crop_start={self.crop_start_date}"
            )
        if requested > self.crop_end_date:
            raise QueryDateError(
                f"QUERY_DATE_AFTER_CROP_END: requested={requested}, crop_end={self.crop_end_date}"
            )
        observation = self._reset()
        days = (requested - self.crop_start_date).days
        if days:
            try:
                self.env.model.run(days=days, action=0)
            except Exception as exc:
                raise WOFOSTRealtimeError("DYNAMIC_RECONSTRUCTION_ERROR") from exc
        sb3_env = self.env.sb3_env
        # Keep the feature counters semantically aligned with the policy's
        # legacy wrapper while timed events remain date-based and independent
        # of RL action boundaries.
        sb3_env.week = days / 7.0
        sb3_env.n_action = sum(
            1 for event in self.management_timeline.fertilizers if event.date <= requested and event.n_rate_kg_ha > 0
        )
        sb3_env.cumulative_fertilization = sum(
            event.n_rate_kg_ha for event in self.management_timeline.fertilizers if event.date <= requested
        )
        output = self.env.model.get_output()
        if not output:
            raise StateValidationError("STATE_EMPTY: WOFOST returned no output")
        recent_output = output[-self.timestep_days:]
        raw = sb3_env._observation(sb3_env._get_observation(recent_output))
        training_observation = self._format_training_observation(raw)
        return {
            "requested_query_date": requested.isoformat(),
            "simulation_date": self.env.date.isoformat(),
            "crop_start_date": self.crop_start_date.isoformat(),
            "crop_end_date": self.crop_end_date.isoformat(),
            "timestep_days": self.timestep_days,
            "steps_executed": days // self.timestep_days,
            "residual_days": days % self.timestep_days,
            "exact_date_supported": days % self.timestep_days == 0,
            "terminated": bool(getattr(self.env.model, "terminated", False)),
            "truncated": False,
            "action_history": [],
            "management_events_applied": [
                event.to_dict()
                for event in self.management_timeline.events_on(requested)
            ],
            "crop_state": self._current_state(),
            **training_observation,
        }

    def close(self) -> None:
        close = getattr(self, "env", None)
        if close is not None:
            close.close()

    def __enter__(self) -> "WOFOSTRealtimeEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
