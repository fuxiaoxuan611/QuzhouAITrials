"""Reconstruct a CN-Maize WOFOST state at a requested date.

This module deliberately builds the current Maize environment through the
project config loader and builders.  It does not load an RL checkpoint or
make policy decisions.  Historical decision actions are replayed through the
same environment step method; a final non-aligned residual is advanced
directly through the project's PCSE Engine with no new fertilizer action.
"""

from __future__ import annotations

import numbers
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from pcse_gym.config import load_config
from pcse_gym.config.builders import build_env_from_config


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
    ) -> None:
        self.config_path = _resolve_path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"config file does not exist: {self.config_path}")

        self.config = load_config(
            str(self.config_path),
            ["logging.comet.enabled=false"],
        )
        self.env_split = env_split
        self.seed = seed if seed is not None else self.config["experiment"].get("seed")
        self.timestep_days = int(self.config["environment"].get("timestep", 7))
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
        return self.env.sb3_env.agmt.crop_start_date

    @property
    def crop_end_date(self) -> date:
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

    def _reset(self) -> None:
        self.env.reset(seed=self.seed)
        if self.env.date != self.crop_start_date:
            raise WOFOSTRealtimeError(
                "RESET_DATE_MISMATCH: "
                f"environment={self.env.date}, crop_start={self.crop_start_date}"
            )

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

        self._reset()
        steps_executed = 0
        terminated = False
        truncated = False
        for action_index in actions:
            _, _, terminated, truncated, _ = self.env.step(action_index)
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
        }

    def close(self) -> None:
        close = getattr(self, "env", None)
        if close is not None:
            close.close()

    def __enter__(self) -> "WOFOSTRealtimeEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
