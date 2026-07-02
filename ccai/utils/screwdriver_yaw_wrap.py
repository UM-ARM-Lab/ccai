"""Yaw wrapping helpers for screwdriver task-model queries."""

from __future__ import annotations

import math

import torch


YAW_WRAP_PERIOD = math.pi / 2.0
_EPISODE_START_KEY = "_screwdriver_yaw_wrap_episode_start"
_COUNT_KEY = "_screwdriver_yaw_wrap_count"


def reset_screwdriver_yaw_wrap(params: dict, state) -> None:
    """Start a new episode yaw-wrap frame from the raw simulator state."""
    state_t = torch.as_tensor(state)
    params[_EPISODE_START_KEY] = float(state_t.reshape(-1)[-1].item())
    params[_COUNT_KEY] = 0


def ensure_screwdriver_yaw_wrap(params: dict, state) -> None:
    if _EPISODE_START_KEY not in params or _COUNT_KEY not in params:
        reset_screwdriver_yaw_wrap(params, state)


def update_screwdriver_yaw_wrap_after_recovery(params: dict, state) -> int:
    """Advance the wrap count after recovery if full clockwise quarter-turns were crossed."""
    ensure_screwdriver_yaw_wrap(params, state)
    state_t = torch.as_tensor(state)
    raw_yaw = float(state_t.reshape(-1)[-1].item())
    episode_start_yaw = float(params[_EPISODE_START_KEY])
    crossed = math.floor((episode_start_yaw - raw_yaw + 1.0e-9) / YAW_WRAP_PERIOD)
    params[_COUNT_KEY] = max(0, int(crossed))
    return int(params[_COUNT_KEY])


def screwdriver_yaw_wrap_offset(params: dict) -> float:
    return float(params.get(_COUNT_KEY, 0)) * YAW_WRAP_PERIOD


def wrap_screwdriver_task_state_yaw(params: dict, state, *, yaw_idx: int = -1):
    """Return a cloned state with yaw wrapped for task-model inputs only."""
    if state is None:
        return None
    ensure_screwdriver_yaw_wrap(params, state)
    state_wrapped = state.clone() if hasattr(state, "clone") else torch.as_tensor(state).clone()
    state_wrapped[..., yaw_idx] = state_wrapped[..., yaw_idx] + screwdriver_yaw_wrap_offset(params)
    return state_wrapped


def unwrap_screwdriver_task_state_yaw(params: dict, state, *, yaw_idx: int = -1):
    """Return a cloned task-model state with yaw transformed back to raw simulator yaw."""
    if state is None:
        return None
    ensure_screwdriver_yaw_wrap(params, state)
    state_unwrapped = state.clone() if hasattr(state, "clone") else torch.as_tensor(state).clone()
    state_unwrapped[..., yaw_idx] = state_unwrapped[..., yaw_idx] - screwdriver_yaw_wrap_offset(params)
    return state_unwrapped
