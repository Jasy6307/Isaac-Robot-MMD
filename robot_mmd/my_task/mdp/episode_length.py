"""Per-env episode-length helpers and random timeout term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ENV_EPISODE_TARGET_STEPS_ATTR = "_g1_episode_target_steps"
_ENV_EPISODE_MIN_SECONDS_ATTR = "_g1_episode_min_seconds"
_ENV_EPISODE_MAX_SECONDS_ATTR = "_g1_episode_max_seconds"


def _ensure_episode_target_steps(env: "ManagerBasedRLEnv") -> torch.Tensor:
    target: torch.Tensor | None = getattr(env, _ENV_EPISODE_TARGET_STEPS_ATTR, None)
    num_envs = int(env.num_envs)
    if target is None or target.shape[0] != num_envs:
        max_steps = int(getattr(env, "max_episode_length", 1))
        target = torch.full((num_envs,), max(max_steps, 1), dtype=torch.long, device=env.device)
        setattr(env, _ENV_EPISODE_TARGET_STEPS_ATTR, target)
        return target
    if target.device != torch.device(env.device):
        target = target.to(device=env.device, dtype=torch.long)
        setattr(env, _ENV_EPISODE_TARGET_STEPS_ATTR, target)
        return target
    return target


def set_episode_target_steps(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, target_steps: torch.Tensor
) -> None:
    """Set per-env target episode steps used by random timeout."""
    if env_ids.numel() == 0:
        return
    target = _ensure_episode_target_steps(env)
    env_ids_i64 = env_ids.to(device=target.device, dtype=torch.long)
    target_i64 = target_steps.to(device=target.device, dtype=torch.long).clamp_min(1)
    target[env_ids_i64] = target_i64


def episode_target_steps(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Get per-env target episode steps (shape [num_envs])."""
    return _ensure_episode_target_steps(env)


def set_runtime_episode_length_seconds(
    env: "ManagerBasedRLEnv", min_seconds: float | None, max_seconds: float | None
) -> None:
    """Set runtime episode-length range used by reset sampling."""
    if min_seconds is not None:
        setattr(env, _ENV_EPISODE_MIN_SECONDS_ATTR, float(min_seconds))
    if max_seconds is not None:
        setattr(env, _ENV_EPISODE_MAX_SECONDS_ATTR, float(max_seconds))


def get_runtime_episode_length_seconds(
    env: "ManagerBasedRLEnv", default_min: float, default_max: float
) -> tuple[float, float]:
    """Resolve runtime episode-length range with safe ordering."""
    min_s = float(getattr(env, _ENV_EPISODE_MIN_SECONDS_ATTR, default_min))
    max_s = float(getattr(env, _ENV_EPISODE_MAX_SECONDS_ATTR, default_max))
    if min_s > max_s:
        min_s, max_s = max_s, min_s
    return min_s, max_s


def sample_episode_target_steps(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    min_seconds: float,
    max_seconds: float,
) -> torch.Tensor:
    """Sample per-env target episode steps from [min_seconds, max_seconds]."""
    if env_ids.numel() == 0:
        return torch.zeros((0,), device=env.device, dtype=torch.long)
    min_steps = max(1, int(round(float(min_seconds) / float(env.step_dt))))
    max_steps = max(1, int(round(float(max_seconds) / float(env.step_dt))))
    if min_steps > max_steps:
        min_steps, max_steps = max_steps, min_steps
    if min_steps == max_steps:
        return torch.full((env_ids.numel(),), min_steps, device=env.device, dtype=torch.long)
    # randint high is exclusive.
    return torch.randint(
        low=min_steps,
        high=max_steps + 1,
        size=(env_ids.numel(),),
        device=env.device,
        dtype=torch.long,
    )


def random_episode_time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env timeout against sampled target episode steps."""
    targets = episode_target_steps(env)
    steps = env.episode_length_buf.to(device=targets.device, dtype=torch.long)
    return steps >= targets

