"""Per-episode spawn anchor for drift / fly terminations.

Isaac Lab scenes with a plane terrain expose ``scene.env_origins`` from the
terrain grid, while articulations are cloned on the GridCloner grid stored in
``scene._default_env_origins``.  For some scene layouts those two grids differ,
so drift measured as ``root_pos_w - terrain.env_origins`` can be non-zero
immediately after reset (episode length = 1, drift_xy = 100%).

We anchor terminations to the robot's expected spawn root in **world frame**:
``default_root_state.pos + cloner_env_origins`` and refresh that anchor on each
reset event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_SPAWN_ROOT_W_ATTR = "_g1_dance_spawn_root_w"


def cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


def expected_spawn_root_w(env: "ManagerBasedRLEnv", asset: Articulation) -> torch.Tensor:
    """Expected root world position at episode start, shape ``[num_envs, 3]``."""
    spawn = asset.data.default_root_state[:, :3].clone()
    spawn += cloner_env_origins(env)
    return spawn


def get_spawn_root_w(env: "ManagerBasedRLEnv", asset: Articulation) -> torch.Tensor:
    """Return cached per-env spawn root (world); lazily init from expected spawn."""
    cache: torch.Tensor | None = getattr(env, _SPAWN_ROOT_W_ATTR, None)
    if cache is None:
        cache = expected_spawn_root_w(env, asset)
        setattr(env, _SPAWN_ROOT_W_ATTR, cache)
    return cache


def cache_spawn_root_w(
    env: "ManagerBasedRLEnv",
    asset: Articulation,
    env_ids: torch.Tensor,
) -> None:
    """Refresh spawn anchor for envs that just reset."""
    cache = get_spawn_root_w(env, asset)
    cache[env_ids] = expected_spawn_root_w(env, asset)[env_ids]
