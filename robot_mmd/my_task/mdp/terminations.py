"""Custom termination terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from robot_mmd.my_task.mdp.spawn_anchor import get_spawn_root_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_above_spawn(
    env: "ManagerBasedRLEnv",
    max_height_above_spawn: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root rises more than ``max_height_above_spawn`` above reset spawn."""
    asset: RigidObject = env.scene[asset_cfg.name]
    spawn_z = get_spawn_root_w(env, asset)[:, 2]
    return asset.data.root_pos_w[:, 2] > spawn_z + float(max_height_above_spawn)


def root_xy_drift_from_spawn(
    env: "ManagerBasedRLEnv",
    max_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when horizontal drift from reset spawn exceeds ``max_distance``."""
    asset: RigidObject = env.scene[asset_cfg.name]
    spawn_xy = get_spawn_root_w(env, asset)[:, :2]
    drift_xy = asset.data.root_pos_w[:, :2] - spawn_xy
    return torch.norm(drift_xy, dim=1) > float(max_distance)
