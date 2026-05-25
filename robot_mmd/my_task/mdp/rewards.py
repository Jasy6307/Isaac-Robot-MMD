"""Custom reward terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_mmd.my_task.motion_reference import get_or_create_motion_buffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_tracking_exp(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp-kernel joint position tracking reward.

    ``r = exp( - mean_squared_error / sigma^2 )``
    Where the error is computed in absolute joint-angle space (rad).
    """
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref_abs = buf.q_ref_abs(env.episode_length_buf)
    q_cur = asset.data.joint_pos
    if asset_cfg.joint_ids != slice(None):
        q_cur = q_cur[:, asset_cfg.joint_ids]
        q_ref_abs = q_ref_abs[:, asset_cfg.joint_ids]
    err_sq = torch.square(q_cur - q_ref_abs)
    mse = torch.mean(err_sq, dim=1)
    return torch.exp(-mse / (float(sigma) ** 2))


def joint_pos_tracking_l2(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared L2 joint tracking penalty (sum over joints), shape ``[num_envs]``."""
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref_abs = buf.q_ref_abs(env.episode_length_buf)
    q_cur = asset.data.joint_pos
    if asset_cfg.joint_ids != slice(None):
        q_cur = q_cur[:, asset_cfg.joint_ids]
        q_ref_abs = q_ref_abs[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(q_cur - q_ref_abs), dim=1)
