"""Custom observation terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_mmd.my_task.motion_reference import get_or_create_motion_buffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ref_joint_pos_rel(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference joint deltas (vs runtime default) at the current episode step.

    Shape: ``[num_envs, num_joints_selected]``.
    """
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    steps = env.episode_length_buf
    q_rel = buf.q_ref_rel(steps)
    if asset_cfg.joint_ids != slice(None):
        q_rel = q_rel[:, asset_cfg.joint_ids]
    return q_rel


def ref_joint_pos_rel_next(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    lookahead: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference joint deltas at ``current_step + lookahead`` (clamped)."""
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    steps = env.episode_length_buf
    q_rel = buf.q_ref_rel(steps, offset=int(lookahead))
    if asset_cfg.joint_ids != slice(None):
        q_rel = q_rel[:, asset_cfg.joint_ids]
    return q_rel


def motion_phase(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalized motion phase in [0, 1], shape ``[num_envs, 1]``."""
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    return buf.motion_phase(env.episode_length_buf)


def joint_pos_tracking_error(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-step joint tracking error (current - reference), for diagnostics."""
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref_abs = buf.q_ref_abs(env.episode_length_buf)
    q_cur = asset.data.joint_pos
    err = q_cur - q_ref_abs
    if asset_cfg.joint_ids != slice(None):
        err = err[:, asset_cfg.joint_ids]
    return err
