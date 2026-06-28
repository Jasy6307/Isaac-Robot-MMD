"""Custom observation terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as lab_mdp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from source.my_task.mdp.joint_groups import get_cached_joint_scales
from source.my_task.mdp.root_reference import root_reference_pose_w, root_yaw_error_rad
from source.my_task.motion_reference import get_or_create_motion_buffer, motion_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _policy_obs_corruption_enabled(env: "ManagerBasedRLEnv") -> bool:
    return bool(getattr(env.cfg.observations.policy, "enable_corruption", False))


def _joint_scales_for_obs(
    env: "ManagerBasedRLEnv",
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    cache_key: str,
    joint_noise_scale_by_expr: dict[str, float],
) -> torch.Tensor:
    """Per-joint noise scales aligned with ``asset_cfg`` joint selection."""
    scales_all = get_cached_joint_scales(
        env, asset, cache_key, joint_noise_scale_by_expr, default=1.0
    )
    if asset_cfg.joint_ids != slice(None):
        return scales_all[asset_cfg.joint_ids]
    return scales_all


def joint_pos_rel_group_noise(
    env: "ManagerBasedRLEnv",
    pos_noise: float,
    joint_noise_scale_by_expr: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``joint_pos_rel`` with optional per-joint uniform noise (C1 upper-body scaling)."""
    obs = lab_mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    if pos_noise <= 0.0 or not _policy_obs_corruption_enabled(env):
        return obs
    asset: Articulation = env.scene[asset_cfg.name]
    scales = _joint_scales_for_obs(env, asset, asset_cfg, "obs_pos", joint_noise_scale_by_expr)
    noise = (torch.rand_like(obs) * 2.0 - 1.0) * float(pos_noise) * scales.unsqueeze(0)
    return obs + noise


def joint_vel_rel_group_noise(
    env: "ManagerBasedRLEnv",
    vel_noise: float,
    joint_noise_scale_by_expr: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``joint_vel_rel`` with optional per-joint uniform noise (C1 upper-body scaling)."""
    obs = lab_mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    if vel_noise <= 0.0 or not _policy_obs_corruption_enabled(env):
        return obs
    asset: Articulation = env.scene[asset_cfg.name]
    scales = _joint_scales_for_obs(env, asset, asset_cfg, "obs_vel", joint_noise_scale_by_expr)
    noise = (torch.rand_like(obs) * 2.0 - 1.0) * float(vel_noise) * scales.unsqueeze(0)
    return obs + noise


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
    steps = motion_steps(env)
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
    steps = motion_steps(env)
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
    return buf.motion_phase(motion_steps(env))


def root_yaw_error_sin_cos(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root yaw tracking error as ``[sin(err), cos(err)]``, shape ``[num_envs, 2]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_err = root_yaw_error_rad(
        env,
        asset,
        h5_path=h5_path,
        window_seconds=window_seconds,
        asset_name=asset_cfg.name,
    )
    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)


def root_xy_error(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root XY tracking error ``current - reference`` in world frame, shape ``[num_envs, 2]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    p_ref_w, _ = root_reference_pose_w(
        asset,
        env,
        h5_path=h5_path,
        window_seconds=window_seconds,
        asset_name=asset_cfg.name,
    )
    return asset.data.root_state_w[:, 0:2] - p_ref_w[:, 0:2]


def joint_pos_tracking_error(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Per-step joint tracking error (current - reference), for diagnostics."""
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref_abs = buf.q_ref_abs(motion_steps(env))
    q_cur = asset.data.joint_pos
    err = q_cur - q_ref_abs
    if asset_cfg.joint_ids != slice(None):
        err = err[:, asset_cfg.joint_ids]
    return err
