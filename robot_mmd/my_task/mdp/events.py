"""Custom reset events for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_mmd.my_task.motion_reference import get_or_create_motion_buffer
from robot_mmd.my_task.mdp.spawn_anchor import cache_spawn_root_w, cloner_env_origins

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_root_to_spawn(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Teleport root back to default spawn (cloner origin + init_state pose), zero velocity.

    Isaac Lab's ``articulation.reset()`` only clears actuators / external wrenches; it
    does **not** restore root pose. Without this, terminated envs keep the last flown-away
    root position even though joints are reset.
    """
    if env_ids.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] += cloner_env_origins(env)[env_ids]
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    cache_spawn_root_w(env, asset, env_ids)


def reset_to_motion_start(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    h5_path: str,
    window_seconds: float = 10.0,
    joint_pos_noise: float = 0.05,
    joint_vel_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root + joints to the reference motion's first frame plus small noise."""
    if env_ids.numel() == 0:
        return

    reset_root_to_spawn(env, env_ids, asset_cfg=asset_cfg)

    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    q0 = buf.q_ref_abs_first().to(asset.device)  # [J]
    num_reset = int(env_ids.numel())

    target_q = q0.unsqueeze(0).expand(num_reset, -1).clone()
    if joint_pos_noise > 0.0:
        target_q = target_q + (
            torch.rand_like(target_q) * (2.0 * joint_pos_noise) - joint_pos_noise
        )
    # Clamp to soft joint position limits.
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    target_q = torch.clamp(target_q, soft_limits[..., 0], soft_limits[..., 1])

    target_qd = torch.zeros_like(target_q)
    if joint_vel_noise > 0.0:
        target_qd = (
            torch.rand_like(target_qd) * (2.0 * joint_vel_noise) - joint_vel_noise
        )

    asset.write_joint_state_to_sim(target_q, target_qd, env_ids=env_ids)
