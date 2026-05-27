"""Custom reset events for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from robot_mmd.my_task.mdp.episode_length import (
    get_runtime_episode_length_seconds,
    sample_episode_target_steps,
    set_episode_target_steps,
)
from robot_mmd.my_task.mdp.joint_groups import get_cached_joint_scales
from robot_mmd.my_task.motion_reference import (
    get_or_create_motion_buffer,
    reset_motion_start_steps,
    set_motion_start_steps,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


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
    root_state[:, 0:3] += _cloner_env_origins(env)[env_ids]
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_to_motion_start(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    h5_path: str,
    window_seconds: float = 10.0,
    joint_pos_noise: float = 0.05,
    joint_vel_noise: float = 0.0,
    reset_root_to_motion_quat: bool = False,
    joint_noise_scale_by_expr: dict[str, float] | None = None,
    random_start: bool = False,
    segment_seconds: float | None = None,
    random_episode_length: bool = False,
    episode_min_seconds: float = 2.0,
    episode_max_seconds: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset root + joints to the reference motion's first frame plus optional noise.

    When ``joint_noise_scale_by_expr`` is set, each joint's reset noise is
    ``joint_pos_noise * scale`` (e.g. arms 0, waist 0.2, legs 1.0).
    """
    if env_ids.numel() == 0:
        return

    reset_root_to_spawn(env, env_ids, asset_cfg=asset_cfg)

    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    if random_episode_length:
        min_s, max_s = get_runtime_episode_length_seconds(env, episode_min_seconds, episode_max_seconds)
        target_steps = sample_episode_target_steps(env, env_ids, min_s, max_s)
    else:
        if segment_seconds is None:
            fixed_steps = int(getattr(env, "max_episode_length", 1))
        else:
            fixed_steps = int(round(float(segment_seconds) / float(env.step_dt)))
        fixed_steps = max(fixed_steps, 1)
        target_steps = torch.full((env_ids.numel(),), fixed_steps, device=asset.device, dtype=torch.long)
    set_episode_target_steps(env, env_ids, target_steps)

    if random_start:
        max_start_each = (int(buf.num_steps) - target_steps).clamp_min(0)
        start_steps = torch.zeros((env_ids.numel(),), device=asset.device, dtype=torch.long)
        has_room = max_start_each > 0
        if has_room.any():
            rand_unit = torch.rand((int(has_room.sum().item()),), device=asset.device)
            start_steps[has_room] = torch.floor(
                rand_unit * (max_start_each[has_room].to(dtype=torch.float32) + 1.0)
            ).to(dtype=torch.long)
        set_motion_start_steps(env, env_ids, start_steps)
    else:
        reset_motion_start_steps(env, env_ids)
        start_steps = torch.zeros((env_ids.numel(),), device=asset.device, dtype=torch.long)

    if reset_root_to_motion_quat:
        root_pose = asset.data.root_state_w[env_ids, :7].clone()
        default_root_quat = asset.data.default_root_state[env_ids, 3:7]
        q_delta0 = buf.root_quat_wxyz(start_steps)
        q_target = torch.stack(
            (
                q_delta0[:, 0] * default_root_quat[:, 0]
                - q_delta0[:, 1] * default_root_quat[:, 1]
                - q_delta0[:, 2] * default_root_quat[:, 2]
                - q_delta0[:, 3] * default_root_quat[:, 3],
                q_delta0[:, 0] * default_root_quat[:, 1]
                + q_delta0[:, 1] * default_root_quat[:, 0]
                + q_delta0[:, 2] * default_root_quat[:, 3]
                - q_delta0[:, 3] * default_root_quat[:, 2],
                q_delta0[:, 0] * default_root_quat[:, 2]
                - q_delta0[:, 1] * default_root_quat[:, 3]
                + q_delta0[:, 2] * default_root_quat[:, 0]
                + q_delta0[:, 3] * default_root_quat[:, 1],
                q_delta0[:, 0] * default_root_quat[:, 3]
                + q_delta0[:, 1] * default_root_quat[:, 2]
                - q_delta0[:, 2] * default_root_quat[:, 1]
                + q_delta0[:, 3] * default_root_quat[:, 0],
            ),
            dim=-1,
        )
        q_target = q_target / torch.linalg.norm(q_target, dim=-1, keepdim=True).clamp_min(1e-8)
        root_pose[:, 3:7] = q_target
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

    q0 = buf.q_ref_abs(start_steps).to(asset.device)  # [N, J]
    target_q = q0.clone()
    if joint_pos_noise > 0.0:
        delta = torch.rand_like(target_q) * (2.0 * joint_pos_noise) - joint_pos_noise
        if joint_noise_scale_by_expr:
            scales = get_cached_joint_scales(
                env,
                asset,
                "reset",
                joint_noise_scale_by_expr,
                default=1.0,
            )
            delta = delta * scales.unsqueeze(0)
        target_q = target_q + delta
    # Clamp to soft joint position limits.
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    target_q = torch.clamp(target_q, soft_limits[..., 0], soft_limits[..., 1])

    target_qd = torch.zeros_like(target_q)
    if joint_vel_noise > 0.0:
        target_qd = (
            torch.rand_like(target_qd) * (2.0 * joint_vel_noise) - joint_vel_noise
        )

    asset.write_joint_state_to_sim(target_q, target_qd, env_ids=env_ids)
