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


_ENV_START_TO_END_MODE_ATTR = "_g1_episode_random_start_to_end"
_ENV_STAGE_END_SECONDS_ATTR = "_g1_episode_end_seconds"
_ENV_START_SAMPLE_LOG_COUNT_ATTR = "_g1_start_sample_log_count"
_ENV_START_SAMPLE_LOG_INTERVAL_ATTR = "_g1_start_sample_log_interval"


def _cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


def _maybe_log_start_steps(
    env: "ManagerBasedRLEnv",
    start_steps: torch.Tensor,
    num_steps: int,
) -> None:
    """Low-frequency logging for reset start-step coverage."""
    if start_steps.numel() == 0 or num_steps <= 0:
        return
    count = int(getattr(env, _ENV_START_SAMPLE_LOG_COUNT_ATTR, 0))
    interval = int(getattr(env, _ENV_START_SAMPLE_LOG_INTERVAL_ATTR, 200))
    setattr(env, _ENV_START_SAMPLE_LOG_COUNT_ATTR, count + 1)
    if interval <= 0 or (count % interval) != 0:
        return
    ss = start_steps.to(dtype=torch.float32)
    tail_threshold = max(0, int(round(0.9 * max(1, num_steps - 1))))
    tail_ratio = float((start_steps >= tail_threshold).to(dtype=torch.float32).mean().item())
    print(
        "[INFO] reset random_start coverage: "
        f"min={int(start_steps.min().item())} "
        f"max={int(start_steps.max().item())} "
        f"mean={float(ss.mean().item()):.1f} "
        f"tail>=90% ratio={tail_ratio:.3f} "
        f"(num_steps={int(num_steps)})"
    )


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
    ``joint_pos_noise * scale`` (e.g. arms/waist 0, legs 1.0).
    """
    if env_ids.numel() == 0:
        return

    reset_root_to_spawn(env, env_ids, asset_cfg=asset_cfg)

    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    start_to_end_mode = bool(getattr(env, _ENV_START_TO_END_MODE_ATTR, False))
    stage_end_seconds = float(
        getattr(env, _ENV_STAGE_END_SECONDS_ATTR, episode_max_seconds if random_episode_length else window_seconds)
    )

    if start_to_end_mode:
        end_steps = max(1, int(round(stage_end_seconds / float(env.step_dt))))
        end_steps = min(end_steps, int(buf.num_steps))
        min_steps_for_start = max(1, int(round(float(episode_min_seconds) / float(env.step_dt))))
        max_start_step = max(0, end_steps - min_steps_for_start)

        start_steps = torch.zeros((env_ids.numel(),), device=asset.device, dtype=torch.long)
        if max_start_step > 0:
            start_steps = torch.randint(
                low=0,
                high=max_start_step + 1,
                size=(env_ids.numel(),),
                device=asset.device,
                dtype=torch.long,
            )

        target_steps = (end_steps - start_steps).clamp_min(1)
        set_episode_target_steps(env, env_ids, target_steps)
        set_motion_start_steps(env, env_ids, start_steps)
    else:
        start_steps: torch.Tensor | None = None

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
    if not start_to_end_mode:
        set_episode_target_steps(env, env_ids, target_steps)

    if start_to_end_mode:
        if start_steps is None:
            start_steps = torch.zeros((env_ids.numel(),), device=asset.device, dtype=torch.long)
    elif random_start:
        # Tail-coverage mode: uniformly sample any start in [0, num_steps-1].
        # Overrun segments are naturally held at the last reference frame via buffer clamping.
        start_steps = torch.randint(
            low=0,
            high=max(1, int(buf.num_steps)),
            size=(env_ids.numel(),),
            device=asset.device,
            dtype=torch.long,
        )
        set_motion_start_steps(env, env_ids, start_steps)
        _maybe_log_start_steps(env, start_steps, int(buf.num_steps))
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
