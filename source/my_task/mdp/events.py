"""Custom reset events for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from source.my_task.mdp.episode_length import (
    get_runtime_episode_length_seconds,
    sample_episode_target_steps,
    set_episode_target_steps,
)
from source.my_task.mdp.joint_groups import get_cached_joint_scales
from source.my_task.motion_reference import (
    get_or_create_motion_buffer,
    reset_motion_start_steps,
    set_motion_start_steps,
)
from source.train_workflow.utils.motion.start_weight import (
    compute_motion_start_weights_from_h5,
    summarize_result,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ENV_START_TO_END_MODE_ATTR = "_g1_episode_random_start_to_end"
_ENV_STAGE_END_SECONDS_ATTR = "_g1_episode_end_seconds"
_ENV_START_WEIGHT_CACHE_ATTR = "_g1_motion_start_weight_cache"
_ENV_START_WEIGHT_LOGGED_KEYS_ATTR = "_g1_motion_start_weight_logged_keys"


def _cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


def _get_or_build_motion_start_weights(
    env: "ManagerBasedRLEnv",
    *,
    h5_path: str,
    steps: int,
    lookahead_seconds: float,
    top_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Build/cached per-step sampling weights for random motion starts."""
    cache: dict[tuple[str, int, float, float], torch.Tensor] | None = getattr(
        env, _ENV_START_WEIGHT_CACHE_ATTR, None
    )
    if cache is None:
        cache = {}
        setattr(env, _ENV_START_WEIGHT_CACHE_ATTR, cache)

    key = (
        str(h5_path),
        int(steps),
        float(lookahead_seconds),
        float(top_ratio),
    )
    if key in cache:
        w = cache[key]
        if w.device != device:
            w = w.to(device=device)
            cache[key] = w
        return w

    result = compute_motion_start_weights_from_h5(
        h5_path=h5_path,
        target_steps=int(steps),
        lookahead_seconds=float(lookahead_seconds),
        top_ratio=float(top_ratio),
    )
    w = torch.as_tensor(result.weights, dtype=torch.float32, device=device).clamp_min(1.0e-8)
    cache[key] = w

    logged_keys: set[tuple[str, int, float, float]] | None = getattr(
        env, _ENV_START_WEIGHT_LOGGED_KEYS_ATTR, None
    )
    if logged_keys is None:
        logged_keys = set()
        setattr(env, _ENV_START_WEIGHT_LOGGED_KEYS_ATTR, logged_keys)
    if key not in logged_keys:
        print("[INFO] Auto motion-start weighting enabled (runtime reset sampling).")
        print(summarize_result(result))
        logged_keys.add(key)

    return w


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
    reset_root_to_motion_pose: bool = False,
    joint_noise_scale_by_expr: dict[str, float] | None = None,
    random_start: bool = False,
    auto_motion_start_weight: bool = False,
    auto_motion_start_weight_lookahead_seconds: float = 3.0,
    auto_motion_start_weight_top_ratio: float = 0.25,
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
        if auto_motion_start_weight:
            weights = _get_or_build_motion_start_weights(
                env,
                h5_path=buf.h5_path,
                steps=int(buf.num_steps),
                lookahead_seconds=float(auto_motion_start_weight_lookahead_seconds),
                top_ratio=float(auto_motion_start_weight_top_ratio),
                device=asset.device,
            )
            # Sample start indices from per-step weights (weights are in [1, 3], not normalized).
            start_steps = torch.multinomial(
                weights,
                num_samples=int(env_ids.numel()),
                replacement=True,
            ).to(device=asset.device, dtype=torch.long)
        else:
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
    else:
        reset_motion_start_steps(env, env_ids)
        start_steps = torch.zeros((env_ids.numel(),), device=asset.device, dtype=torch.long)

    if reset_root_to_motion_pose or reset_root_to_motion_quat:
        root_pose = asset.data.root_state_w[env_ids, :7].clone()
        env_origin = _cloner_env_origins(env)[env_ids]

        if reset_root_to_motion_pose:
            p_anchor = asset.data.default_root_state[env_ids, 0:3]
            p_delta = buf.root_pos_delta(start_steps).to(asset.device)
            root_pose[:, 0:3] = p_anchor + env_origin + p_delta

        if reset_root_to_motion_quat:
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
        asset.write_root_velocity_to_sim(
            torch.zeros((env_ids.numel(), 6), device=asset.device, dtype=torch.float32),
            env_ids=env_ids,
        )

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
