"""Custom reward terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

from source.my_task.mdp.root_reference import root_yaw_error_rad
from source.my_task.motion_reference import get_or_create_motion_buffer, motion_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


def joint_pos_tracking_exp(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    sigma: float = 0.25,
    joint_weight_by_expr: dict[str, float] | None = None,
    joint_weight_default: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp-kernel joint position tracking reward.

    ``r = exp( - mean_squared_error / sigma^2 )``
    Where the error is computed in absolute joint-angle space (rad).
    """
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]
    q_ref_abs = buf.q_ref_abs(motion_steps(env))
    q_cur = asset.data.joint_pos
    if asset_cfg.joint_ids != slice(None):
        q_cur = q_cur[:, asset_cfg.joint_ids]
        q_ref_abs = q_ref_abs[:, asset_cfg.joint_ids]
    err_sq = torch.square(q_cur - q_ref_abs)
    if joint_weight_by_expr:
        # Build per-joint weights once and cache on env for fast reward calls.
        cache = getattr(env, "_g1_joint_tracking_weight_cache", None)
        if cache is None:
            cache = {}
            setattr(env, "_g1_joint_tracking_weight_cache", cache)
        cache_key = (
            asset_cfg.name,
            tuple(sorted((str(k), float(v)) for k, v in joint_weight_by_expr.items())),
            float(joint_weight_default),
        )
        weights_all: torch.Tensor | None = cache.get(cache_key)
        if weights_all is None or weights_all.device != asset.device:
            weights_all = torch.full(
                (asset.num_joints,),
                float(joint_weight_default),
                device=asset.device,
                dtype=torch.float32,
            )
            for expr, weight in joint_weight_by_expr.items():
                joint_ids, _ = asset.find_joints(expr)
                if len(joint_ids) == 0:
                    continue
                weights_all[joint_ids] = float(weight)
            cache[cache_key] = weights_all

        if asset_cfg.joint_ids != slice(None):
            weights = weights_all[asset_cfg.joint_ids]
        else:
            weights = weights_all
        weights = weights.clamp_min(0.0)
        denom = torch.sum(weights).clamp_min(1e-8)
        mse = torch.sum(err_sq * weights.unsqueeze(0), dim=1) / denom
    else:
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
    q_ref_abs = buf.q_ref_abs(motion_steps(env))
    q_cur = asset.data.joint_pos
    if asset_cfg.joint_ids != slice(None):
        q_cur = q_cur[:, asset_cfg.joint_ids]
        q_ref_abs = q_ref_abs[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(q_cur - q_ref_abs), dim=1)


def root_yaw_tracking_exp(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    sigma: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp-kernel reward on root yaw alignment (world-frame) against H5 root reference.

    Reference root orientation is reconstructed as:
    ``q_ref_world = q_delta_h5 * q_anchor_default`` where all quaternions are in wxyz.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    asset: Articulation = env.scene[asset_cfg.name]
    yaw_err = root_yaw_error_rad(
        env,
        asset,
        h5_path=h5_path,
        window_seconds=window_seconds,
        asset_name=asset_cfg.name,
    )
    return torch.exp(-torch.square(yaw_err) / (float(sigma) ** 2))


def root_xy_tracking_exp(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    sigma: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp-kernel reward on root XY alignment (world-frame) against H5 root reference.

    Reference root position is reconstructed as:
    ``p_ref_world = p_anchor_default + env_origin + p_delta_h5``.
    Z is intentionally ignored; only XY is tracked.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    p_cur_xy = asset.data.root_state_w[:, 0:2]
    p_anchor_xy = asset.data.default_root_state[:, 0:2]
    env_origin_xy = _cloner_env_origins(env)[:, 0:2]
    p_delta_xy = buf.root_pos_delta(motion_steps(env))[:, 0:2]
    p_ref_xy = p_anchor_xy + env_origin_xy + p_delta_xy

    err_sq = torch.square(p_cur_xy - p_ref_xy)
    mse_xy = torch.mean(err_sq, dim=1)
    return torch.exp(-mse_xy / (float(sigma) ** 2))


def root_z_tracking_exp(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float = 10.0,
    sigma: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp-kernel reward on root Z alignment (world-frame) against H5 root reference.

    Reference root position is reconstructed as:
    ``p_ref_world = p_anchor_default + env_origin + p_delta_h5``.
    Only Z is tracked to softly keep body height close to the source motion.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    buf = get_or_create_motion_buffer(env, h5_path, window_seconds, asset_name=asset_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    p_cur_z = asset.data.root_state_w[:, 2]
    p_anchor_z = asset.data.default_root_state[:, 2]
    env_origin_z = _cloner_env_origins(env)[:, 2]
    p_delta_z = buf.root_pos_delta(motion_steps(env))[:, 2]
    p_ref_z = p_anchor_z + env_origin_z + p_delta_z

    err_sq_z = torch.square(p_cur_z - p_ref_z)
    return torch.exp(-err_sq_z / (float(sigma) ** 2))
