"""G1 joint groups and per-group noise / control scales for dance tracking."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation

# Arms (+ hands): frozen in C1 — follow H5 reference, no policy / reset / obs noise.
G1_ARM_JOINT_EXPR: list[str] = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_index_.*",
    ".*_middle_.*",
    ".*_thumb_.*",
]

G1_WAIST_JOINT_EXPR: list[str] = ["waist_.*_joint"]

G1_LEG_JOINT_EXPR: list[str] = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

# Multiplier on C0 base reset noise (0.05 rad) and obs noise in C1.
C1_RESET_NOISE_SCALE_BY_EXPR: dict[str, float] = {
    **{expr: 0.0 for expr in G1_ARM_JOINT_EXPR},
    **{expr: 0.0 for expr in G1_WAIST_JOINT_EXPR},
}

C1_OBS_NOISE_SCALE_BY_EXPR: dict[str, float] = {
    **{expr: 0.0 for expr in G1_ARM_JOINT_EXPR},
    **{expr: 0.0 for expr in G1_WAIST_JOINT_EXPR},
}

C1_JOINT_POS_OBS_NOISE = 0.01
C1_JOINT_VEL_OBS_NOISE = 0.1

_ENV_JOINT_SCALE_CACHE_ATTR = "_g1_dance_joint_noise_scales"


def build_joint_scale_vector(
    asset: Articulation,
    scale_by_expr: dict[str, float],
    *,
    default: float = 1.0,
) -> torch.Tensor:
    """Per-joint scale vector aligned with ``asset.joint_names``, shape ``[num_joints]``."""
    scales = torch.full((asset.num_joints,), float(default), device=asset.device, dtype=torch.float32)
    for expr, scale in scale_by_expr.items():
        joint_ids, _ = asset.find_joints(expr)
        if len(joint_ids) == 0:
            continue
        scales[joint_ids] = float(scale)
    return scales


def get_cached_joint_scales(
    env,
    asset: Articulation,
    cache_key: str,
    scale_by_expr: dict[str, float],
    *,
    default: float = 1.0,
) -> torch.Tensor:
    """Cache per-joint scales on ``env`` keyed by ``cache_key``."""
    cache: dict[str, torch.Tensor] | None = getattr(env, _ENV_JOINT_SCALE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, _ENV_JOINT_SCALE_CACHE_ATTR, cache)
    if cache_key not in cache:
        cache[cache_key] = build_joint_scale_vector(asset, scale_by_expr, default=default)
    return cache[cache_key]


def resolve_joint_ids(asset: Articulation, joint_name_expr: list[str]) -> torch.Tensor:
    """Return sorted unique joint indices matching any expression."""
    joint_ids, _ = asset.find_joints(joint_name_expr, preserve_order=False)
    if isinstance(joint_ids, slice):
        return torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    return torch.as_tensor(sorted(set(joint_ids)), device=asset.device, dtype=torch.long)
