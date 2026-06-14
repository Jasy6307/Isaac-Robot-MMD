"""G1 29-DOF planar foot IK geometry from ``g1_29dof_mode_15_brainco_hand.urdf``.

Collapses hip_pitch -> knee as thigh and knee -> ankle_roll_link as shin.
Hip pivot is the left/right ``*_hip_pitch_joint`` origin in pelvis frame.
"""

from __future__ import annotations

import math

# pelvis -> left_hip_pitch_joint origin
G1_FOOT_IK_HIP_OFFSET_Y_M = 0.064452
G1_FOOT_IK_HIP_OFFSET_Z_M = -0.1027

# |hip_pitch -> knee_joint| along hip_roll/yaw/knee chain (T-pose vector sum)
G1_FOOT_IK_THIGH_LENGTH_M = 0.340506

# |knee_joint -> ankle_roll_joint| along ankle_pitch + ankle_roll chain
G1_FOOT_IK_SHIN_LENGTH_M = 0.317568

# Viz-only: ignore tiny exceedances from near-full leg extension / numeric noise.
FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M = 0.015


def g1_foot_ik_max_reach_m(*, max_reach_ratio: float = 0.985) -> float:
    return (G1_FOOT_IK_THIGH_LENGTH_M + G1_FOOT_IK_SHIN_LENGTH_M) * float(max_reach_ratio)


def planar_leg_ik_forward_m(vx: float, *, side: str) -> float:
    """Sagittal forward in the hip pitch plane (both legs negate root +X)."""
    return -float(vx)


def planar_leg_ik_reach_debug(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    hip_offset_y: float,
    hip_offset_z: float,
    thigh_length: float,
    shin_length: float,
    max_reach_ratio: float,
) -> dict[str, float]:
    """Pre/post clamp reach metrics for IK debug logging."""
    side_sign = 1.0 if side == "left" else -1.0
    hx, hy, hz = 0.0, side_sign * float(hip_offset_y), float(hip_offset_z)
    tx, ty, tz = target_local_xyz
    vx, vy, vz = tx - hx, ty - hy, tz - hz
    forward = planar_leg_ik_forward_m(vx, side=side)
    down = float(max(1e-6, -vz))
    d_raw = float(math.sqrt(forward * forward + down * down))
    l1 = max(1e-6, float(thigh_length))
    l2 = max(1e-6, float(shin_length))
    d_max = (l1 + l2) * max(0.2, float(max_reach_ratio))
    d_used = float(max(1e-6, min(d_raw, d_max)))
    return {
        "forward_m": forward,
        "down_m": down,
        "lateral_m": float(vy),
        "d_raw_m": d_raw,
        "d_used_m": d_used,
        "d_max_m": d_max,
    }


def planar_leg_ik_reach_clamped(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    hip_offset_y: float,
    hip_offset_z: float,
    thigh_length: float,
    shin_length: float,
    max_reach_ratio: float,
    margin_m: float = 0.0,
) -> bool:
    """True when planar reach distance exceeds ``d_max`` by more than ``margin_m``."""
    reach = planar_leg_ik_reach_debug(
        target_local_xyz,
        side=side,
        hip_offset_y=hip_offset_y,
        hip_offset_z=hip_offset_z,
        thigh_length=thigh_length,
        shin_length=shin_length,
        max_reach_ratio=max_reach_ratio,
    )
    return float(reach["d_raw_m"]) > float(reach["d_max_m"]) + max(0.0, float(margin_m))
