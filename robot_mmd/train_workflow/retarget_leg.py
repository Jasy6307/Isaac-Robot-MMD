"""
G1 下肢重定向（MMD -> G1）。

- Hip 3DOF:  pitch(Y) -> roll(X) -> yaw(Z)   => Ry(p) * Rx(r) * Rz(y)
- Ankle 2DOF: pitch(Y) -> roll(X)             => Ry(p) * Rx(r)

公用基变换与分解见 ``retarget_basis``；腿部 tune 与肩独立（默认全 0）。
"""
from __future__ import annotations

import math

from robot_mmd.train_workflow.retarget_basis import (
    decompose_rotmat_yx,
    decompose_rotmat_yxz,
    get_tune_axes_deg as _get_tune_axes_deg_ns,
    normalize_quat_xyzw_short_arc,
    quat_xyzw_to_mat3,
    reset_tune_axes as _reset_tune_axes_ns,
    rotmat_mmd_to_g1,
    set_tune_axes_deg as _set_tune_axes_deg_ns,
)

_NS_LEG = "leg"


def get_leg_tune_axes_deg(side: str) -> tuple[float, float, float]:
    return _get_tune_axes_deg_ns(_NS_LEG, side)


def set_leg_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    _set_tune_axes_deg_ns(_NS_LEG, side, rx, ry, rz)


def reset_leg_tune_axes(side: str | None = None) -> None:
    _reset_tune_axes_ns(_NS_LEG, side)


def compute_hip_angles(side: str, q_leg_xyzw: tuple[float, float, float, float] | None) -> tuple[float, float, float]:
    if q_leg_xyzw is None:
        return (0.0, 0.0, 0.0)
    q = normalize_quat_xyzw_short_arc(q_leg_xyzw)
    R_mmd = quat_xyzw_to_mat3(q)
    R_g1 = rotmat_mmd_to_g1(_NS_LEG, side, R_mmd)
    return decompose_rotmat_yxz(R_g1)


def compute_ankle_angles(side: str, q_ank_xyzw: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if q_ank_xyzw is None:
        return (0.0, 0.0)
    q = normalize_quat_xyzw_short_arc(q_ank_xyzw)
    R_mmd = quat_xyzw_to_mat3(q)
    R_g1 = rotmat_mmd_to_g1(_NS_LEG, side, R_mmd)
    return decompose_rotmat_yx(R_g1)


def leg_debug_info(frame_data_raw: dict[str, dict], read_bone_quat_fn) -> dict[str, str]:
    out: dict[str, str] = {}
    if not frame_data_raw:
        return out
    for side, leg_bone, ank_bone, hk, ak in (
        ("left", "左足", "左足首", "__leg_left_hip_raw", "__leg_left_ank_raw"),
        ("right", "右足", "右足首", "__leg_right_hip_raw", "__leg_right_ank_raw"),
    ):
        q_leg = read_bone_quat_fn(frame_data_raw, leg_bone)
        q_ank = read_bone_quat_fn(frame_data_raw, ank_bone)
        hp, hr, hy = compute_hip_angles(side, q_leg)
        ap, ar = compute_ankle_angles(side, q_ank)
        out[hk] = f"P:{math.degrees(hp):+.1f}° R:{math.degrees(hr):+.1f}° Y:{math.degrees(hy):+.1f}°"
        out[ak] = f"P:{math.degrees(ap):+.1f}° R:{math.degrees(ar):+.1f}°"
    return out


HIP_JOINT_TO_AXIS_INDEX: dict[str, int] = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "right_hip_pitch_joint": 0,
    "right_hip_roll_joint": 1,
    "right_hip_yaw_joint": 2,
}

HIP_JOINT_TO_SIDE_BONE: dict[str, tuple[str, str]] = {
    "left_hip_pitch_joint": ("left", "左足"),
    "left_hip_roll_joint": ("left", "左足"),
    "left_hip_yaw_joint": ("left", "左足"),
    "right_hip_pitch_joint": ("right", "右足"),
    "right_hip_roll_joint": ("right", "右足"),
    "right_hip_yaw_joint": ("right", "右足"),
}

ANKLE_JOINT_TO_AXIS_INDEX: dict[str, int] = {
    "left_ankle_pitch_joint": 0,
    "left_ankle_roll_joint": 1,
    "right_ankle_pitch_joint": 0,
    "right_ankle_roll_joint": 1,
}

ANKLE_JOINT_TO_SIDE_BONE: dict[str, tuple[str, str]] = {
    "left_ankle_pitch_joint": ("left", "左足首"),
    "left_ankle_roll_joint": ("left", "左足首"),
    "right_ankle_pitch_joint": ("right", "右足首"),
    "right_ankle_roll_joint": ("right", "右足首"),
}
