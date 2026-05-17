"""
G1 肩部 3DOF 重定向（MMD -> G1）。

输入: MMD 局部四元数 q_肩(xyzw)、q_腕(xyzw)（均为相对父骨的本地旋转）
输出: G1 (shoulder_pitch, shoulder_roll, shoulder_yaw) 三个关节角增量（弧度）

固定基变换、tune 缓存与欧拉反解实现见 ``retarget_basis``；本模块仅持有 shoulder 语义与映射表。
"""
from __future__ import annotations

import math

from robot_mmd.train_workflow.retarget_basis import (
    decompose_rotmat_yxz,
    get_tune_axes_deg as _get_tune_axes_deg_ns,
    normalize_quat_xyzw_short_arc,
    quat_mul_xyzw,
    quat_xyzw_to_mat3,
    reset_tune_axes as _reset_tune_axes_ns,
    rotmat_mmd_to_g1,
    set_tune_axes_deg as _set_tune_axes_deg_ns,
)

_NS_ARM = "arm"


def get_tune_axes_deg(side: str) -> tuple[float, float, float]:
    """返回 (rx, ry, rz) 度数调整值，side='left'|'right'。"""
    return _get_tune_axes_deg_ns(_NS_ARM, side)


def set_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    """设置 tune 参数（度），自动使 basis 缓存失效。"""
    _set_tune_axes_deg_ns(_NS_ARM, side, rx, ry, rz)


def reset_tune_axes(side: str | None = None) -> None:
    """重置 tune 参数为默认值（L:−30°/R:+30° Rx），side=None 时重置两侧。"""
    _reset_tune_axes_ns(_NS_ARM, side)


def compute_shoulder_angles(
    side: str,
    q_shoulder_xyzw: tuple[float, float, float, float] | None,
    q_arm_xyzw: tuple[float, float, float, float] | None,
) -> tuple[float, float, float]:
    """
    返回 G1 (shoulder_pitch, shoulder_roll, shoulder_yaw) 关节角增量（弧度）。
    side: 'left' | 'right'。任一输入 None 时视作单位四元数。
    """
    if q_shoulder_xyzw is None and q_arm_xyzw is None:
        return 0.0, 0.0, 0.0
    if q_shoulder_xyzw is None:
        q: tuple[float, float, float, float] = q_arm_xyzw  # type: ignore[assignment]
    elif q_arm_xyzw is None:
        q = q_shoulder_xyzw
    else:
        q = quat_mul_xyzw(q_shoulder_xyzw, q_arm_xyzw)

    qn = normalize_quat_xyzw_short_arc(q)
    R_mmd = quat_xyzw_to_mat3(qn)
    R_g1 = rotmat_mmd_to_g1(_NS_ARM, side, R_mmd)
    return decompose_rotmat_yxz(R_g1)


def shoulder_debug_info(
    frame_data_raw: dict[str, dict],
    read_bone_quat_fn,  # callable(frame_data, bone) -> xyzw | None
) -> dict[str, str]:
    """
    返回供 UI 显示的肩部调试信息字典。
    键格式: '__sho_left_raw' / '__sho_right_raw'
    值: 'P:±XX.X° R:±XX.X° Y:±XX.X°'（计算出的 raw pitch/roll/yaw，未施加 scale）
    """
    out: dict[str, str] = {}
    if not frame_data_raw:
        return out
    for side, sho_bone, arm_bone, key in (
        ("left", "左肩", "左腕", "__sho_left_raw"),
        ("right", "右肩", "右腕", "__sho_right_raw"),
    ):
        q_sho = read_bone_quat_fn(frame_data_raw, sho_bone)
        q_arm = read_bone_quat_fn(frame_data_raw, arm_bone)
        if q_sho is None and q_arm is None:
            out[key] = "no bone data"
            continue
        p, r, y = compute_shoulder_angles(side, q_sho, q_arm)
        out[key] = f"P:{math.degrees(p):+.1f}° R:{math.degrees(r):+.1f}° Y:{math.degrees(y):+.1f}°"
    return out


SHOULDER_JOINT_TO_AXIS_INDEX: dict[str, int] = {
    "left_shoulder_pitch_joint": 0,
    "left_shoulder_roll_joint": 1,
    "left_shoulder_yaw_joint": 2,
    "right_shoulder_pitch_joint": 0,
    "right_shoulder_roll_joint": 1,
    "right_shoulder_yaw_joint": 2,
}

SHOULDER_JOINT_TO_SIDE_BONES: dict[str, tuple[str, str, str]] = {
    "left_shoulder_pitch_joint": ("left", "左肩", "左腕"),
    "left_shoulder_roll_joint": ("left", "左肩", "左腕"),
    "left_shoulder_yaw_joint": ("left", "左肩", "左腕"),
    "right_shoulder_pitch_joint": ("right", "右肩", "右腕"),
    "right_shoulder_roll_joint": ("right", "右肩", "右腕"),
    "right_shoulder_yaw_joint": ("right", "右肩", "右腕"),
}
