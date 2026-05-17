"""
固定轴（外旋 / static / extrinsic）欧拉角：旋转矩阵 -> 三个角度。

实现摘自 Matthew Brett 的 transforms3d（BSD 许可证），仅保留 ``mat2euler`` 中非重复轴分支对
``syxz``、``szxy`` 两类的支持，供 G1 肩/腰与 URDF 关节链顺序对齐。

- 肩（torso 外展顺序 pitch Y → roll X → yaw Z）对应乘积 R ≈ Rz * Rx * Ry，
  使用 ``syxz``：返回 (绕固定 Y 角, 绕固定 X 角, 绕固定 Z 角) = (pitch_y, roll_x, yaw_z)。
- 腰（pelvis 顺序 yaw Z → roll X → pitch Y）对应乘积 R ≈ Ry * Rx * Rz，
  使用 ``szxy``：返回 (绕固定 Z, 绕固定 X, 绕固定 Y) = (yaw_z, roll_x, pitch_y)。

再按 X/Y/Z 物理轴重排成 (θx, θy, θz)，与 ``csv_motion_loader._AXIS_INDEX_TO_VEC`` 的 0/1/2 一致。
"""
from __future__ import annotations

import math

import numpy as np

# transforms3d euler.py
_NEXT_AXIS = [1, 2, 0, 1]

# 仅用到的两种约定：(first_axis, parity, repetition, frame)
_AXES_META: dict[str, tuple[int, int, int, int]] = {
    "syxz": (1, 1, 0, 0),  # shoulder-like: Y then X then Z (static/extrinsic)
    "szxy": (2, 0, 0, 0),  # waist-like: Z then X then Y
}

_EPS4 = float(np.finfo(float).eps * 4.0)


def _mat2euler_extrinsic(mat: np.ndarray, axes_key: str) -> tuple[float, float, float]:
    firstaxis, parity, repetition, frame = _AXES_META[axes_key]
    assert repetition == 0 and frame == 0

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(mat, dtype=np.float64)[:3, :3]

    cy = math.sqrt(float(M[i, i] * M[i, i] + M[j, i] * M[j, i]))
    if cy > _EPS4:
        ax = math.atan2(float(M[k, j]), float(M[k, k]))
        ay = math.atan2(float(-M[k, i]), cy)
        az = math.atan2(float(M[j, i]), float(M[i, i]))
    else:
        ax = math.atan2(float(-M[j, k]), float(M[j, j]))
        ay = math.atan2(float(-M[k, i]), cy)
        az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az

    return ax, ay, az


def quaternion_xyzw_to_mat33_xyzw(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    """Hamilton quat xyzw（与 scipy Rotation.from_quat 约定一致），旋转 Active: v' = R @ v."""
    qx, qy, qz, qw = quat_xyzw
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = float(qx / n), float(qy / n), float(qz / n), float(qw / n)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def euler_xyz_rad_shoulder_extrinsic(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """
    G1 肩链外旋等价 R ≈ Rz*Rx*Ry：``syxz`` 三分量 → 按物理 X/Y/Z 排成 (θx, θy, θz)。
    """
    r = quaternion_xyzw_to_mat33_xyzw(quat_xyzw)
    py_fix, rx_fix, yz_fix = _mat2euler_extrinsic(r, "syxz")
    # syxz triple = (绕固定 Y, 固定 X, 固定 Z)，映射到 θx、θy、θz：
    theta_x_roll = rx_fix
    theta_y_pitch = py_fix
    theta_z_yaw = yz_fix
    return (theta_x_roll, theta_y_pitch, theta_z_yaw)


def euler_xyz_rad_waist_extrinsic(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """
    G1 腰链外旋等价 R ≈ Ry*Rx*Rz：``szxy`` 三分量 → 按物理 X/Y/Z 排成 (θx, θy, θz)。
    """
    r = quaternion_xyzw_to_mat33_xyzw(quat_xyzw)
    yz_fix, rx_fix, py_fix = _mat2euler_extrinsic(r, "szxy")
    # triple = (绕固定 Z, 固定 X, 固定 Y)
    theta_x_roll = rx_fix
    theta_y_pitch = py_fix
    theta_z_yaw = yz_fix
    return (theta_x_roll, theta_y_pitch, theta_z_yaw)
