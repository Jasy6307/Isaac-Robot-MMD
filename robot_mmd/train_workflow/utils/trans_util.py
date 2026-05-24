# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""标量四元数与 MMD→仿真轴向的旋转工具。

约定：四元数均为 Isaac ``root_state_w`` 使用的 (w, x, y, z) 顺序。
"""

from __future__ import annotations

import math
from typing import Any


def coerce_quat(q: Any, fallback_wxyz4: list[float]) -> list[float]:
    """入参为 wxyz；若无效则使用 Isaac root_state 切片 fallback（wxyz）。"""
    if q is None:
        return [float(v) for v in fallback_wxyz4]
    try:
        out = [float(v) for v in q]
        if len(out) != 4:
            raise ValueError
        return out
    except Exception:
        return [float(v) for v in fallback_wxyz4]


def root_quat_from_state_row(state_row: Any) -> list[float]:
    """单环境 root_state 一行：索引 3:7 为 wxyz。"""
    return [float(state_row[i].item()) for i in (3, 4, 5, 6)]


def quat_normalize(q: list[float]) -> list[float]:
    """归一化 wxyz 四元数。"""
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def quat_mul(q1: list[float], q2: list[float]) -> list[float]:
    """四元数乘法 q1 * q2（wxyz）。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_inv(q: list[float]) -> list[float]:
    """单位四元数逆（wxyz）。"""
    qn = quat_normalize(q)
    return [qn[0], -qn[1], -qn[2], -qn[3]]


def remap_root_csv_euler_xyz(
    roll: float,
    pitch: float,
    yaw: float,
    axis_idx_out: tuple[int, int, int],
    scale_out: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Per output row (X/Y/Z physical-angle triple): pick source component then scale."""
    src = (float(roll), float(pitch), float(yaw))
    out: list[float] = []
    for i in range(3):
        si = max(0, min(2, int(axis_idx_out[i])))
        out.append(src[si] * float(scale_out[i]))
    return out[0], out[1], out[2]


def quat_from_waist_extrinsic_xyz(theta_x: float, theta_y: float, theta_z: float) -> list[float]:
    """Physical-axis (X, Y, Z) angles -> quat(wxyz) using waist-style extrinsic chain.

    Waist chain uses fixed-axis order Z -> X -> Y, equivalent matrix product:
    ``R = Ry(theta_y) * Rx(theta_x) * Rz(theta_z)``.
    """
    cx, sx = math.cos(theta_x), math.sin(theta_x)
    cy, sy = math.cos(theta_y), math.sin(theta_y)
    cz, sz = math.cos(theta_z), math.sin(theta_z)
    r_x = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    r_y = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    r_z = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]
    r = _mat3_mul(r_y, _mat3_mul(r_x, r_z))
    return rotmat_to_quat(r)


def _mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat3_transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def rotate_vec_by_quat_wxyz(q_wxyz: list[float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """用单位四元数 (wxyz) 旋转三维向量（与 ``quat_to_rotmat`` 同一右手约定）。"""
    r = quat_to_rotmat(quat_normalize(q_wxyz))
    x, y, z = v
    return (
        r[0][0] * x + r[0][1] * y + r[0][2] * z,
        r[1][0] * x + r[1][1] * y + r[1][2] * z,
        r[2][0] * x + r[2][1] * y + r[2][2] * z,
    )


def quat_to_rotmat(q: list[float]) -> list[list[float]]:
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def rotmat_to_quat(r: list[list[float]]) -> list[float]:
    """旋转矩阵 -> 单位四元数 wxyz（Shepperd）。"""
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
    return quat_normalize([qw, qx, qy, qz])


def mmd_quat_to_world(q_mmd_wxyz: list[float]) -> list[float]:
    """MMD 四元数(wxyz) -> 仿真系(wxyz)（与平移 X->X, Z->Y, Y->Z 一致：R_w = B R_m B^T）。"""
    b = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    r_m = quat_to_rotmat(q_mmd_wxyz)
    r_w = _mat3_mul(b, _mat3_mul(r_m, _mat3_transpose(b)))
    return rotmat_to_quat(r_w)


def mmd_root_offset_quat_to_world(q_mmd_wxyz: list[float]) -> list[float]:
    """センター/グルーブ根旋转：轴向变换后再取逆，使仿真里俯仰与 MMD 一致（避免俯身变仰身）。

    平移已用 B 与 + 号对齐；根姿态若仅用 ``mmd_quat_to_world`` 与 MMD 视觉差一个「前后弯」符号，
    对相似变换后的旋转取逆即可与身体骨骼链的弯腰方向一致。
    """
    return quat_inv(mmd_quat_to_world(q_mmd_wxyz))
