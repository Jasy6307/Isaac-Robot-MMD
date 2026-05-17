"""
MMD 骨骼局部系 → G1 机体系的固定基变换与公用旋转数学。

供 ``retarget_arm`` / ``retarget_leg`` 共用：两侧各自持有独立的 tune（度）与 basis 缓存，
通过 ``namespace``（``\"arm\"`` | ``\"leg\"``）区分，避免肩、腿互相污染。
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

RetargetNamespace = Literal["arm", "leg"]

# G1 torso row vectors expressed in MMD limb-root local space（与肩/腿注释一致）
B_FIXED_MMD_TO_G1 = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_DEFAULT_TUNE_DEG: dict[str, dict[str, tuple[float, float, float]]] = {
    "arm": {"left": (-30.0, 0.0, 0.0), "right": (30.0, 0.0, 0.0)},
    "leg": {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
}

_tune_deg: dict[str, dict[str, list[float]]] = {
    ns: {side: list(vals[side]) for side in ("left", "right")}
    for ns, vals in _DEFAULT_TUNE_DEG.items()
}
_basis_cache: dict[str, dict[str, np.ndarray | None]] = {
    ns: {"left": None, "right": None} for ns in _DEFAULT_TUNE_DEG
}


def get_tune_axes_deg(namespace: RetargetNamespace, side: str) -> tuple[float, float, float]:
    """返回 (rx, ry, rz) 度数；``side`` 为 ``left`` | ``right``。"""
    t = _tune_deg[namespace][side]
    return (t[0], t[1], t[2])


def set_tune_axes_deg(namespace: RetargetNamespace, side: str, rx: float, ry: float, rz: float) -> None:
    t = _tune_deg[namespace][side]
    t[0], t[1], t[2] = float(rx), float(ry), float(rz)
    _basis_cache[namespace][side] = None


def reset_tune_axes(namespace: RetargetNamespace, side: str | None = None) -> None:
    sides = ["left", "right"] if side is None else [side]
    for s in sides:
        default = _DEFAULT_TUNE_DEG[namespace][s]
        t = _tune_deg[namespace][s]
        t[0], t[1], t[2] = default
        _basis_cache[namespace][s] = None


def make_tune_rotation_mat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """R_tune = Rz(rz)·Ry(ry)·Rx(rx)，输入为度。"""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def get_basis(namespace: RetargetNamespace, side: str) -> np.ndarray:
    cached = _basis_cache[namespace][side]
    if cached is not None:
        return cached
    t = _tune_deg[namespace][side]
    b = make_tune_rotation_mat(t[0], t[1], t[2]) @ B_FIXED_MMD_TO_G1
    _basis_cache[namespace][side] = b
    return b


def quat_xyzw_to_mat3(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    qx, qy, qz, qw = q_xyzw
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-24:
        return np.eye(3, dtype=np.float64)
    n = math.sqrt(n2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def quat_mul_xyzw(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def normalize_quat_xyzw_short_arc(q_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    qx, qy, qz, qw = q_xyzw
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-24:
        return (0.0, 0.0, 0.0, 1.0)
    n = math.sqrt(n2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return (qx, qy, qz, qw)


def decompose_rotmat_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """R = Ry(p)·Rx(r)·Rz(y) -> (pitch, roll, yaw)，弧度。"""
    s_roll = max(-1.0, min(1.0, -float(R[1, 2])))
    if abs(s_roll) > 0.999999:
        roll = math.copysign(math.pi / 2.0, s_roll)
        pitch = math.atan2(-float(R[2, 0]), float(R[0, 0]))
        yaw = 0.0
        return pitch, roll, yaw
    roll = math.asin(s_roll)
    pitch = math.atan2(float(R[0, 2]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[1, 1]))
    return pitch, roll, yaw


def decompose_rotmat_yx(R: np.ndarray) -> tuple[float, float]:
    """R = Ry(p)·Rx(r) -> (pitch, roll)，弧度。"""
    roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
    pitch = math.atan2(float(-R[2, 0]), float(R[0, 0]))
    return pitch, roll


def rotmat_mmd_to_g1(namespace: RetargetNamespace, side: str, R_mmd: np.ndarray) -> np.ndarray:
    """R_g1 = B · R_mmd · B^T。"""
    b = get_basis(namespace, side)
    return b @ R_mmd @ b.T
