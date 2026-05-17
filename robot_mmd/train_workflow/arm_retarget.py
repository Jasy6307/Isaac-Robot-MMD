"""
G1 肩部 3DOF 重定向（MMD -> G1）。

输入: MMD 局部四元数 q_肩(xyzw)、q_腕(xyzw)（均为相对父骨的本地旋转）
输出: G1 (shoulder_pitch, shoulder_roll, shoulder_yaw) 三个关节角增量（弧度）

数学推导
========
G1 URDF 肩部关节链（左右相同轴序）：
    pitch_joint(axis=Y)  →  roll_joint(axis=X)  →  yaw_joint(axis=Z)  →  elbow
令 p,r,y 分别为三个关节角度，则上臂相对 torso 的旋转矩阵为 intrinsic YXZ：
    R_g1 = Ry(p) · Rx(r) · Rz(y)

展开后可唯一反解（|sin r| < 1 时无万向锁）：
    r     = asin( -R[1,2] )
    p     = atan2( R[0,2],  R[2,2] )
    y     = atan2( R[1,0],  R[1,1] )

MMD 坐标系
==========
MMD 骨骼四元数存储在父骨局部系中：
    q_combined = q_肩 * q_腕       # 上臂相对「上半身2」的整体旋转

将其转为旋转矩阵 R_mmd，再通过基变换 B 换到 G1 torso 系：
    R_g1 = B · R_mmd · B^T

基变换 B（固定部分 _B_FIXED）
==============================
MMD 模型空间: X=右(character right), Y=上, Z=后(away from screen)
  注: VMD 四元数以 PMX 局部系存储，但 body/上半身2 局部系在 bind pose 下
  与世界系的差异仅有 Y-up 约定一致，X/Z 朝向与模型正面方向有关。

G1 torso_link 系(URDF, bind pose): X=前, Y=左, Z=上

默认基变换（可通过 set_tune_axes_deg 在运行时叠加额外旋转）：
    B_fixed = [[0, -1, 0],    # G1 X(前)  = MMD -Z(屏幕侧)
               [1,  0, 0],    # G1 Y(左)  = MMD +X(character right, 但对称后同左)
               [0,  0, 1]]    # G1 Z(上)  = MMD +Y(上)

tune 层: R_tune = Rz(rz)·Ry(ry)·Rx(rx)（度），最终 B = R_tune · B_fixed。
遇到方向整体偏转时，先试 rx/ry/rz 各 ±90° 整数倍，再细调。

per-joint scale
===============
g1_joint_axis_map_raw.py 中 scale 默认 +1.0，遇到某轴方向反了在 UI 里 Flip 即可。
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# ──────────────────────────────────────────────────────────────
# 固定基变换
# ──────────────────────────────────────────────────────────────
# G1 torso row vectors expressed in MMD arm-root local space.
#   Row 0 (G1 +X = forward)  <- MMD -Z  → [0, 0, -1]
#   Row 1 (G1 +Y = left)     <- MMD +X  → [1, 0,  0]
#   Row 2 (G1 +Z = up)       <- MMD +Y  → [0, 1,  0]
_B_FIXED = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# ──────────────────────────────────────────────────────────────
# 运行时可调 tune 参数（度）—— 供 UI 实时修改
# ──────────────────────────────────────────────────────────────
# R_tune = Rz(rz) · Ry(ry) · Rx(rx)
#
# 默认值 ±30° Rx 的来源（两个叠加）：
#   1. G1 URDF 肩部安装倾角：left_shoulder_pitch_joint origin rpy[0] ≈ +16°，
#      与 roll_joint rpy[0] ≈ −16° 不完全对消，在 torso→pitch 坐标变换中留下净偏转。
#   2. MMD「腕」骨绑定姿态倾角：T-pose 下腕骨本地"上"方向相对 MMD 世界坐标系
#      有模型特定的小角度偏斜（约 14°）。
#   两项合计实测约 30°；左侧 −30°、右侧 +30° 是因为两侧 roll 轴在 torso 系里
#   方向相反（+Y 展左、−Y 展右），Rx 修正效果因此互为反号。
#
# 如换了 MMD 模型仍有偏差，可在 UI「Shoulder Retarget Tune」面板微调 Rx，
# 调好后把新值写回下面两行即可作为新默认。
_DEFAULT_LEFT_TUNE: tuple[float, float, float] = (-30.0, 0.0, 0.0)
_DEFAULT_RIGHT_TUNE: tuple[float, float, float] = (30.0, 0.0, 0.0)

_left_tune_deg: list[float] = list(_DEFAULT_LEFT_TUNE)
_right_tune_deg: list[float] = list(_DEFAULT_RIGHT_TUNE)

# 缓存 basis（None = 需重算）
_basis_cache: dict[str, np.ndarray | None] = {"left": None, "right": None}


def get_tune_axes_deg(side: str) -> tuple[float, float, float]:
    """返回 (rx, ry, rz) 度数调整值，side='left'|'right'。"""
    t = _left_tune_deg if side == "left" else _right_tune_deg
    return (t[0], t[1], t[2])


def set_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    """设置 tune 参数（度），自动使 basis 缓存失效。"""
    t = _left_tune_deg if side == "left" else _right_tune_deg
    t[0], t[1], t[2] = float(rx), float(ry), float(rz)
    _basis_cache[side] = None


def reset_tune_axes(side: str | None = None) -> None:
    """重置 tune 参数为默认值（L:−30°/R:+30° Rx），side=None 时重置两侧。"""
    for s in (["left", "right"] if side is None else [side]):
        default = _DEFAULT_LEFT_TUNE if s == "left" else _DEFAULT_RIGHT_TUNE
        t = _left_tune_deg if s == "left" else _right_tune_deg
        t[0], t[1], t[2] = default
        _basis_cache[s] = None


# ──────────────────────────────────────────────────────────────
# 内部数学工具
# ──────────────────────────────────────────────────────────────

def _make_tune_mat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
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


def _get_basis(side: str) -> np.ndarray:
    cached = _basis_cache.get(side)
    if cached is not None:
        return cached
    t = _left_tune_deg if side == "left" else _right_tune_deg
    B = _make_tune_mat(*t) @ _B_FIXED
    _basis_cache[side] = B
    return B


def _quat_to_mat3(q_xyzw: tuple[float, float, float, float]) -> np.ndarray:
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


def _quat_mul_xyzw(
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


def _decompose_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """从 R = Ry(p)·Rx(r)·Rz(y) 反解 (pitch, roll, yaw)（弧度）。"""
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


# ──────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────

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
        q = _quat_mul_xyzw(q_shoulder_xyzw, q_arm_xyzw)

    qx, qy, qz, qw = q
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-24:
        return 0.0, 0.0, 0.0
    n = math.sqrt(n2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    R_mmd = _quat_to_mat3((qx, qy, qz, qw))
    B = _get_basis(side)
    R_g1 = B @ R_mmd @ B.T
    return _decompose_yxz(R_g1)


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


# ──────────────────────────────────────────────────────────────
# csv_motion_loader 内部用的查找表
# ──────────────────────────────────────────────────────────────

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
