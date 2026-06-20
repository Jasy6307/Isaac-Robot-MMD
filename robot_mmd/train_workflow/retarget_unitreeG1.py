"""
Unitree G1 MMD → G1 关节重定向（肩 / 腿 / 腰）。

- 肩、髋：MMD 局部四元数经固定基变换 B + 可选 tune，再按 G1 URDF 链反解 (pitch, roll, yaw)。
- 踝：同上，反解 (pitch, roll)。
- 腰：上半身+上半身2 组合，相对下半身抵消后外旋 szxy 分解，再映射为语义 (pitch, roll, yaw)。
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

RetargetNamespace = Literal["arm", "leg", "wrist"]
Side = Literal["left", "right"]
QuatXYZW = tuple[float, float, float, float]

_NS_ARM: RetargetNamespace = "arm"
_NS_LEG: RetargetNamespace = "leg"
_NS_WRIST: RetargetNamespace = "wrist"

# G1 torso row vectors in MMD limb-root local space
B_FIXED_MMD_TO_G1 = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_DEFAULT_TUNE_DEG: dict[RetargetNamespace, dict[str, tuple[float, float, float]]] = {
    "arm": {"left": (-30.0, 0.0, 0.0), "right": (30.0, 0.0, 0.0)},
    "leg": {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
    "wrist": {"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
}

_tune_deg: dict[str, dict[str, list[float]]] = {
    ns: {side: list(vals[side]) for side in ("left", "right")}
    for ns, vals in _DEFAULT_TUNE_DEG.items()
}
_basis_cache: dict[str, dict[str, np.ndarray | None]] = {
    ns: {"left": None, "right": None} for ns in _DEFAULT_TUNE_DEG
}


# ---------------------------------------------------------------------------
# Shared quaternion / rotation-matrix math
# ---------------------------------------------------------------------------


def quat_xyzw_to_mat3(q_xyzw: QuatXYZW) -> np.ndarray:
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


def quat_mul_xyzw(q1: QuatXYZW, q2: QuatXYZW) -> QuatXYZW:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def normalize_quat_xyzw_short_arc(q_xyzw: QuatXYZW) -> QuatXYZW:
    qx, qy, qz, qw = q_xyzw
    n2 = qx * qx + qy * qy + qz * qz + qw * qw
    if n2 < 1e-24:
        return (0.0, 0.0, 0.0, 1.0)
    n = math.sqrt(n2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return (qx, qy, qz, qw)


def quat_conjugate_xyzw(q_xyzw: QuatXYZW) -> QuatXYZW:
    x, y, z, w = normalize_quat_xyzw_short_arc(q_xyzw)
    return (-x, -y, -z, w)


def _combine_bone_quats(
    q_first: QuatXYZW | None,
    q_second: QuatXYZW | None,
) -> QuatXYZW | None:
    if q_first is None and q_second is None:
        return None
    if q_first is None:
        return normalize_quat_xyzw_short_arc(q_second)  # type: ignore[arg-type]
    if q_second is None:
        return normalize_quat_xyzw_short_arc(q_first)
    return normalize_quat_xyzw_short_arc(quat_mul_xyzw(q_first, q_second))


def decompose_rotmat_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """R = Ry(p)·Rx(r)·Rz(y) -> (pitch, roll, yaw)，弧度。"""
    s_roll = max(-1.0, min(1.0, -float(R[1, 2])))
    if abs(s_roll) > 0.999999:
        roll = math.copysign(math.pi / 2.0, s_roll)
        pitch = math.atan2(-float(R[2, 0]), float(R[0, 0]))
        return pitch, roll, 0.0
    roll = math.asin(s_roll)
    pitch = math.atan2(float(R[0, 2]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[1, 1]))
    return pitch, roll, yaw


def decompose_rotmat_yx(R: np.ndarray) -> tuple[float, float]:
    """R = Ry(p)·Rx(r) -> (pitch, roll)，弧度。"""
    roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
    pitch = math.atan2(float(-R[2, 0]), float(R[0, 0]))
    return pitch, roll


def decompose_rotmat_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rx(roll)·Ry(pitch)·Rz(yaw) -> (roll, pitch, yaw)，弧度。

    用于 G1 手腕链（URDF 顺序 roll(X)→pitch(Y)→yaw(Z)）。
    """
    s_pitch = max(-1.0, min(1.0, float(R[0, 2])))
    if abs(s_pitch) > 0.999999:
        pitch = math.copysign(math.pi / 2.0, s_pitch)
        roll = math.atan2(float(R[2, 1]), float(R[1, 1]))
        return roll, pitch, 0.0
    pitch = math.asin(s_pitch)
    roll = math.atan2(-float(R[1, 2]), float(R[2, 2]))
    yaw = math.atan2(-float(R[0, 1]), float(R[0, 0]))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Extrinsic (fixed-axis) Euler decomposition for waist / root chains
# (transforms3d mat2euler, BSD — syxz/szxy subset)
# ---------------------------------------------------------------------------

_NEXT_AXIS = [1, 2, 0, 1]
_AXES_META: dict[str, tuple[int, int, int, int]] = {
    "szxy": (2, 0, 0, 0),  # waist/root: Z then X then Y
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


def euler_xyz_rad_waist_extrinsic(quat_xyzw: QuatXYZW) -> tuple[float, float, float]:
    """G1 腰/根链外旋 R ≈ Ry·Rx·Rz：szxy → 物理轴 (θx roll, θy pitch, θz yaw)。"""
    r = quat_xyzw_to_mat3(quat_xyzw)
    yaw_z, roll_x, pitch_y = _mat2euler_extrinsic(r, "szxy")
    return roll_x, pitch_y, yaw_z


def _make_tune_rotation_mat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """R_tune = Rz(rz)·Ry(ry)·Rx(rx)，输入为度。"""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mz @ my @ mx


def _get_basis(namespace: RetargetNamespace, side: str) -> np.ndarray:
    cached = _basis_cache[namespace][side]
    if cached is not None:
        return cached
    t = _tune_deg[namespace][side]
    b = _make_tune_rotation_mat(t[0], t[1], t[2]) @ B_FIXED_MMD_TO_G1
    _basis_cache[namespace][side] = b
    return b


def _rotmat_mmd_to_g1(namespace: RetargetNamespace, side: str, R_mmd: np.ndarray) -> np.ndarray:
    b = _get_basis(namespace, side)
    return b @ R_mmd @ b.T


def _retarget_yxz(
    namespace: RetargetNamespace,
    side: str,
    q_xyzw: QuatXYZW | None,
) -> tuple[float, float, float]:
    if q_xyzw is None:
        return (0.0, 0.0, 0.0)
    q = normalize_quat_xyzw_short_arc(q_xyzw)
    R_g1 = _rotmat_mmd_to_g1(namespace, side, quat_xyzw_to_mat3(q))
    return decompose_rotmat_yxz(R_g1)


def _retarget_yx(
    namespace: RetargetNamespace,
    side: str,
    q_xyzw: QuatXYZW | None,
) -> tuple[float, float]:
    if q_xyzw is None:
        return (0.0, 0.0)
    q = normalize_quat_xyzw_short_arc(q_xyzw)
    R_g1 = _rotmat_mmd_to_g1(namespace, side, quat_xyzw_to_mat3(q))
    return decompose_rotmat_yx(R_g1)


# ---------------------------------------------------------------------------
# Tune API (arm / leg namespaces)
# ---------------------------------------------------------------------------


def _get_tune_axes_deg_ns(namespace: RetargetNamespace, side: str) -> tuple[float, float, float]:
    t = _tune_deg[namespace][side]
    return (t[0], t[1], t[2])


def _set_tune_axes_deg_ns(
    namespace: RetargetNamespace,
    side: str,
    rx: float,
    ry: float,
    rz: float,
) -> None:
    t = _tune_deg[namespace][side]
    t[0], t[1], t[2] = float(rx), float(ry), float(rz)
    _basis_cache[namespace][side] = None


def _reset_tune_axes_ns(namespace: RetargetNamespace, side: str | None = None) -> None:
    sides = ["left", "right"] if side is None else [side]
    for s in sides:
        default = _DEFAULT_TUNE_DEG[namespace][s]
        t = _tune_deg[namespace][s]
        t[0], t[1], t[2] = default
        _basis_cache[namespace][s] = None


def get_tune_axes_deg(side: str) -> tuple[float, float, float]:
    """Shoulder tune (rx, ry, rz) in degrees."""
    return _get_tune_axes_deg_ns(_NS_ARM, side)


def set_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    _set_tune_axes_deg_ns(_NS_ARM, side, rx, ry, rz)


def reset_tune_axes(side: str | None = None) -> None:
    _reset_tune_axes_ns(_NS_ARM, side)


def get_leg_tune_axes_deg(side: str) -> tuple[float, float, float]:
    return _get_tune_axes_deg_ns(_NS_LEG, side)


def set_leg_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    _set_tune_axes_deg_ns(_NS_LEG, side, rx, ry, rz)


def reset_leg_tune_axes(side: str | None = None) -> None:
    _reset_tune_axes_ns(_NS_LEG, side)


# ---------------------------------------------------------------------------
# Shoulder 3DOF
# ---------------------------------------------------------------------------


def compute_shoulder_angles(
    side: str,
    q_shoulder_xyzw: QuatXYZW | None,
    q_arm_xyzw: QuatXYZW | None,
    q_elbow_xyzw: QuatXYZW | None = None,
) -> tuple[float, float, float]:
    q = _combine_bone_quats(q_shoulder_xyzw, q_arm_xyzw)
    if q is None:
        return (0.0, 0.0, 0.0)
    pitch, roll, yaw = _retarget_yxz(_NS_ARM, side, q)
    if q_elbow_xyzw is not None:
        # Shoulder side-drift correction for near-straight forward reach:
        # MMD upper-arm local twist around Y can leak into shoulder roll after
        # basis transform/decomposition, causing arm direction to drift toward ±Y.
        # Instead of brute-force roll damping, estimate the leaked twist term
        # from q_arm and subtract it from shoulder roll.
        bend_deg = math.degrees(compute_elbow_angle(side, q_elbow_xyzw))
        pitch_abs_deg = abs(math.degrees(pitch))
        if q_arm_xyzw is not None and bend_deg < 20.0 and pitch_abs_deg > 60.0:
            w_ext = max(0.0, min(1.0, (20.0 - bend_deg) / 20.0))
            w_fwd = max(0.0, min(1.0, (pitch_abs_deg - 60.0) / 50.0))
            leak_w = w_ext * w_fwd
            arm_twist_y = _signed_twist_angle_about_axis_xyzw(q_arm_xyzw, (0.0, 1.0, 0.0))
            # Tuned leak gain: enough to pull IRIS straight-reach back to +X,
            # while keeping gokuraku shoulder behavior near original.
            roll -= 0.18 * arm_twist_y * leak_w
    if q_arm_xyzw is not None and q_elbow_xyzw is not None:
        # Re-distribute a portion of upper-arm Z-twist from shoulder roll to yaw.
        # This targets bent-arm side-lift poses (e.g. gokuraku ~435/441) where
        # roll is over-assigned while yaw is under-assigned.
        # Use smooth weights (instead of hard if-gates) to avoid frame-to-frame
        # on/off jumps that look like twitching.
        def _smoothstep(edge0: float, edge1: float, x: float) -> float:
            if edge1 <= edge0:
                return 0.0
            t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        bend_deg = math.degrees(compute_elbow_angle(side, q_elbow_xyzw))
        pitch_abs_deg = abs(math.degrees(pitch))
        yaw_abs_deg = abs(math.degrees(yaw))
        arm_twist_z_deg = math.degrees(_signed_twist_angle_about_axis_xyzw(q_arm_xyzw, (0.0, 0.0, 1.0)))
        w_bend_in = _smoothstep(45.0, 65.0, bend_deg)
        w_bend_out = 1.0 - _smoothstep(120.0, 145.0, bend_deg)
        w_pitch = 1.0 - _smoothstep(40.0, 65.0, pitch_abs_deg)
        w_yaw = 1.0 - _smoothstep(25.0, 45.0, yaw_abs_deg)
        w_twist = _smoothstep(0.0, 30.0, abs(arm_twist_z_deg))
        redist_w = max(0.0, min(1.0, w_bend_in * w_bend_out * w_pitch * w_yaw * w_twist))
        dz = math.radians(0.45 * arm_twist_z_deg * redist_w)
        roll -= dz
        yaw += dz
    return (pitch, roll, yaw)


def shoulder_debug_info(
    frame_data_raw: dict[str, dict],
    read_bone_quat_fn,
) -> dict[str, str]:
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


# ---------------------------------------------------------------------------
# Wrist 3DOF
# ---------------------------------------------------------------------------


def _twist_quat_about_axis_xyzw(q_xyzw: QuatXYZW, axis: tuple[float, float, float]) -> QuatXYZW:
    """提取 q 绕单位轴 axis 的扭转分量（swing-twist 的 twist 部分，xyzw）。"""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = ax / n, ay / n, az / n
    qx, qy, qz, qw = normalize_quat_xyzw_short_arc(q_xyzw)
    dot = ax * qx + ay * qy + az * qz
    return normalize_quat_xyzw_short_arc((ax * dot, ay * dot, az * dot, qw))


def _signed_twist_angle_about_axis_xyzw(q_xyzw: QuatXYZW, axis: tuple[float, float, float]) -> float:
    """Signed twist angle (rad) of q around axis, in [-pi, pi]."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return 0.0
    ax, ay, az = ax / n, ay / n, az / n
    qx, qy, qz, qw = normalize_quat_xyzw_short_arc(q_xyzw)
    dot = ax * qx + ay * qy + az * qz
    tx, ty, tz, tw = (ax * dot, ay * dot, az * dot, qw)
    tn = math.sqrt(tx * tx + ty * ty + tz * tz + tw * tw)
    if tn < 1e-12:
        return 0.0
    tx, ty, tz, tw = tx / tn, ty / tn, tz / tn, tw / tn
    sin_half = math.sqrt(tx * tx + ty * ty + tz * tz)
    angle = 2.0 * math.atan2(sin_half, tw)
    if dot < 0.0:
        angle = -angle
    if angle > math.pi:
        angle -= 2.0 * math.pi
    elif angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# 把肘骨自转(pronation)注入腕时的符号（按模型/侧可调，sim 里看手心翻转方向）。
_WRIST_PRONATION_SIGN: dict[str, float] = {"left": 1.0, "right": 1.0}
_WRIST_ROLL_LIMIT = (-1.972222054, 1.972222054)
_WRIST_PITCH_LIMIT = (-1.614429558, 1.614429558)
_WRIST_YAW_LIMIT = (-1.614429558, 1.614429558)


def get_wrist_pronation_sign(side: str) -> float:
    return float(_WRIST_PRONATION_SIGN[side])


def set_wrist_pronation_sign(side: str, sign: float) -> None:
    _WRIST_PRONATION_SIGN[side] = float(sign)


def _clamp_wrist_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    r = min(max(float(roll), _WRIST_ROLL_LIMIT[0]), _WRIST_ROLL_LIMIT[1])
    p = min(max(float(pitch), _WRIST_PITCH_LIMIT[0]), _WRIST_PITCH_LIMIT[1])
    y = min(max(float(yaw), _WRIST_YAW_LIMIT[0]), _WRIST_YAW_LIMIT[1])
    return r, p, y


def _wrist_basis(side: str) -> np.ndarray:
    """MMD 世界对齐系 → G1 前臂系 的基 B（行向量为 G1 轴在 MMD 系中的坐标）。

    G1 前臂系：X=前臂长轴(=wrist_roll 轴, 指向手), Z≈竖直向上(投影到⊥前臂), Y=Z×X。
    叠加 _NS_WRIST 的 tune 便于运行时微调。
    """
    d = _normalize_vec3(tuple(_elbow_forearm_axis[side]))  # type: ignore[arg-type]
    x_hat = np.array(d, dtype=np.float64)
    # Isaac / USD uses Z-up. Using Y-up here mixes wrist pitch/yaw decomposition.
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(x_hat @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_tmp = up - float(up @ x_hat) * x_hat
    z_hat = z_tmp / float(np.linalg.norm(z_tmp))
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / float(np.linalg.norm(y_hat))
    b0 = np.array([x_hat, y_hat, z_hat], dtype=np.float64)
    t = _tune_deg[_NS_WRIST][side]
    return _make_tune_rotation_mat(t[0], t[1], t[2]) @ b0


def compute_wrist_angles(
    side: str,
    q_wrist_xyzw: QuatXYZW | None,
    q_elbow_xyzw: QuatXYZW | None = None,
) -> tuple[float, float, float]:
    """MMD 手首局部四元数 → G1 腕 (pitch, roll, yaw)。

    G1 手腕链 URDF 顺序为 roll(X)→pitch(Y)→yaw(Z)，故用 **XYZ** 分解（区别于肩的 YXZ），
    且 roll 轴 = 前臂长轴。专用基 ``_wrist_basis`` 把前臂长轴对到 G1 的 X(roll)。

    若给定 ``q_elbow_xyzw``，把肘骨绕前臂长轴的自转(pronation)前乘到手首：
    G1 肘是纯铰链(只弯曲)，MMD 烘焙进 ひじ 的前臂自转转交腕 roll，恢复手心朝向。
    """
    q_in = q_wrist_xyzw if q_wrist_xyzw is not None else (0.0, 0.0, 0.0, 1.0)
    if q_elbow_xyzw is not None:
        d = _normalize_vec3(tuple(_elbow_forearm_axis[side]))  # type: ignore[arg-type]
        q_pron = _twist_quat_about_axis_xyzw(q_elbow_xyzw, d)
        if _WRIST_PRONATION_SIGN[side] < 0.0:
            q_pron = quat_conjugate_xyzw(q_pron)
        # Elbow->wrist pronation transfer confidence:
        # - very low bend (~straight elbow): elbow bone pronation is often noisy,
        #   avoid injecting it to prevent wrist outward-flip artifacts.
        # - very high bend (~90+ deg): swing-twist around forearm axis also gets
        #   unstable, attenuate to avoid over-bent wrist.
        bend_deg = math.degrees(compute_elbow_angle(side, q_elbow_xyzw))
        w_low_bend = max(0.0, min(1.0, (bend_deg - 8.0) / 20.0))
        w_high_bend = max(0.0, min(1.0, (105.0 - bend_deg) / 55.0))
        pron_w = w_low_bend * w_high_bend
        if pron_w < 1.0:
            # Slerp(identity, q_pron, pron_w)
            px, py, pz, pw = q_pron
            if pw < 0.0:
                px, py, pz, pw = -px, -py, -pz, -pw
            vn = math.sqrt(px * px + py * py + pz * pz)
            if vn > 1e-12:
                half = math.atan2(vn, pw)
                nh = half * pron_w
                s = math.sin(nh) / vn
                q_pron = normalize_quat_xyzw_short_arc((px * s, py * s, pz * s, math.cos(nh)))
            else:
                q_pron = (0.0, 0.0, 0.0, 1.0)
        q_in = normalize_quat_xyzw_short_arc(quat_mul_xyzw(q_pron, q_in))
    b = _wrist_basis(side)
    r_g1 = b @ quat_xyzw_to_mat3(normalize_quat_xyzw_short_arc(q_in)) @ b.T
    roll, pitch, yaw = decompose_rotmat_xyz(r_g1)
    # XYZ Euler has an equivalent branch:
    # (r,p,y) and (r+pi, pi-p, y+pi). Pick branch closer to single-axis twist
    # reference from raw wrist local quat to reduce frame-to-frame branch flips.
    if q_wrist_xyzw is not None:
        p_ref = _signed_twist_angle_about_axis_xyzw(q_wrist_xyzw, (1.0, 0.0, 0.0))
        r_ref = _signed_twist_angle_about_axis_xyzw(q_wrist_xyzw, (0.0, 1.0, 0.0))
        y_ref = _signed_twist_angle_about_axis_xyzw(q_wrist_xyzw, (0.0, 0.0, 1.0))
        roll2 = _wrap_pi(roll + math.pi)
        pitch2 = _wrap_pi(math.pi - pitch)
        yaw2 = _wrap_pi(yaw + math.pi)
        d1 = abs(_wrap_pi(roll - r_ref)) + abs(_wrap_pi(pitch - p_ref)) + abs(_wrap_pi(yaw - y_ref))
        d2 = (
            abs(_wrap_pi(roll2 - r_ref))
            + abs(_wrap_pi(pitch2 - p_ref))
            + abs(_wrap_pi(yaw2 - y_ref))
        )
        # Only switch branch near obvious wrap/singularity zones; avoid
        # re-routing normal frames where main branch is already stable.
        near_wrap = (
            abs(math.degrees(roll)) > 85.0
            or abs(math.degrees(pitch)) > 85.0
            or abs(math.degrees(yaw)) > 85.0
        )
        if near_wrap and d2 + 1e-6 < d1:
            roll, pitch, yaw = roll2, pitch2, yaw2
    roll, pitch, yaw = _clamp_wrist_xyz(roll, pitch, yaw)
    return (pitch, roll, yaw)


def get_wrist_tune_axes_deg(side: str) -> tuple[float, float, float]:
    return _get_tune_axes_deg_ns(_NS_WRIST, side)


def set_wrist_tune_axes_deg(side: str, rx: float, ry: float, rz: float) -> None:
    _set_tune_axes_deg_ns(_NS_WRIST, side, rx, ry, rz)


def reset_wrist_tune_axes(side: str | None = None) -> None:
    _reset_tune_axes_ns(_NS_WRIST, side)


WRIST_JOINT_TO_AXIS_INDEX: dict[str, int] = {
    "left_wrist_pitch_joint": 0,
    "left_wrist_roll_joint": 1,
    "left_wrist_yaw_joint": 2,
    "right_wrist_pitch_joint": 0,
    "right_wrist_roll_joint": 1,
    "right_wrist_yaw_joint": 2,
}

WRIST_JOINT_TO_SIDE_BONE: dict[str, tuple[str, str]] = {
    "left_wrist_pitch_joint": ("left", "左手首"),
    "left_wrist_roll_joint": ("left", "左手首"),
    "left_wrist_yaw_joint": ("left", "左手首"),
    "right_wrist_pitch_joint": ("right", "右手首"),
    "right_wrist_roll_joint": ("right", "右手首"),
    "right_wrist_yaw_joint": ("right", "右手首"),
}


# ---------------------------------------------------------------------------
# Elbow 1DOF (hinge flexion)
# ---------------------------------------------------------------------------

# MMD 肘骨常把"前臂自转(pronation)"和真实弯曲(flexion)一起烘焙进 ひじ 局部旋转，
# 旋转轴在帧间漂移，故"绕固定轴 twist"会低估弯曲（O 型手势等帧手臂摊开）。
# 改用"前臂长轴在旋转前后的夹角"作为弯曲角：绕前臂长轴的自转不改变该方向→自动剔除，
# 仅保留把前臂掰弯的成分。前臂长轴在 ひじ 局部系（MMD rest 下与世界轴对齐）中，
# 大致沿手臂指向（左臂 +X 偏下、右臂 -X 偏下），做成可调以便按模型微调。
_DEFAULT_ELBOW_FOREARM_AXIS: dict[str, tuple[float, float, float]] = {
    "left": (0.77, -0.64, 0.0),
    "right": (-0.77, -0.64, 0.0),
}
_elbow_forearm_axis: dict[str, list[float]] = {
    s: list(v) for s, v in _DEFAULT_ELBOW_FOREARM_AXIS.items()
}


def _normalize_vec3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return (1.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def compute_elbow_angle(side: str, q_elbow_xyzw: QuatXYZW | None) -> float:
    """肘弯曲角（无符号，弧度）：前臂长轴 d 在肘骨旋转 R 前后的夹角 acos(d·Rd)。"""
    if q_elbow_xyzw is None:
        return 0.0
    d = _normalize_vec3(tuple(_elbow_forearm_axis[side]))  # type: ignore[arg-type]
    q = normalize_quat_xyzw_short_arc(q_elbow_xyzw)
    r = quat_xyzw_to_mat3(q)
    rd = (
        r[0, 0] * d[0] + r[0, 1] * d[1] + r[0, 2] * d[2],
        r[1, 0] * d[0] + r[1, 1] * d[1] + r[1, 2] * d[2],
        r[2, 0] * d[0] + r[2, 1] * d[1] + r[2, 2] * d[2],
    )
    dot = max(-1.0, min(1.0, d[0] * rd[0] + d[1] * rd[1] + d[2] * rd[2]))
    return math.acos(dot)


def get_elbow_forearm_axis(side: str) -> tuple[float, float, float]:
    v = _elbow_forearm_axis[side]
    return (float(v[0]), float(v[1]), float(v[2]))


def set_elbow_forearm_axis(side: str, x: float, y: float, z: float) -> None:
    _elbow_forearm_axis[side] = [float(x), float(y), float(z)]


def reset_elbow_forearm_axis(side: str | None = None) -> None:
    sides = ["left", "right"] if side is None else [side]
    for s in sides:
        _elbow_forearm_axis[s] = list(_DEFAULT_ELBOW_FOREARM_AXIS[s])


ELBOW_JOINT_TO_SIDE_BONE: dict[str, tuple[str, str]] = {
    "left_elbow_joint": ("left", "左ひじ"),
    "right_elbow_joint": ("right", "右ひじ"),
}


# ---------------------------------------------------------------------------
# Leg hip 3DOF + ankle 2DOF
# ---------------------------------------------------------------------------


def rotate_mmd_vec_by_leg_retarget_basis(
    side: str,
    q_mmd_wxyz: list[float] | tuple[float, float, float, float],
    v_mmd: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector with lower-body quat using leg hip basis: v' = B·R_mmd·Bᵀ·v."""
    w = float(q_mmd_wxyz[0])
    x = float(q_mmd_wxyz[1])
    y = float(q_mmd_wxyz[2])
    z = float(q_mmd_wxyz[3])
    q_xyzw = normalize_quat_xyzw_short_arc((x, y, z, w))
    r_mmd = quat_xyzw_to_mat3(q_xyzw)
    r = _rotmat_mmd_to_g1(_NS_LEG, side, r_mmd)
    vx, vy, vz = float(v_mmd[0]), float(v_mmd[1]), float(v_mmd[2])
    return (
        float(r[0, 0] * vx + r[0, 1] * vy + r[0, 2] * vz),
        float(r[1, 0] * vx + r[1, 1] * vy + r[1, 2] * vz),
        float(r[2, 0] * vx + r[2, 1] * vy + r[2, 2] * vz),
    )


def compute_hip_angles(side: str, q_leg_xyzw: QuatXYZW | None) -> tuple[float, float, float]:
    return _retarget_yxz(_NS_LEG, side, q_leg_xyzw)


def compute_ankle_angles(side: str, q_ank_xyzw: QuatXYZW | None) -> tuple[float, float]:
    return _retarget_yx(_NS_LEG, side, q_ank_xyzw)


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


# ---------------------------------------------------------------------------
# Waist 3DOF
# ---------------------------------------------------------------------------

# G1 waist joint limits from URDF (radians).
_WAIST_PITCH_LIMIT = (-0.52, 0.52)
_WAIST_ROLL_LIMIT = (-0.52, 0.52)
_WAIST_YAW_LIMIT = (-2.618, 2.618)


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _clamp_waist_semantic(pitch: float, roll: float, yaw: float) -> tuple[float, float, float]:
    p = min(max(float(pitch), _WAIST_PITCH_LIMIT[0]), _WAIST_PITCH_LIMIT[1])
    r = min(max(float(roll), _WAIST_ROLL_LIMIT[0]), _WAIST_ROLL_LIMIT[1])
    y = min(max(float(yaw), _WAIST_YAW_LIMIT[0]), _WAIST_YAW_LIMIT[1])
    return p, r, y


def _waist_rot_from_semantic(pitch: float, roll: float, yaw: float) -> np.ndarray:
    # semantic -> physical mapping used by compute_waist_angles:
    # pitch = -theta_x, roll = theta_z, yaw = -theta_y
    theta_x = -float(pitch)
    theta_y = -float(yaw)
    theta_z = float(roll)
    # Extrinsic Z->X->Y, equivalent to matrix product Ry * Rx * Rz.
    return _rot_y(theta_y) @ _rot_x(theta_x) @ _rot_z(theta_z)


def _geodesic_angle_between_rotmat(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r = r_a.T @ r_b
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return math.acos(c)


def _project_waist_semantic_with_limits(
    raw_pitch: float,
    raw_roll: float,
    raw_yaw: float,
    r_target: np.ndarray,
) -> tuple[float, float, float]:
    # Initialization: raw result clipped to URDF bounds.
    s0 = np.array(_clamp_waist_semantic(raw_pitch, raw_roll, raw_yaw), dtype=np.float64)
    s = s0.copy()
    reg = 1.0e-4
    eps = 1.0e-4
    step = 0.08

    def _objective(v: np.ndarray) -> float:
        r_model = _waist_rot_from_semantic(v[0], v[1], v[2])
        ang = _geodesic_angle_between_rotmat(r_model, r_target)
        dev = v - s0
        return ang * ang + reg * float(dev @ dev)

    best = _objective(s)
    for _ in range(40):
        grad = np.zeros((3,), dtype=np.float64)
        for i in range(3):
            vp = s.copy()
            vm = s.copy()
            vp[i] += eps
            vm[i] -= eps
            grad[i] = (_objective(vp) - _objective(vm)) / (2.0 * eps)

        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1.0e-6:
            break

        accepted = False
        local_step = step
        for _ in range(8):
            cand = s - local_step * grad
            cand = np.array(_clamp_waist_semantic(cand[0], cand[1], cand[2]), dtype=np.float64)
            val = _objective(cand)
            if val <= best:
                s = cand
                best = val
                step = min(0.2, local_step * 1.2)
                accepted = True
                break
            local_step *= 0.5
        if not accepted:
            break

    return float(s[0]), float(s[1]), float(s[2])


def compute_waist_angles(
    q_upper_xyzw: QuatXYZW | None,
    q_upper2_xyzw: QuatXYZW | None,
    q_lower_xyzw: QuatXYZW | None = None,
    upper_conj: tuple[bool, bool] = (False, False),
) -> tuple[float, float, float]:
    q_first, q_second = q_upper_xyzw, q_upper2_xyzw
    if q_first is not None and upper_conj[0]:
        q_first = quat_conjugate_xyzw(q_first)
    if q_second is not None and upper_conj[1]:
        q_second = quat_conjugate_xyzw(q_second)

    q = _combine_bone_quats(q_first, q_second)
    if q is None:
        return (0.0, 0.0, 0.0)
    if q_lower_xyzw is not None:
        q = normalize_quat_xyzw_short_arc(quat_mul_xyzw(quat_conjugate_xyzw(q_lower_xyzw), q))

    theta_x, theta_y, theta_z = euler_xyz_rad_waist_extrinsic(q)
    raw_pitch, raw_roll, raw_yaw = -theta_x, theta_z, -theta_y
    r_target = quat_xyzw_to_mat3(q)
    return _project_waist_semantic_with_limits(raw_pitch, raw_roll, raw_yaw, r_target)


WAIST_JOINT_TO_AXIS_INDEX: dict[str, int] = {
    "waist_pitch_joint": 0,
    "waist_roll_joint": 1,
    "waist_yaw_joint": 2,
}
