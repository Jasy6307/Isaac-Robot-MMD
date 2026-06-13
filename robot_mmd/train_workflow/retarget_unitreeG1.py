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

RetargetNamespace = Literal["arm", "leg"]
Side = Literal["left", "right"]
QuatXYZW = tuple[float, float, float, float]

_NS_ARM: RetargetNamespace = "arm"
_NS_LEG: RetargetNamespace = "leg"

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
) -> tuple[float, float, float]:
    q = _combine_bone_quats(q_shoulder_xyzw, q_arm_xyzw)
    if q is None:
        return (0.0, 0.0, 0.0)
    return _retarget_yxz(_NS_ARM, side, q)


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
