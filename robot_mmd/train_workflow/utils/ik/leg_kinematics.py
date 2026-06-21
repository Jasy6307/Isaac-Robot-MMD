"""G1 6-DOF leg FK + damped least-squares IK (pelvis -> ankle_roll_link).

Joint chain constants from ``g1_29dof_mode_15_brainco_hand.urdf``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_mmd.train_workflow.utils.ik.geometry import (
    FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M,
    G1_FOOT_IK_SHIN_LENGTH_M,
    G1_FOOT_IK_THIGH_LENGTH_M,
)

# (origin_xyz, origin_rpy, axis_xyz, lo, hi) per joint — hip_pitch .. ankle_roll
_LegJointSpec = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
]

_G1_LEG_CHAIN: dict[str, tuple[_LegJointSpec, ...]] = {
    "left": (
        ((0.0, 0.064452, -0.1027), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.5307, 2.8798),
        ((0.0, 0.052, -0.030465), (0.0, -0.1749, 0.0), (1.0, 0.0, 0.0), -0.5236, 2.9671),
        ((0.025001, 0.0, -0.12412), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.7576, 2.7576),
        ((-0.078273, 0.0021489, -0.17734), (0.0, 0.1749, 0.0), (0.0, 1.0, 0.0), -0.087267, 2.8798),
        ((0.0, -9.4445e-05, -0.30001), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -0.87267, 0.5236),
        ((0.0, 0.0, -0.017558), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.2618, 0.2618),
    ),
    "right": (
        ((0.0, -0.064452, -0.1027), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.5307, 2.8798),
        ((0.0, -0.052, -0.030465), (0.0, -0.1749, 0.0), (1.0, 0.0, 0.0), -2.9671, 0.5236),
        ((0.025001, 0.0, -0.12412), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.7576, 2.7576),
        ((-0.078273, -0.0021489, -0.17734), (0.0, 0.1749, 0.0), (0.0, 1.0, 0.0), -0.087267, 2.8798),
        ((0.0, 9.4445e-05, -0.30001), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -0.87267, 0.5236),
        ((0.0, 0.0, -0.017558), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -0.2618, 0.2618),
    ),
}

_IDENTITY4 = np.eye(4, dtype=np.float64)


_IDENTITY3 = np.eye(3, dtype=np.float64)
_IDENTITY6 = np.eye(6, dtype=np.float64)


@dataclass
class LegIkResult:
    q: tuple[float, float, float, float, float, float]
    converged: bool
    iterations: int
    residual_m: float
    foot_z_local: float = 0.0


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_rot(axis: tuple[float, float, float], theta: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _make_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    rot = _rpy_to_rot(rpy[0], rpy[1], rpy[2])
    out = np.empty((4, 4), dtype=np.float64)
    out[:] = _IDENTITY4
    out[:3, :3] = rot
    out[0, 3] = xyz[0]
    out[1, 3] = xyz[1]
    out[2, 3] = xyz[2]
    return out


def _joint_motion_into(axis: tuple[float, float, float], theta: float, out: np.ndarray) -> None:
    out[:] = _IDENTITY4
    out[:3, :3] = _axis_angle_rot(axis, theta)


def _precompute_side(side: str) -> dict[str, Any]:
    chain = _G1_LEG_CHAIN[side]
    fixed = tuple(_make_transform(spec[0], spec[1]) for spec in chain)
    axes = tuple(np.asarray(spec[2], dtype=np.float64) for spec in chain)
    limits = tuple((spec[3], spec[4]) for spec in chain)
    joint_buf = np.empty((4, 4), dtype=np.float64)
    return {
        "fixed": fixed,
        "axes": axes,
        "limits": limits,
        "joint_buf": joint_buf,
        "work_T": np.empty((4, 4), dtype=np.float64),
    }


_SIDE_CACHE: dict[str, dict[str, Any]] = {
    "left": _precompute_side("left"),
    "right": _precompute_side("right"),
}


def _as_q6(q: tuple[float, float, float, float, float, float] | np.ndarray) -> np.ndarray:
    if isinstance(q, np.ndarray):
        return q.astype(np.float64, copy=False)
    return np.asarray(q, dtype=np.float64)


def _leg_fk_jacobian(q6: np.ndarray, *, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass FK + analytic 3x6 position Jacobian in pelvis frame."""
    cache = _SIDE_CACHE[side]
    fixed = cache["fixed"]
    axes = cache["axes"]
    joint_buf = cache["joint_buf"]
    work_T = cache["work_T"]
    work_T[:] = _IDENTITY4

    origins = np.empty((6, 3), dtype=np.float64)
    axes_w = np.empty((6, 3), dtype=np.float64)

    for i in range(6):
        work_T[:] = work_T @ fixed[i]
        origins[i, 0] = work_T[0, 3]
        origins[i, 1] = work_T[1, 3]
        origins[i, 2] = work_T[2, 3]
        rot = work_T[:3, :3]
        ax = axes[i]
        axes_w[i, 0] = rot[0, 0] * ax[0] + rot[0, 1] * ax[1] + rot[0, 2] * ax[2]
        axes_w[i, 1] = rot[1, 0] * ax[0] + rot[1, 1] * ax[1] + rot[1, 2] * ax[2]
        axes_w[i, 2] = rot[2, 0] * ax[0] + rot[2, 1] * ax[1] + rot[2, 2] * ax[2]
        _joint_motion_into((float(ax[0]), float(ax[1]), float(ax[2])), float(q6[i]), joint_buf)
        work_T[:] = work_T @ joint_buf

    pos = work_T[:3, 3].copy()
    jac = np.empty((3, 6), dtype=np.float64)
    for i in range(6):
        dx = pos[0] - origins[i, 0]
        dy = pos[1] - origins[i, 1]
        dz = pos[2] - origins[i, 2]
        ax0, ax1, ax2 = axes_w[i]
        jac[0, i] = ax1 * dz - ax2 * dy
        jac[1, i] = ax2 * dx - ax0 * dz
        jac[2, i] = ax0 * dy - ax1 * dx
    return pos, jac


def g1_leg_fk_transform(q6: tuple[float, float, float, float, float, float], *, side: str) -> np.ndarray:
    """Pelvis frame -> ankle_roll_link frame (4x4, rotation + translation)."""
    cache = _SIDE_CACHE[side]
    fixed = cache["fixed"]
    axes = cache["axes"]
    joint_buf = cache["joint_buf"]
    work_T = cache["work_T"]
    q = _as_q6(q6)
    work_T[:] = _IDENTITY4
    for i in range(6):
        work_T[:] = work_T @ fixed[i]
        _joint_motion_into(tuple(axes[i]), float(q[i]), joint_buf)
        work_T[:] = work_T @ joint_buf
    return work_T.copy()


def g1_leg_fk_transform_upto_joint(
    q6: tuple[float, float, float, float, float, float] | np.ndarray,
    *,
    side: str,
    last_joint_index: int = 3,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """FK through ``last_joint_index`` inclusive (0=hip_pitch .. 5=ankle_roll)."""
    cache = _SIDE_CACHE[side]
    fixed = cache["fixed"]
    axes = cache["axes"]
    joint_buf = cache["joint_buf"]
    work_T = out if out is not None else np.empty((4, 4), dtype=np.float64)
    q = _as_q6(q6)
    work_T[:] = _IDENTITY4
    last_i = max(0, min(5, int(last_joint_index)))
    for i in range(last_i + 1):
        work_T[:] = work_T @ fixed[i]
        _joint_motion_into(tuple(axes[i]), float(q[i]), joint_buf)
        work_T[:] = work_T @ joint_buf
    return work_T


def g1_leg_fk_compose_ankle_from_knee(
    T_upto_knee: np.ndarray,
    ankle_pitch: float,
    ankle_roll: float,
    *,
    side: str,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Append ankle_pitch + ankle_roll to a FK chain that already ends at knee."""
    cache = _SIDE_CACHE[side]
    fixed = cache["fixed"]
    axes = cache["axes"]
    joint_buf = cache["joint_buf"]
    work_T = out if out is not None else np.empty((4, 4), dtype=np.float64)
    work_T[:] = T_upto_knee
    work_T[:] = work_T @ fixed[4]
    _joint_motion_into(tuple(axes[4]), float(ankle_pitch), joint_buf)
    work_T[:] = work_T @ joint_buf
    work_T[:] = work_T @ fixed[5]
    _joint_motion_into(tuple(axes[5]), float(ankle_roll), joint_buf)
    work_T[:] = work_T @ joint_buf
    return work_T


def g1_leg_fk_pos(q6: tuple[float, float, float, float, float, float], *, side: str) -> tuple[float, float, float]:
    """Ankle_roll_link origin in pelvis frame."""
    pos, _ = _leg_fk_jacobian(_as_q6(q6), side=side)
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def g1_leg_joint_limits(side: str) -> list[tuple[float, float]]:
    return list(_SIDE_CACHE[side]["limits"])


def g1_leg_hip_pitch_origin(side: str) -> tuple[float, float, float]:
    xyz = _G1_LEG_CHAIN[str(side)][0][0]
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def g1_leg_max_reach_m(*, max_reach_ratio: float = 1.0) -> float:
    """Conservative straight-leg reach from hip_pitch origin (both legs similar)."""
    base = float(G1_FOOT_IK_THIGH_LENGTH_M + G1_FOOT_IK_SHIN_LENGTH_M + 0.017558)
    return base * max(0.2, float(max_reach_ratio))


def g1_leg_reach_clamped(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    max_reach_ratio: float,
    margin_m: float = FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M,
) -> bool:
    hip = g1_leg_hip_pitch_origin(side)
    dx = float(target_local_xyz[0]) - hip[0]
    dy = float(target_local_xyz[1]) - hip[1]
    dz = float(target_local_xyz[2]) - hip[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist > g1_leg_max_reach_m(max_reach_ratio=max_reach_ratio) + max(0.0, float(margin_m))


def g1_leg_ground_z_ref_local(*, side: str, max_reach_ratio: float = 1.0) -> float:
    """Nominal lowest foot Z in pelvis frame when the leg is extended downward."""
    hip_z = float(g1_leg_hip_pitch_origin(side)[2])
    return hip_z - g1_leg_max_reach_m(max_reach_ratio=max_reach_ratio) * 0.95


def g1_leg_remap_foot_ik_target(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    xy_scale: float,
    max_reach_ratio: float = 1.0,
    ground_clearance_m: float = 0.012,
    z_compress_power: float = 2.0,
) -> tuple[float, float, float]:
    """Remap foot IK target: XY scaled from hip; Z unchanged when on ground band, else compressed toward floor."""
    hip = g1_leg_hip_pitch_origin(side)
    tx, ty, tz = float(target_local_xyz[0]), float(target_local_xyz[1]), float(target_local_xyz[2])
    s_xy = max(0.05, min(2.0, float(xy_scale)))
    ox = float(hip[0]) + (tx - float(hip[0])) * s_xy
    oy = float(hip[1]) + (ty - float(hip[1])) * s_xy

    ground_z = g1_leg_ground_z_ref_local(side=side, max_reach_ratio=max_reach_ratio)
    hip_z = float(hip[2])
    clearance = max(0.0, float(ground_clearance_m))
    ground_band_top = ground_z + clearance

    if tz <= ground_band_top:
        if abs(s_xy - 1.0) < 1e-9:
            return target_local_xyz
        return (ox, oy, tz)

    hip_span = hip_z - ground_band_top
    if hip_span <= 1e-6:
        return (ox, oy, tz)

    dz = tz - ground_band_top
    if tz <= hip_z:
        t = dz / hip_span
    else:
        t = 1.0 + (tz - hip_z) / max(hip_span, 0.05)

    power = max(0.5, float(z_compress_power))
    z_frac = s_xy ** (1.0 + t * power)
    oz = ground_band_top + dz * z_frac

    if tz > hip_z:
        oz = min(oz, hip_z - clearance)
    oz = max(oz, ground_band_top)

    return (ox, oy, float(oz))


def g1_leg_scale_foot_target_from_hip(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    scale: float,
    max_reach_ratio: float = 1.0,
    ground_clearance_m: float = 0.012,
    z_compress_power: float = 2.0,
) -> tuple[float, float, float]:
    """Backward-compatible wrapper around ``g1_leg_remap_foot_ik_target``."""
    return g1_leg_remap_foot_ik_target(
        target_local_xyz,
        side=side,
        xy_scale=float(scale),
        max_reach_ratio=float(max_reach_ratio),
        ground_clearance_m=float(ground_clearance_m),
        z_compress_power=float(z_compress_power),
    )


def g1_leg_clamp_target_to_reach(
    target_local_xyz: tuple[float, float, float],
    *,
    side: str,
    max_reach_ratio: float,
) -> tuple[tuple[float, float, float], bool]:
    """Project target onto the leg reach sphere (hip_pitch origin). Returns (clamped, was_clamped)."""
    hip = g1_leg_hip_pitch_origin(side)
    tx, ty, tz = target_local_xyz
    dx = float(tx) - hip[0]
    dy = float(ty) - hip[1]
    dz = float(tz) - hip[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_r = g1_leg_max_reach_m(max_reach_ratio=max_reach_ratio)
    if dist <= max_r or dist <= 1e-9:
        return target_local_xyz, False
    scale = max_r / dist
    return (
        (hip[0] + dx * scale, hip[1] + dy * scale, hip[2] + dz * scale),
        True,
    )


def g1_leg_jacobian_pos(
    q6: tuple[float, float, float, float, float, float],
    *,
    side: str,
    eps: float = 1e-6,
) -> np.ndarray:
    """3x6 position Jacobian (analytic; ``eps`` ignored, kept for API compat)."""
    del eps
    _, jac = _leg_fk_jacobian(_as_q6(q6), side=side)
    return jac


def _clamp_q(q: np.ndarray, limits: tuple[tuple[float, float], ...]) -> None:
    for i, (lo, hi) in enumerate(limits):
        v = q[i]
        if v < lo:
            q[i] = lo
        elif v > hi:
            q[i] = hi


def _reg_weights(cfg: Any) -> np.ndarray:
    base = max(0.0, float(getattr(cfg, "ik_reg_weight", 0.15)))
    hy = max(0.0, float(getattr(cfg, "ik_reg_hip_yaw", 0.8)))
    ar = max(0.0, float(getattr(cfg, "ik_reg_ankle_roll", 0.8)))
    return np.array([base, base, base * hy, base, base, base * ar], dtype=np.float64)


def _q_tuple(q: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))


def solve_g1_leg_ik_dls(
    target_local_xyz: tuple[float, float, float],
    q_seed: tuple[float, float, float, float, float, float],
    *,
    q_reg: tuple[float, float, float, float, float, float] | None = None,
    side: str,
    cfg: Any,
    lock_ankle_from_fk: bool = False,
) -> LegIkResult:
    """DLS position IK with null-space pull toward ``q_reg`` (FK retarget).

    When ``lock_ankle_from_fk`` is True, ``ankle_pitch`` / ``ankle_roll`` (indices 4–5)
    stay fixed to ``q_reg`` and only hip + knee joints are updated.
    """
    target = np.asarray(target_local_xyz, dtype=np.float64)
    q = _as_q6(q_seed).copy()
    q_fk = _as_q6(q_reg if q_reg is not None else q_seed)
    limits = _SIDE_CACHE[side]["limits"]
    lock_ankle = bool(lock_ankle_from_fk)

    def _apply_ankle_lock() -> None:
        if lock_ankle:
            q[4] = q_fk[4]
            q[5] = q_fk[5]

    max_iters = max(1, int(getattr(cfg, "ik_max_iters", 20)))
    pos_tol = max(1e-6, float(getattr(cfg, "ik_pos_tol_m", 1e-3)))
    lam = max(1e-6, float(getattr(cfg, "ik_dls_lambda", 0.05)))
    step_scale = max(0.05, min(1.0, float(getattr(cfg, "ik_step_scale", 0.85))))
    reg_w = _reg_weights(cfg)
    lam2 = lam * lam

    _clamp_q(q, limits)
    _apply_ankle_lock()
    pos, _ = _leg_fk_jacobian(q, side=side)
    err = target - pos
    residual = float(np.linalg.norm(err))
    if residual <= pos_tol:
        return LegIkResult(_q_tuple(q), True, 0, residual, float(pos[2]))

    converged = False
    it_done = 0
    reg_pull = np.empty(6, dtype=np.float64)
    knee_min = max(
        float(q_fk[3]),
        float(getattr(cfg, "ik_min_knee_rad", 0.12)),
    )

    for it in range(max_iters):
        it_done = it + 1
        pos, jac = _leg_fk_jacobian(q, side=side)
        err = target - pos
        residual = float(np.linalg.norm(err))
        if residual <= pos_tol:
            converged = True
            break

        jj_t = jac @ jac.T
        jj_t[0, 0] += lam2
        jj_t[1, 1] += lam2
        jj_t[2, 2] += lam2
        inv_jjt = np.linalg.solve(jj_t, _IDENTITY3)
        dq_pos = jac.T @ (inv_jjt @ err)

        j_pinv = jac.T @ inv_jjt
        np.subtract(q_fk, q, out=reg_pull)
        reg_pull *= reg_w
        if float(q[3]) < knee_min:
            reg_pull[3] += reg_w[3] * (knee_min - float(q[3]))
        dq_reg = (_IDENTITY6 - j_pinv @ jac) @ reg_pull

        dq = step_scale * (dq_pos + dq_reg)
        if lock_ankle:
            dq[4] = 0.0
            dq[5] = 0.0
        q += dq
        _clamp_q(q, limits)
        _apply_ankle_lock()

    return LegIkResult(_q_tuple(q), converged, it_done, residual, float(pos[2]))
