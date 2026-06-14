"""Ankle-only ground compensation: tilt foot when sole penetrates floor, else unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_mmd.train_workflow.utils.g1_leg_kinematics import (
    g1_leg_fk_compose_ankle_from_knee,
    g1_leg_fk_transform_upto_joint,
    g1_leg_joint_limits,
)
from robot_mmd.train_workflow.utils.root_z_edit import (
    FOOT_COLLISION_SPHERE_RADIUS,
    FOOT_COLLISION_SPHERES_LOCAL,
)
from robot_mmd.train_workflow.utils.trans_util import quat_to_rotmat

_LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_SPHERES_LOCAL_NP = np.asarray(FOOT_COLLISION_SPHERES_LOCAL, dtype=np.float64)
_FLOOR_R = float(FOOT_COLLISION_SPHERE_RADIUS)


@dataclass
class FootAnkleGroundCompConfig:
    enable: bool = True
    ground_z: float = 0.0
    clearance_m: float = 0.005
    max_pitch_delta_deg: float = 28.0
    max_roll_delta_deg: float = 18.0
    pitch_search_steps: int = 11
    roll_search_steps: int = 5


@dataclass
class FootAnkleGroundCompState:
    last_left_pitch_delta_deg: float = 0.0
    last_right_pitch_delta_deg: float = 0.0
    last_left_roll_delta_deg: float = 0.0
    last_right_roll_delta_deg: float = 0.0


@dataclass
class _LegGroundSearchCtx:
    side: str
    root_z: float
    floor_z: float
    clearance: float
    R_root: np.ndarray
    T_knee: np.ndarray = field(default_factory=lambda: np.empty((4, 4), dtype=np.float64))
    T_foot: np.ndarray = field(default_factory=lambda: np.empty((4, 4), dtype=np.float64))


def foot_ankle_ground_comp_config_from_namespace(ns: Any) -> FootAnkleGroundCompConfig:
    return FootAnkleGroundCompConfig(
        enable=bool(getattr(ns, "foot_ankle_ground_comp", True)),
        ground_z=float(getattr(ns, "foot_ground_z", 0.0)),
        clearance_m=float(getattr(ns, "foot_ground_clearance", 0.005)),
        max_pitch_delta_deg=float(getattr(ns, "foot_ankle_ground_max_pitch_deg", 28.0)),
        max_roll_delta_deg=float(getattr(ns, "foot_ankle_ground_max_roll_deg", 18.0)),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _q6_from_joint_cmd(
    joint_pos_cmd: Any,
    joint_names: list[str],
    default_joint_pos: Any,
    side: str,
) -> tuple[float, float, float, float, float, float] | None:
    jidx = {str(n): i for i, n in enumerate(joint_names)}
    vals: list[float] = []
    for n in _LEG_JOINTS:
        jn = f"{side}_{n}_joint"
        if jn not in jidx:
            return None
        vals.append(float(joint_pos_cmd[jidx[jn]]) - float(default_joint_pos[jidx[jn]]))
    return tuple(vals)  # type: ignore[return-value]


def _write_ankle_to_joint_cmd(
    joint_pos_cmd: Any,
    joint_names: list[str],
    default_joint_pos: Any,
    side: str,
    ankle_pitch: float,
    ankle_roll: float,
) -> None:
    jidx = {str(n): i for i, n in enumerate(joint_names)}
    for jn, val in (
        (f"{side}_ankle_pitch_joint", ankle_pitch),
        (f"{side}_ankle_roll_joint", ankle_roll),
    ):
        if jn in jidx:
            joint_pos_cmd[jidx[jn]] = float(default_joint_pos[jidx[jn]]) + float(val)


def _make_search_ctx(
    q6: tuple[float, float, float, float, float, float],
    *,
    side: str,
    root_pos: tuple[float, float, float],
    root_quat_wxyz: list[float],
    cfg: FootAnkleGroundCompConfig,
) -> _LegGroundSearchCtx:
    ctx = _LegGroundSearchCtx(
        side=str(side),
        root_z=float(root_pos[2]),
        floor_z=float(cfg.ground_z) + _FLOOR_R,
        clearance=float(cfg.clearance_m),
        R_root=np.asarray(quat_to_rotmat(root_quat_wxyz), dtype=np.float64),
    )
    g1_leg_fk_transform_upto_joint(q6, side=side, last_joint_index=3, out=ctx.T_knee)
    return ctx


def _min_clearance_from_ctx(ctx: _LegGroundSearchCtx, ankle_pitch: float, ankle_roll: float) -> float:
    """Min distance from foot collision spheres to floor (m); negative = penetration."""
    g1_leg_fk_compose_ankle_from_knee(
        ctx.T_knee,
        float(ankle_pitch),
        float(ankle_roll),
        side=ctx.side,
        out=ctx.T_foot,
    )
    R = ctx.T_foot[:3, :3]
    t = ctx.T_foot[:3, 3]
    p_pelvis = (_SPHERES_LOCAL_NP @ R.T) + t
    z_world = ctx.root_z + (ctx.R_root @ p_pelvis.T)[2, :]
    return float(np.min(z_world) - ctx.floor_z)


def _consider_candidate(
    ap: float,
    ar: float,
    ap0: float,
    ar0: float,
    ctx: _LegGroundSearchCtx,
    *,
    best: dict[str, float | bool],
) -> None:
    clr = _min_clearance_from_ctx(ctx, ap, ar)
    cost = abs(ap - ap0) + 0.35 * abs(ar - ar0)
    found_ok = bool(best["found_ok"])
    if clr >= ctx.clearance and cost < float(best["best_cost"]):
        best["best_ap"] = ap
        best["best_ar"] = ar
        best["best_clear"] = clr
        best["best_cost"] = cost
        best["found_ok"] = True
        return
    if found_ok:
        return
    if clr > float(best["best_clear"]) + 1e-9:
        best["best_ap"] = ap
        best["best_ar"] = ar
        best["best_clear"] = clr
        best["best_cost"] = cost
    elif abs(clr - float(best["best_clear"])) < 1e-9 and cost < float(best["best_cost"]):
        best["best_ap"] = ap
        best["best_ar"] = ar
        best["best_cost"] = cost


def _search_ankle_angles(
    ctx: _LegGroundSearchCtx,
    ap0: float,
    ar0: float,
    ap_lo: float,
    ap_hi: float,
    ar_lo: float,
    ar_hi: float,
    max_dp: float,
    max_dr: float,
    n_pitch: int,
    n_roll: int,
) -> tuple[float, float]:
    best: dict[str, float | bool] = {
        "best_ap": ap0,
        "best_ar": ar0,
        "best_clear": _min_clearance_from_ctx(ctx, ap0, ar0),
        "best_cost": float("inf"),
        "found_ok": False,
    }
    n_p = max(3, int(n_pitch))
    n_r = max(1, int(n_roll))

    for i in range(n_p):
        t = 0.0 if n_p <= 1 else float(i) / float(n_p - 1)
        ap = _clamp(ap0 + (2.0 * t - 1.0) * max_dp, ap_lo, ap_hi)
        _consider_candidate(ap, ar0, ap0, ar0, ctx, best=best)

    if not bool(best["found_ok"]) and n_r > 1:
        ap_seed = float(best["best_ap"])
        for j in range(n_r):
            u = float(j) / float(n_r - 1)
            ar = _clamp(ar0 + (2.0 * u - 1.0) * max_dr, ar_lo, ar_hi)
            _consider_candidate(ap_seed, ar, ap0, ar0, ctx, best=best)

    if not bool(best["found_ok"]) and n_r > 1 and n_p > 3:
        ap_seed = float(best["best_ap"])
        ar_seed = float(best["best_ar"])
        refine_dp = max_dp / max(3.0, float(n_p) / 3.0)
        refine_dr = max_dr / max(2.0, float(n_r) / 2.0)
        for i in range(3):
            t = 0.0 if n_p <= 1 else float(i) / 2.0
            ap = _clamp(ap_seed + (2.0 * t - 1.0) * refine_dp, ap_lo, ap_hi)
            for j in range(2):
                u = float(j)
                ar = _clamp(ar_seed + (2.0 * u - 1.0) * refine_dr, ar_lo, ar_hi)
                _consider_candidate(ap, ar, ap0, ar0, ctx, best=best)

    return float(best["best_ap"]), float(best["best_ar"])


def min_foot_sole_clearance_m(
    q6: tuple[float, float, float, float, float, float],
    *,
    side: str,
    root_pos: tuple[float, float, float],
    root_quat_wxyz: list[float],
    ground_z: float,
) -> float:
    """Min distance from foot collision spheres to ground (m); negative = penetration."""
    cfg = FootAnkleGroundCompConfig(ground_z=float(ground_z))
    ctx = _make_search_ctx(
        q6,
        side=side,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        cfg=cfg,
    )
    return _min_clearance_from_ctx(ctx, float(q6[4]), float(q6[5]))


def compensate_leg_ankle_for_ground(
    q6: tuple[float, float, float, float, float, float],
    *,
    side: str,
    root_pos: tuple[float, float, float],
    root_quat_wxyz: list[float],
    cfg: FootAnkleGroundCompConfig,
) -> tuple[tuple[float, float, float, float, float, float], float, float, bool]:
    """Return (q6_out, pitch_delta_rad, roll_delta_rad, adjusted)."""
    ap0 = float(q6[4])
    ar0 = float(q6[5])
    ctx = _make_search_ctx(
        q6,
        side=side,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        cfg=cfg,
    )
    base_clear = _min_clearance_from_ctx(ctx, ap0, ar0)
    if base_clear >= ctx.clearance:
        return q6, 0.0, 0.0, False

    limits = g1_leg_joint_limits(side)
    ap_lo, ap_hi = limits[4]
    ar_lo, ar_hi = limits[5]
    max_dp = math.radians(float(cfg.max_pitch_delta_deg))
    max_dr = math.radians(float(cfg.max_roll_delta_deg))
    best_ap, best_ar = _search_ankle_angles(
        ctx,
        ap0,
        ar0,
        ap_lo,
        ap_hi,
        ar_lo,
        ar_hi,
        max_dp,
        max_dr,
        int(cfg.pitch_search_steps),
        int(cfg.roll_search_steps),
    )
    q_out = (q6[0], q6[1], q6[2], q6[3], float(best_ap), float(best_ar))
    return q_out, float(best_ap - ap0), float(best_ar - ar0), True


def apply_ankle_ground_comp_to_joint_cmd(
    joint_pos_cmd: Any,
    joint_names: list[str],
    default_joint_pos: Any,
    *,
    root_pos: tuple[float, float, float] | None,
    root_quat_wxyz: list[float] | None,
    cfg: FootAnkleGroundCompConfig | None,
    state: FootAnkleGroundCompState | None = None,
) -> None:
    """In-place ankle pitch/roll fix when foot sole spheres penetrate the floor."""
    if cfg is None or not bool(cfg.enable):
        return
    if root_pos is None or root_quat_wxyz is None:
        return

    for side in ("left", "right"):
        q6 = _q6_from_joint_cmd(joint_pos_cmd, joint_names, default_joint_pos, side)
        if q6 is None:
            continue
        q_out, dp, dr, adjusted = compensate_leg_ankle_for_ground(
            q6,
            side=side,
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
            cfg=cfg,
        )
        if not adjusted:
            if state is not None:
                if side == "left":
                    state.last_left_pitch_delta_deg = 0.0
                    state.last_left_roll_delta_deg = 0.0
                else:
                    state.last_right_pitch_delta_deg = 0.0
                    state.last_right_roll_delta_deg = 0.0
            continue
        _write_ankle_to_joint_cmd(
            joint_pos_cmd,
            joint_names,
            default_joint_pos,
            side,
            q_out[4],
            q_out[5],
        )
        if state is not None:
            if side == "left":
                state.last_left_pitch_delta_deg = math.degrees(dp)
                state.last_left_roll_delta_deg = math.degrees(dr)
            else:
                state.last_right_pitch_delta_deg = math.degrees(dp)
                state.last_right_roll_delta_deg = math.degrees(dr)
