"""Offline start-weight estimation for motion HDF5.

This module computes per-start-step sampling weights in [1.0, 3.0] from
motion difficulty heuristics, without any simulator dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from source.train_workflow.utils.format.hdf5 import Hdf5Motion, load_hdf5_motion


_LEG_JOINT_WEIGHTED_PATTERNS: tuple[tuple[str, float], ...] = (
    (r".*_hip_pitch_joint", 1.00),
    (r".*_hip_roll_joint", 1.00),
    (r".*_hip_yaw_joint", 1.00),
    (r".*_knee_joint", 1.00),
    (r".*_ankle_pitch_joint", 0.20),
    (r".*_ankle_roll_joint", 0.20),
)

_UPPER_BODY_WEIGHTED_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"waist_.*_joint", 1.30),
    (r".*_shoulder_pitch_joint", 1.00),
    (r".*_shoulder_roll_joint", 1.00),
    (r".*_shoulder_yaw_joint", 1.00),
    (r".*_elbow_joint", 0.90),
    (r".*_wrist_.*_joint", 0.60),
)


@dataclass
class MotionStartWeightResult:
    """Computed difficulty and sampling weights."""

    weights: np.ndarray
    difficulty: np.ndarray
    start_difficulty: np.ndarray
    sample_hz: float
    lookahead_steps: int
    threshold: float
    high_ranges: list[tuple[int, int, float, float]]


def _robust_scale01(x: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 95.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    y = (arr - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def _resample_linear(arr: np.ndarray, target_steps: int) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    if src.ndim == 1:
        src = src[:, None]
    n = int(src.shape[0])
    t = int(target_steps)
    if t <= 0:
        raise ValueError(f"target_steps must be > 0, got {target_steps}")
    if n == t:
        out = src
    else:
        x_old = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, t, endpoint=True, dtype=np.float64)
        out = np.zeros((t, src.shape[1]), dtype=np.float32)
        for j in range(src.shape[1]):
            out[:, j] = np.interp(x_new, x_old, src[:, j]).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return out[:, 0]
    return out


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    qc = np.asarray(q, dtype=np.float32).copy()
    qc[..., 1:] *= -1.0
    return qc


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_angular_speed_wxyz(q: np.ndarray, sample_hz: float) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float32)
    if qq.shape[0] <= 1:
        return np.zeros((qq.shape[0],), dtype=np.float32)
    norm = np.linalg.norm(qq, axis=1, keepdims=True)
    qq = qq / np.clip(norm, 1e-8, None)
    dq = _quat_mul_wxyz(_quat_conj_wxyz(qq[:-1]), qq[1:])
    w = np.abs(np.clip(dq[:, 0], -1.0, 1.0))
    ang = 2.0 * np.arccos(w)
    speed = np.zeros((qq.shape[0],), dtype=np.float32)
    speed[1:] = (ang * float(sample_hz)).astype(np.float32, copy=False)
    speed[0] = speed[1] if speed.shape[0] > 1 else 0.0
    return speed


def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    w = max(1, int(win))
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        j = min(arr.shape[0], i + w)
        out[i] = float(np.mean(arr[i:j]))
    return out


def _moving_max(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    w = max(1, int(win))
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        j = min(arr.shape[0], i + w)
        out[i] = float(np.max(arr[i:j]))
    return out


def _select_weighted_leg_joints(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Select leg joints and per-joint contribution weights.

    Design:
    - hip/knee are primary stability contributors
    - ankle joints are down-weighted
    """
    ids: list[int] = []
    ws: list[float] = []
    for i, name in enumerate(joint_names):
        s = str(name)
        for pat, w in _LEG_JOINT_WEIGHTED_PATTERNS:
            if re.fullmatch(pat, s):
                ids.append(i)
                ws.append(float(w))
                break
    if not ids:
        all_ids = np.arange(len(joint_names), dtype=np.int64)
        all_ws = np.ones((len(joint_names),), dtype=np.float32)
        return all_ids, all_ws
    return np.asarray(ids, dtype=np.int64), np.asarray(ws, dtype=np.float32)


def _extract_high_ranges(weights: np.ndarray, top_ratio: float) -> tuple[float, list[tuple[int, int, float, float]]]:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    ratio = float(np.clip(top_ratio, 1e-3, 0.99))
    threshold = float(np.percentile(w, 100.0 * (1.0 - ratio)))
    flag = w >= threshold
    ranges: list[tuple[int, int, float, float]] = []
    i = 0
    n = int(w.shape[0])
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flag[j]:
            j += 1
        seg = w[i:j]
        ranges.append((i, j, float(np.mean(seg)), float(np.max(seg))))
        i = j
    ranges.sort(key=lambda item: item[3], reverse=True)
    return threshold, ranges


def _upper_body_pseudo_com_offset(joint: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Approximate upper-body COM offset using weighted joint-angle magnitude.

    This is a kinematics-free proxy:
    - select waist/arms joints
    - compute weighted RMS of absolute joint deltas per frame
    Higher value means upper body is farther from neutral/root-aligned posture.
    """
    ids: list[int] = []
    ws: list[float] = []
    for i, name in enumerate(joint_names):
        s = str(name)
        for pat, w in _UPPER_BODY_WEIGHTED_PATTERNS:
            if re.fullmatch(pat, s):
                ids.append(i)
                ws.append(float(w))
                break
    if not ids:
        return np.zeros((joint.shape[0],), dtype=np.float32)
    j = np.asarray(joint[:, np.asarray(ids, dtype=np.int64)], dtype=np.float32)
    w = np.asarray(ws, dtype=np.float32).reshape(1, -1)
    rms = np.sqrt(np.mean((j * w) ** 2, axis=1))
    return rms.astype(np.float32, copy=False)


def compute_motion_start_weights(
    motion: Hdf5Motion,
    *,
    target_steps: int | None = None,
    lookahead_seconds: float = 3.0,
    top_ratio: float = 0.25,
) -> MotionStartWeightResult:
    """Compute per-start-step weights in [1.0, 3.0]."""
    if lookahead_seconds <= 0.0:
        raise ValueError(f"lookahead_seconds must be > 0, got {lookahead_seconds}")

    joint = np.asarray(motion.joint_pos_delta, dtype=np.float32)
    root_pos = np.asarray(motion.root_pos_delta, dtype=np.float32)
    root_quat = np.asarray(motion.root_quat_delta_wxyz, dtype=np.float32)
    root_rpy_deg = np.asarray(motion.root_rpy_deg, dtype=np.float32)

    src_steps = int(joint.shape[0])
    if src_steps <= 1:
        raise ValueError("H5 motion must have at least 2 frames")

    h5_fps = float(motion.fps if motion.fps > 0 else 30.0)
    steps = int(target_steps) if target_steps is not None else src_steps
    if steps <= 1:
        raise ValueError(f"target steps must be > 1, got {steps}")

    joint = _resample_linear(joint, steps)
    root_pos = _resample_linear(root_pos, steps)
    root_quat = _resample_linear(root_quat, steps)
    root_rpy_deg = _resample_linear(root_rpy_deg, steps)

    sample_hz = float(h5_fps * (float(steps) / float(src_steps)))
    leg_ids, leg_w = _select_weighted_leg_joints(list(motion.joint_names))
    leg_w = leg_w.reshape(1, -1)
    leg_w_sum = float(np.sum(leg_w))
    leg_w_sum = max(leg_w_sum, 1e-8)

    d_joint = np.diff(joint, axis=0, prepend=joint[0:1])
    joint_vel = (np.sum(np.abs(d_joint[:, leg_ids]) * leg_w, axis=1) / leg_w_sum) * sample_hz
    d_joint_vel = np.diff(d_joint, axis=0, prepend=d_joint[0:1])
    joint_acc = (
        np.sum(np.abs(d_joint_vel[:, leg_ids]) * leg_w, axis=1) / leg_w_sum
    ) * (sample_hz * sample_hz)

    d_pos = np.diff(root_pos, axis=0, prepend=root_pos[0:1])
    root_lin_speed = np.linalg.norm(d_pos, axis=1) * sample_hz
    root_ang_speed = _quat_angular_speed_wxyz(root_quat, sample_hz)
    tilt = np.linalg.norm(np.deg2rad(root_rpy_deg[:, 0:2]), axis=1)
    upper_pseudo_com_offset = _upper_body_pseudo_com_offset(joint, list(motion.joint_names))

    score = (
        0.13 * _robust_scale01(joint_vel)
        + 0.09 * _robust_scale01(joint_acc)
        + 0.14 * _robust_scale01(root_lin_speed)
        + 0.12 * _robust_scale01(root_ang_speed)
        + 0.10 * _robust_scale01(tilt)
        + 0.20 * _robust_scale01(upper_pseudo_com_offset)
    ).astype(np.float32, copy=False)

    lookahead_steps = max(1, int(round(float(lookahead_seconds) * sample_hz)))
    start_score = 0.5 * _moving_mean(score, lookahead_steps) + 0.5 * _moving_max(score, lookahead_steps)
    start_score = _robust_scale01(start_score, lo_pct=5.0, hi_pct=95.0)

    weights = (1.0 + 2.0 * start_score).astype(np.float32, copy=False)
    weights = np.clip(weights, 1.0, 3.0).astype(np.float32, copy=False)

    threshold, high_ranges = _extract_high_ranges(weights, top_ratio=top_ratio)
    return MotionStartWeightResult(
        weights=weights,
        difficulty=score,
        start_difficulty=start_score,
        sample_hz=sample_hz,
        lookahead_steps=lookahead_steps,
        threshold=threshold,
        high_ranges=high_ranges,
    )


def compute_motion_start_weights_from_h5(
    h5_path: str,
    *,
    target_steps: int | None = None,
    lookahead_seconds: float = 3.0,
    top_ratio: float = 0.25,
) -> MotionStartWeightResult:
    """Load H5 and compute start weights in [1.0, 3.0]."""
    motion = load_hdf5_motion(h5_path)
    return compute_motion_start_weights(
        motion,
        target_steps=target_steps,
        lookahead_seconds=lookahead_seconds,
        top_ratio=top_ratio,
    )


def summarize_result(result: MotionStartWeightResult, max_ranges: int = 8) -> str:
    """Render a concise terminal summary."""
    w = result.weights
    lines = [
        "[INFO] Auto motion-start weighting (offline)",
        (
            f"[INFO] Weights: steps={w.shape[0]} min={float(np.min(w)):.3f} "
            f"mean={float(np.mean(w)):.3f} max={float(np.max(w)):.3f}"
        ),
        (
            f"[INFO] Difficulty: sample_hz={result.sample_hz:.3f} "
            f"lookahead_steps={result.lookahead_steps} threshold={result.threshold:.3f}"
        ),
        "[INFO] High-weight ranges (start,end,mean,max):",
    ]
    shown = 0
    for st, ed, mean_w, max_w in result.high_ranges:
        lines.append(f"  - [{st}, {ed}) mean={mean_w:.3f} max={max_w:.3f}")
        shown += 1
        if shown >= max(1, int(max_ranges)):
            break
    if shown == 0:
        lines.append("  - none")
    return "\n".join(lines)


def save_weights_csv(path: str, weights: np.ndarray) -> None:
    """Save one-row-per-step weights as CSV."""
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("step,weight\n")
        for i, w in enumerate(arr):
            f.write(f"{i},{float(w):.6f}\n")


def save_weights_json(path: str, result: MotionStartWeightResult) -> None:
    """Save summary + full weights in JSON."""
    import json

    payload = {
        "steps": int(result.weights.shape[0]),
        "sample_hz": float(result.sample_hz),
        "lookahead_steps": int(result.lookahead_steps),
        "weight_min": float(np.min(result.weights)),
        "weight_mean": float(np.mean(result.weights)),
        "weight_max": float(np.max(result.weights)),
        "threshold": float(result.threshold),
        "high_ranges": [
            {"start": int(st), "end": int(ed), "mean": float(mn), "max": float(mx)}
            for st, ed, mn, mx in result.high_ranges
        ],
        "weights": [float(v) for v in result.weights.tolist()],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

