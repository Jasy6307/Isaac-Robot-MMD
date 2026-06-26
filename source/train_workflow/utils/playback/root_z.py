"""Root Z compensation for CSV/HDF5 motion (emit ``*_z_editted.*``)."""

from __future__ import annotations

import bisect
import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from source.train_workflow.utils.retarget.joint_axis_map import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from source.train_workflow.utils.format.csv_loader import FootIkConfig, FootIkState
from source.train_workflow.utils.ik.geometry import (
    G1_FOOT_IK_HIP_OFFSET_Y_M,
    G1_FOOT_IK_HIP_OFFSET_Z_M,
    G1_FOOT_IK_SHIN_LENGTH_M,
    G1_FOOT_IK_THIGH_LENGTH_M,
)
from source.train_workflow.utils.ik.mmd_fk import (
    FOOT_IK_VIZ_AXIS_IDX,
    FOOT_IK_VIZ_AXIS_SIGN,
    FOOT_IK_VIZ_AXIS_SIGN_POSE,
    FOOT_IK_VIZ_LEFT_REF_ORIGIN_M,
    FOOT_IK_VIZ_POS_SCALE,
    FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M,
    FootIkVizConfig,
)
from source.train_workflow.utils.format.hdf5 import Hdf5Motion, write_hdf5_motion
from source.train_workflow.utils.motion.loader import load_motion, z_editted_sibling_path
from source.train_workflow.utils.playback.targets import (
    MotionRootTrackState,
    PlaybackUiDebugState,
    compute_targets_for_hdf5_frame,
    compute_targets_for_motion_frame,
    motion_is_static_pose,
)
from source.train_workflow.utils.playback.sim_robot import (
    apply_joint_state_instant,
    apply_root_pos_instant,
    robot_root_row_clone,
)
from source.train_workflow.utils.math.trans_util import rotate_vec_by_quat_wxyz

VMD_FPS = 30
LEFT_FOOT_LINK = "left_ankle_roll_link"
RIGHT_FOOT_LINK = "right_ankle_roll_link"
FOOT_COLLISION_SPHERES_LOCAL = [
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
]
FOOT_COLLISION_SPHERE_RADIUS = 0.005


@dataclass
class RootZEditConfig:
    output_path: str | None = None
    clearance: float = 0.005
    ground_z: float = 0.0
    frame_step: int = 1
    mode: str = "per-frame"
    airborne_threshold: float = 0.03
    airborne_hold: bool = True
    dry_run: bool = False
    groove_pos_to_world: float = 0.1
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_quat_rpy_scale: tuple[float, float, float] = tuple(MMD_ROOT_QUAT_RPY_SCALE_DEFAULT)
    root_quat_rpy_axis_idx: tuple[int, int, int] = tuple(MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT)
    knee_hinge_projection: bool = True
    mmd_foot_ik_enable: bool = True
    mmd_foot_ik_max_reach_ratio: float = 1.0
    mmd_sphere_map_scale: float = FOOT_IK_VIZ_POS_SCALE
    mmd_sphere_map_axis_idx: tuple[int, int, int] = FOOT_IK_VIZ_AXIS_IDX
    mmd_sphere_map_axis_sign: tuple[float, float, float] = FOOT_IK_VIZ_AXIS_SIGN
    mmd_sphere_map_axis_sign_pose: tuple[float, float, float] = FOOT_IK_VIZ_AXIS_SIGN_POSE
    mmd_sphere_map_left_ref_origin: tuple[float, float, float] = FOOT_IK_VIZ_LEFT_REF_ORIGIN_M
    mmd_sphere_map_right_ref_origin: tuple[float, float, float] = FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M
    mmd_foot_ik_hip_offset_y: float = G1_FOOT_IK_HIP_OFFSET_Y_M
    mmd_foot_ik_hip_offset_z: float = G1_FOOT_IK_HIP_OFFSET_Z_M
    mmd_foot_ik_thigh_length: float = G1_FOOT_IK_THIGH_LENGTH_M
    mmd_foot_ik_shin_length: float = G1_FOOT_IK_SHIN_LENGTH_M
    mmd_foot_ik_hip_roll_gain: float = 0.85
    mmd_foot_ik_debug_every: int = 0
    mmd_foot_ik_ik_max_iters: int = 20
    mmd_foot_ik_ik_pos_tol: float = 1e-3
    mmd_foot_ik_ik_reg_weight: float = 0.15
    mmd_foot_ik_ik_reg_hip_yaw: float = 0.8
    mmd_foot_ik_ik_reg_ankle_roll: float = 0.8


def resolve_z_editted_output_path(input_path: str, output: str | None = None) -> str:
    if output and str(output).strip():
        return os.path.abspath(output)
    return z_editted_sibling_path(os.path.abspath(input_path))


def _choose_root_translation_bone(bone_frame_lists: dict[str, list[int]]) -> str:
    g_list = bone_frame_lists.get("グルーブ") or []
    c_list = bone_frame_lists.get("センター") or []
    if c_list and len(c_list) > len(g_list):
        return "センター"
    return "グルーブ"


def _try_get_body_names(robot: Any) -> list[str]:
    for holder in (getattr(robot, "data", None), robot):
        if holder is None:
            continue
        for name in ("body_names", "link_names"):
            v = getattr(holder, name, None)
            if isinstance(v, (list, tuple)) and v:
                return [str(x) for x in v]
    return []


def _resolve_body_indices(robot: Any, needed_names: list[str]) -> dict[str, int]:
    body_names = _try_get_body_names(robot)
    if not body_names:
        raise RuntimeError("Robot body/link name list not found; cannot locate foot links.")
    out: dict[str, int] = {}
    for name in needed_names:
        try:
            out[name] = int(body_names.index(name))
        except ValueError as exc:
            raise RuntimeError(f"Foot link not found: {name}") from exc
    return out


def _body_pos_world_tensor(robot: Any) -> torch.Tensor | None:
    data = robot.data
    for pos_name in ("body_pos_w", "body_link_pos_w", "link_pos_w"):
        v = getattr(data, pos_name, None)
        if torch.is_tensor(v):
            return v
    return None


def _extract_body_pose_wxyz(robot: Any, body_idx: int) -> tuple[tuple[float, float, float], list[float]]:
    data = robot.data
    state = None
    for field in ("body_state_w", "body_link_state_w", "link_state_w"):
        v = getattr(data, field, None)
        if torch.is_tensor(v):
            state = v
            break
    if state is not None:
        if state.ndim == 3:
            row = state[0, body_idx]
        elif state.ndim == 2:
            row = state[body_idx]
        else:
            raise RuntimeError(f"Unsupported body state shape: {tuple(state.shape)}")
        pos = (float(row[0].item()), float(row[1].item()), float(row[2].item()))
        quat = [float(row[3].item()), float(row[4].item()), float(row[5].item()), float(row[6].item())]
        return pos, quat

    pos_t = None
    quat_t = None
    for pos_name in ("body_pos_w", "body_link_pos_w", "link_pos_w"):
        v = getattr(data, pos_name, None)
        if torch.is_tensor(v):
            pos_t = v
            break
    for quat_name in ("body_quat_w", "body_link_quat_w", "link_quat_w"):
        v = getattr(data, quat_name, None)
        if torch.is_tensor(v):
            quat_t = v
            break
    if pos_t is None or quat_t is None:
        raise RuntimeError("Body world pose tensors not found (state_w or pos_w/quat_w).")

    if pos_t.ndim == 3:
        prow = pos_t[0, body_idx]
    elif pos_t.ndim == 2:
        prow = pos_t[body_idx]
    else:
        raise RuntimeError(f"Unsupported body pos shape: {tuple(pos_t.shape)}")
    if quat_t.ndim == 3:
        qrow = quat_t[0, body_idx]
    elif quat_t.ndim == 2:
        qrow = quat_t[body_idx]
    else:
        raise RuntimeError(f"Unsupported body quat shape: {tuple(quat_t.shape)}")
    pos = (float(prow[0].item()), float(prow[1].item()), float(prow[2].item()))
    quat = [float(qrow[0].item()), float(qrow[1].item()), float(qrow[2].item()), float(qrow[3].item())]
    return pos, quat


def resolve_ankle_roll_link_body_indices(robot: Any) -> dict[str, int]:
    """Return body indices for G1 ``left_ankle_roll_link`` / ``right_ankle_roll_link``."""
    return _resolve_body_indices(robot, [LEFT_FOOT_LINK, RIGHT_FOOT_LINK])


def read_ankle_roll_link_world_positions(
    robot: Any,
    *,
    body_indices: dict[str, int] | None = None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Read ankle roll link origins in Isaac world frame (for debug viz)."""
    try:
        idx = body_indices or resolve_ankle_roll_link_body_indices(robot)
        pos_t = _body_pos_world_tensor(robot)
        if pos_t is not None:
            rows = pos_t[0] if pos_t.ndim == 3 else pos_t
            l_idx = int(idx[LEFT_FOOT_LINK])
            r_idx = int(idx[RIGHT_FOOT_LINK])
            # One GPU->CPU copy for both ankles (avoids many scalar .item() syncs).
            lr = rows[[l_idx, r_idx], :3].detach().cpu().tolist()
            return (
                (float(lr[0][0]), float(lr[0][1]), float(lr[0][2])),
                (float(lr[1][0]), float(lr[1][1]), float(lr[1][2])),
            )
        left_pos, _ = _extract_body_pose_wxyz(robot, idx[LEFT_FOOT_LINK])
        right_pos, _ = _extract_body_pose_wxyz(robot, idx[RIGHT_FOOT_LINK])
        return left_pos, right_pos
    except Exception:
        return None, None


def _foot_min_distance_to_ground(robot: Any, foot_body_indices: list[int], ground_z: float) -> float:
    min_d = float("inf")
    for body_idx in foot_body_indices:
        body_pos, body_quat = _extract_body_pose_wxyz(robot, body_idx)
        for local_offset in FOOT_COLLISION_SPHERES_LOCAL:
            dv = rotate_vec_by_quat_wxyz(body_quat, local_offset)
            wz = body_pos[2] + dv[2]
            d = float(wz - FOOT_COLLISION_SPHERE_RADIUS - ground_z)
            if d < min_d:
                min_d = d
    return min_d


def _foot_distances_to_ground(robot: Any, foot_body_indices: list[int], ground_z: float) -> list[float]:
    out: list[float] = []
    for body_idx in foot_body_indices:
        body_pos, body_quat = _extract_body_pose_wxyz(robot, body_idx)
        min_d = float("inf")
        for local_offset in FOOT_COLLISION_SPHERES_LOCAL:
            dv = rotate_vec_by_quat_wxyz(body_quat, local_offset)
            wz = body_pos[2] + dv[2]
            d = float(wz - FOOT_COLLISION_SPHERE_RADIUS - ground_z)
            if d < min_d:
                min_d = d
        out.append(float(min_d))
    return out


def _write_csv_with_root_pos_y_offset(
    input_csv: str,
    output_csv: str,
    root_translation_bone: str,
    csv_pos_y_delta: float,
) -> int:
    with open(input_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"CSV missing header: {input_csv}")
    if "bone" not in fieldnames or "pos_y" not in fieldnames:
        raise RuntimeError(f"CSV missing bone/pos_y columns: {input_csv}")

    changed = 0
    for row in rows:
        if str(row.get("bone", "")) != root_translation_bone:
            continue
        try:
            y = float(row.get("pos_y", "0"))
        except Exception:
            continue
        row["pos_y"] = f"{y + csv_pos_y_delta:.6f}"
        changed += 1

    if changed <= 0:
        raise RuntimeError(f"No root translation rows for bone: {root_translation_bone}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def _interp_delta_by_frame(
    frame: int,
    sampled_frames: list[int],
    sampled_deltas: list[float],
) -> float:
    if not sampled_frames:
        return 0.0
    if len(sampled_frames) == 1:
        return float(sampled_deltas[0])
    if frame <= sampled_frames[0]:
        return float(sampled_deltas[0])
    if frame >= sampled_frames[-1]:
        return float(sampled_deltas[-1])
    i = bisect.bisect_left(sampled_frames, frame)
    if i < len(sampled_frames) and sampled_frames[i] == frame:
        return float(sampled_deltas[i])
    lo = i - 1
    hi = i
    f0 = float(sampled_frames[lo])
    f1 = float(sampled_frames[hi])
    t = 0.0 if (f1 - f0) <= 1e-12 else (float(frame) - f0) / (f1 - f0)
    d0 = float(sampled_deltas[lo])
    d1 = float(sampled_deltas[hi])
    return d0 * (1.0 - t) + d1 * t


def _write_csv_with_per_frame_root_pos_y_offset(
    input_csv: str,
    output_csv: str,
    root_translation_bone: str,
    sampled_frames: list[int],
    sampled_csv_pos_y_deltas: list[float],
) -> tuple[int, float, float]:
    with open(input_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"CSV missing header: {input_csv}")
    if "bone" not in fieldnames or "pos_y" not in fieldnames or "frame" not in fieldnames:
        raise RuntimeError(f"CSV missing bone/pos_y/frame columns: {input_csv}")

    changed = 0
    min_delta = float("inf")
    max_delta = float("-inf")
    for row in rows:
        if str(row.get("bone", "")) != root_translation_bone:
            continue
        try:
            y = float(row.get("pos_y", "0"))
            fr = int(float(row.get("frame", "0")))
        except Exception:
            continue
        dy = _interp_delta_by_frame(fr, sampled_frames, sampled_csv_pos_y_deltas)
        row["pos_y"] = f"{y + dy:.6f}"
        changed += 1
        if dy < min_delta:
            min_delta = dy
        if dy > max_delta:
            max_delta = dy

    if changed <= 0:
        raise RuntimeError(f"No root translation rows for bone: {root_translation_bone}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed, float(min_delta), float(max_delta)


def _clone_hdf5_motion(motion: Hdf5Motion) -> Hdf5Motion:
    return Hdf5Motion(
        frames=np.asarray(motion.frames, dtype=np.int32).copy(),
        joint_names=[str(n) for n in motion.joint_names],
        joint_pos_delta=np.asarray(motion.joint_pos_delta, dtype=np.float32).copy(),
        root_pos_delta=np.asarray(motion.root_pos_delta, dtype=np.float32).copy(),
        root_quat_delta_wxyz=np.asarray(motion.root_quat_delta_wxyz, dtype=np.float32).copy(),
        root_valid=np.asarray(motion.root_valid, dtype=np.bool_).copy(),
        root_rot_bone=[str(v) for v in motion.root_rot_bone],
        root_rpy_deg=np.asarray(motion.root_rpy_deg, dtype=np.float32).copy(),
        source_csv=str(motion.source_csv),
        fps=float(motion.fps),
        knee_hinge_projection=bool(motion.knee_hinge_projection),
        root_quat_rpy_scale=tuple(float(v) for v in motion.root_quat_rpy_scale),
        root_quat_rpy_axis_idx=tuple(int(v) for v in motion.root_quat_rpy_axis_idx),
        mmd_center_to_root_offset_local_xyz=tuple(float(v) for v in motion.mmd_center_to_root_offset_local_xyz),
        groove_pos_to_world=float(motion.groove_pos_to_world),
    )


def _write_hdf5_with_root_z_offsets(
    output_h5: str,
    source_motion: Hdf5Motion,
    sampled_frames: list[int],
    sampled_z_deltas_world: list[float],
) -> tuple[int, float, float]:
    out_motion = _clone_hdf5_motion(source_motion)
    if out_motion.frames.size <= 0:
        raise RuntimeError("HDF5 motion is empty; cannot write compensation.")

    changed = 0
    min_delta = float("inf")
    max_delta = float("-inf")
    for i, fr in enumerate(out_motion.frames.tolist()):
        dz = _interp_delta_by_frame(int(fr), sampled_frames, sampled_z_deltas_world)
        out_motion.root_pos_delta[i, 2] = float(out_motion.root_pos_delta[i, 2] + dz)
        changed += 1
        if dz < min_delta:
            min_delta = dz
        if dz > max_delta:
            max_delta = dz
    write_hdf5_motion(output_h5, out_motion)
    return changed, float(min_delta), float(max_delta)


def generate_z_editted_motion(
    env: Any,
    env_cfg: Any,
    input_motion_path: str,
    *,
    config: RootZEditConfig | None = None,
) -> str:
    """Scan motion in *env*, compute foot clearance, write ``*_z_editted.*`` sibling."""
    cfg = config or RootZEditConfig()
    input_motion_path = os.path.abspath(input_motion_path)
    if not os.path.isfile(input_motion_path):
        raise FileNotFoundError(f"Input motion not found: {input_motion_path}")

    output_path = resolve_z_editted_output_path(input_motion_path, cfg.output_path)
    frame_step = max(1, int(cfg.frame_step))

    motion = load_motion(input_motion_path)
    if motion is None:
        raise RuntimeError(f"Cannot load motion: {input_motion_path}")
    kind = str(motion.get("kind", "")).strip().lower()
    if kind not in ("csv", "hdf5"):
        raise RuntimeError(f"Only CSV/HDF5 supported, got: {kind or 'unknown'}")

    frame_list = motion["frame_list"]
    if not frame_list:
        raise RuntimeError("Motion has no frames.")

    if kind == "csv":
        frames = motion["frames"]
        bone_frame_lists = motion["bone_frame_lists"]
        all_bones = motion["all_bones"]
        root_translation_bone = _choose_root_translation_bone(bone_frame_lists)
        h5_motion = None
    else:
        frames = None
        bone_frame_lists = None
        all_bones = None
        root_translation_bone = ""
        h5_motion = motion["hdf5"]

    env.reset()
    robot = env.unwrapped.scene["robot"]
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_root_snapshot_row = robot_root_row_clone(env)
    ui_debug = PlaybackUiDebugState()
    root_state = MotionRootTrackState()

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_names = action_term._joint_names
    joint_ids = action_term._joint_ids
    default_joint_pos = (
        env.unwrapped.scene["robot"]
        .data.default_joint_pos[0, action_term._joint_ids]
        .cpu()
        .numpy()
    )
    action_scale = env_cfg.actions.joint_pos.scale

    body_indices = _resolve_body_indices(robot, [LEFT_FOOT_LINK, RIGHT_FOOT_LINK])
    foot_body_indices = [body_indices[LEFT_FOOT_LINK], body_indices[RIGHT_FOOT_LINK]]

    max_frame = int(frame_list[-1])
    min_foot_distance = float("inf")
    max_foot_distance = float("-inf")
    scanned = 0
    play_hz = float(VMD_FPS)
    if kind == "csv" and motion_is_static_pose(frames):
        play_hz = 1.0
    sampled_frames: list[int] = []
    sampled_distances: list[float] = []
    sampled_left_distances: list[float] = []
    sampled_right_distances: list[float] = []

    center_off = tuple(float(v) for v in cfg.mmd_center_to_root_offset_local_xyz)
    root_rpy_scale = tuple(float(v) for v in cfg.root_quat_rpy_scale)
    root_rpy_axis_idx = tuple(int(v) for v in cfg.root_quat_rpy_axis_idx)
    foot_ik_cfg = FootIkConfig(
        enable=bool(cfg.mmd_foot_ik_enable),
        groove_pos_to_world=float(cfg.groove_pos_to_world),
        max_reach_ratio=float(cfg.mmd_foot_ik_max_reach_ratio),
        hip_offset_y=float(cfg.mmd_foot_ik_hip_offset_y),
        hip_offset_z=float(cfg.mmd_foot_ik_hip_offset_z),
        thigh_length=float(cfg.mmd_foot_ik_thigh_length),
        shin_length=float(cfg.mmd_foot_ik_shin_length),
        hip_roll_gain=float(cfg.mmd_foot_ik_hip_roll_gain),
        debug_every_n_frames=max(0, int(cfg.mmd_foot_ik_debug_every)),
        ik_max_iters=max(1, int(cfg.mmd_foot_ik_ik_max_iters)),
        ik_pos_tol_m=float(cfg.mmd_foot_ik_ik_pos_tol),
        ik_reg_weight=float(cfg.mmd_foot_ik_ik_reg_weight),
        ik_reg_hip_yaw=float(cfg.mmd_foot_ik_ik_reg_hip_yaw),
        ik_reg_ankle_roll=float(cfg.mmd_foot_ik_ik_reg_ankle_roll),
        is_static_pose=bool(kind == "csv" and motion_is_static_pose(frames)),
    )
    foot_ik_viz_cfg = FootIkVizConfig(
        pos_scale=float(cfg.mmd_sphere_map_scale),
        axis_idx=tuple(int(v) for v in cfg.mmd_sphere_map_axis_idx),
        axis_sign=tuple(float(v) for v in cfg.mmd_sphere_map_axis_sign),
        axis_sign_pose=tuple(float(v) for v in cfg.mmd_sphere_map_axis_sign_pose),
        left_ref_origin_m=tuple(float(v) for v in cfg.mmd_sphere_map_left_ref_origin),
        right_ref_origin_m=tuple(float(v) for v in cfg.mmd_sphere_map_right_ref_origin),
    )
    foot_ik_state = FootIkState()

    for frame in range(0, max_frame + 1, frame_step):
        if kind == "csv":
            (
                joint_pos_cmd,
                target_root_pos,
                target_root_quat_wxyz,
                _result,
                _mmd_root_trans_bone,
                _csv_root_rotation_lookup,
            ) = compute_targets_for_motion_frame(
                frame,
                frames,
                bone_frame_lists,
                all_bones,
                joint_names,
                default_joint_pos,
                action_scale,
                float(cfg.groove_pos_to_world),
                robot,
                root_state,
                ui_debug,
                root_snapshot_row=initial_root_snapshot_row,
                knee_hinge_projection=bool(cfg.knee_hinge_projection),
                mmd_center_to_root_offset_local_xyz=center_off,
                root_quat_rpy_scale=root_rpy_scale,
                root_quat_rpy_axis_idx=root_rpy_axis_idx,
                foot_ik_cfg=foot_ik_cfg,
                foot_ik_state=foot_ik_state,
                foot_ik_viz_cfg=foot_ik_viz_cfg,
            )
        else:
            (
                joint_pos_cmd,
                target_root_pos,
                target_root_quat_wxyz,
                _result,
                _mmd_root_trans_bone,
                _csv_root_rotation_lookup,
            ) = compute_targets_for_hdf5_frame(
                frame,
                h5_motion,
                joint_names,
                default_joint_pos,
                action_scale,
                root_state,
                robot,
                ui_debug,
                root_snapshot_row=initial_root_snapshot_row,
            )

        if target_root_pos is not None and target_root_quat_wxyz is not None:
            apply_root_pos_instant(env, target_root_pos, target_root_quat_wxyz)
        if joint_pos_cmd is not None:
            apply_joint_state_instant(env, joint_pos_cmd, joint_ids)

        env.step(zero_action)
        d = _foot_min_distance_to_ground(robot, foot_body_indices, float(cfg.ground_z))
        feet_d = _foot_distances_to_ground(robot, foot_body_indices, float(cfg.ground_z))
        if d < min_foot_distance:
            min_foot_distance = d
        if d > max_foot_distance:
            max_foot_distance = d
        sampled_frames.append(int(frame))
        sampled_distances.append(float(d))
        sampled_left_distances.append(float(feet_d[0]))
        sampled_right_distances.append(float(feet_d[1]))
        scanned += 1
        if scanned % 300 == 0:
            print(
                f"[INFO] Z edit scan: frame={frame}/{max_frame}, "
                f"min_foot_distance={min_foot_distance:.6f}m"
            )

    if abs(float(cfg.groove_pos_to_world)) < 1e-12:
        raise RuntimeError("groove_pos_to_world must be non-zero")
    if not sampled_frames:
        raise RuntimeError("No frames sampled; cannot compute compensation.")

    mode = str(cfg.mode).strip().lower()
    airborne_count = 0
    if mode == "global":
        z_delta_world = max(0.0, float(cfg.clearance) - float(min_foot_distance))
        sampled_z_deltas = [float(z_delta_world) for _ in sampled_frames]
    else:
        sampled_z_deltas = [max(0.0, float(cfg.clearance) - float(d)) for d in sampled_distances]
        if bool(cfg.airborne_hold):
            th = float(cfg.airborne_threshold)
            hold = 0.0
            for i in range(len(sampled_z_deltas)):
                left_d = float(sampled_left_distances[i])
                right_d = float(sampled_right_distances[i])
                airborne = left_d > th and right_d > th
                if airborne:
                    sampled_z_deltas[i] = max(float(sampled_z_deltas[i]), float(hold))
                    airborne_count += 1
                else:
                    hold = float(sampled_z_deltas[i])
        z_delta_world = float(max(sampled_z_deltas))

    sampled_csv_pos_y_deltas = [d / float(cfg.groove_pos_to_world) for d in sampled_z_deltas]

    print(f"[INFO] input_motion={input_motion_path}")
    print(f"[INFO] output_path={output_path}")
    print(f"[INFO] motion_kind={kind}")
    if kind == "csv":
        print(f"[INFO] root_translation_bone={root_translation_bone}")
    print(f"[INFO] mode={mode}, scanned={scanned}, frames=0..{max_frame}")
    print(f"[INFO] min_foot_distance={min_foot_distance:.6f}m")
    print(f"[INFO] z_delta_world_max={z_delta_world:.6f}m")
    print(f"[INFO] airborne_frames_detected={airborne_count}")

    def _restore_initial_pose() -> None:
        row = initial_root_snapshot_row
        apply_root_pos_instant(
            env,
            (float(row[0]), float(row[1]), float(row[2])),
            [float(row[3]), float(row[4]), float(row[5]), float(row[6])],
        )
        apply_joint_state_instant(env, default_joint_pos, joint_ids)

    if cfg.dry_run:
        print("[INFO] dry-run: no output file written.")
        _restore_initial_pose()
        return output_path

    if kind == "csv":
        if mode == "global":
            changed_rows = _write_csv_with_root_pos_y_offset(
                input_motion_path,
                output_path,
                root_translation_bone,
                sampled_csv_pos_y_deltas[0],
            )
        else:
            changed_rows, _wmin, _wmax = _write_csv_with_per_frame_root_pos_y_offset(
                input_motion_path,
                output_path,
                root_translation_bone,
                sampled_frames,
                sampled_csv_pos_y_deltas,
            )
        print(f"[INFO] Wrote compensated CSV: {output_path} ({changed_rows} rows)")
    else:
        changed_rows, _wmin, _wmax = _write_hdf5_with_root_z_offsets(
            output_path,
            h5_motion,
            sampled_frames,
            sampled_z_deltas,
        )
        print(f"[INFO] Wrote compensated H5: {output_path} ({changed_rows} frames)")

    _restore_initial_pose()
    return output_path
