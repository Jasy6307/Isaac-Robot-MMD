"""Per-frame joint/root target computation for CSV and HDF5 playback."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.retarget_unitreeG1 import euler_xyz_rad_waist_extrinsic
from robot_mmd.train_workflow.utils.csv_motion_loader import (
    build_joint_positions_from_frame,
    get_frame_indices,
    interpolate_bone,
)
from robot_mmd.train_workflow.utils.hdf5_motion import Hdf5Motion, sample_hdf5_frame
from robot_mmd.train_workflow.utils.trans_util import (
    mmd_root_offset_quat_to_world,
    quat_from_waist_extrinsic_xyz,
    quat_mul,
    quat_normalize,
    remap_root_csv_euler_xyz,
    root_quat_from_state_row,
    rotate_vec_by_quat_wxyz,
)


@dataclass
class PlaybackUiDebugState:
    """Mutable debug state consumed by mapping UI during playback."""

    last_interp_frame_data: dict[str, dict] | None = None
    root_rpy_euler_scaled_deg: tuple[float | None, float | None, float | None] = (
        None,
        None,
        None,
    )
    root_rot_bone_name: str | None = None

    def reset(self) -> None:
        self.last_interp_frame_data = None
        self.root_rpy_euler_scaled_deg = (None, None, None)
        self.root_rot_bone_name = None


@dataclass
class MotionRootTrackState:
    """Root anchor cached when a motion segment starts."""

    root_origin_pos: tuple[float, float, float] | None = None
    root_quat_wxyz: list[float] | None = None


def build_joint_pos_deg_cache(joint_names: list[str], joint_pos_cmd: Any) -> dict[str, float]:
    """Convert joint radians to UI degree cache."""
    return {j: float(deg) for j, deg in zip(joint_names, joint_pos_cmd * (180.0 / math.pi))}


def compute_action_for_frame(
    frame: int,
    current_frames: Any,
    current_bone_frame_lists: dict[str, list[int]],
    current_all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    knee_hinge_projection: bool = True,
) -> tuple[Any | None, dict[str, dict]]:
    """Interpolate one frame and return (action_delta, interpolated bone dict)."""
    frame_data: dict[str, dict] = {}
    for bone in current_all_bones:
        d = interpolate_bone(frame, bone, current_frames, current_bone_frame_lists.get(bone))
        if d is not None:
            frame_data[bone] = d

    if not frame_data or joint_names is None or default_joint_pos is None:
        return None, frame_data

    target_pos = build_joint_positions_from_frame(
        frame_data,
        joint_names,
        default_joint_pos,
        knee_hinge_projection=knee_hinge_projection,
    )
    target_action = (target_pos - default_joint_pos) / action_scale
    return target_action.copy(), frame_data


def motion_is_static_pose(frames: Any) -> bool:
    """True when CSV has at most one keyed frame (pose snapshot)."""
    return len(get_frame_indices(frames)) <= 1


def get_csv_root_quat_with_bone(
    frame: int,
    frames: Any,
    bone_frame_lists: dict[str, list[int]],
) -> tuple[str | None, list[float] | None]:
    """Extract root orientation quaternion (wxyz) and source bone name from CSV."""
    candidates = ("下半身", "グルーブ", "センター親", "腰", "センター")

    for require_dynamic in (True, False):
        for bone in candidates:
            keyframes = bone_frame_lists.get(bone) or []
            if require_dynamic and len(keyframes) <= 1:
                continue
            d = interpolate_bone(frame, bone, frames, keyframes)
            if d is None:
                continue
            quat_wxyz = d.get("quat_wxyz")
            if quat_wxyz is None or len(quat_wxyz) != 4:
                continue
            try:
                return bone, quat_normalize([float(v) for v in quat_wxyz])
            except Exception:
                continue
    return None, None


def interpolate_mmd_root_translation_bone(
    frame: int,
    frames: Any,
    bone_frame_lists: dict[str, list[int]],
) -> tuple[str | None, dict | None]:
    """Pick グルーブ vs センター for root translation based on keyframe density."""
    g_list = bone_frame_lists.get("グルーブ") or []
    c_list = bone_frame_lists.get("センター") or []
    if c_list and len(c_list) > len(g_list):
        order: tuple[str, ...] = ("センター", "グルーブ")
    else:
        order = ("グルーブ", "センター")
    for bone in order:
        d = interpolate_bone(frame, bone, frames, bone_frame_lists.get(bone))
        if d is not None and "pos" in d:
            return bone, d
    return None, None


def _ensure_root_anchor(
    state: MotionRootTrackState,
    robot: Any,
    root_snapshot_row: Any | None,
) -> None:
    if state.root_origin_pos is not None:
        return
    if root_snapshot_row is not None:
        state.root_origin_pos = (
            float(root_snapshot_row[0].item()),
            float(root_snapshot_row[1].item()),
            float(root_snapshot_row[2].item()),
        )
        state.root_quat_wxyz = root_quat_from_state_row(root_snapshot_row)
        return
    root_state = getattr(robot.data, "root_state_w", None)
    if torch.is_tensor(root_state) and root_state.shape[1] >= 7:
        state.root_origin_pos = (
            float(root_state[0, 0].item()),
            float(root_state[0, 1].item()),
            float(root_state[0, 2].item()),
        )
        state.root_quat_wxyz = root_quat_from_state_row(root_state[0])


def compute_targets_for_motion_frame(
    frame: int,
    frames: Any,
    bone_frame_lists: dict[str, list[int]],
    all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    groove_pos_to_world: float,
    robot: Any,
    state: MotionRootTrackState,
    ui_debug: PlaybackUiDebugState,
    root_snapshot_row: Any | None = None,
    knee_hinge_projection: bool = True,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_quat_rpy_scale: tuple[float, float, float] = MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
    root_quat_rpy_axis_idx: tuple[int, int, int] = MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
) -> tuple[Any, tuple[float, float, float] | None, list[float] | None, Any, str | None, bool | None]:
    """Compute joint/root targets for one CSV motion frame."""
    ui_debug.root_rpy_euler_scaled_deg = (None, None, None)
    ui_debug.root_rot_bone_name = None
    result, interp_fd = compute_action_for_frame(
        frame,
        frames,
        bone_frame_lists,
        all_bones,
        joint_names,
        default_joint_pos,
        action_scale,
        knee_hinge_projection=knee_hinge_projection,
    )
    ui_debug.last_interp_frame_data = interp_fd
    if result is not None:
        joint_pos_cmd = default_joint_pos + action_scale * result
    else:
        joint_pos_cmd = default_joint_pos.copy()

    target_root_pos: tuple[float, float, float] | None = None
    target_root_quat_wxyz: list[float] | None = None
    mmd_root_trans_bone: str | None = None
    csv_root_rotation_lookup: bool | None = None

    root_bone_name, root_mmd = interpolate_mmd_root_translation_bone(frame, frames, bone_frame_lists)
    if root_mmd is not None and "pos" in root_mmd:
        mmd_root_trans_bone = root_bone_name
        try:
            gx, gy, gz = root_mmd["pos"]
            mmd_pos = (float(gx), float(gy), float(gz))
            _ensure_root_anchor(state, robot, root_snapshot_row)

            is_pose = motion_is_static_pose(frames)

            if state.root_origin_pos is not None:
                s = float(groove_pos_to_world)
                dx = mmd_pos[0] * s
                dy = mmd_pos[1] * s
                dz = mmd_pos[2] * s
                ox, oy, oz = state.root_origin_pos
                if is_pose:
                    target_root_pos = (ox - dx, oy + dz, oz + dy)
                else:
                    target_root_pos = (ox - dx, oy - dz, oz - dy)
                target_root_quat_wxyz = list(state.root_quat_wxyz) if state.root_quat_wxyz else None
                csv_root_bone, csv_root_quat_wxyz = get_csv_root_quat_with_bone(
                    frame, frames, bone_frame_lists
                )
                ui_debug.root_rot_bone_name = csv_root_bone
                csv_root_rotation_lookup = csv_root_quat_wxyz is not None
                if csv_root_quat_wxyz is not None and state.root_quat_wxyz is not None:
                    q_w = mmd_root_offset_quat_to_world(csv_root_quat_wxyz)
                    qx, qy, qz, qw = q_w[1], q_w[2], q_w[3], q_w[0]
                    rr, rp, ry = euler_xyz_rad_waist_extrinsic((qx, qy, qz, qw))
                    out_r, out_p, out_y = remap_root_csv_euler_xyz(
                        rr, rp, ry, root_quat_rpy_axis_idx, root_quat_rpy_scale
                    )
                    ui_debug.root_rpy_euler_scaled_deg = (
                        math.degrees(out_r),
                        math.degrees(out_p),
                        math.degrees(out_y),
                    )
                    q_w = quat_from_waist_extrinsic_xyz(out_r, out_p, out_y)
                    target_root_quat_wxyz = quat_normalize(quat_mul(q_w, state.root_quat_wxyz))
                off_l = mmd_center_to_root_offset_local_xyz
                if (
                    target_root_pos is not None
                    and target_root_quat_wxyz is not None
                    and (abs(off_l[0]) > 1e-12 or abs(off_l[1]) > 1e-12 or abs(off_l[2]) > 1e-12)
                ):
                    dv = rotate_vec_by_quat_wxyz(target_root_quat_wxyz, off_l)
                    target_root_pos = (
                        target_root_pos[0] + dv[0],
                        target_root_pos[1] + dv[1],
                        target_root_pos[2] + dv[2],
                    )
        except Exception:
            pass

    return joint_pos_cmd, target_root_pos, target_root_quat_wxyz, result, mmd_root_trans_bone, csv_root_rotation_lookup


def compute_targets_for_hdf5_frame(
    frame: int,
    motion: Hdf5Motion,
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    state: MotionRootTrackState,
    robot: Any,
    ui_debug: PlaybackUiDebugState,
    root_snapshot_row: Any | None = None,
) -> tuple[Any, tuple[float, float, float] | None, list[float] | None, Any, str | None, bool | None]:
    """Compute joint/root targets for one precompiled HDF5 frame."""
    ui_debug.last_interp_frame_data = None
    ui_debug.root_rpy_euler_scaled_deg = (None, None, None)
    ui_debug.root_rot_bone_name = None

    _ensure_root_anchor(state, robot, root_snapshot_row)

    joint_pos_cmd, target_root_pos, target_root_quat_wxyz, debug = sample_hdf5_frame(
        motion,
        frame,
        joint_names,
        np.asarray(default_joint_pos, dtype=np.float32),
        state.root_origin_pos,
        state.root_quat_wxyz,
    )
    rr = debug.get("root_rpy_deg")
    if isinstance(rr, tuple) and len(rr) == 3 and bool(debug.get("root_valid")):
        ui_debug.root_rpy_euler_scaled_deg = (float(rr[0]), float(rr[1]), float(rr[2]))
    rb = str(debug.get("root_rot_bone") or "")
    ui_debug.root_rot_bone_name = rb if rb else None

    result = (joint_pos_cmd - np.asarray(default_joint_pos, dtype=np.float32)) / float(action_scale)
    csv_root_rotation_lookup: bool | None = bool(debug.get("root_valid"))
    return joint_pos_cmd, target_root_pos, target_root_quat_wxyz, result, ui_debug.root_rot_bone_name, csv_root_rotation_lookup
