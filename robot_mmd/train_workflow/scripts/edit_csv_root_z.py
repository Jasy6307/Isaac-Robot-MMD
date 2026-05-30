#!/usr/bin/env python3
"""Compute root Z compensation for CSV/HDF5 motion and emit *_z_editted.*."""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKFLOW_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_WORKFLOW_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.utils.hdf5_motion import Hdf5Motion, write_hdf5_motion
from robot_mmd.train_workflow.utils.motion_loader import load_motion
from robot_mmd.train_workflow.utils.playback_cli import parse_center_to_root_offset
from robot_mmd.train_workflow.utils.playback_targets import (
    MotionRootTrackState,
    PlaybackUiDebugState,
    compute_targets_for_hdf5_frame,
    compute_targets_for_motion_frame,
)
from robot_mmd.train_workflow.utils.sim_robot import (
    apply_joint_state_instant,
    apply_root_pos_instant,
    robot_root_row_clone,
)
from robot_mmd.train_workflow.utils.trans_util import rotate_vec_by_quat_wxyz


def _parse_triplet_float(text: str, name: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} 需为 x,y,z 三个浮点数（逗号分隔）")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_triplet_int(text: str, name: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} 需为 x,y,z 三个整数（逗号分隔）")
    out = tuple(max(0, min(2, int(v))) for v in parts)
    return int(out[0]), int(out[1]), int(out[2])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSV/H5 root Z 自动补偿（输出 *_z_editted.csv|.h5）")
    p.add_argument("--motion", type=str, default="", help="输入 motion 路径（.csv/.h5/.hdf5）")
    p.add_argument("--csv", type=str, default="", help="兼容旧参数：等价于 --motion")
    p.add_argument("-o", "--output", type=str, default=None, help="输出路径（默认 *_z_editted.<ext>）")
    p.add_argument("--clearance", type=float, default=0.005, help="目标脚底最小离地距离（米）")
    p.add_argument("--ground-z", type=float, default=0.0, help="地面高度 Z（米）")
    p.add_argument("--frame-step", type=int, default=1, help="逐帧扫描步长（默认 1）")
    p.add_argument(
        "--mode",
        type=str,
        choices=("per-frame", "global"),
        default="per-frame",
        help="补偿模式：per-frame 逐帧补偿（默认）或 global 全局常量补偿",
    )
    p.add_argument(
        "--airborne-threshold",
        type=float,
        default=0.03,
        help="判定双脚离地的阈值（米）。当两脚最低点都高于该值时进入离地段。",
    )
    p.add_argument(
        "--airborne-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="逐帧模式下：离地段是否保持起跳前补偿（默认开启）。",
    )
    p.add_argument("--dry-run", action="store_true", help="仅计算补偿，不写输出文件")
    p.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
    p.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
    p.add_argument(
        "--groove-pos-to-world",
        type=float,
        default=0.1,
        help="CSV 根平移 pos 到世界米制的缩放（默认 0.1）",
    )
    p.add_argument(
        "--mmd-center-to-root-offset-local",
        type=str,
        default="0,0,0.0",
        help="根局部偏移 x,y,z（米），逗号分隔",
    )
    p.add_argument(
        "--root-rpy-scale",
        type=str,
        default=",".join(str(v) for v in MMD_ROOT_QUAT_RPY_SCALE_DEFAULT),
        help="root RPY 输出缩放，格式 r,p,y",
    )
    p.add_argument(
        "--root-rpy-axis-idx",
        type=str,
        default=",".join(str(v) for v in MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT),
        help="root RPY 轴索引，格式 r,p,y（各值 0/1/2）",
    )
    p.add_argument(
        "--mmd-knee-hinge-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="关节重定向时是否启用膝铰链投影（默认开启）",
    )
    p.add_argument("--sim-fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    AppLauncher.add_app_launcher_args(p)
    return p


_PARSER = _build_parser()
_ARGS = _PARSER.parse_args()
_ARGS.device = "cpu"
_APP = AppLauncher(_ARGS).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import robot_mmd.my_task  # noqa: F401
from robot_mmd.train_workflow.utils.playback_targets import motion_is_static_pose

TASK_ID = "Isaac-G1-Stand-v0"
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


def _resolve_output_path(input_path: str, output: str | None) -> str:
    if output and str(output).strip():
        return os.path.abspath(output)
    stem, ext = os.path.splitext(os.path.abspath(input_path))
    ext = ext or ".csv"
    return stem + "_z_editted" + ext


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
        raise RuntimeError("未找到 robot body/link 名称列表，无法定位脚部 link。")
    out: dict[str, int] = {}
    for name in needed_names:
        try:
            out[name] = int(body_names.index(name))
        except ValueError as exc:
            raise RuntimeError(f"未找到脚部 link: {name}") from exc
    return out


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
            raise RuntimeError(f"不支持的 body state 维度: {tuple(state.shape)}")
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
        raise RuntimeError("未找到 body 世界位姿张量（state_w 或 pos_w/quat_w）。")

    if pos_t.ndim == 3:
        prow = pos_t[0, body_idx]
    elif pos_t.ndim == 2:
        prow = pos_t[body_idx]
    else:
        raise RuntimeError(f"不支持的 body pos 维度: {tuple(pos_t.shape)}")
    if quat_t.ndim == 3:
        qrow = quat_t[0, body_idx]
    elif quat_t.ndim == 2:
        qrow = quat_t[body_idx]
    else:
        raise RuntimeError(f"不支持的 body quat 维度: {tuple(quat_t.shape)}")
    pos = (float(prow[0].item()), float(prow[1].item()), float(prow[2].item()))
    quat = [float(qrow[0].item()), float(qrow[1].item()), float(qrow[2].item()), float(qrow[3].item())]
    return pos, quat


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
        raise RuntimeError(f"CSV 缺少表头: {input_csv}")
    if "bone" not in fieldnames or "pos_y" not in fieldnames:
        raise RuntimeError(f"CSV 缺少必需列 bone/pos_y: {input_csv}")

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
        raise RuntimeError(f"未在 CSV 中找到可修改的 root 平移骨骼行: {root_translation_bone}")

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
        raise RuntimeError(f"CSV 缺少表头: {input_csv}")
    if "bone" not in fieldnames or "pos_y" not in fieldnames or "frame" not in fieldnames:
        raise RuntimeError(f"CSV 缺少必需列 bone/pos_y/frame: {input_csv}")

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
        raise RuntimeError(f"未在 CSV 中找到可修改的 root 平移骨骼行: {root_translation_bone}")

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
        raise RuntimeError("HDF5 轨迹为空，无法写出补偿。")

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


def main() -> None:
    args = _ARGS
    motion_arg = str(args.motion or "").strip() or str(args.csv or "").strip()
    if not motion_arg:
        raise SystemExit("请提供 --motion（或兼容参数 --csv）。")
    input_motion_path = os.path.abspath(motion_arg)
    if not os.path.isfile(input_motion_path):
        raise SystemExit(f"输入 motion 不存在: {input_motion_path}")
    output_path = _resolve_output_path(input_motion_path, args.output)
    frame_step = max(1, int(args.frame_step))

    center_off = parse_center_to_root_offset(args.mmd_center_to_root_offset_local)
    root_rpy_scale = _parse_triplet_float(args.root_rpy_scale, "--root-rpy-scale")
    root_rpy_axis_idx = _parse_triplet_int(args.root_rpy_axis_idx, "--root-rpy-axis-idx")

    motion = load_motion(input_motion_path)
    if motion is None:
        raise SystemExit(f"无法加载 motion: {input_motion_path}")
    kind = str(motion.get("kind", "")).strip().lower()
    if kind not in ("csv", "hdf5"):
        raise SystemExit(f"该工具仅支持 CSV/HDF5 输入，当前: {kind or 'unknown'}")

    frame_list = motion["frame_list"]
    if not frame_list:
        raise SystemExit("motion 无有效帧。")
    if kind == "csv":
        frames = motion["frames"]
        bone_frame_lists = motion["bone_frame_lists"]
        all_bones = motion["all_bones"]
        root_translation_bone = _choose_root_translation_bone(bone_frame_lists)
    else:
        frames = None
        bone_frame_lists = None
        all_bones = None
        root_translation_bone = ""
        h5_motion = motion["hdf5"]

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    from robot_mmd.my_task.g1_stand_env_cfg import G1_TPOSE_INIT_STATE

    env_cfg.scene.robot.init_state = G1_TPOSE_INIT_STATE
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = False
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.robot.spawn.rigid_props.linear_damping = 10.0
    env_cfg.scene.robot.spawn.rigid_props.angular_damping = 10.0

    if int(args.sim_fps) > 0:
        control_dt = 1.0 / int(args.sim_fps)
        env_cfg.sim.dt = control_dt / 2
        env_cfg.decimation = 2
        env_cfg.sim.render_interval = env_cfg.decimation

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
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
                    float(args.groove_pos_to_world),
                    robot,
                    root_state,
                    ui_debug,
                    root_snapshot_row=initial_root_snapshot_row,
                    knee_hinge_projection=bool(args.mmd_knee_hinge_projection),
                    mmd_center_to_root_offset_local_xyz=center_off,
                    root_quat_rpy_scale=root_rpy_scale,
                    root_quat_rpy_axis_idx=root_rpy_axis_idx,
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
            d = _foot_min_distance_to_ground(robot, foot_body_indices, float(args.ground_z))
            feet_d = _foot_distances_to_ground(robot, foot_body_indices, float(args.ground_z))
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
                    f"[INFO] 扫描中: frame={frame}/{max_frame}, "
                    f"current_min_foot_distance={min_foot_distance:.6f}m"
                )

        if abs(float(args.groove_pos_to_world)) < 1e-12:
            raise RuntimeError("--groove-pos-to-world 不能为 0")
        if not sampled_frames:
            raise RuntimeError("未采样到有效帧，无法计算补偿。")

        mode = str(args.mode).strip().lower()
        if mode == "global":
            z_delta_world = max(0.0, float(args.clearance) - float(min_foot_distance))
            sampled_z_deltas = [float(z_delta_world) for _ in sampled_frames]
        else:
            sampled_z_deltas = [
                max(0.0, float(args.clearance) - float(d))
                for d in sampled_distances
            ]
            airborne_count = 0
            if bool(args.airborne_hold):
                # When both feet are airborne, keep the takeoff compensation instead of collapsing to 0.
                # This avoids distorting jump trajectories in per-frame mode.
                th = float(args.airborne_threshold)
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

        sampled_csv_pos_y_deltas = [d / float(args.groove_pos_to_world) for d in sampled_z_deltas]
        min_csv_delta = float(min(sampled_csv_pos_y_deltas))
        max_csv_delta = float(max(sampled_csv_pos_y_deltas))

        print(f"[INFO] input_motion={input_motion_path}")
        print(f"[INFO] output_path={output_path}")
        print(f"[INFO] motion_kind={kind}")
        if kind == "csv":
            print(f"[INFO] root_translation_bone={root_translation_bone}")
        print(f"[INFO] mode={mode}")
        print(f"[INFO] frame_range=0..{max_frame}, frame_step={frame_step}, scanned={scanned}")
        print(f"[INFO] min_foot_distance={min_foot_distance:.6f}m")
        print(f"[INFO] max_foot_distance={max_foot_distance:.6f}m")
        print(f"[INFO] target_clearance={float(args.clearance):.6f}m")
        print(f"[INFO] z_delta_world_max={z_delta_world:.6f}m")
        print(f"[INFO] delta_world_z_min={float(min(sampled_z_deltas)):.6f}m")
        print(f"[INFO] delta_world_z_max={float(max(sampled_z_deltas)):.6f}m")
        print(f"[INFO] airborne_hold={bool(args.airborne_hold)}")
        print(f"[INFO] airborne_threshold={float(args.airborne_threshold):.6f}m")
        print(f"[INFO] airborne_frames_detected={int(airborne_count)}")
        if kind == "csv":
            print(f"[INFO] csv_pos_y_delta_min={min_csv_delta:.6f}")
            print(f"[INFO] csv_pos_y_delta_max={max_csv_delta:.6f}")
        print(f"[INFO] vmd_fps={VMD_FPS}, play_hz={play_hz:.3f}")

        if args.dry_run:
            print("[INFO] dry-run 模式：未写输出文件。")
            return

        if kind == "csv":
            if mode == "global":
                changed_rows = _write_csv_with_root_pos_y_offset(
                    input_motion_path,
                    output_path,
                    root_translation_bone,
                    sampled_csv_pos_y_deltas[0],
                )
                written_min_delta = float(sampled_csv_pos_y_deltas[0])
                written_max_delta = float(sampled_csv_pos_y_deltas[0])
            else:
                changed_rows, written_min_delta, written_max_delta = _write_csv_with_per_frame_root_pos_y_offset(
                    input_motion_path,
                    output_path,
                    root_translation_bone,
                    sampled_frames,
                    sampled_csv_pos_y_deltas,
                )
            print(f"[INFO] 已生成补偿 CSV: {output_path}")
            print(f"[INFO] 修改行数: {changed_rows}")
            print(
                f"[INFO] 写入 root pos_y 偏移范围: "
                f"[{written_min_delta:.6f}, {written_max_delta:.6f}]"
            )
        else:
            changed_rows, written_min_delta, written_max_delta = _write_hdf5_with_root_z_offsets(
                output_path,
                h5_motion,
                sampled_frames,
                sampled_z_deltas,
            )
            print(f"[INFO] 已生成补偿 H5: {output_path}")
            print(f"[INFO] 修改帧数: {changed_rows}")
            print(
                f"[INFO] 写入 root_pos_delta[:,2] 偏移范围(米): "
                f"[{written_min_delta:.6f}, {written_max_delta:.6f}]"
            )
    finally:
        env.close()
        _APP.close()


if __name__ == "__main__":
    main()
