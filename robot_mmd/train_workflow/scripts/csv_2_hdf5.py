#!/usr/bin/env python3
"""Compile retargeted robot trajectory from CSV into HDF5."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKFLOW_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_WORKFLOW_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.utils.hdf5_motion import (
    compile_csv_motion_to_hdf5_motion,
    write_hdf5_motion,
)


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


def _parse_joint_names(args_joint_names: str | None) -> list[str]:
    if args_joint_names and str(args_joint_names).strip():
        return [s.strip() for s in str(args_joint_names).split(",") if s.strip()]
    # 默认 G1 23 关节顺序：与当前映射表一致
    return [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
        "waist_yaw_joint",
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSV -> HDF5 机器人关节/根轨迹预编译")
    p.add_argument("input_csv", type=str, help="输入 CSV（bone 轨迹）")
    p.add_argument("-o", "--output", type=str, default=None, help="输出 HDF5 路径，默认同名 .h5")
    p.add_argument(
        "--joint-names",
        type=str,
        default="",
        help="运行时 joint_names 顺序，逗号分隔；不传则用内置 G1 默认顺序",
    )
    p.add_argument("--fps", type=float, default=30.0, help="写入元数据 fps（默认 30）")
    p.add_argument(
        "--knee-hinge-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="编译关节角时是否启用膝铰链投影（默认开启）",
    )
    p.add_argument(
        "--groove-pos-to-world",
        type=float,
        default=0.1,
        help="MMD 根平移到世界系的缩放（默认 0.1）",
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
        help="root RPY 输出缩放，格式 r,p,y（默认映射默认值）",
    )
    p.add_argument(
        "--root-rpy-axis-idx",
        type=str,
        default=",".join(str(v) for v in MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT),
        help="root RPY 轴索引，格式 r,p,y（各值 0/1/2）",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    input_csv = os.path.abspath(args.input_csv)
    if not os.path.isfile(input_csv):
        raise SystemExit(f"输入 CSV 不存在: {input_csv}")

    out = args.output
    if not out:
        out = os.path.splitext(input_csv)[0] + ".h5"
    out = os.path.abspath(out)

    joint_names = _parse_joint_names(args.joint_names)
    center_off = _parse_triplet_float(args.mmd_center_to_root_offset_local, "--mmd-center-to-root-offset-local")
    root_scale = _parse_triplet_float(args.root_rpy_scale, "--root-rpy-scale")
    root_idx = _parse_triplet_int(args.root_rpy_axis_idx, "--root-rpy-axis-idx")

    motion = compile_csv_motion_to_hdf5_motion(
        input_csv,
        joint_names,
        fps=float(args.fps),
        knee_hinge_projection=bool(args.knee_hinge_projection),
        groove_pos_to_world=float(args.groove_pos_to_world),
        mmd_center_to_root_offset_local_xyz=center_off,
        root_quat_rpy_scale=root_scale,
        root_quat_rpy_axis_idx=root_idx,
    )
    out_path = write_hdf5_motion(out, motion)
    print(f"[INFO] 已生成 HDF5: {out_path}")
    print(
        "[INFO] frames=%d joints=%d root_valid=%d"
        % (
            int(motion.frames.shape[0]),
            int(motion.joint_pos_delta.shape[1]),
            int(np.count_nonzero(motion.root_valid)),
        )
    )


if __name__ == "__main__":
    main()

