#!/usr/bin/env python3
"""Compute root Z compensation for CSV/HDF5 motion and emit *_z_editted.*."""

from __future__ import annotations

import argparse
import os
import sys

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
from robot_mmd.train_workflow.utils.playback_cli import parse_center_to_root_offset
from robot_mmd.train_workflow.utils.root_z_edit import RootZEditConfig, generate_z_editted_motion


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
    p.add_argument(
        "--mmd-foot-ik-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 VMD 足IK目标驱动腿部 IK 覆盖（默认开启）",
    )
    p.add_argument(
        "--mmd-foot-ik-scale",
        type=float,
        default=1.0,
        help="足IK位移缩放（默认 1.0）",
    )
    p.add_argument(
        "--mmd-foot-ik-weight",
        type=float,
        default=1.0,
        help="FK/IK 混合权重，0=纯FK，1=纯IK（默认 1.0）",
    )
    p.add_argument(
        "--mmd-foot-ik-max-reach-ratio",
        type=float,
        default=0.985,
        help="IK 最远可达比例（相对 thigh+shin，默认 0.985）",
    )
    p.add_argument(
        "--mmd-foot-ik-axis-idx",
        type=str,
        default="0,2,1",
        help="MMD->foot target 轴索引 x,y,z（每项 0/1/2）",
    )
    p.add_argument(
        "--mmd-foot-ik-axis-sign",
        type=str,
        default="-1,-1,1",
        help="MMD->foot target 轴符号 x,y,z（建议 ±1）",
    )
    p.add_argument(
        "--mmd-foot-ik-axis-sign-pose",
        type=str,
        default="-1,1,1",
        help="静态 pose 时的轴符号 x,y,z",
    )
    p.add_argument(
        "--mmd-foot-ik-left-ref-local",
        type=str,
        default="0.0,0.095,-0.42",
        help="左脚参考点（root local，米）x,y,z",
    )
    p.add_argument(
        "--mmd-foot-ik-right-ref-local",
        type=str,
        default="0.0,-0.095,-0.42",
        help="右脚参考点（root local，米）x,y,z",
    )
    p.add_argument("--mmd-foot-ik-hip-offset-y", type=float, default=0.095, help="髋关节左右偏置（米）")
    p.add_argument("--mmd-foot-ik-hip-offset-z", type=float, default=0.0, help="髋关节高度偏置（米）")
    p.add_argument("--mmd-foot-ik-thigh-length", type=float, default=0.213, help="大腿长度（米）")
    p.add_argument("--mmd-foot-ik-shin-length", type=float, default=0.213, help="小腿长度（米）")
    p.add_argument("--mmd-foot-ik-hip-roll-gain", type=float, default=0.85, help="侧向 hip roll 增益")
    p.add_argument("--mmd-foot-ik-debug-every", type=int, default=0, help="每 N 帧打印 IK debug；0=关闭")
    p.add_argument("--sim-fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    AppLauncher.add_app_launcher_args(p)
    return p


TASK_ID = "Isaac-G1-Stand-v0"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.device = "cpu"
    app = AppLauncher(args).app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import robot_mmd.my_task  # noqa: F401

    motion_arg = str(args.motion or "").strip() or str(args.csv or "").strip()
    if not motion_arg:
        raise SystemExit("请提供 --motion（或兼容参数 --csv）。")
    input_motion_path = os.path.abspath(motion_arg)
    if not os.path.isfile(input_motion_path):
        raise SystemExit(f"输入 motion 不存在: {input_motion_path}")

    center_off = parse_center_to_root_offset(args.mmd_center_to_root_offset_local)
    root_rpy_scale = _parse_triplet_float(args.root_rpy_scale, "--root-rpy-scale")
    root_rpy_axis_idx = _parse_triplet_int(args.root_rpy_axis_idx, "--root-rpy-axis-idx")
    foot_axis_idx = _parse_triplet_int(args.mmd_foot_ik_axis_idx, "--mmd-foot-ik-axis-idx")
    foot_axis_sign = _parse_triplet_float(args.mmd_foot_ik_axis_sign, "--mmd-foot-ik-axis-sign")
    foot_axis_sign_pose = _parse_triplet_float(args.mmd_foot_ik_axis_sign_pose, "--mmd-foot-ik-axis-sign-pose")
    left_ref_local = _parse_triplet_float(args.mmd_foot_ik_left_ref_local, "--mmd-foot-ik-left-ref-local")
    right_ref_local = _parse_triplet_float(args.mmd_foot_ik_right_ref_local, "--mmd-foot-ik-right-ref-local")

    config = RootZEditConfig(
        output_path=args.output,
        clearance=float(args.clearance),
        ground_z=float(args.ground_z),
        frame_step=int(args.frame_step),
        mode=str(args.mode),
        airborne_threshold=float(args.airborne_threshold),
        airborne_hold=bool(args.airborne_hold),
        dry_run=bool(args.dry_run),
        groove_pos_to_world=float(args.groove_pos_to_world),
        mmd_center_to_root_offset_local_xyz=center_off,
        root_quat_rpy_scale=root_rpy_scale,
        root_quat_rpy_axis_idx=root_rpy_axis_idx,
        knee_hinge_projection=bool(args.mmd_knee_hinge_projection),
        mmd_foot_ik_enable=bool(args.mmd_foot_ik_enable),
        mmd_foot_ik_scale=float(args.mmd_foot_ik_scale),
        mmd_foot_ik_weight=float(args.mmd_foot_ik_weight),
        mmd_foot_ik_max_reach_ratio=float(args.mmd_foot_ik_max_reach_ratio),
        mmd_foot_ik_axis_idx=tuple(int(v) for v in foot_axis_idx),
        mmd_foot_ik_axis_sign=tuple(float(v) for v in foot_axis_sign),
        mmd_foot_ik_axis_sign_pose=tuple(float(v) for v in foot_axis_sign_pose),
        mmd_foot_ik_left_ref_local=tuple(float(v) for v in left_ref_local),
        mmd_foot_ik_right_ref_local=tuple(float(v) for v in right_ref_local),
        mmd_foot_ik_hip_offset_y=float(args.mmd_foot_ik_hip_offset_y),
        mmd_foot_ik_hip_offset_z=float(args.mmd_foot_ik_hip_offset_z),
        mmd_foot_ik_thigh_length=float(args.mmd_foot_ik_thigh_length),
        mmd_foot_ik_shin_length=float(args.mmd_foot_ik_shin_length),
        mmd_foot_ik_hip_roll_gain=float(args.mmd_foot_ik_hip_roll_gain),
        mmd_foot_ik_debug_every=max(0, int(args.mmd_foot_ik_debug_every)),
    )

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
        generate_z_editted_motion(env, env_cfg, input_motion_path, config=config)
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
