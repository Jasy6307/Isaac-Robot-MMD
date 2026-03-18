# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""运行宇树 G1 站立任务 - 零动作，机器人在场景正中以默认姿态站立。Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDIA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../media"))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
parser.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
parser.add_argument(
    "--motion_csv",
    type=str,
    default=os.path.join(_MEDIA_DIR, "333_euler.csv"),
    help="CSV 动作文件路径（U 键播放）",
)
parser.add_argument(
    "--dance_motion_csv",
    type=str,
    default=os.path.join(_MEDIA_DIR, "you_are_important_euler.csv"),
    help="CSV 动作文件路径（I 键播放）",
)
parser.add_argument(
    "--motion_playback",
    action="store_true",
    default=True,
    help="动作回放模式：不固定根链接、禁用重力、增加阻尼",
)
parser.add_argument("--play_speed", type=float, default=1.0, help="播放速度倍率")
parser.add_argument("--smooth_alpha", type=float, default=1.0, help="动作平滑系数 0~1")
parser.add_argument("--sim_fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
parser.add_argument("--mapping_ui", action="store_true", default=True, help="开启 G1 关节映射编辑窗口")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cpu"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robot_mmd.my_task  # noqa: F401
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from robot_mmd.my_task.g1_stand_env_cfg import get_robot_cfg_for_motion_playback
from robot_mmd.train_workflow.csv_motion_loader import (
    build_joint_positions_from_frame,
    get_bone_frame_lists,
    get_frame_indices,
    interpolate_bone,
    load_csv_motion,
)
from robot_mmd.train_workflow.mapping_ui import create_mapping_ui

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


def _load_motion(filepath: str) -> tuple | None:
    """加载 CSV 动作，返回 (frames, frame_list, bone_frame_lists, all_bones) 或 None"""
    if not os.path.isfile(filepath):
        return None
    frames = load_csv_motion(filepath)
    frame_list = get_frame_indices(frames)
    all_bones = set()
    for f in frames.values():
        all_bones.update(f.keys())
    bone_frame_lists = get_bone_frame_lists(frames, frame_list, all_bones)
    return (frames, frame_list, bone_frame_lists, all_bones)


def _compute_action_for_frame(
    frame: int,
    current_frames: Any,
    current_bone_frame_lists: dict[str, list[int]],
    current_all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    smooth_alpha: float,
    smoothed_action: Any,
) -> tuple[Any, Any]:
    """根据帧号插值得到动作，并做平滑。返回 (action_tensor, new_smoothed_action)"""
    frame_data = {}
    for bone in current_all_bones:
        d = interpolate_bone(frame, bone, current_frames, current_bone_frame_lists.get(bone))
        if d is not None:
            frame_data[bone] = d

    if not frame_data or joint_names is None or default_joint_pos is None:
        return None, smoothed_action

    target_pos = build_joint_positions_from_frame(frame_data, joint_names, default_joint_pos)
    target_action = (target_pos - default_joint_pos) / action_scale

    if smoothed_action is None:
        smoothed_action = target_action.copy()
    else:
        smoothed_action = smooth_alpha * target_action + (1.0 - smooth_alpha) * smoothed_action
    return smoothed_action, smoothed_action


def main():
    """零动作运行 G1 站立环境。"""
    motion_data = _load_motion(args_cli.motion_csv)
    dance_data = _load_motion(args_cli.dance_motion_csv)

    if motion_data is None:
        print(f"[WARN] 未找到动作文件: {args_cli.motion_csv}，U 键不可用")
    else:
        print(f"[INFO] 已加载动作: {args_cli.motion_csv}，共 {len(motion_data[1])} 帧")

    if dance_data is None:
        print(f"[WARN] 未找到舞蹈动作: {args_cli.dance_motion_csv}，I 键不可用")
    else:
        print(f"[INFO] 已加载舞蹈动作: {args_cli.dance_motion_csv}，共 {len(dance_data[1])} 帧")

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.motion_playback:
        env_cfg.scene.robot = get_robot_cfg_for_motion_playback().replace(
            prim_path=env_cfg.scene.robot.prim_path
        )
        print("[INFO] 已启用动作回放模式")

    step_dt = 0.02
    if args_cli.sim_fps > 0:
        step_dt = 1.0 / args_cli.sim_fps
        env_cfg.sim.dt = step_dt / 2
        env_cfg.decimation = 2
        env_cfg.sim.render_interval = env_cfg.decimation
        print(f"[INFO] 仿真控制: {args_cli.sim_fps} FPS")

    env = gym.make(TASK_ID, cfg=env_cfg)
    if args_cli.mapping_ui:
        create_mapping_ui()

    print(f"[INFO] 观测: {env.observation_space}, 动作: {env.action_space}")
    print("[INFO] L=重置, U=播放动作, I=播放舞蹈")

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    reset_requested = False
    play_motion_requested = False
    play_dance_requested = False

    def _on_reset():
        nonlocal reset_requested
        reset_requested = True

    def _on_play_motion():
        nonlocal play_motion_requested
        play_motion_requested = True

    def _on_play_dance():
        nonlocal play_dance_requested
        play_dance_requested = True

    keyboard.add_callback("L", _on_reset)
    keyboard.add_callback("U", _on_play_motion)
    keyboard.add_callback("I", _on_play_dance)

    env.reset()
    keyboard.reset()

    current_motion = None  # (frames, frame_list, bone_frame_lists, all_bones)
    play_start_step = 0
    is_playing = False
    last_printed_frame = -1
    action_scale = env_cfg.actions.joint_pos.scale
    joint_names: list[str] = []
    default_joint_pos: Any = None
    smoothed_action = None
    step_count = 0

    def _ensure_joint_info():
        nonlocal joint_names, default_joint_pos
        if not joint_names:
            action_term = env.unwrapped.action_manager.get_term("joint_pos")
            joint_names = action_term._joint_names
            default_joint_pos = (
                env.unwrapped.scene["robot"]
                .data.default_joint_pos[0, action_term._joint_ids]
                .cpu()
                .numpy()
            )

    def _switch_to_motion(data, label: str):
        nonlocal current_motion, play_start_step, is_playing, last_printed_frame, smoothed_action
        if data is None:
            return
        current_motion = data
        play_start_step = step_count
        is_playing = True
        last_printed_frame = -1
        smoothed_action = None
        _ensure_joint_info()
        print(f"[INFO] 开始播放 {label}")

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                env.reset()
                keyboard.reset()
                is_playing = False
                smoothed_action = None
                print("[INFO] 环境已重置")

            if play_motion_requested and motion_data:
                play_motion_requested = False
                _switch_to_motion(motion_data, "CSV 动作 (U)")

            if play_dance_requested and dance_data:
                play_dance_requested = False
                _switch_to_motion(dance_data, "舞蹈动作 (I)")

            if is_playing and current_motion:
                frames, frame_list, bone_frame_lists, all_bones = current_motion  # type: ignore
                elapsed_steps = step_count - play_start_step
                frame = int(elapsed_steps * step_dt * VMD_FPS * args_cli.play_speed)
                max_frame = frame_list[-1]
                frame = min(frame, max_frame)

                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    print(f"[播放] 帧 {frame}/{max_frame}")

                result, smoothed_action = _compute_action_for_frame(  
                    frame, frames, bone_frame_lists, all_bones,
                    joint_names, default_joint_pos, action_scale,
                    args_cli.smooth_alpha, smoothed_action,
                )
                if result is not None:
                    actions = torch.tensor(
                        result, dtype=torch.float32, device=env.unwrapped.device
                    ).unsqueeze(0)
                else:
                    actions = zero_action
            else:
                actions = zero_action

            env.step(actions)
            step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
