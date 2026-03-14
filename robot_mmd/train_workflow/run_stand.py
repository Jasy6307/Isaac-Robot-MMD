# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""运行宇树 G1 站立任务 - 零动作，机器人在场景正中以默认姿态站立。"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# 将工作区根目录加入路径，以便导入 robot_mmd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDIA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../media"))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1，场景正中）")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用 fabric，使用 USD I/O。"
)
parser.add_argument(
    "--motion_csv",
    type=str,
    default=os.path.join(_MEDIA_DIR, "you_are_important.csv"),
    help="CSV 动作文件路径（默认: media/you_are_important.csv）",
)
parser.add_argument(
    "--motion_playback",
    default=True,
    action="store_true",
    help="动作回放模式：不固定根链接、禁用重力、增加阻尼，便于观察动作且不摔倒",
)
parser.add_argument(
    "--play_speed",
    type=float,
    default=0.5,
    help="动作播放速度倍率（默认 0.5，即半速；1.0 为原速）",
)
parser.add_argument(
    "--smooth_alpha",
    type=float,
    default=0.3,
    help="动作平滑系数 0~1（默认 0.3，越小越平滑；0 表示不更新，1 表示无平滑）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cpu"


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import robot_mmd.my_task  # noqa: F401 - 注册任务
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from robot_mmd.my_task.g1_stand_env_cfg import get_robot_cfg_for_motion_playback
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from robot_mmd.train_workflow.csv_motion_loader import (
    build_joint_positions_from_frame,
    get_bone_frame_lists,
    get_frame_indices,
    interpolate_bone,
    load_csv_motion,
)

TASK_ID = "Isaac-G1-Stand-v0"

reset_requested = False
play_motion_requested = False


def reset_cb():
    global reset_requested
    reset_requested = True


def play_motion_cb():
    global play_motion_requested
    play_motion_requested = True


def main():
    """零动作运行 G1 站立环境。"""
    global reset_requested, play_motion_requested

    motion_csv = args_cli.motion_csv
    if motion_csv is None:
        motion_csv = os.path.join(_MEDIA_DIR, "you_are_important.csv")
    if not os.path.isfile(motion_csv):
        print(f"[WARN] 未找到动作文件: {motion_csv}，U 键播放功能不可用")
    else:
        print(f"[INFO] 未成功加载动作: {motion_csv}")

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
        print("[INFO] 已启用动作回放模式：不固定根链接、禁用重力、增加阻尼")
    env = gym.make(TASK_ID, cfg=env_cfg)

    print(f"[INFO] 观测空间: {env.observation_space}")
    print(f"[INFO] 动作空间: {env.action_space}")
    print(
        f"[INFO] G1 站立任务 - 零动作运行中... 按 L 键重置环境，按 U 键播放 CSV 动作"
    )
    if args_cli.motion_playback:
        print(f"[INFO] 播放速度: {args_cli.play_speed}x, 平滑系数: {args_cli.smooth_alpha}")

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    keyboard.add_callback("L", reset_cb)
    keyboard.add_callback("U", play_motion_cb)
    env.reset()
    keyboard.reset()

    # 动作播放状态
    motion_frames = None
    frame_list: list[int] = []
    bone_frame_lists: dict[str, list[int]] = {}
    all_bones: set[str] = set()
    play_start_step = 0
    is_playing = False
    last_printed_frame = -1
    action_scale = env_cfg.actions.joint_pos.scale
    joint_names = []
    default_joint_pos = None
    smoothed_action = None  # 用于动作平滑

    if os.path.isfile(motion_csv):
        motion_frames = load_csv_motion(motion_csv)
        frame_list = get_frame_indices(motion_frames)
        for f in motion_frames.values():
            all_bones.update(f.keys())
        bone_frame_lists = get_bone_frame_lists(motion_frames, frame_list, all_bones)
        print(f"[INFO] 已加载动作: {motion_csv}，共 {len(frame_list)} 个关键帧")

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                env.reset()
                keyboard.reset()
                is_playing = False
                smoothed_action = None
                print("[INFO] 环境已重置")

            if play_motion_requested and motion_frames is not None:
                play_motion_requested = False
                is_playing = True
                play_start_step = step_count
                last_printed_frame = -1
                smoothed_action = None  # 新播放时重置平滑状态
                # 获取关节信息（首次播放时）
                if not joint_names:
                    action_term = env.unwrapped.action_manager.get_term("joint_pos")
                    joint_names = action_term._joint_names
                    default_joint_pos = (
                        env.unwrapped.scene["robot"]
                        .data.default_joint_pos[0, action_term._joint_ids]
                        .cpu()
                        .numpy()
                    )
                print("[INFO] 开始播放 CSV 动作")

            if is_playing and motion_frames is not None and frame_list:
                # 按时间计算当前帧：VMD 30fps，sim 每步约 0.02s，play_speed 控制播放速度
                elapsed_steps = step_count - play_start_step
                frame = int(elapsed_steps * 0.02 * 30 * args_cli.play_speed)
                max_frame = frame_list[-1]
                if frame > max_frame:
                    frame = max_frame

                # 每 10 帧打印一次
                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    print(f"[播放] 帧 {frame}/{max_frame}")

                # 插值得到该帧的骨骼数据（使用预计算的 bone_frame_lists 加速）
                frame_data = {}
                for bone in all_bones:
                    d = interpolate_bone(
                        frame, bone, motion_frames, bone_frame_lists.get(bone)
                    )
                    if d is not None:
                        frame_data[bone] = d

                if frame_data and joint_names is not None and default_joint_pos is not None:
                    target_pos = build_joint_positions_from_frame(
                        frame_data, joint_names, default_joint_pos
                    )
                    # action = (target - default) / scale
                    target_action = (target_pos - default_joint_pos) / action_scale
                    # 动作平滑：指数移动平均，减少突变
                    alpha = args_cli.smooth_alpha
                    if smoothed_action is None:
                        smoothed_action = target_action.copy()
                    else:
                        smoothed_action = alpha * target_action + (1.0 - alpha) * smoothed_action
                    action = smoothed_action
                    actions = torch.tensor(
                        action,
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    ).unsqueeze(0)
                else:
                    actions = torch.zeros(
                        env.action_space.shape, device=env.unwrapped.device
                    )
            else:
                actions = torch.zeros(
                    env.action_space.shape, device=env.unwrapped.device
                )

            env.step(actions)
            step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
