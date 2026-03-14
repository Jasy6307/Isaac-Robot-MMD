# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""运行宇树 G1 站立任务 - 零动作，机器人在场景正中以默认姿态站立。"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# 将工作区根目录加入路径，以便导入 robot_mmd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1，场景正中）")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用 fabric，使用 USD I/O。"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import robot_mmd.my_task  # noqa: F401 - 注册任务
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

TASK_ID = "Isaac-G1-Stand-v0"

reset_requested = False


def reset_cb():
    global reset_requested
    reset_requested = True


def main():
    """零动作运行 G1 站立环境。"""
    global reset_requested
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(TASK_ID, cfg=env_cfg)

    print(f"[INFO] 观测空间: {env.observation_space}")
    print(f"[INFO] 动作空间: {env.action_space}")
    print(f"[INFO] G1 站立任务 - 零动作运行中... 按 L 键重置环境")

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    keyboard.add_callback("L", reset_cb)
    env.reset()
    keyboard.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                env.reset()
                keyboard.reset()
                print("[INFO] 环境已重置")
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
