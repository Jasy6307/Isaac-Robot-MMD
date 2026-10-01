"""Minimal local G1 scene on Sim 6.1 / Lab 3; no dance assets required.

Run ``run_isaac61.bat`` for a window, or pass this file with ``--viz none``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from source import sim_compat
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=300, help="Number of physics steps; 0 keeps the window open.")
parser.add_argument("--task", choices=("replay", "train"), default="replay")
parser.add_argument("--h5", help="Optional H5 reference for a small training-task smoke test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--screenshot", action="store_true", help="Save the Kit viewport to output/isaac61/smoke.png.")
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args = parser.parse_args()
sim_compat.prepare_launcher_args(args)
app_launcher = AppLauncher(args)
app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
import source.my_task
from isaaclab_tasks.utils import parse_env_cfg
from source.my_task.robots.g1_29dof_o6_cfg import G1_29DOF_O6_CFG
from source.my_task.robots.actuator_pd import apply_robot_pd_profile
from source.train_workflow.utils.playback.sim_robot import apply_joint_state_instant, apply_root_pos_instant


def main():
    task_id = "Isaac-G1-Vmd-Replay-v0" if args.task == "replay" else "Isaac-G1-Vmd-Train-C1-v0"
    cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=1500.0)
    cfg.sim.default_visualizer_cfg.eye = (0.0, 7.0, 3.0)
    cfg.sim.default_visualizer_cfg.lookat = (0.0, 0.0, 1.2)
    cfg.scene.robot = apply_robot_pd_profile(
        G1_29DOF_O6_CFG.replace(prim_path=cfg.scene.robot.prim_path, init_state=cfg.scene.robot.init_state),
        "deploy", o6_hands=True,
    )
    if args.task == "replay":
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    else:
        if args.h5:
            for container in (cfg.observations.policy, cfg.rewards, cfg.events):
                for term in vars(container).values():
                    params = getattr(term, "params", None)
                    if params and "h5_path" in params:
                        params["h5_path"] = str(Path(args.h5).resolve())
            cfg.actions.joint_pos.motion_h5_path = str(Path(args.h5).resolve())
        # A single environment does not need the large training contact allocation.
        cfg.sim.physics.gpu_max_rigid_patch_count = 2**16
        cfg.sim.physics.gpu_max_rigid_contact_count = 2**20
    cfg.sim.log_dir = str(REPO_ROOT / "output" / "isaac61")
    env = gym.make(task_id, cfg=cfg)
    try:
        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        actions = torch.zeros((env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
        if args.task == "replay":
            # Exercise the migrated playback writers and a non-identity WXYZ root rotation.
            root = robot.data.root_state_w.torch[0].detach().cpu()
            assert apply_root_pos_instant(env, tuple(root[:3].tolist()), [0.92387953, 0.0, 0.0, 0.38268343])
            expected_q = torch.tensor([0.0, 0.0, 0.38268343, 0.92387953], device=env.unwrapped.device)
            actual_q = robot.data.root_state_w.torch[0, 3:7]
            if abs(float(torch.dot(actual_q, expected_q))) < 0.9999:
                raise RuntimeError("WXYZ motion pose was not written as the expected XYZW simulation pose")
            assert apply_joint_state_instant(env, robot.data.default_joint_pos.torch[0].tolist(), slice(None))
        completed = 0
        while app.is_running() and (args.steps == 0 or completed < args.steps):
            obs, rewards, terminated, truncated, _ = env.step(actions)
            if not torch.isfinite(robot.data.root_state_w.torch).all():
                raise RuntimeError("Non-finite robot root state")
            if not torch.isfinite(robot.data.joint_pos.torch).all():
                raise RuntimeError("Non-finite joint positions")
            completed += 1
            if not args.headless:
                time.sleep(cfg.sim.dt)
        if args.steps and completed < args.steps:
            raise RuntimeError(f"Window closed before smoke test finished: {completed}/{args.steps}")
        report = {
            "task": task_id, "steps": completed, "num_envs": env.unwrapped.num_envs,
            "python": sys.version.split()[0],
            "isaacsim_build": (Path(os.environ["ISAACSIM_PATH"]) / "VERSION").read_text().strip(),
            "visualizer": args.visualizer,
            "joint_count": robot.num_joints, "action_dim": actions.shape[1],
            "device": str(env.unwrapped.device), "root_xyzw": robot.data.root_state_w.torch[0, :7].detach().cpu().tolist(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        output = REPO_ROOT / "output" / "isaac61"
        output.mkdir(parents=True, exist_ok=True)
        if args.screenshot:
            if args.headless:
                raise ValueError("--screenshot requires --viz kit")
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            capture = capture_viewport_to_file(get_active_viewport(), str(output / "smoke.png"))
            for _ in range(30):
                app.update()
            if not (output / "smoke.png").is_file():
                raise RuntimeError("Viewport screenshot was not written")
        (output / f"smoke_{args.task}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("ISAAC61_SMOKE_PASS " + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
