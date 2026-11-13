"""Build a fresh Auto-Jump config and check rewards at a reference time."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from source import sim_compat
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--h5", required=True)
parser.add_argument("--task", default="Isaac-G1-Vmd-Auto-Jump-Conditional-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--motion_start_seconds", type=float, default=0.)
parser.add_argument("--window_seconds", type=float)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args = parser.parse_args()
from source.paths import resolve_dance_asset
if args.h5:
    args.h5 = str(resolve_dance_asset(args.h5))
sim_compat.prepare_launcher_args(args, default_visualizer="none")
app = AppLauncher(args).app

import gymnasium as gym
import source.my_task
from isaaclab_tasks.utils import parse_env_cfg
from source.train_workflow.utils.auto_jump_training import prepare_auto_config


def _set_h5(cfg, h5_path, duration):
    h5_path = str(Path(h5_path).resolve())
    cfg.actions.joint_pos.motion_h5_path = h5_path
    cfg.actions.joint_pos.motion_window_seconds = duration
    for container in (cfg.observations.policy, cfg.rewards, cfg.events):
        for term in vars(container).values():
            params = getattr(term, "params", None)
            if params is not None and "h5_path" in params:
                params["h5_path"] = h5_path
                if "window_seconds" in params:
                    params["window_seconds"] = duration


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    import h5py
    with h5py.File(args.h5) as motion:
        duration = args.window_seconds or len(motion["joint_pos_delta"])/float(motion.attrs["fps"])
    _set_h5(cfg, args.h5, duration)
    cfg.sim.physics.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physics.gpu_max_rigid_contact_count = 2**20
    report = prepare_auto_config(cfg, str(Path(args.h5).resolve()), duration, hold_seconds=2.0)
    cfg.events.reset_robot_joints.params.update(motion_start_seconds=args.motion_start_seconds)
    env = gym.make(args.task, cfg=cfg)
    try:
        obs, _ = env.reset()
        names = list(getattr(env.unwrapped.reward_manager, "active_terms", []))
        print("AUTO_JUMP_CFG_PASS", {"gate": {
            "pre": cfg.jump_gate_pre_seconds,
            "post": cfg.jump_gate_post_seconds,
            "ramp": cfg.jump_gate_ramp_seconds,
        }, "reward_terms": names, "steps": args.steps}, flush=True)
        import torch
        from source.train_workflow.utils.auto_jump_training import AutoJumpController
        env.unwrapped._auto_jump_controller = AutoJumpController(env.unwrapped, report, REPO_ROOT/"output/reference-reward-smoke")
        env.unwrapped._auto_jump_controller.configure(.25)
        action_dim = env.unwrapped.action_manager.total_action_dim
        actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        for _ in range(args.steps):
            env.step(actions)
        from source.my_task.mdp.auto_jump import swing_metrics
        metrics = swing_metrics(env.unwrapped)
        print("SWING_RUNTIME_PASS", {key: value.detach().cpu().tolist() for key,value in metrics.items()}, flush=True)
        from source.my_task.mdp.reference_rewards import metrics as reference_metrics
        scores = reference_metrics(env.unwrapped)
        assert all(torch.isfinite(value).all() for value in scores.values()), scores
        print("REFERENCE_REWARDS_PASS", {key: value.detach().cpu().tolist() for key,value in scores.items()}, flush=True)
        assert len(names) == 13, names
        if args.motion_start_seconds + args.steps*env.unwrapped.step_dt >= duration:
            from source.my_task.motion_reference import motion_steps
            from source.my_task.mdp.reference_rewards import _tables
            table = _tables(env.unwrapped)
            current = motion_steps(env.unwrapped)
            assert (current >= table["buf"].num_steps).all(), current
            assert (table["vel"][-1] == 0).all()
            assert (table["buf"].root_pos_delta(current+1)-table["buf"].root_pos_delta(current) == 0).all()
            print("REFERENCE_HOLD_PASS", current.detach().cpu().tolist(), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
