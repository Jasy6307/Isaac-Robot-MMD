# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 dance tracking PPO trainer (Isaac Lab + RSL-RL).

Defaults to the C0 fixed-root smoke task, training on the first 10 seconds of
``robot_mmd/media/dance/you_are_important.h5``.

Example
-------
.. code-block:: powershell

    & C:/Users/Administrator/.conda/envs/env_isaaclab_mmd/python.exe `
      i:/robot_isaac/robot_mmd/train_workflow/scripts/train_g1_dance_track.py `
      --task Isaac-G1-Dance-Track-C0-v0 --num_envs 512 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train G1 dance tracking PPO with RSL-RL.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-G1-Dance-Track-C0-v0",
    help="Registered Isaac Lab task ID.",
)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel envs.")
parser.add_argument(
    "--max_iterations", type=int, default=None, help="Override PPO iterations."
)
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument(
    "--motion_h5",
    type=str,
    default=None,
    help="Override the dance HDF5 path. Defaults to env cfg's value.",
)
parser.add_argument(
    "--window_seconds",
    type=float,
    default=None,
    help="Override the reference window length (seconds). Must match episode length.",
)
parser.add_argument(
    "--experiment_suffix",
    type=str,
    default=None,
    help="Optional suffix appended to the experiment name (subfolder under logs/).",
)
parser.add_argument(
    "--resume", action="store_true", default=False, help="Resume from a checkpoint."
)
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file.")
parser.add_argument(
    "--video",
    action="store_true",
    default=True,
    help="Record videos during training (requires rendering; do not use --headless).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=450,
    help="Recorded clip length in env steps (300 ≈ 10s at 30Hz control).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="Env-step interval between recordings. Overrides --video_every_save if set.",
)
parser.add_argument(
    "--video_every_save",
    action="store_true",
    default=True,
    help="Record once per checkpoint save (interval = save_interval * num_steps_per_env).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Launch Isaac Sim (must happen before importing isaaclab modules with USD bindings).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import robot_mmd.my_task  # noqa: F401, E402  -- register Isaac-G1-* tasks


def _apply_motion_overrides(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Patch env_cfg with --motion_h5 / --window_seconds overrides."""
    if args_cli.motion_h5 is None and args_cli.window_seconds is None:
        return

    new_h5 = (
        os.path.abspath(args_cli.motion_h5)
        if args_cli.motion_h5 is not None
        else None
    )
    new_ws = args_cli.window_seconds

    if new_ws is not None:
        env_cfg.episode_length_s = float(new_ws)

    def _patch(params: dict) -> None:
        if "h5_path" in params and new_h5 is not None:
            params["h5_path"] = new_h5
        if "window_seconds" in params and new_ws is not None:
            params["window_seconds"] = float(new_ws)

    # observations
    pol = env_cfg.observations.policy
    for term_name in dir(pol):
        term = getattr(pol, term_name, None)
        if term is not None and hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
    # rewards
    for term_name in vars(env_cfg.rewards):
        term = getattr(env_cfg.rewards, term_name)
        if hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
    # events
    for term_name in vars(env_cfg.events):
        term = getattr(env_cfg.events, term_name)
        if hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(  # type: ignore[assignment]
        args_cli.task, "env_cfg_entry_point"
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(  # type: ignore[assignment]
        args_cli.task, "rsl_rl_cfg_entry_point"
    )

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    _apply_motion_overrides(env_cfg)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.seed = int(args_cli.seed)
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    exp_name = agent_cfg.experiment_name
    if args_cli.experiment_suffix:
        exp_name = f"{exp_name}_{args_cli.experiment_suffix}"
        agent_cfg.experiment_name = exp_name
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", exp_name))
    os.makedirs(log_root_path, exist_ok=True)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Run directory: {log_dir}")

    env_cfg.log_dir = log_dir
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = False
        env_cfg.io_descriptors_output_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    else:
        resume_path = None

    if args_cli.video:
        if args_cli.video_interval is not None:
            video_interval = int(args_cli.video_interval)
        elif args_cli.video_every_save:
            video_interval = int(agent_cfg.save_interval) * int(agent_cfg.num_steps_per_env)
        else:
            video_interval = 2000
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step, interval=video_interval: step % interval == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
