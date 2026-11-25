# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 dance tracking PPO trainer (Isaac Lab + RSL-RL).

Trains the G1 dance tracking task with an automatic curriculum and validation.
Select a reference with ``--dance`` or ``--motion_h5``.

Example
-------
.. code-block:: powershell

    python source/train_workflow/g1_vmd_1_train.py `
      --num_envs 2048 `
      --dance IRIS_OUT `
      --physics_hz 120 `
      --decimation 4 `
      --window_frames 230 `
      --max_iterations 10000 `
      --run_note standard_30hz_half `
      --auto_motion_feedback `
      --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from source.train_workflow.task_ids import G1_TRAIN_TASK_ID, normalize_g1_train_task_id  # noqa: E402
from source.train_workflow.utils.motion.resolve import (  # noqa: E402
    resolve_training_log_root,
    resolve_training_motion,
)
from source.train_workflow.utils.run_note import create_run_note, run_note_filename  # noqa: E402
from source.train_workflow.utils.checkpoints import resolve_training_checkpoint  # noqa: E402
from source.train_workflow.utils.validation_reference import (  # noqa: E402
    add_reference_video_arguments, reference_video_from_args,
)
from isaaclab.app import AppLauncher  # noqa: E402
from source import sim_compat

parser = argparse.ArgumentParser(description="Train G1 dance tracking PPO with RSL-RL.")
parser.add_argument(
    "--task",
    type=normalize_g1_train_task_id,
    choices=(G1_TRAIN_TASK_ID,),
    default=G1_TRAIN_TASK_ID,
    help="G1 dance tracking task (default).",
)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel envs.")
parser.add_argument(
    "--max_iterations", type=int, default=None, help="Override PPO iterations."
)
parser.add_argument("--num_steps_per_env",type=int,default=None,
                    help="Control steps collected per environment per PPO iteration; episodes continue across iterations. Auto-Jump default: 48.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument(
    "--physics_hz",
    type=float,
    default=None,
    help="Override training physics frequency (sim.dt = 1 / physics_hz).",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Override physics steps per policy action (120 Hz physics / 4 = 30 Hz control).",
)
parser.add_argument(
    "--dance",
    type=str,
    default=None,
    help=(
        "Dance name under media/dance/ (e.g. IRIS_OUT). "
        "Auto-resolves HDF5, preferring *_z_editted.h5. With --motion_h5, use this as the dance label."
    ),
)
parser.add_argument(
    "--motion_h5",
    type=str,
    help="Explicit reference H5 path, for example the arm_stabilized.h5 from g1-prepare-motion.",
)
parser.add_argument(
    "--window_frames",
    type=int,
    default=None,
    help="Override reference motion window length (control-step frame count).",
)
parser.add_argument(
    "--episode_full_hold_seconds",
    type=float,
    default=None,
    help="Seconds to hold the last reference frame after the full motion (default: 2.0).",
)
parser.add_argument(
    "--residual_alpha",
    type=float,
    default=None,
    help="Override residual gain alpha for residual joint actions.",
)
parser.add_argument(
    "--use_reference_residual",
    dest="use_reference_residual",
    action="store_true",
    default=None,
    help="Enable reference residual action mode when supported by task action cfg.",
)
parser.add_argument(
    "--no_use_reference_residual",
    dest="use_reference_residual",
    action="store_false",
    help="Disable reference residual action mode when supported by task action cfg.",
)
parser.add_argument(
    "--run_note",
    type=run_note_filename,
    default=None,
    help="Create an empty file with this exact name inside the new run log directory; does not change the experiment name.",
)
parser.add_argument(
    "--experiment_suffix",
    type=str,
    default=None,
    help="Optional suffix appended to the experiment name (subfolder under logs/).",
)
parser.add_argument(
    "--pd_profile",
    type=str,
    choices=("deploy", "isaaclab"),
    default="deploy",
    help=(
        "Robot actuator PD: deploy=Unitree rl_lab FixStand (default, sim-to-real); "
        "isaaclab=G1_29DOF locomanipulation defaults (legacy checkpoints only)."
    ),
)
parser.add_argument(
    "--self_collisions",
    dest="self_collisions",
    action="store_true",
    default=None,
    help="Enable robot self-collisions during training.",
)
parser.add_argument(
    "--no_self_collisions",
    dest="self_collisions",
    action="store_false",
    help="Disable robot self-collisions during training.",
)
parser.add_argument(
    "--resume", action="store_true", default=False, help="Resume from a checkpoint."
)
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file.")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Source run directory; omitted --checkpoint selects its latest .pt (requires --resume).")
parser.add_argument("--auto_jump_interval", type=int, default=250,
                    help="Shared PPO iteration interval for sampling updates and validation (positive; default 250).")
parser.add_argument("--auto_jump_probe_start_fraction", type=float, default=0.2,
                    help="Skip full validation and video until this fraction of the cumulative iteration budget. 0 keeps every probe.")
parser.add_argument("--auto_jump_review", type=str, default=None, help="Reference-hash-bound support review JSON.")
parser.add_argument("--auto_jump_repair", action="store_true", help="Low-LR continuation mixing support reviews, retained jumps and full motion.")
parser.add_argument("--auto_jump_tail_only", action="store_true",
                    help="Train and validate only the final high-confidence jump window; no earlier motion is sampled.")
parser.add_argument("--auto_jump_tail_start_seconds", type=float, default=None,
                    help="Tail-only global H5 start time in seconds (default: 1s before the last high-confidence jump).")
parser.add_argument("--turn_smoothing",action="store_true",help="Opt in to supported-turn residual rate/acceleration rewards; preserves jump transition margins.")
parser.add_argument("--auto_motion_feedback",action="store_true",help="Reference-aware full-motion slip/chatter candidates, bounded rehearsal feedback and auxiliary best ranking.")
parser.add_argument("--auto_jump_check_starts", type=float, nargs="+", default=None,
                    help="Optional independent regression clip starts before/after repair.")
parser.add_argument("--auto_quality_profile", type=str, default=None,
                    help="Frozen reference-bound quality JSON; default uses a matching accepted profile or generic limits.")
parser.add_argument("--auto_quality_patience", type=int, default=3,
                    help="Consecutive qualified checkpoints required to recommend manual visual review; never stops training.")
parser.add_argument("--no_auto_validation_record",dest="auto_validation_record",action="store_false",default=True,
                    help="Disable saving actual full-validation poses for later video rendering (enabled by default, headless-compatible).")
parser.add_argument("--no_auto_validation_video",dest="auto_validation_video",action="store_false",default=True,
                    help="Only save validation poses; disable the single synchronized multi-seed MP4 after each parallel full check.")
add_reference_video_arguments(parser)
parser.add_argument("--no_auto_quality_early_stop", dest="auto_quality_early_stop",action="store_false",default=False,
                    help="Compatibility option: training always continues to the iteration limit or a manual stop.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record videos during training (requires rendering; do not use --headless).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Recorded clip length in env steps (300 = 10s at the default 30Hz control).",
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
sim_compat.add_legacy_headless_arg(parser)
args_cli = parser.parse_args()
if args_cli.num_steps_per_env is not None and args_cli.num_steps_per_env < 1:
    parser.error("--num_steps_per_env must be positive")
if args_cli.checkpoint_dir and not args_cli.resume:
    parser.error("--checkpoint_dir requires --resume")
if args_cli.auto_jump_interval < 1:
    parser.error("--auto_jump_interval must be positive")
if not 0 <= args_cli.auto_jump_probe_start_fraction <= 1:
    parser.error("--auto_jump_probe_start_fraction must be between 0 and 1")
if args_cli.auto_quality_patience < 1:
    parser.error("--auto_quality_patience must be positive")
if args_cli.physics_hz is not None and (
    not math.isfinite(args_cli.physics_hz) or args_cli.physics_hz <= 0.0
):
    parser.error("--physics_hz must be finite and > 0")
if args_cli.decimation is not None and args_cli.decimation < 1:
    parser.error("--decimation must be >= 1")
sim_compat.prepare_launcher_args(args_cli, default_visualizer="kit" if args_cli.video else "none")
if args_cli.video and args_cli.headless:
    parser.error("--video requires the Kit visualizer; omit --headless/--viz none")

try:
    MOTION_H5_PATH, DANCE_NAME = resolve_training_motion(args_cli.dance, args_cli.motion_h5)
except (OSError, ValueError, TypeError) as error:
    parser.error(str(error))
if MOTION_H5_PATH is None:
    parser.error("G1 dance tracking requires --dance or --motion_h5")
try:
    args_cli.validation_reference = reference_video_from_args(args_cli, MOTION_H5_PATH)
except (OSError, ValueError, TypeError) as error:
    parser.error(str(error))
if args_cli.motion_h5 is not None:
    print(f"[INFO] Training reference motion: {MOTION_H5_PATH}", flush=True)
if args_cli.validation_reference:
    print(f"[PMX对照视频] {args_cli.validation_reference['video']} | "
          f"动作零点对应视频 {args_cli.validation_reference['start_seconds']:g}s", flush=True)

if args_cli.video:
    args_cli.enable_cameras = True

# Launch Isaac Sim (must happen before importing isaaclab modules with USD bindings).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import source.my_task  # noqa: F401, E402  -- register Isaac-G1-* tasks
from source.my_task.g1_train_env_cfg import C1_FULL_WINDOW_HOLD_SECONDS  # noqa: E402
from source.my_task.mdp.jump import validate_ground_contacts  # noqa: E402
from source.my_task.robots.actuator_pd import (  # noqa: E402
    apply_pd_profile_to_scene_robot,
    log_pd_profile_summary,
)
from source.train_workflow.utils.auto_jump_training import prepare_auto_config, train_auto  # noqa: E402
from source.train_workflow.utils.motion.window import (  # noqa: E402
    control_hz_from_env_cfg,
    log_window_frames_override,
    window_seconds_from_frames,
)


def _apply_motion_overrides(env_cfg: ManagerBasedRLEnvCfg, window_seconds: float) -> None:
    """Use the same H5 and window for all reference consumers."""
    for container in (env_cfg.observations.policy, env_cfg.rewards, env_cfg.events, env_cfg.terminations):
        for name in dir(container):
            params = getattr(getattr(container, name), "params", None)
            if isinstance(params, dict):
                if "h5_path" in params:
                    params["h5_path"] = MOTION_H5_PATH
                if "window_seconds" in params:
                    params["window_seconds"] = window_seconds

    action = env_cfg.actions.joint_pos
    action.motion_h5_path = MOTION_H5_PATH
    action.motion_window_seconds = window_seconds
    if args_cli.residual_alpha is not None:
        action.residual_alpha = args_cli.residual_alpha
    if args_cli.use_reference_residual is not None:
        action.use_reference_residual = args_cli.use_reference_residual


def _apply_self_collision_override(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Optionally override robot self-collision from CLI."""
    if args_cli.self_collisions is None:
        return
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is None:
        print("[WARN] --self_collisions ignored: env_cfg.scene.robot unavailable.")
        return
    spawn = getattr(robot_cfg, "spawn", None)
    if spawn is None:
        print("[WARN] --self_collisions ignored: robot spawn unavailable.")
        return
    articulation_props = getattr(spawn, "articulation_props", None)
    if articulation_props is None:
        print("[WARN] --self_collisions ignored: robot spawn.articulation_props unavailable.")
        return
    enabled = bool(args_cli.self_collisions)
    env_cfg.scene.robot = robot_cfg.replace(
        spawn=spawn.replace(
            articulation_props=articulation_props.replace(enabled_self_collisions=enabled)
        )
    )
    print(f"[INFO] Robot self-collisions override: enabled_self_collisions={enabled}")


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task)
    if args_cli.physics_hz is not None:
        env_cfg.sim.dt = 1.0 / args_cli.physics_hz
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    print(
        f"[INFO] Train timing: sim.dt = {env_cfg.sim.dt:.9f}s; "
        f"physics_hz = {1.0 / env_cfg.sim.dt:.1f}; decimation = {env_cfg.decimation}; "
        f"render_interval = {env_cfg.sim.render_interval}; "
        f"render_hz = {1.0 / (env_cfg.sim.dt * env_cfg.sim.render_interval):.1f}; "
        f"control_hz = {control_hz_from_env_cfg(env_cfg):.1f}"
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(  # type: ignore[assignment]
        args_cli.task, "rsl_rl_cfg_entry_point"
    )
    agent_cfg = sim_compat.prepare_rsl_rl_cfg(agent_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot = apply_pd_profile_to_scene_robot(
        env_cfg.scene.robot, args_cli.pd_profile, o6_hands=True
    )
    _apply_self_collision_override(env_cfg)
    log_pd_profile_summary(args_cli.pd_profile, o6_hands=True)

    control_hz = control_hz_from_env_cfg(env_cfg)
    window_frames = args_cli.window_frames
    if window_frames is None:
        with h5py.File(MOTION_H5_PATH) as motion:
            duration = len(motion["joint_pos_delta"]) / float(motion.attrs["fps"])
        window_frames = math.floor(duration * control_hz + 1e-6)
    window_seconds = window_seconds_from_frames(window_frames, control_hz)
    log_window_frames_override(window_frames, window_seconds, control_hz)
    _apply_motion_overrides(env_cfg, window_seconds)

    hold_seconds = args_cli.episode_full_hold_seconds
    if hold_seconds is None:
        hold_seconds = C1_FULL_WINDOW_HOLD_SECONDS
    auto_report = prepare_auto_config(
        env_cfg,
        MOTION_H5_PATH,
        window_seconds,
        args_cli.auto_jump_review,
        hold_seconds=hold_seconds,
        tail_only=args_cli.auto_jump_tail_only,
        tail_start_seconds=args_cli.auto_jump_tail_start_seconds,
    )
    print(
        f"[AUTO HOLD] Motion {auto_report['duration_s']:.2f}s + last-frame hold "
        f"{auto_report['end_hold_seconds']:.2f}s = full episode {auto_report['episode_duration_s']:.2f}s"
    )
    if args_cli.auto_jump_repair:
        if not args_cli.resume or not auto_report["support_windows"]:
            raise ValueError("Repair requires --resume and --auto_jump_review with support windows")
        auto_report["repair_training"] = True
        agent_cfg.num_steps_per_env = 30

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = int(args_cli.max_iterations)
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
    if args_cli.turn_smoothing:
        from source.my_task.mdp.turn_smoothing import configure

        configure(env_cfg)
    if args_cli.auto_motion_feedback:
        env_cfg.auto_motion_feedback = True
    if agent_cfg.max_iterations < 1:
        raise ValueError("G1 dance tracking requires at least one learning iteration")
    batch_steps=int(env_cfg.scene.num_envs)*int(agent_cfg.num_steps_per_env)
    print(f"[PPO采样] 每环境每iter {agent_cfg.num_steps_per_env} 控制步 "
          f"({agent_cfg.num_steps_per_env/control_hz_from_env_cfg(env_cfg):.2f}s) | "
          f"{env_cfg.scene.num_envs}环境 | 每iter {batch_steps:,}条样本 | "
          f"{agent_cfg.algorithm.num_learning_epochs}轮学习 / {agent_cfg.algorithm.num_mini_batches}个小批次")
    print(f"[PPO预算] 本次 {agent_cfg.max_iterations} iter，共 {batch_steps*int(agent_cfg.max_iterations):,}条训练样本（不含验收）；"
          "采样段结束后继续原回合，不因iter结束重置机器人")
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
    resume_path = None
    if agent_cfg.resume:
        resume_path = resolve_training_checkpoint(
            exp_name,
            DANCE_NAME,
            load_run=args_cli.load_run,
            checkpoint=args_cli.checkpoint,
            checkpoint_dir=args_cli.checkpoint_dir,
            run_pattern=agent_cfg.load_run,
            checkpoint_pattern=agent_cfg.load_checkpoint,
        )
    log_root_path = resolve_training_log_root(exp_name, DANCE_NAME)
    os.makedirs(log_root_path, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_stamp += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, run_stamp)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if DANCE_NAME:
        print(f"[INFO] Dance: {DANCE_NAME}")
    print(f"[INFO] Run directory: {log_dir}")
    if args_cli.run_note is not None:
        note_path = create_run_note(log_dir, args_cli.run_note)
        print(f"[INFO] Run note: {note_path}")

    env_cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "auto_jump_candidates.json"), "w", encoding="utf-8") as report_file:
        json.dump(auto_report, report_file, indent=2)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = False
        env_cfg.io_descriptors_output_dir = log_dir

    if args_cli.video:
        if args_cli.video_interval is not None:
            video_interval = int(args_cli.video_interval)
        elif args_cli.video_every_save:
            video_interval = int(agent_cfg.save_interval) * int(agent_cfg.num_steps_per_env)
        else:
            video_interval = 2000
        env_cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:kit", output_dir=os.path.join(log_dir, "videos", "train"),
            video_interval=video_interval, video_length=int(args_cli.video_length),
        )]

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    validate_ground_contacts(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    resume_infos = None
    if resume_path is not None:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        if args_cli.auto_jump_repair:
            resume_infos = runner.load(
                resume_path,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True},
            )
        else:
            resume_infos = runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    train_auto(runner, env, env_cfg, agent_cfg, auto_report, log_dir, args_cli, resume_infos)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
