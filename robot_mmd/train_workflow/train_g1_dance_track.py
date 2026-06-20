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
from dataclasses import dataclass

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
    "--window_frames",
    type=int,
    default=None,
    help="Override reference motion window length (control-step frame count).",
)
parser.add_argument(
    "--episode_seconds",
    type=float,
    default=None,
    help="Override episode length (seconds), independent from reference window.",
)
parser.add_argument(
    "--random_episode_length",
    dest="random_episode_length",
    action="store_true",
    default=None,
    help="Enable per-reset random episode length sampling.",
)
parser.add_argument(
    "--no_random_episode_length",
    dest="random_episode_length",
    action="store_false",
    help="Disable random episode length sampling.",
)
parser.add_argument(
    "--episode_min_seconds",
    type=float,
    default=None,
    help="Minimum sampled episode length when random episode length is enabled.",
)
parser.add_argument(
    "--episode_max_seconds",
    type=float,
    default=None,
    help="Maximum sampled episode length when random episode length is enabled.",
)
parser.add_argument(
    "--episode_length_curriculum",
    action="store_true",
    default=False,
    help="Enable automatic episode-length curriculum in one training run.",
)
parser.add_argument(
    "--episode_length_curriculum_spec",
    type=str,
    default=None,
    help=(
        "Curriculum spec: start:sec for fixed length, start:min:max for random range, "
        "or start alone for start-to-end (end = --window_frames). "
        "e.g. 0:3,20000:6 (3s then 6s fixed stages)"
    ),
)
parser.add_argument(
    "--random_motion_start",
    dest="random_motion_start",
    action="store_true",
    default=None,
    help="Enable random motion start at reset (C1 random short-segment training).",
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
    "--no_random_motion_start",
    dest="random_motion_start",
    action="store_false",
    help="Disable random motion start and always reset from frame 0.",
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
    "--resume", action="store_true", default=False, help="Resume from a checkpoint."
)
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file.")
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
from robot_mmd.train_workflow.g1_deploy_actuator_cfg import (  # noqa: E402
    apply_pd_profile_to_scene_robot,
    log_pd_profile_summary,
)
from robot_mmd.train_workflow.utils.motion_window import (  # noqa: E402
    control_hz_from_env_cfg,
    default_window_seconds_from_env_cfg,
    log_window_frames_override,
    resolve_motion_window_seconds,
)


@dataclass(frozen=True)
class EpisodeLengthStage:
    start_iter: int
    min_seconds: float | None = None
    max_seconds: float | None = None

    @property
    def start_to_end(self) -> bool:
        return self.min_seconds is None and self.max_seconds is None

    @property
    def is_fixed_length(self) -> bool:
        if self.start_to_end or self.min_seconds is None or self.max_seconds is None:
            return False
        return abs(self.min_seconds - self.max_seconds) < 1e-6


def _parse_episode_length_curriculum(spec: str) -> list[EpisodeLengthStage]:
    stages: list[EpisodeLengthStage] = []
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            start_iter = int(parts[0])
            if start_iter < 0:
                raise ValueError(f"start_iter must be >= 0, got {start_iter}")
            stages.append(EpisodeLengthStage(start_iter=start_iter))
            continue
        if len(parts) == 2:
            start_iter = int(parts[0])
            fixed_s = float(parts[1])
            if start_iter < 0:
                raise ValueError(f"start_iter must be >= 0, got {start_iter}")
            if fixed_s <= 0.0:
                raise ValueError(f"Episode seconds must be > 0, got {fixed_s}")
            stages.append(
                EpisodeLengthStage(
                    start_iter=start_iter, min_seconds=fixed_s, max_seconds=fixed_s
                )
            )
            continue
        if len(parts) != 3:
            raise ValueError(
                f"Invalid curriculum chunk '{item}'. "
                "Expected start:seconds (fixed), start:min:max (random range), "
                "or start alone (start-to-end)."
            )
        start_iter = int(parts[0])
        min_s = float(parts[1])
        max_s = float(parts[2])
        if start_iter < 0:
            raise ValueError(f"start_iter must be >= 0, got {start_iter}")
        if min_s <= 0.0 or max_s <= 0.0:
            raise ValueError(f"Episode seconds must be > 0, got {min_s}, {max_s}")
        if min_s > max_s:
            min_s, max_s = max_s, min_s
        stages.append(
            EpisodeLengthStage(start_iter=start_iter, min_seconds=min_s, max_seconds=max_s)
        )
    if not stages:
        raise ValueError("Curriculum spec produced no stages.")
    stages.sort(key=lambda s: s.start_iter)
    if stages[0].start_iter != 0:
        raise ValueError("Curriculum spec must start at iteration 0.")
    for i in range(1, len(stages)):
        if stages[i].start_iter == stages[i - 1].start_iter:
            raise ValueError("Duplicate stage start_iter in curriculum spec.")
    return stages


def _curriculum_schedule(
    stages: list[EpisodeLengthStage], total_iterations: int
) -> list[tuple[EpisodeLengthStage, int]]:
    total = int(total_iterations)
    if total <= 0:
        raise ValueError(f"total_iterations must be > 0, got {total_iterations}")
    schedule: list[tuple[EpisodeLengthStage, int]] = []
    for i, stage in enumerate(stages):
        start = stage.start_iter
        if start >= total:
            break
        next_start = stages[i + 1].start_iter if i + 1 < len(stages) else total
        end = min(next_start, total)
        count = end - start
        if count > 0:
            schedule.append((stage, count))
    if not schedule:
        raise ValueError("Curriculum schedule is empty for the requested total iterations.")
    return schedule


def _set_env_runtime_episode_range(env, min_seconds: float, max_seconds: float) -> None:
    unwrapped = env.unwrapped
    setattr(unwrapped, "_g1_episode_min_seconds", float(min_seconds))
    setattr(unwrapped, "_g1_episode_max_seconds", float(max_seconds))


def _set_env_runtime_start_to_end_mode(
    env, *, enabled: bool, end_seconds: float | None = None
) -> None:
    """Toggle runtime mode: random start, then run until motion-window end."""
    unwrapped = env.unwrapped
    setattr(unwrapped, "_g1_episode_random_start_to_end", bool(enabled))
    if end_seconds is not None:
        setattr(unwrapped, "_g1_episode_end_seconds", float(end_seconds))


def _resolve_motion_window_seconds(env_cfg: ManagerBasedRLEnvCfg) -> float:
    """Motion reference window length used as start-to-end episode end."""
    ws = resolve_motion_window_seconds(env_cfg, window_frames=args_cli.window_frames)
    if ws is not None:
        return ws
    return default_window_seconds_from_env_cfg(env_cfg)


def _resolve_motion_h5_path(env_cfg: ManagerBasedRLEnvCfg) -> str:
    if args_cli.motion_h5 is not None:
        return os.path.abspath(args_cli.motion_h5)
    reset_evt = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_evt is not None and hasattr(reset_evt, "params"):
        hp = reset_evt.params.get("h5_path")
        if hp is not None:
            return str(hp)
    raise ValueError("无法解析 motion_h5，请显式传入 --motion_h5")


def _apply_motion_overrides(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Patch env_cfg with reference-window/episode/random-start overrides."""
    if (
        args_cli.motion_h5 is None
        and args_cli.window_frames is None
        and args_cli.episode_seconds is None
        and args_cli.random_motion_start is None
        and args_cli.random_episode_length is None
        and args_cli.episode_min_seconds is None
        and args_cli.episode_max_seconds is None
        and args_cli.residual_alpha is None
        and args_cli.use_reference_residual is None
    ):
        return

    new_h5 = (
        os.path.abspath(args_cli.motion_h5)
        if args_cli.motion_h5 is not None
        else None
    )
    new_ws = resolve_motion_window_seconds(env_cfg, window_frames=args_cli.window_frames)
    if new_ws is not None and args_cli.window_frames is not None:
        log_window_frames_override(
            int(args_cli.window_frames),
            new_ws,
            control_hz_from_env_cfg(env_cfg),
        )
    new_episode_s = args_cli.episode_seconds
    new_random_start = args_cli.random_motion_start
    new_random_episode_length = args_cli.random_episode_length
    new_episode_min_s = args_cli.episode_min_seconds
    new_episode_max_s = args_cli.episode_max_seconds
    new_residual_alpha = args_cli.residual_alpha
    new_use_reference_residual = args_cli.use_reference_residual

    if new_episode_s is not None:
        env_cfg.episode_length_s = float(new_episode_s)
    elif new_ws is not None:
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
            if new_random_start is not None and "random_start" in term.params:
                term.params["random_start"] = bool(new_random_start)
            if new_episode_s is not None and "segment_seconds" in term.params:
                term.params["segment_seconds"] = float(new_episode_s)
            elif new_ws is not None and "segment_seconds" in term.params:
                term.params["segment_seconds"] = float(new_ws)
            if (
                new_random_episode_length is not None
                and "random_episode_length" in term.params
            ):
                term.params["random_episode_length"] = bool(new_random_episode_length)
            if new_episode_min_s is not None and "episode_min_seconds" in term.params:
                term.params["episode_min_seconds"] = float(new_episode_min_s)
            elif new_ws is not None and "episode_min_seconds" in term.params:
                term.params["episode_min_seconds"] = float(new_ws)
            if new_episode_max_s is not None and "episode_max_seconds" in term.params:
                term.params["episode_max_seconds"] = float(new_episode_max_s)
            elif new_ws is not None and "episode_max_seconds" in term.params:
                term.params["episode_max_seconds"] = float(new_ws)
    # terminations (C2 motion_end_with_hold_time_out must share motion h5/window)
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None:
        for term_name in vars(terminations_cfg):
            term = getattr(terminations_cfg, term_name)
            if hasattr(term, "params") and isinstance(term.params, dict):
                _patch(term.params)
    # C1 residual action (motion reference path)
    joint_pos_action = getattr(env_cfg.actions, "joint_pos", None)
    if joint_pos_action is not None:
        if new_h5 is not None and hasattr(joint_pos_action, "motion_h5_path"):
            joint_pos_action.motion_h5_path = new_h5
        if new_ws is not None and hasattr(joint_pos_action, "motion_window_seconds"):
            joint_pos_action.motion_window_seconds = float(new_ws)
        if new_residual_alpha is not None and hasattr(joint_pos_action, "residual_alpha"):
            joint_pos_action.residual_alpha = float(new_residual_alpha)
        if (
            new_use_reference_residual is not None
            and hasattr(joint_pos_action, "use_reference_residual")
        ):
            joint_pos_action.use_reference_residual = bool(new_use_reference_residual)


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
    env_cfg.scene.robot = apply_pd_profile_to_scene_robot(
        env_cfg.scene.robot, args_cli.pd_profile, o6_hands=True
    )
    log_pd_profile_summary(args_cli.pd_profile, o6_hands=True)
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

    if args_cli.episode_length_curriculum:
        spec = args_cli.episode_length_curriculum_spec
        if spec is None:
            spec = "0:2:4,3000:3:5,6000"
        motion_window_seconds = _resolve_motion_window_seconds(env_cfg)
        stages = _parse_episode_length_curriculum(spec)
        schedule = _curriculum_schedule(stages, int(agent_cfg.max_iterations))
        print(f"[INFO] Episode-length curriculum enabled: {spec}")
        print(f"[INFO] Start-to-end stages use motion window end: {motion_window_seconds:.2f}s")
        for stage, stage_iters in schedule:
            if stage.start_to_end:
                _set_env_runtime_start_to_end_mode(
                    env,
                    enabled=True,
                    end_seconds=motion_window_seconds,
                )
                print(
                    f"[INFO] Curriculum stage start={stage.start_iter} "
                    f"mode=start_to_end end={motion_window_seconds:.2f}s "
                    f"iters={stage_iters}"
                )
            else:
                assert stage.min_seconds is not None and stage.max_seconds is not None
                _set_env_runtime_episode_range(
                    env, min_seconds=stage.min_seconds, max_seconds=stage.max_seconds
                )
                _set_env_runtime_start_to_end_mode(env, enabled=False)
                if stage.is_fixed_length:
                    length_msg = f"fixed={stage.min_seconds:.2f}s"
                else:
                    length_msg = (
                        f"random=[{stage.min_seconds:.2f}, {stage.max_seconds:.2f}]s"
                    )
                print(
                    f"[INFO] Curriculum stage start={stage.start_iter} "
                    f"{length_msg} iters={stage_iters} start_to_end=off"
                )
            runner.learn(
                num_learning_iterations=stage_iters, init_at_random_ep_len=True
            )
    else:
        _set_env_runtime_start_to_end_mode(env, enabled=False)
        if args_cli.episode_min_seconds is not None or args_cli.episode_max_seconds is not None:
            min_s = (
                float(args_cli.episode_min_seconds)
                if args_cli.episode_min_seconds is not None
                else float(args_cli.episode_max_seconds)
            )
            max_s = (
                float(args_cli.episode_max_seconds)
                if args_cli.episode_max_seconds is not None
                else float(args_cli.episode_min_seconds)
            )
            _set_env_runtime_episode_range(env, min_seconds=min_s, max_seconds=max_s)
            print(f"[INFO] Runtime episode range set to [{min_s:.2f}, {max_s:.2f}]s")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
