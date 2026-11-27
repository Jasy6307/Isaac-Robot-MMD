# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a trained G1 dance tracking checkpoint and report joint
tracking error against the reference HDF5 motion.

Example
-------
.. code-block:: powershell

    python source/train_workflow/g1_vmd_2_eval.py `
      --dance IRIS_OUT `
      --window_frames 230 `
      --num_envs 16

Press ``--start_key`` (default ``P``) or the **G1 Policy Eval** UI Play button to begin
each policy rollout from a stable standing pose; Stop ends rollout and audio early.
When ``--dance`` has a companion WAV, audio starts with Play. Use ``--auto_start`` to
run immediately without waiting for Play. Enable **Record AVI** to capture the viewport
live during rollout at 30 fps (with dance WAV muxed when available).
By default ``media/stage/station.usdz`` is spawned as the scene backdrop; use
``--no_stage`` to disable or ``--stage <path>`` to override.
When a backdrop is loaded, the default ground is hidden but still collidable.
Passing ``--dance`` limits this eval run to that dance; other motions are not loaded.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from source import sim_compat  # noqa: E402
from source.train_workflow.task_ids import G1_TRAIN_TASK_ID, normalize_g1_train_task_id  # noqa: E402
from source.train_workflow.utils.checkpoints import checkpoint_search_roots, resolve_training_checkpoint  # noqa: E402
from source.train_workflow.utils.motion.resolve import (  # noqa: E402
    infer_dance_name_from_motion_path,
    normalize_dance_stem,
    resolve_dance_h5_by_name,
    resolve_training_motion,
)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a G1 dance tracking checkpoint.")
parser.add_argument("--robot", choices=("g1plus", "g1"),
                    help="Dance body: G1+ O6 (default) or original G1 O6. Bundles use their embedded body.")
parser.add_argument("--motion_h5", help="Use an explicit reference H5, as in training.")
parser.add_argument("--eval_motion_start", type=float, default=None,
                    help="Evaluate a 2.6s clip at this source time; default full motion.")
parser.add_argument("--eval_motion_starts", type=float, nargs="+", default=None,
                    help="Cycle clips, repeating each seed across the clip list.")
parser.add_argument("--jump_diagnostics", action="store_true",
                    help="Record physics-step jump telemetry and export CSV/NPZ/PNG per rollout.")
parser.add_argument("--jump_diagnostics_dir", default="output/jump-diagnostics")
parser.add_argument("--jump_diagnostics_env", type=int, default=0)
parser.add_argument("--jump_diagnostics_start", type=float, default=0., help="First recorded rollout second.")
parser.add_argument("--jump_diagnostics_end", type=float, default=None, help="Last recorded rollout second.")
parser.add_argument("--jump_contact_threshold", type=float, default=5., help="Foot contact force threshold, N.")
parser.add_argument("--auto_jump_review", type=str, default=None)
parser.add_argument("--deployment_bundle", type=str, default=None,
                    help="Load a trusted single-file deployment bundle without reading H5.")
parser.add_argument("--export_deployment_bundle", type=str, default=None,
                    help="Export checkpoint, current reference and effective configs, then exit.")
parser.add_argument("--task", type=normalize_g1_train_task_id, choices=(G1_TRAIN_TASK_ID,),
                    default=G1_TRAIN_TASK_ID,
                    help="G1 dance tracking task (default). Deployment bundles use their embedded configuration.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--live_anon",
    action="store_true",
    default=False,
    help="Show and synchronize the Anon PMX dancer with the G1 policy in the same live Isaac Sim stage.",
)
parser.add_argument(
    "--live_anon_usd",
    type=str,
    default=os.path.join(_WORKSPACE_ROOT, "output", "iris_out_policy_20261115", "IRIS_OUT_anon_physx_baked.usdc"),
    help="Baked PMX PhysX USD cache synchronized with the live policy rollout.",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="Optional reproducible rollout reset seed (incremented per episode).",
)
parser.add_argument(
    "--benchmark_inference",
    action="store_true",
    default=False,
    help="Only benchmark policy forward pass (no env.step loop) and exit.",
)
parser.add_argument(
    "--benchmark_steps",
    type=int,
    default=2000,
    help="Forward iterations for --benchmark_inference.",
)
parser.add_argument(
    "--benchmark_warmup",
    type=int,
    default=200,
    help="Warmup forward iterations before --benchmark_inference timing.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint path, or a file name within --load_run or --checkpoint_dir.",
)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Source run directory; omitted --checkpoint selects its latest .pt.")
parser.add_argument(
    "--dance",
    type=str,
    default=None,
    help=(
        "Dance name under media/dance/ (e.g. IRIS_OUT). "
        "Loads only this dance; resolves HDF5 and checkpoint log subfolder "
        "logs/rsl_rl/<exp>/<dance>/."
    ),
)
parser.add_argument(
    "--window_frames",
    type=int,
    default=None,
    help=(
        "Override reference motion window (control-step frame count). "
        "If omitted, eval auto-uses the selected dance H5 full length."
    ),
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
    "--full_window_episode",
    dest="full_window_episode",
    action="store_true",
    default=True,
    help="Play uses full window as episode length (disable random episode length timeout).",
)
parser.add_argument(
    "--no_full_window_episode",
    dest="full_window_episode",
    action="store_false",
    help="Keep env default episode-length behavior during play.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=0,
    help="Number of rollouts to play, including early terminations. Use <= 0 for unlimited playback.",
)
parser.add_argument(
    "--real_time",
    action="store_true",
    default=False,
    help=(
        "Pace policy steps on a wall-clock grid at control_hz (audio stays 1x). "
        "Uses cumulative perf_counter deadlines instead of per-step sleep deltas."
    ),
)
parser.add_argument(
    "--action_smooth_alpha",
    type=float,
    default=1.0,
    help=(
        "EMA action smoothing in (0, 1]. 1.0 = off (raw policy output). "
        "Try 0.25~0.5 to reduce play-time jitter; lowers apparent tracking sharpness."
    ),
)
parser.add_argument(
    "--perf_log_interval",
    type=float,
    default=1.0,
    help=(
        "Print rolling sim performance every N wall seconds. "
        "0 disables periodic logs (episode summary still printed)."
    ),
)
parser.add_argument(
    "--start_key",
    type=str,
    default="P",
    help="Keyboard key to start each policy rollout from standing (default: P).",
)
parser.add_argument(
    "--auto_start",
    action="store_true",
    default=False,
    help="Start policy playback immediately (skip standing wait for start key).",
)
parser.add_argument(
    "--pd_profile",
    type=str,
    choices=("deploy", "isaaclab"),
    default=None,
    help=(
        "Robot actuator PD: deploy=Unitree rl_lab FixStand (default); "
        "isaaclab=legacy G1_29DOF defaults for old checkpoints."
    ),
)
parser.add_argument(
    "--self_collisions",
    dest="self_collisions",
    action="store_true",
    default=None,
    help="Enable robot self-collisions during evaluation (match training).",
)
parser.add_argument(
    "--no_self_collisions",
    dest="self_collisions",
    action="store_false",
    help="Disable robot self-collisions during evaluation.",
)
parser.add_argument(
    "--play",
    action="store_true",
    default=False,
    help=(
        "Play mode: skip reward compute (zero weights) and disable obs corruption. "
        "Does not change render rate; use for slightly lower MDP overhead only."
    ),
)
parser.add_argument(
    "--post_play_seconds",
    type=float,
    default=2.0,
    help=(
        "Extra tail duration after reference window ends (seconds). "
        "Applied in --play mode, default 2.0s."
    ),
)
parser.add_argument(
    "--record_avi",
    action="store_true",
    default=False,
    help=(
        "Record viewport AVI live at 30 fps during each policy rollout. "
        "Muxes the dance WAV when available (ffmpeg or imageio-ffmpeg)."
    ),
)
parser.add_argument(
    "--record_policy_h5",
    type=str,
    default=None,
    help=(
        "Export the actual simulated robot joint/root poses during one --play rollout "
        "to a replay-compatible HDF5 file. Requires --auto_start, --num_envs 1, "
        "and --num_episodes 1."
    ),
)
parser.add_argument(
    "--record_avi_fps",
    type=float,
    default=60.0,
    help="AVI recording FPS for --record_avi .",
)
parser.add_argument(
    "--stage",
    type=str,
    default=os.path.join(_WORKSPACE_ROOT, "media", "stage", "station.usdz"),
    help=(
        "USD/USDZ stage background asset (absolute or repo-relative path). "
        "Default: media/stage/station.usdz."
    ),
)
parser.add_argument(
    "--no_stage",
    action="store_true",
    default=False,
    help="Do not spawn the stage background asset.",
)
parser.add_argument(
    "--physics_hz",
    type=float,
    default=None,
    help="Override eval physics frequency (sim.dt = 1 / physics_hz).",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Override physics steps per policy action (120 Hz physics / 4 = 30 Hz control).",
)
parser.add_argument(
    "--render_interval",
    type=int,
    default=None,
    help="Render every N physics steps without changing physics/control timing (2 = 60 Hz at 120 Hz physics).",
)
parser.add_argument(
    "--viewport_resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
    help="Fix viewport render resolution independently of window size (e.g. 960 540).",
)
parser.add_argument(
    "--profile_render", action="store_true",
    help="Report environment-step and render/UI wall times (no GPU synchronization).",
)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args_cli = parser.parse_args()
if args_cli.physics_hz is not None and (
    not math.isfinite(args_cli.physics_hz) or args_cli.physics_hz <= 0.0
):
    parser.error("--physics_hz must be finite and > 0")
if args_cli.decimation is not None and args_cli.decimation < 1:
    parser.error("--decimation must be >= 1")
if args_cli.render_interval is not None and args_cli.render_interval < 1:
    parser.error("--render_interval must be >= 1")
if args_cli.viewport_resolution is not None and min(args_cli.viewport_resolution) < 1:
    parser.error("--viewport_resolution dimensions must be >= 1")
if args_cli.live_anon:
    if not args_cli.play:
        parser.error("--live_anon requires --play")
    if not os.path.isfile(args_cli.live_anon_usd):
        parser.error(
            f"--live_anon_usd baked cache not found: {args_cli.live_anon_usd}. "
            "Build it with source/pmx_prototype/physx_dance.py --bake-usd first."
        )
if args_cli.record_policy_h5:
    if not args_cli.play or not args_cli.auto_start:
        parser.error("--record_policy_h5 requires --play and --auto_start")
    if args_cli.num_envs != 1 or args_cli.num_episodes != 1:
        parser.error("--record_policy_h5 requires --num_envs 1 and --num_episodes 1")
    if args_cli.eval_motion_starts or args_cli.eval_motion_start is not None:
        parser.error("--record_policy_h5 does not support motion-start overrides")
    if os.path.exists(args_cli.record_policy_h5):
        parser.error(f"--record_policy_h5 output already exists: {args_cli.record_policy_h5}")
sim_compat.prepare_launcher_args(args_cli)
if args_cli.jump_diagnostics:
    if not 0 <= args_cli.jump_diagnostics_env < args_cli.num_envs:
        parser.error("--jump_diagnostics_env must select an existing environment")
    if not math.isfinite(args_cli.jump_contact_threshold) or args_cli.jump_contact_threshold <= 0:
        parser.error("--jump_contact_threshold must be finite and positive")
    if not math.isfinite(args_cli.jump_diagnostics_start) or args_cli.jump_diagnostics_start < 0:
        parser.error("--jump_diagnostics_start must be finite and nonnegative")
    if args_cli.jump_diagnostics_end is not None and (
        not math.isfinite(args_cli.jump_diagnostics_end)
        or args_cli.jump_diagnostics_end <= args_cli.jump_diagnostics_start
    ):
        parser.error("--jump_diagnostics_end must exceed start")
    if args_cli.export_deployment_bundle or args_cli.benchmark_inference:
        parser.error("Jump diagnostics requires a rollout, not export/benchmark mode")

DANCE_NAME: str | None = None
MOTION_H5_PATH: str | None = None
if args_cli.dance:
    DANCE_NAME = normalize_dance_stem(args_cli.dance)
WINDOW_FRAMES_OVERRIDE: int | None = (
    int(args_cli.window_frames) if args_cli.window_frames is not None else None
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import h5py  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from source.my_task.robots.dance_body import configure_dance_body  # noqa: E402

import source.my_task  # noqa: F401, E402

# Play-only default viewport (overrides task env_cfg.viewer).
_PLAY_VIEWER_EYE = (0.0, 4.0, 1.0)
_PLAY_VIEWER_LOOKAT = (0.0, -4.0, 0.0)

from source.my_task.mdp.events import reset_root_to_spawn  # noqa: E402
from source.my_task.mdp.observations import joint_pos_tracking_error  # noqa: E402
from source.my_task.motion_reference import (  # noqa: E402
    get_or_create_motion_buffer,
    reset_motion_start_steps,
)
from source.my_task.robots.actuator_pd import (  # noqa: E402
    apply_pd_profile_to_scene_robot,
    log_pd_profile_summary,
)
from source.paths import DANCE_DIR, MEDIA_DIR, REPO_ROOT, resolve_dance_asset, iter_dance_assets  # noqa: E402
from source.train_workflow.utils.motion.window import (  # noqa: E402
    control_hz_from_env_cfg,
    log_window_frames_override,
    resolve_motion_window_seconds,
    window_seconds_from_frames,
)
from source.train_workflow.utils.media import audio_util  # noqa: E402
from source.train_workflow.utils.media.avi_audio_mux import mux_wav_into_avi  # noqa: E402
from source.train_workflow.utils.media.viewport_avi_recorder import (  # noqa: E402
    ViewportAviRecorder,
    default_playback_avi_path,
)

_DANCES_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "dances_config.yaml")
_POLICY_AUTO_KEY = "__AUTO_LATEST__"
_DANCE_ASSET_CATALOG: dict[str, tuple[str, str | None]] | None = None


def _wait_until_wall_deadline(deadline: float) -> None:
    """Block until perf_counter() >= deadline; retries sleep to absorb Windows overshoot."""
    while True:
        remaining = float(deadline) - time.perf_counter()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.004))


class _PlayPerfTracker:
    """Rolling wall-clock stats to distinguish sim speed from display FPS."""

    def __init__(self, *, control_hz: float, log_interval_s: float) -> None:
        self.control_hz = float(control_hz)
        self.log_interval_s = float(log_interval_s)
        self._window_start = time.time()
        self._window_steps = 0
        self._episode_start = time.time()
        self._episode_steps = 0

    def on_step(self) -> None:
        self._window_steps += 1
        self._episode_steps += 1
        if self.log_interval_s <= 0.0:
            return
        elapsed = time.time() - self._window_start
        if elapsed < self.log_interval_s:
            return
        self._print_window(elapsed)
        self._window_start = time.time()
        self._window_steps = 0

    def on_episode_end(self, episode: int) -> None:
        elapsed = time.time() - self._episode_start
        steps = max(self._episode_steps, 1)
        steps_per_s = steps / max(elapsed, 1e-6)
        ratio = steps_per_s / max(self.control_hz, 1e-6)
        print(
            f"[PERF] episode={episode} summary: "
            f"sim_steps/s={steps_per_s:.1f}  control_hz={self.control_hz:.1f}  "
            f"realtime_ratio={ratio:.2f}x  avg_step_ms={1000.0 * elapsed / steps:.1f}"
        )
        self._episode_start = time.time()
        self._episode_steps = 0
        self._window_start = time.time()
        self._window_steps = 0

    def _print_window(self, elapsed: float) -> None:
        steps = max(self._window_steps, 1)
        steps_per_s = steps / max(elapsed, 1e-6)
        ratio = steps_per_s / max(self.control_hz, 1e-6)
        print(
            f"[PERF] rolling: sim_steps/s={steps_per_s:.1f}  "
            f"control_hz={self.control_hz:.1f}  realtime_ratio={ratio:.2f}x  "
            f"avg_step_ms={1000.0 * elapsed / steps:.1f}"
        )


def _apply_viewport_performance_options() -> None:
    if args_cli.headless:
        return
    if args_cli.viewport_resolution is not None:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            print("[WARN] No active viewport; resolution override skipped.")
        else:
            viewport.resolution = tuple(args_cli.viewport_resolution)
            print(f"[INFO] Viewport render resolution = {viewport.resolution}")


class _EvalStepProfiler:
    """Measure wall time spent inside env.step and its render/UI calls.

    GPU work is asynchronous: these numbers identify host-side waits/costs,
    rather than independent GPU kernel durations. Policy inference is outside step.
    """

    def __init__(self, wrapped, env) -> None:
        self.wrapped, self.sim = wrapped, env.sim
        self.original_step, self.original_render = wrapped.step, env.sim.render
        self.started = time.perf_counter()
        self.steps = self.frames = 0
        self.step_seconds = self.render_seconds = 0.0
        self.stage_seconds = {}
        self.stage_originals = []
        for name, obj, attr in (
            ("actions", env.action_manager, "process_action"),
            ("actions", env.action_manager, "apply_action"),
            ("scene_write", env.scene, "write_data_to_sim"),
            ("physics", env.sim, "step"),
            ("scene_update", env.scene, "update"),
            ("obs", env.observation_manager, "compute"),
            ("terms", env.termination_manager, "compute"),
            ("rewards", env.reward_manager, "compute"),
        ):
            self.stage_seconds.setdefault(name, 0.0)
            original = getattr(obj, attr)
            self.stage_originals.append((obj, attr, original))
            setattr(obj, attr, self._time_stage(name, original))
        self.detail_seconds = {}
        for group, names in env.observation_manager.active_terms.items():
            configs = env.observation_manager._group_obs_term_cfgs[group]
            for name, cfg in zip(names, configs):
                original = cfg.func
                key = f"obs_{name}"
                self.detail_seconds[key] = 0.0
                self.stage_originals.append((cfg, "func", original))
                cfg.func = self._time_stage(key, original, self.detail_seconds)
        if type(env.sim.physics_manager).__module__.startswith("isaaclab_physx") or getattr(
            env.sim.physics_manager, "__module__", ""
        ).startswith("isaaclab_physx"):
            import omni.physx

            original_get = omni.physx.get_physx_simulation_interface
            native_interface = original_get()
            # Native bindings need not allow method assignment. Intercept the
            # Python accessor only while profiling; all other calls forward.
            class TimedPhysxInterface:
                def __getattr__(proxy_self, name):
                    value = getattr(native_interface, name)
                    if name in ("simulate", "fetch_results"):
                        return self._time_stage(f"physx_{name}", value, self.detail_seconds)
                    return value

            for name in ("simulate", "fetch_results"):
                self.detail_seconds[f"physx_{name}"] = 0.0
            proxy = TimedPhysxInterface()
            self.stage_originals.append((omni.physx, "get_physx_simulation_interface", original_get))
            omni.physx.get_physx_simulation_interface = lambda: proxy
        wrapped.step, env.sim.render = self.step, self.render

    def _time_stage(self, name, original, counters=None):
        if counters is None:
            counters = self.stage_seconds
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                counters[name] += time.perf_counter() - started
        return timed

    def render(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.original_render(*args, **kwargs)
        finally:
            self.render_seconds += time.perf_counter() - started
            self.frames += 1

    def step(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.original_step(*args, **kwargs)
        self.step_seconds += time.perf_counter() - started
        self.steps += 1
        now = time.perf_counter()
        if now - self.started >= 2.0:
            print(
                f"[PROFILE] env_step_ms={1000 * self.step_seconds / self.steps:.2f} "
                f"render_ui_ms/frame={1000 * self.render_seconds / max(self.frames, 1):.2f} "
                f"render_calls={self.frames} steps={self.steps} "
                "(wall time; GPU async; policy/wall pacing excluded)"
            )
            print("[PROFILE] stages_ms/step: " + " ".join(
                f"{name}={1000.0 * seconds / self.steps:.2f}"
                for name, seconds in self.stage_seconds.items()
            ))
            print("[PROFILE] details_ms/step (nested within stages): " + " ".join(
                f"{name}={1000.0 * seconds / self.steps:.2f}"
                for name, seconds in self.detail_seconds.items()
            ))
            self.started = now
            self.steps = self.frames = 0
            self.step_seconds = self.render_seconds = 0.0
            for name in self.stage_seconds:
                self.stage_seconds[name] = 0.0
            for name in self.detail_seconds:
                self.detail_seconds[name] = 0.0
        return result

    def close(self) -> None:
        self.wrapped.step, self.sim.render = self.original_step, self.original_render
        for obj, attr, original in self.stage_originals:
            setattr(obj, attr, original)


class _StartKeyListener:
    """Fire once per key press to gate policy rollout start."""

    def __init__(self, key_char: str) -> None:
        ch = str(key_char or "P").strip().upper()[:1]
        if len(ch) != 1 or not ("A" <= ch <= "Z"):
            raise ValueError(f"start_key must be a single A-Z letter, got {key_char!r}")
        self._key_char = ch
        self._key_input = getattr(carb.input.KeyboardInput, ch)
        self._pending = False
        self._input_iface = carb.input.acquire_input_interface()
        self._kb_dev: object | None = None
        self._kb_sub: object | None = None

    @property
    def key_char(self) -> str:
        return self._key_char

    def subscribe(self) -> bool:
        import omni

        app_window = omni.appwindow.get_default_app_window()
        self._kb_dev = app_window.get_keyboard() if app_window is not None else None
        if self._kb_dev is None:
            print("[WARN] Keyboard device unavailable; use --auto_start for headless play.")
            return False
        self._kb_sub = self._input_iface.subscribe_to_keyboard_events(self._kb_dev, self._on_event)
        return True

    def unsubscribe(self) -> None:
        if self._kb_sub is not None and self._kb_dev is not None:
            self._input_iface.unsubscribe_to_keyboard_events(self._kb_dev, self._kb_sub)
        self._kb_sub = None
        self._kb_dev = None

    def consume(self) -> bool:
        if not self._pending:
            return False
        self._pending = False
        return True

    def _on_event(self, event, *args):  # type: ignore[no-untyped-def]
        if getattr(event, "type", None) != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if getattr(event, "input", None) == self._key_input:
            self._pending = True
        return True


def _reset_to_standing_pose(env_unwrapped) -> None:
    """Teleport to spawn and hold default (T-pose) joint targets."""
    with torch.no_grad():
        env_ids = torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device)
        reset_root_to_spawn(env_unwrapped, env_ids)
        asset = env_unwrapped.scene["robot"]
        q = asset.data.default_joint_pos.torch[env_ids].clone()
        qd = torch.zeros_like(q)
        sim_compat.write_joint_state(asset, q, qd, env_ids=env_ids)
        sim_compat.set_joint_position_target(asset, q, env_ids=env_ids)
        sim_compat.write_root_velocity(asset,
            torch.zeros((env_ids.numel(), 6), device=env_unwrapped.device, dtype=torch.float32),
            env_ids=env_ids,
        )
        env_unwrapped.episode_length_buf[:] = 0
        reset_motion_start_steps(env_unwrapped, env_ids)


def _idle_hold_step(env_unwrapped, wrapped, zero_actions: torch.Tensor):
    """Advance sim one step while kinematically holding the default standing pose."""
    _reset_to_standing_pose(env_unwrapped)
    result = wrapped.step(zero_actions)
    # Residual / reference action terms may pull joints during the step; re-anchor after physics.
    _reset_to_standing_pose(env_unwrapped)
    return result


def _env_reset(wrapped, *, seed: int | None = None):
    """Env reset must run outside ``torch.inference_mode`` (Isaac articulation buffers)."""
    with torch.no_grad():
        if seed is not None:
            try:
                wrapped.seed(int(seed))
            except Exception as exc:
                print(f"[WARN] Failed to apply reset seed {int(seed)}: {exc}")
        return wrapped.reset()


def _active_termination_terms_for_env(env_unwrapped, env_idx: int) -> list[str]:
    """Best-effort read active termination term names for one env index."""
    tm = getattr(env_unwrapped, "termination_manager", None)
    if tm is None:
        return []
    try:
        pairs = tm.get_active_iterable_terms(int(env_idx))
    except Exception:
        return []
    terms: list[str] = []
    for name, values in pairs:
        try:
            v0 = float(values[0]) if values else 0.0
        except Exception:
            v0 = 0.0
        if v0 > 0.5:
            terms.append(str(name))
    return terms


def _read_dance_asset_catalog() -> dict[str, tuple[str, str | None]]:
    """Read motion/audio paths from YAML without opening any motion or audio files."""
    global _DANCE_ASSET_CATALOG
    if _DANCE_ASSET_CATALOG is not None:
        return _DANCE_ASSET_CATALOG

    import yaml

    catalog: dict[str, tuple[str, str | None]] = {}
    if os.path.isfile(_DANCES_CONFIG_PATH):
        with open(_DANCES_CONFIG_PATH, encoding="utf-8") as stream:
            doc = yaml.safe_load(stream) or {}
        items = doc.get("dances", [])
        if not isinstance(items, list):
            raise ValueError("Dance YAML must contain a dances list.")
        for entry in items:
            if not isinstance(entry, dict):
                continue
            motion = str(entry.get("motion") or "").strip().replace("\\", "/")
            if not motion:
                continue
            name = infer_dance_name_from_motion_path(motion)
            motion_path = str(resolve_dance_asset(os.path.join(MEDIA_DIR, motion)))
            audio = str(entry.get("audio") or "").strip().replace("\\", "/")
            wav_path = str(resolve_dance_asset(os.path.join(MEDIA_DIR, audio))) if audio else None
            catalog.setdefault(name, (motion_path, wav_path))
    _DANCE_ASSET_CATALOG = catalog
    return catalog


def _resolve_dance_wav_path(dance_name: str | None) -> str | None:
    """Resolve only this dance's companion WAV using path metadata."""
    if not dance_name:
        return None
    stem = normalize_dance_stem(dance_name)
    from source.train_workflow.utils.motion.audio import resolve_motion_audio
    for suffix in ("_z_editted.h5", ".h5", "_z_editted.hdf5", ".hdf5"):
        motion = str(resolve_dance_asset(os.path.join(DANCE_DIR, stem + suffix)))
        if os.path.isfile(motion):
            resolved = resolve_motion_audio(motion)
            if resolved:
                return resolved
            break
    try:
        entry = _read_dance_asset_catalog().get(stem)
        configured = entry[1] if entry else None
        if configured and os.path.isfile(configured):
            return configured
    except Exception as exc:
        print(f"[WARN] Failed to read dance WAV path from YAML: {exc}")

    direct = str(resolve_dance_asset(os.path.join(DANCE_DIR, f"{stem}.wav")))
    if os.path.isfile(direct):
        return direct
    return None


def _collect_available_dances() -> list[str]:
    """Collect dance names from dances_config.yaml and media/dance assets."""
    out: set[str] = set()
    try:
        for name, (path, _wav_path) in _read_dance_asset_catalog().items():
            if name and os.path.isfile(path):
                out.add(name)
    except Exception as exc:
        print(f"[WARN] Failed to read dances from YAML: {exc}")

    for pat in ("*.h5", "*.hdf5", "*.csv"):
        for path in iter_dance_assets(pat):
            try:
                stem = normalize_dance_stem(os.path.basename(path))
            except Exception:
                stem = ""
            if stem:
                out.add(stem)
    return sorted(out)


def _list_policy_paths_for_dance(experiment_name: str, dance_name: str) -> list[str]:
    out: list[str] = []
    for root in checkpoint_search_roots(experiment_name, dance_name):
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                low = name.lower()
                if low.endswith(".pt") and (low.startswith("model_") or low == "model.pt"):
                    out.append(os.path.abspath(os.path.join(dirpath, name)))
    out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out


def _collect_policy_entries(experiment_name: str, dance_names: list[str]) -> list[tuple[str, str]]:
    """Build UI policy entries: key=abs path or AUTO key, label=repo-relative."""
    entries: list[tuple[str, str]] = [(_POLICY_AUTO_KEY, "Auto latest for dance")]
    seen: set[str] = {_POLICY_AUTO_KEY}

    for dance_name in dance_names:
        for abs_path in _list_policy_paths_for_dance(experiment_name, dance_name):
            if abs_path in seen:
                continue
            seen.add(abs_path)
            entries.append((abs_path, _checkpoint_relpath(abs_path)))

    policy_dir = os.path.join(REPO_ROOT, "policy")
    if os.path.isdir(policy_dir):
        for abs_path in sorted(glob.glob(os.path.join(policy_dir, "*.pt"))):
            abs_norm = os.path.abspath(abs_path)
            if abs_norm in seen:
                continue
            seen.add(abs_norm)
            entries.append((abs_norm, _checkpoint_relpath(abs_norm)))
    return entries


def _resolve_checkpoint_for_dance(
    agent_cfg: RslRlOnPolicyRunnerCfg,
    dance_name: str,
    *,
    checkpoint: str | None = None,
) -> str:
    if checkpoint and not (args_cli.load_run or args_cli.checkpoint_dir):
        # Preserve the eval CLI's relative path interpretation without a run selector.
        checkpoint = os.path.abspath(checkpoint)
    try:
        return resolve_training_checkpoint(
            agent_cfg.experiment_name,
            dance_name,
            load_run=args_cli.load_run,
            checkpoint=checkpoint,
            checkpoint_dir=args_cli.checkpoint_dir,
            run_pattern=agent_cfg.load_run,
            checkpoint_pattern=agent_cfg.load_checkpoint,
        )
    except FileNotFoundError:
        if not (checkpoint or args_cli.load_run or args_cli.checkpoint_dir):
            candidates = _list_policy_paths_for_dance(agent_cfg.experiment_name, dance_name)
            if candidates:
                return candidates[0]
        raise


class _PolicyPlaybackGate:
    """Shared play/stop state for keyboard, UI, and the eval main loop."""

    def __init__(self) -> None:
        self.playing = False
        self.play_requested = False
        self.stop_requested = False
        self.policy_step = 0
        self.policy_total = 0
        self.control_hz = 0.0
        self.wav_path: str | None = None
        self.audio_enabled = False
        self.record_avi_enabled = False
        self.dance_title = "Policy eval"
        self.policy_relpath = ""
        self.dance_selected = ""
        self.policy_selected = _POLICY_AUTO_KEY
        self.play_enabled = True
        self.policy_error = ""

    def request_play(self) -> None:
        if not self.play_enabled:
            print("[WARN] Play ignored: no runnable policy for current selection.")
            return
        self.play_requested = True

    def request_stop(self) -> None:
        self.stop_requested = True

    def ui_status(self) -> dict[str, object]:
        return {
            "dance_title": self.dance_title,
            "playing": self.playing,
            "policy_step": self.policy_step,
            "policy_total": self.policy_total,
            "control_hz": self.control_hz,
            "has_audio": bool(self.wav_path),
            "audio_enabled": self.audio_enabled,
            "record_avi_enabled": self.record_avi_enabled,
            "policy_relpath": self.policy_relpath,
            "dance_selected": self.dance_selected,
            "policy_selected": self.policy_selected,
            "play_enabled": self.play_enabled,
            "policy_error": self.policy_error,
        }


def _checkpoint_relpath(abs_path: str) -> str:
    """Display path relative to repo root (forward slashes)."""
    try:
        rel = os.path.relpath(os.path.abspath(abs_path), REPO_ROOT)
    except ValueError:
        rel = os.path.abspath(abs_path)
    return rel.replace("\\", "/")


def _find_h5_window(env_cfg: ManagerBasedRLEnvCfg) -> tuple[str, float]:
    """Recover (h5_path, window_seconds) from the env cfg's reward term params."""
    tracking = env_cfg.rewards.joint_pos_tracking
    return str(tracking.params["h5_path"]), float(tracking.params["window_seconds"])


def _read_h5_frames_and_fps(h5_path: str) -> tuple[int, float]:
    """Read H5 frame count and source fps without loading full motion arrays."""
    if args_cli.deployment_bundle:
        ref = _DEPLOYMENT_BUNDLE["reference"]
        return int(ref["tables"]["_phase"].shape[0]), float(ref["control_hz"])
    import h5py

    with h5py.File(os.path.abspath(h5_path), "r") as f:
        if "frames" not in f:
            raise ValueError(f"H5 missing 'frames' dataset: {h5_path}")
        frames = int(f["frames"].shape[0])
        fps = float(f.attrs.get("fps", 30.0))
        if fps <= 0.0:
            fps = 30.0
        return frames, fps


def _resolve_eval_window_seconds(
    env_cfg: ManagerBasedRLEnvCfg,
    h5_path: str,
    window_frames_override: int | None,
) -> tuple[float, int, str]:
    """Resolve eval window seconds from override or current dance full H5 length."""
    control_hz = control_hz_from_env_cfg(env_cfg)
    if args_cli.deployment_bundle:
        ref = _DEPLOYMENT_BUNDLE["reference"]
        return float(ref["window_seconds"]), int(ref["tables"]["_phase"].shape[0]), "deployment_bundle"
    if window_frames_override is not None:
        frames = max(1, int(window_frames_override))
        return float(window_seconds_from_frames(frames, control_hz)), frames, "override"
    h5_frames, h5_fps = _read_h5_frames_and_fps(h5_path)
    # H5 frames are authored at source fps (typically 30). Runtime uses control_hz (typically 60).
    # Convert to control-step count so playback duration matches source clip time.
    control_steps = max(1, int(round(float(h5_frames) * float(control_hz) / float(h5_fps))))
    return (
        float(window_seconds_from_frames(control_steps, control_hz)),
        int(control_steps),
        "auto_full_h5",
    )


def _infer_obs_device(obs_obj) -> torch.device:
    """Best-effort infer device from nested obs object."""
    if torch.is_tensor(obs_obj):
        return obs_obj.device
    if hasattr(obs_obj, "get"):
        try:
            pol = obs_obj.get("policy", None)
            if torch.is_tensor(pol):
                return pol.device
        except Exception:
            pass
    if hasattr(obs_obj, "values"):
        try:
            for value in obs_obj.values():
                try:
                    return _infer_obs_device(value)
                except ValueError:
                    pass
        except Exception:
            pass
    if isinstance(obs_obj, dict):
        for value in obs_obj.values():
            try:
                return _infer_obs_device(value)
            except ValueError:
                pass
    if isinstance(obs_obj, (tuple, list)):
        for item in obs_obj:
            try:
                return _infer_obs_device(item)
            except ValueError:
                pass
    raise ValueError(f"Cannot infer device from obs type: {type(obs_obj)}")


def _benchmark_policy_forward(policy, obs_obj, *, warmup: int, steps: int) -> None:
    warmup_i = max(0, int(warmup))
    steps_i = max(1, int(steps))
    device = _infer_obs_device(obs_obj)
    is_cuda = device.type == "cuda"
    print(
        f"[BENCH] Running policy forward benchmark: warmup={warmup_i}, steps={steps_i}, device={device}"
    )
    with torch.inference_mode():
        for _ in range(warmup_i):
            _ = policy(obs_obj)
        if is_cuda:
            torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()
        for _ in range(steps_i):
            _ = policy(obs_obj)
        if is_cuda:
            torch.cuda.synchronize(device=device)
        elapsed = max(time.perf_counter() - t0, 1e-9)
    ms = 1000.0 * elapsed / steps_i
    hz = steps_i / elapsed
    print(f"[BENCH] policy_forward: avg={ms:.3f} ms  hz={hz:.1f}")
    if is_cuda:
        # Isolated per-call wall latency, including a CUDA completion wait.
        # Unlike the throughput measurement above, each call must finish
        # before the next is submitted; this is a diagnostic, not rollout pacing.
        latency_steps = min(steps_i, 200)
        with torch.inference_mode():
            torch.cuda.synchronize(device=device)
            t0 = time.perf_counter()
            for _ in range(latency_steps):
                _ = policy(obs_obj)
                torch.cuda.synchronize(device=device)
            elapsed = time.perf_counter() - t0
        print(
            f"[BENCH] policy_forward_sync: avg={1000.0 * elapsed / latency_steps:.3f} ms "
            f"(isolated latency including CUDA wait; samples={latency_steps})"
        )


def _build_policy_obs_adapter(policy, sample_obs, sample_tensor: torch.Tensor):
    """Build an adapter so policy(...) accepts tensor or group-style obs."""
    with torch.inference_mode():
        accepts_tensor = False
        accepts_group = False
        try:
            _ = policy(sample_tensor)
            accepts_tensor = True
        except Exception:
            pass
        try:
            _ = policy({"policy": sample_tensor})
            accepts_group = True
        except Exception:
            pass
        if accepts_tensor and not accepts_group:
            def _as_tensor(x):
                return x if torch.is_tensor(x) else _extract_policy_obs_tensor(x)
            return _as_tensor
        if accepts_group and not accepts_tensor:
            def _as_group(x):
                if isinstance(x, dict) or hasattr(x, "get"):
                    return x
                return {"policy": x}
            return _as_group
        if accepts_tensor and accepts_group:
            # Prefer tensor path to avoid dictionary packing overhead.
            def _prefer_tensor(x):
                return x if torch.is_tensor(x) else _extract_policy_obs_tensor(x)
            return _prefer_tensor
    raise ValueError(
        f"Unable to adapt observation format for policy input type: {type(sample_obs)}"
    )


def _extract_policy_obs_tensor(obs_obj) -> torch.Tensor:
    """Normalize wrapped.get_observations() output to a policy tensor."""
    if torch.is_tensor(obs_obj):
        return obs_obj
    # RSL-RL wrappers may return a TensorDict-like object.
    if hasattr(obs_obj, "get"):
        try:
            pol = obs_obj.get("policy", None)
            if torch.is_tensor(pol):
                return pol
        except Exception:
            pass
    if hasattr(obs_obj, "values"):
        try:
            for value in obs_obj.values():
                if torch.is_tensor(value):
                    return value
                if isinstance(value, (dict, tuple, list)) or hasattr(value, "get") or hasattr(value, "values"):
                    try:
                        return _extract_policy_obs_tensor(value)
                    except ValueError:
                        pass
        except Exception:
            pass
    if isinstance(obs_obj, dict):
        if "policy" in obs_obj and torch.is_tensor(obs_obj["policy"]):
            return obs_obj["policy"]
        for value in obs_obj.values():
            if torch.is_tensor(value):
                return value
            if isinstance(value, (dict, tuple, list)):
                try:
                    return _extract_policy_obs_tensor(value)
                except ValueError:
                    pass
    if isinstance(obs_obj, (tuple, list)):
        for item in obs_obj:
            try:
                return _extract_policy_obs_tensor(item)
            except ValueError:
                pass
    raise ValueError(f"Cannot extract policy observation tensor from type: {type(obs_obj)}")


def _apply_motion_overrides(env_cfg: ManagerBasedRLEnvCfg) -> None:
    global WINDOW_FRAMES_OVERRIDE
    if (
        MOTION_H5_PATH is None
        and WINDOW_FRAMES_OVERRIDE is None
        and args_cli.residual_alpha is None
        and args_cli.use_reference_residual is None
    ):
        return
    new_h5 = MOTION_H5_PATH
    new_ws = None
    new_wf = None
    ws_mode = "default"
    if new_h5 is not None:
        new_ws, new_wf, ws_mode = _resolve_eval_window_seconds(
            env_cfg, new_h5, WINDOW_FRAMES_OVERRIDE
        )
    elif WINDOW_FRAMES_OVERRIDE is not None:
        # Fallback when dance is unresolved (normally not hit).
        new_ws = resolve_motion_window_seconds(env_cfg, window_frames=WINDOW_FRAMES_OVERRIDE)
        new_wf = int(WINDOW_FRAMES_OVERRIDE)
        ws_mode = "override"
    post_tail_s = max(0.0, float(getattr(args_cli, "post_play_seconds", 0.0)))
    extend_episode_tail = bool(args_cli.play and args_cli.full_window_episode and post_tail_s > 0.0)
    if new_ws is not None and new_wf is not None:
        if ws_mode == "override":
            log_window_frames_override(new_wf, new_ws, control_hz_from_env_cfg(env_cfg))
        else:
            src_frames, src_fps = _read_h5_frames_and_fps(new_h5) if new_h5 is not None else (0, 0.0)
            print(
                f"[INFO] Motion window ({ws_mode}): src_frames={src_frames}@{src_fps:.1f}fps "
                f"=> control_steps={new_wf} "
                f"=> window_seconds={new_ws:.6f} "
                f"(control_hz={control_hz_from_env_cfg(env_cfg):.1f})"
            )
    new_residual_alpha = args_cli.residual_alpha
    new_use_reference_residual = args_cli.use_reference_residual

    def _patch(params: dict) -> None:
        if "h5_path" in params and new_h5 is not None:
            params["h5_path"] = new_h5
        if "window_seconds" in params and new_ws is not None:
            params["window_seconds"] = float(new_ws)
        if "random_start" in params:
            params["random_start"] = False
        if getattr(env_cfg, "jump_specialist", False) and "segment_seconds" in params:
            params["segment_seconds"] = float(env_cfg.episode_length_s)
        elif args_cli.full_window_episode:
            if "segment_seconds" in params:
                if new_ws is not None:
                    seg_s = float(new_ws)
                elif "window_seconds" in params:
                    seg_s = float(params["window_seconds"])
                else:
                    seg_s = float(params["segment_seconds"])
                if extend_episode_tail:
                    seg_s += post_tail_s
                params["segment_seconds"] = seg_s
        elif "segment_seconds" in params and new_ws is not None:
            params["segment_seconds"] = float(new_ws)

    pol = env_cfg.observations.policy
    for term_name in dir(pol):
        term = getattr(pol, term_name, None)
        if term is not None and hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
    for term_name in vars(env_cfg.rewards):
        term = getattr(env_cfg.rewards, term_name)
        if hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
    for term_name in vars(env_cfg.events):
        term = getattr(env_cfg.events, term_name)
        if hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
    for term_name in vars(env_cfg.terminations):
        term = getattr(env_cfg.terminations, term_name)
        if hasattr(term, "params") and isinstance(term.params, dict):
            _patch(term.params)
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

    base_episode_s = float(new_ws) if new_ws is not None else float(getattr(env_cfg, "episode_length_s", 0.0))
    if getattr(env_cfg, "jump_specialist", False):
        base_episode_s = float(env_cfg.episode_length_s)
    if extend_episode_tail:
        env_cfg.episode_length_s = float(base_episode_s + post_tail_s)
    elif new_ws is not None:
        env_cfg.episode_length_s = base_episode_s


def _sync_runtime_motion_overrides(env_unwrapped, *, h5_path: str, window_seconds: float) -> None:
    """Best-effort runtime patch for managers that cache term/action params."""
    ws = float(window_seconds)
    h5 = str(h5_path)

    def _patch_obj(obj: object, depth: int = 0, seen: set[int] | None = None) -> None:
        if obj is None:
            return
        if seen is None:
            seen = set()
        oid = id(obj)
        if oid in seen or depth > 3:
            return
        seen.add(oid)

        if isinstance(obj, dict):
            if "h5_path" in obj:
                obj["h5_path"] = h5
            if "window_seconds" in obj:
                obj["window_seconds"] = ws
            if "motion_h5_path" in obj:
                obj["motion_h5_path"] = h5
            if "motion_window_seconds" in obj:
                obj["motion_window_seconds"] = ws
            for v in obj.values():
                _patch_obj(v, depth + 1, seen)
            return

        if hasattr(obj, "params") and isinstance(getattr(obj, "params"), dict):
            params = getattr(obj, "params")
            if "h5_path" in params:
                params["h5_path"] = h5
            if "window_seconds" in params:
                params["window_seconds"] = ws

        if hasattr(obj, "motion_h5_path"):
            try:
                setattr(obj, "motion_h5_path", h5)
            except Exception:
                pass
        if hasattr(obj, "motion_window_seconds"):
            try:
                setattr(obj, "motion_window_seconds", ws)
            except Exception:
                pass

        try:
            children = vars(obj).values()
        except Exception:
            children = []
        for child in children:
            _patch_obj(child, depth + 1, seen)

    for attr in (
        "observation_manager",
        "reward_manager",
        "event_manager",
        "termination_manager",
        "action_manager",
        "command_manager",
    ):
        _patch_obj(getattr(env_unwrapped, attr, None))


def _resolve_stage_usd_path(stage_path: str) -> str:
    """Resolve stage asset path (absolute or repo-relative)."""
    if os.path.isabs(stage_path):
        return os.path.abspath(stage_path)
    return os.path.abspath(os.path.join(REPO_ROOT, stage_path))


def _apply_stage_background(env_cfg: ManagerBasedRLEnvCfg, stage_path: str) -> bool:
    """Spawn a shared USD/USDZ stage backdrop at /World/station."""
    resolved = _resolve_stage_usd_path(stage_path)
    if not os.path.isfile(resolved):
        print(f"[WARN] Stage background not found, skipping: {resolved}")
        return False

    env_cfg.scene.stage = AssetBaseCfg(
        prim_path="/World/station",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolved,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    print(f"[INFO] Stage background: {resolved}")
    return True


def _hide_default_ground_visuals(env_unwrapped) -> None:
    """Hide the default plane's rendering while retaining its physics collider."""
    from pxr import UsdGeom

    terrain = env_unwrapped.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "plane":
        return
    for prim_path in terrain.terrain_prim_paths:
        imageable = UsdGeom.Imageable(env_unwrapped.sim.stage.GetPrimAtPath(prim_path))
        if not imageable:
            print(f"[WARN] Cannot hide default ground: {prim_path}")
            continue
        imageable.MakeInvisible()
        print(f"[INFO] Default ground hidden (collision retained): {prim_path}")


def _apply_play_viewer(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Override Isaac Lab viewer camera for policy playback."""
    env_cfg.sim.default_visualizer_cfg.eye = _PLAY_VIEWER_EYE
    env_cfg.sim.default_visualizer_cfg.lookat = _PLAY_VIEWER_LOOKAT


def _apply_play_mode(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Zero reward weights and disable obs corruption for interactive playback."""
    zeroed = 0
    for term_name in vars(env_cfg.rewards):
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None and hasattr(term, "weight"):
            term.weight = 0.0
            zeroed += 1

    policy_obs = getattr(env_cfg.observations, "policy", None)
    if policy_obs is not None and hasattr(policy_obs, "enable_corruption"):
        policy_obs.enable_corruption = False

    # Play mode has its own rollout length control (rollout_steps),
    # so env timeout should not preempt playback and auto-reset.
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "time_out"):
        terminations.time_out = None

    print(
        f"[INFO] Play mode: rewards zeroed ({zeroed} terms), obs corruption off, "
        "time_out termination disabled, per-step joint_pos_tracking_error skipped during rollout."
    )


def _apply_self_collision_override(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply the same articulation setting as training before env creation."""
    if args_cli.self_collisions is None:
        return
    robot = env_cfg.scene.robot
    spawn = robot.spawn
    if spawn is None or spawn.articulation_props is None:
        raise ValueError("Self-collision override requires robot articulation properties")
    enabled = bool(args_cli.self_collisions)
    env_cfg.scene.robot = robot.replace(
        spawn=spawn.replace(
            articulation_props=spawn.articulation_props.replace(enabled_self_collisions=enabled)
        )
    )
    print(f"[INFO] Robot self-collisions override: enabled_self_collisions={enabled}")


def main() -> None:
    global DANCE_NAME, MOTION_H5_PATH, WINDOW_FRAMES_OVERRIDE
    global _DEPLOYMENT_BUNDLE

    from source.train_workflow.utils.deployment import load_bundle, save_bundle
    _DEPLOYMENT_BUNDLE = None
    if args_cli.deployment_bundle:
        if args_cli.export_deployment_bundle:
            raise ValueError("Use export and deployment loading in separate runs")
        # A bundle fixes its policy, motion and timing; switching requires another bundle.
        for name in ("dance", "checkpoint", "checkpoint_dir", "load_run", "window_frames",
                     "physics_hz", "decimation", "residual_alpha", "use_reference_residual",
                     "self_collisions", "pd_profile", "robot", "motion_h5"):
            if getattr(args_cli, name, None) is not None:
                raise ValueError(f"--{name} cannot override a deployment bundle")
        _DEPLOYMENT_BUNDLE = load_bundle(args_cli.deployment_bundle)
        args_cli.task = _DEPLOYMENT_BUNDLE["task"]
    elif args_cli.pd_profile is None:
        args_cli.pd_profile = "deploy"

    env_cfg: ManagerBasedRLEnvCfg
    agent_cfg: RslRlOnPolicyRunnerCfg
    if _DEPLOYMENT_BUNDLE is None:
        env_cfg = parse_env_cfg(args_cli.task)
        agent_cfg = sim_compat.prepare_rsl_rl_cfg(
            load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        )
    else:
        env_cfg = _DEPLOYMENT_BUNDLE["env_cfg"]
        agent_cfg = _DEPLOYMENT_BUNDLE["agent_cfg"]
    # Timing overrides must precede motion-window resolution and env creation.
    if args_cli.physics_hz is not None:
        env_cfg.sim.dt = 1.0 / args_cli.physics_hz
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    if _DEPLOYMENT_BUNDLE is not None:
        from source.my_task.motion_reference.motion_buffer import register_packaged_reference
        DANCE_NAME = _DEPLOYMENT_BUNDLE["dance"]
        MOTION_H5_PATH = os.path.abspath(args_cli.deployment_bundle)
        register_packaged_reference(MOTION_H5_PATH, _DEPLOYMENT_BUNDLE["reference"])
        dance_options = [DANCE_NAME]
    elif args_cli.motion_h5:
        MOTION_H5_PATH, DANCE_NAME = resolve_training_motion(args_cli.dance, args_cli.motion_h5)
        if not DANCE_NAME:
            raise ValueError("Use --dance to name the explicit --motion_h5 reference")
        dance_options = [DANCE_NAME]
    elif args_cli.dance:
        dance_options = [normalize_dance_stem(args_cli.dance)]
        print(f"[INFO] Single-dance eval: {dance_options[0]} (other dance assets skipped).")
    else:
        dance_options = _collect_available_dances()
    if not dance_options:
        raise ValueError(f"No dances found under {DANCE_DIR}.")
    selected_dance = normalize_dance_stem(DANCE_NAME or dance_options[0])
    if selected_dance not in dance_options:
        dance_options.insert(0, selected_dance)
    if _DEPLOYMENT_BUNDLE is None and not args_cli.motion_h5:
        MOTION_H5_PATH, DANCE_NAME = resolve_dance_h5_by_name(selected_dance)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        # Keep inference on the selected device as well (especially for the
        # single-env CPU diagnostic; avoid implicit CPU/CUDA transfers).
        agent_cfg.device = args_cli.device
    if _DEPLOYMENT_BUNDLE is None:
        env_cfg.scene.robot = apply_pd_profile_to_scene_robot(
            env_cfg.scene.robot, args_cli.pd_profile, o6_hands=True
        )
        configure_dance_body(env_cfg, args_cli.robot or "g1plus")
        _apply_self_collision_override(env_cfg)
    if _DEPLOYMENT_BUNDLE is None:
        log_pd_profile_summary(args_cli.pd_profile, o6_hands=True)
    else:
        print("[INFO] Using actuator and collision configuration embedded in deployment bundle.")
    _apply_motion_overrides(env_cfg)
    if getattr(env_cfg, "auto_jump", False) and _DEPLOYMENT_BUNDLE is None:
        from source.train_workflow.utils.auto_jump_training import prepare_auto_config
        auto_h5, auto_window = _find_h5_window(env_cfg)
        prepare_auto_config(env_cfg, auto_h5, auto_window, args_cli.auto_jump_review)
    if getattr(env_cfg, "jump_multi", False) or (getattr(env_cfg, "auto_jump", False) and
                                               (args_cli.eval_motion_start is not None or args_cli.eval_motion_starts)):
        if args_cli.eval_motion_starts:
            args_cli.eval_motion_start = args_cli.eval_motion_starts[0]
        start = args_cli.eval_motion_start
        env_cfg.episode_length_s = 2.6 if start is not None else 2500/30.
        env_cfg.events.reset_robot_joints.params.update(motion_start_choices=None,
            motion_start_seconds=float(start or 0.), segment_seconds=env_cfg.episode_length_s)
    _apply_play_viewer(env_cfg)
    stage_background_loaded = False
    if not args_cli.no_stage:
        stage_background_loaded = _apply_stage_background(env_cfg, args_cli.stage)
    if args_cli.play:
        _apply_play_mode(env_cfg)
    if args_cli.jump_diagnostics:
        from source.train_workflow.utils.jump_diagnostics import configure_jump_sensors
        configure_jump_sensors(env_cfg)

    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval
    print(
        f"[INFO] Eval timing: sim.dt = {env_cfg.sim.dt:.9f}s; "
        f"physics_hz = {1.0 / env_cfg.sim.dt:.1f}; decimation = {env_cfg.decimation}; "
        f"render_interval = {env_cfg.sim.render_interval}; "
        f"render_hz = {1.0 / (env_cfg.sim.dt * env_cfg.sim.render_interval):.1f}; "
        f"control_hz = {control_hz_from_env_cfg(env_cfg):.1f}"
    )

    if _DEPLOYMENT_BUNDLE is None:
        roots = checkpoint_search_roots(agent_cfg.experiment_name, DANCE_NAME)
        print(f"[INFO] Looking for checkpoints under: {', '.join(map(str, roots))}")
    print(f"[INFO] Dance: {DANCE_NAME}")
    resume_path = ""
    initial_policy_error = ""
    if _DEPLOYMENT_BUNDLE is not None:
        import atexit
        import tempfile
        # Runner expects a checkpoint path; materialize embedded bytes in a private
        # temporary directory, never consult the original checkpoint or logs.
        checkpoint_temp = tempfile.TemporaryDirectory(prefix="g1-deployment-")
        atexit.register(checkpoint_temp.cleanup)
        resume_path = os.path.join(checkpoint_temp.name, "policy.pt")
        with open(resume_path, "wb") as checkpoint_file:
            checkpoint_file.write(_DEPLOYMENT_BUNDLE["checkpoint"])
        selected_policy_key = resume_path
    elif args_cli.checkpoint:
        resume_path = _resolve_checkpoint_for_dance(
            agent_cfg, DANCE_NAME or selected_dance, checkpoint=args_cli.checkpoint
        )
        selected_policy_key = resume_path
    else:
        selected_policy_key = _POLICY_AUTO_KEY
        try:
            resume_path = _resolve_checkpoint_for_dance(agent_cfg, DANCE_NAME or selected_dance)
        except Exception as exc:
            initial_policy_error = str(exc)
            print(f"[WARN] {initial_policy_error}")
    if resume_path:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        env_cfg.log_dir = os.path.dirname(resume_path)
        if getattr(env_cfg,"auto_jump",False):
            header = torch.load(resume_path,map_location="cpu",weights_only=False)
            if any(name.startswith("jump_mlp.") for name in header["actor_state_dict"]):
                env_cfg.auto_jump_exact_residual = True
                agent_cfg.actor.class_name = "source.train_workflow.utils.exact_phase_policy:ExactPhaseResidualPolicy"
            del header
    else:
        print("[WARN] No runnable policy selected at startup; Play will stay disabled.")

    if _DEPLOYMENT_BUNDLE is None:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    else:
        # Bundles carry both configs, including those from retired task IDs.
        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    if stage_background_loaded:
        _hide_default_ground_visuals(env_unwrapped)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    live_anon = None
    if args_cli.live_anon:
        from source.pmx_prototype.runtime.live_anon_scene import LiveAnonScene
        live_anon = LiveAnonScene(
            env_unwrapped.sim.stage,
            pmx_usd=args_cli.live_anon_usd,
        )

    _apply_viewport_performance_options()

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    obs_raw = wrapped.get_observations()
    policy = None
    obs_adapter = None
    if resume_path:
        try:
            runner.load(resume_path, load_cfg={"actor": True, "critic": True})
            policy = runner.get_inference_policy(device=env_unwrapped.device)
            obs = _extract_policy_obs_tensor(obs_raw)
            obs_adapter = _build_policy_obs_adapter(policy, obs_raw, obs)
        except Exception as exc:
            initial_policy_error = str(exc)
            print(f"[WARN] Failed to load runnable policy: {resume_path}")
            print(f"[WARN] {initial_policy_error}")
            resume_path = ""

    if args_cli.export_deployment_bundle:
        if policy is None or obs_adapter is None:
            wrapped.close()
            raise ValueError("Cannot export: checkpoint did not load successfully")
        export_h5, export_window = _find_h5_window(env_cfg)
        export_buf = get_or_create_motion_buffer(env_unwrapped, export_h5, export_window)
        save_bundle(
            args_cli.export_deployment_bundle, checkpoint_path=resume_path,
            reference=export_buf.export_reference(), env_cfg=env_cfg, agent_cfg=agent_cfg,
            task=args_cli.task, dance=DANCE_NAME,
        )
        print(f"[INFO] Deployment bundle exported: {os.path.abspath(args_cli.export_deployment_bundle)}")
        wrapped.close()
        return

    if args_cli.benchmark_inference:
        if policy is None or obs_adapter is None:
            print("[WARN] --benchmark_inference skipped: no runnable policy.")
            wrapped.close()
            return
        _benchmark_policy_forward(
            policy,
            obs_adapter(obs_raw),
            warmup=int(args_cli.benchmark_warmup),
            steps=int(args_cli.benchmark_steps),
        )
        wrapped.close()
        return

    h5_path, window_s = _find_h5_window(env_cfg)
    buf = get_or_create_motion_buffer(env_unwrapped, h5_path, window_s)
    T = buf.num_steps
    control_hz = float(buf.control_hz)
    window_frames_current = int(T)
    print(f"[INFO] Reference window steps = {T}; control_hz = {control_hz:.1f}")
    post_play_seconds = max(0.0, float(args_cli.post_play_seconds))
    if post_play_seconds != float(args_cli.post_play_seconds):
        print(f"[WARN] --post_play_seconds clamped to {post_play_seconds:.3f}")
    post_play_steps = int(round(control_hz * post_play_seconds)) if args_cli.play else 0
    rollout_steps = int(T + post_play_steps)
    if getattr(env_cfg, "jump_specialist", False) or (getattr(env_cfg, "auto_jump", False) and
                                                   (args_cli.eval_motion_start is not None or args_cli.eval_motion_starts)):
        rollout_steps = int(round(env_cfg.episode_length_s * control_hz))
    if post_play_steps > 0:
        print(
            f"[INFO] --play tail enabled: +{post_play_steps} steps "
            f"({post_play_seconds:.2f}s), rollout steps={rollout_steps}"
        )

    policy_pose_records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    policy_joint_names: list[str] = []

    def _capture_policy_pose() -> None:
        """Snapshot the articulated state after policy control for H5 replay."""
        robot = env_unwrapped.scene["robot"]
        joint_pos = sim_compat.as_torch(robot.data.joint_pos)[0].detach().cpu().numpy().astype(np.float32)
        root_state = sim_compat.as_torch(getattr(robot.data, "root_state_w", None))
        if root_state is None or root_state.ndim != 2 or root_state.shape[1] < 7:
            raise RuntimeError("Robot articulation does not expose a valid root_state_w")
        root_xyzw = root_state[0, 3:7].detach().cpu().numpy().astype(np.float32)
        root_pos = root_state[0, :3].detach().cpu().numpy().astype(np.float32)
        if not policy_joint_names:
            policy_joint_names.extend(str(name) for name in robot.joint_names)
            if len(policy_joint_names) != len(joint_pos):
                raise RuntimeError(
                    f"joint name/state size mismatch: {len(policy_joint_names)} vs {len(joint_pos)}"
                )
        # IsaacLab's root_state_w tensor is XYZW; this motion H5 schema is WXYZ.
        root_wxyz = root_xyzw[[3, 0, 1, 2]]
        policy_pose_records.append((joint_pos, root_pos.copy(), root_wxyz.copy()))
    if float(args_cli.perf_log_interval) > 0.0:
        print(
            f"[INFO] Perf logging every {float(args_cli.perf_log_interval):.1f}s "
            f"(sim_steps/s vs control_hz={control_hz:.1f}; "
            f"realtime_ratio=1.0 means real-time sim)."
        )
    if args_cli.real_time:
        print(
            f"[INFO] --real_time: pacing steps on wall clock at {control_hz:.1f} Hz "
            "(audio 1x; cumulative deadline, not per-step sleep delta)."
        )

    dt = env_unwrapped.step_dt
    perf = _PlayPerfTracker(
        control_hz=control_hz,
        log_interval_s=float(args_cli.perf_log_interval),
    )
    smooth_alpha = float(args_cli.action_smooth_alpha)
    if not (0.0 < smooth_alpha <= 1.0):
        raise ValueError(f"--action_smooth_alpha must be in (0, 1], got {smooth_alpha}")
    if smooth_alpha < 1.0:
        print(f"[INFO] Action EMA smoothing enabled: alpha={smooth_alpha:.3f}")
    record_avi_fps = max(1.0, float(args_cli.record_avi_fps))
    if record_avi_fps != float(args_cli.record_avi_fps):
        print(f"[WARN] --record_avi_fps clamped to {record_avi_fps:.1f}")

    action_dim = int(env_unwrapped.action_manager.total_action_dim)
    jump_diagnostics = None
    if args_cli.jump_diagnostics:
        from source.train_workflow.utils.jump_diagnostics import JumpDiagnostics
        jump_diagnostics = JumpDiagnostics(
            env_unwrapped, args_cli.jump_diagnostics_dir, env_index=args_cli.jump_diagnostics_env,
            threshold=args_cli.jump_contact_threshold, start=args_cli.jump_diagnostics_start,
            end=args_cli.jump_diagnostics_end,
        )
    zero_actions = torch.zeros(env_unwrapped.num_envs, action_dim, device=env_unwrapped.device)
    policy_entries = ([] if _DEPLOYMENT_BUNDLE is not None else
                      _collect_policy_entries(agent_cfg.experiment_name, dance_options))
    if resume_path and (not any(key == resume_path for key, _ in policy_entries)):
        policy_entries.append((resume_path, _checkpoint_relpath(resume_path)))

    playback = _PolicyPlaybackGate()
    playback.policy_relpath = _checkpoint_relpath(resume_path) if resume_path else ""
    playback.policy_total = rollout_steps
    playback.control_hz = control_hz
    playback.dance_title = DANCE_NAME or os.path.splitext(os.path.basename(h5_path))[0]
    playback.dance_selected = DANCE_NAME or ""
    playback.policy_selected = selected_policy_key
    playback.play_enabled = bool(policy is not None and obs_adapter is not None)
    playback.policy_error = initial_policy_error if (not playback.play_enabled) else ""
    playback.wav_path = _resolve_dance_wav_path(DANCE_NAME)
    playback.audio_enabled = bool(playback.wav_path)
    playback.record_avi_enabled = bool(args_cli.record_avi)
    if playback.wav_path:
        audio_util.warn_if_no_pygame_sync()
        print(f"[INFO] Dance audio: {playback.wav_path}")
    else:
        print("[INFO] No dance WAV configured for this eval run.")
    if playback.record_avi_enabled and getattr(args_cli, "headless", False):
        print("[WARN] --record_avi ignored in headless mode (no viewport).")
        playback.record_avi_enabled = False
    if playback.record_avi_enabled and playback.audio_enabled:
        playback.audio_enabled = False
        print("[INFO] Record AVI enabled: dance audio disabled during rollout.")

    if playback.record_avi_enabled and not args_cli.real_time:
        print(
            "[INFO] Record AVI: policy rollout runs at full sim speed; "
            f"viewport AVI captured live at {record_avi_fps:.0f} fps."
        )

    avi_recorder: ViewportAviRecorder | None = None
    avi_recording_active = False
    avi_synced_video_frame = 0

    def _avi_pre_render() -> None:
        try:
            env_unwrapped.sim.render()
        except Exception:
            pass

    def _start_live_avi_recording(*, episode_idx: int) -> None:
        nonlocal avi_recorder, avi_recording_active, avi_synced_video_frame
        if avi_recording_active:
            _finalize_live_avi_recording(reason="restart")
        label = f"eval_{playback.dance_title}"
        out_path = default_playback_avi_path(media_dir=MEDIA_DIR, motion_label=label)
        avi_recorder = ViewportAviRecorder(
            fps=record_avi_fps,
            pre_render=_avi_pre_render,
            policy_safe_capture=True,
        )
        try:
            avi_recorder.start(out_path)
        except Exception as exc:
            avi_recorder = None
            avi_recording_active = False
            avi_synced_video_frame = 0
            print(f"[ERROR] AVI recording failed to start: {exc}")
            return
        print(
            f"[INFO] AVI recording started (episode={episode_idx}): {out_path} "
            f"@ {record_avi_fps:.0f} fps (live capture from rollout)"
        )
        avi_recording_active = True
        avi_synced_video_frame = 0

    def _sync_live_avi_recording(*, policy_step: int) -> None:
        """Sample live rollout at fixed AVI fps, relative to policy control_hz."""
        nonlocal avi_synced_video_frame
        if not avi_recording_active or avi_recorder is None:
            return
        ps = max(0, int(policy_step))
        target_video_frame = int(round(ps * record_avi_fps / max(control_hz, 1e-6)))
        if target_video_frame <= avi_synced_video_frame:
            return
        gap = target_video_frame - avi_synced_video_frame - 1
        if gap > 0:
            avi_recorder.write_duplicate_frames(gap)
        if avi_recorder.capture_frame():
            avi_synced_video_frame = target_video_frame

    def _finalize_live_avi_recording(*, reason: str) -> None:
        nonlocal avi_recorder, avi_recording_active, avi_synced_video_frame
        recorder = avi_recorder
        avi_recorder = None
        avi_recording_active = False
        avi_synced_video_frame = 0
        if recorder is None:
            return
        frame_count = int(recorder.frame_count)
        recorder_fps = float(recorder.fps)
        try:
            out_final = recorder.stop()
        except Exception as exc:
            print(f"[ERROR] AVI recording finalize failed ({reason}): {exc}")
            return
        if not out_final:
            print(f"[WARN] AVI recording produced no frames ({reason})")
            return
        if playback.wav_path and os.path.isfile(playback.wav_path):
            mux_wav_into_avi(out_final, playback.wav_path, replace_original=True)
        expected_s = float(frame_count) / max(recorder_fps, 1e-6)
        print(
            f"[INFO] AVI recording complete ({reason}): {out_final} "
            f"({frame_count} frames @ {recorder_fps:.1f} fps, "
            f"expected duration {expected_s:.2f}s)"
        )

    pending_dance_name: str | None = None
    pending_policy_key: str | None = None
    pending_window_frames: int | None = None

    def _request_dance_from_ui(dance_name: str) -> None:
        nonlocal pending_dance_name, pending_policy_key
        pending_dance_name = normalize_dance_stem(dance_name)
        pending_policy_key = _POLICY_AUTO_KEY

    def _request_policy_from_ui(policy_key: str) -> None:
        nonlocal pending_policy_key
        pending_policy_key = str(policy_key or "").strip() or _POLICY_AUTO_KEY

    def _request_window_frames_from_ui(window_frames: int) -> None:
        nonlocal pending_window_frames
        pending_window_frames = int(window_frames)

    def _apply_pending_selection() -> None:
        global DANCE_NAME, MOTION_H5_PATH, WINDOW_FRAMES_OVERRIDE
        nonlocal pending_dance_name, pending_policy_key, pending_window_frames
        nonlocal resume_path, selected_policy_key, policy, obs_adapter, obs_raw
        nonlocal h5_path, window_s, buf, T, control_hz, post_play_steps, rollout_steps, window_frames_current
        nonlocal perf
        if (
            pending_dance_name is None
            and pending_policy_key is None
            and pending_window_frames is None
        ):
            return
        if playback.playing:
            return
        if _DEPLOYMENT_BUNDLE is not None:
            print("[WARN] Deployment bundle fixes policy, dance and window; load another bundle to change them.")
            pending_dance_name = pending_policy_key = pending_window_frames = None
            return

        target_dance = normalize_dance_stem(pending_dance_name or DANCE_NAME or "")
        if not target_dance:
            target_dance = DANCE_NAME or selected_dance
        dance_changed = bool(target_dance and target_dance != (DANCE_NAME or ""))
        window_changed = pending_window_frames is not None
        if window_changed:
            wf = int(pending_window_frames)
            WINDOW_FRAMES_OVERRIDE = None if wf <= 0 else wf
        if dance_changed:
            try:
                new_h5, canonical = resolve_dance_h5_by_name(target_dance)
            except Exception as exc:
                print(f"[WARN] Dance switch rejected: {exc}")
                pending_dance_name = None
                return
            DANCE_NAME = canonical
            MOTION_H5_PATH = new_h5
        if dance_changed or window_changed:
            _apply_motion_overrides(env_cfg)
            if getattr(env_cfg, "auto_jump", False):
                from source.train_workflow.utils.auto_jump_training import prepare_auto_config
                auto_h5, auto_window = _find_h5_window(env_cfg)
                prepare_auto_config(env_cfg, auto_h5, auto_window, args_cli.auto_jump_review)
            h5_cfg, ws_cfg = _find_h5_window(env_cfg)
            _sync_runtime_motion_overrides(env_unwrapped, h5_path=h5_cfg, window_seconds=ws_cfg)
            h5_path = str(h5_cfg)
            window_s = float(ws_cfg)
            buf = get_or_create_motion_buffer(env_unwrapped, h5_path, window_s)
            T = int(buf.num_steps)
            window_frames_current = int(T)
            control_hz = float(buf.control_hz)
            post_play_steps = int(round(control_hz * post_play_seconds)) if args_cli.play else 0
            rollout_steps = int(T + post_play_steps)
            perf = _PlayPerfTracker(
                control_hz=control_hz,
                log_interval_s=float(args_cli.perf_log_interval),
            )
            playback.policy_total = rollout_steps
            playback.control_hz = control_hz
            playback.dance_title = DANCE_NAME or os.path.splitext(os.path.basename(h5_path))[0]
            playback.wav_path = _resolve_dance_wav_path(DANCE_NAME)
            playback.dance_selected = DANCE_NAME or ""
            if playback.record_avi_enabled and playback.audio_enabled:
                playback.audio_enabled = False
            elif playback.wav_path is None:
                playback.audio_enabled = False
            if dance_changed:
                print(f"[INFO] Switched dance: {DANCE_NAME} ({_checkpoint_relpath(h5_path)})")
            if window_changed:
                mode_msg = "auto_full_h5" if WINDOW_FRAMES_OVERRIDE is None else str(WINDOW_FRAMES_OVERRIDE)
                print(
                    f"[INFO] Updated eval window: mode={mode_msg} "
                    f"effective_frames={window_frames_current} window_s={window_s:.3f}"
                )

        target_policy_key = pending_policy_key or selected_policy_key
        target_policy_path: str
        if target_policy_key == _POLICY_AUTO_KEY:
            try:
                target_policy_path = _resolve_checkpoint_for_dance(
                    agent_cfg,
                    DANCE_NAME or selected_dance,
                )
            except Exception as exc:
                playback.play_enabled = False
                playback.policy_error = str(exc)
                playback.policy_relpath = ""
                playback.policy_selected = _POLICY_AUTO_KEY
                print(f"[WARN] {playback.policy_error}")
                pending_policy_key = None
                pending_dance_name = None
                return
        else:
            target_policy_path = os.path.abspath(target_policy_key)
            if not os.path.isfile(target_policy_path):
                print(f"[WARN] Policy not found: {target_policy_path}")
                playback.play_enabled = False
                playback.policy_error = f"missing file ({_checkpoint_relpath(target_policy_path)})"
                playback.policy_relpath = ""
                playback.policy_selected = target_policy_key
                pending_policy_key = None
                pending_dance_name = None
                return

        if target_policy_path != resume_path:
            try:
                header = torch.load(target_policy_path,map_location="cpu",weights_only=False)
                target_exact = any(name.startswith("jump_mlp.") for name in header["actor_state_dict"])
                del header
                if target_exact != bool(getattr(env_cfg,"auto_jump_exact_residual",False)):
                    raise ValueError("Policy observation/architecture differs; restart eval with the target checkpoint")
                runner.load(target_policy_path, load_cfg={"actor": True, "critic": True})
                policy = runner.get_inference_policy(device=env_unwrapped.device)
                obs = _extract_policy_obs_tensor(obs_raw)
                obs_adapter = _build_policy_obs_adapter(policy, obs_raw, obs)
                resume_path = target_policy_path
                print(f"[INFO] Switched policy: {resume_path}")
            except Exception as exc:
                playback.play_enabled = False
                playback.policy_error = str(exc)
                playback.policy_relpath = _checkpoint_relpath(target_policy_path)
                playback.policy_selected = target_policy_key
                print(f"[WARN] Policy rejected: {playback.policy_relpath}")
                print(f"[WARN] {playback.policy_error}")
                pending_dance_name = None
                pending_policy_key = None
                return

        selected_policy_key = target_policy_key
        playback.policy_relpath = _checkpoint_relpath(resume_path)
        playback.policy_selected = selected_policy_key
        playback.play_enabled = bool(policy is not None and obs_adapter is not None and resume_path)
        playback.policy_error = ""
        pending_dance_name = None
        pending_policy_key = None
        pending_window_frames = None

    if not getattr(args_cli, "headless", False):
        from source.train_workflow.ui import eval_play_ui

        def _set_audio_enabled_from_ui(enabled: bool) -> None:
            playback.audio_enabled = bool(enabled) and not bool(playback.record_avi_enabled)

        def _set_record_avi_enabled_from_ui(enabled: bool) -> None:
            playback.record_avi_enabled = bool(enabled)
            if playback.record_avi_enabled:
                playback.audio_enabled = False
                if playback.playing:
                    audio_util.stop_wav()

        eval_play_ui.set_play_callback(playback.request_play)
        eval_play_ui.set_stop_callback(playback.request_stop)
        eval_play_ui.set_status_provider(playback.ui_status)
        eval_play_ui.set_audio_enabled_provider(lambda: playback.audio_enabled)
        eval_play_ui.set_audio_enabled_setter(_set_audio_enabled_from_ui)
        eval_play_ui.set_audio_volume_callbacks(audio_util.get_volume, audio_util.set_volume)
        eval_play_ui.set_record_avi_callbacks(
            lambda: playback.record_avi_enabled,
            _set_record_avi_enabled_from_ui,
        )
        eval_play_ui.set_dance_selector_callbacks(
            lambda: [(name, name) for name in dance_options],
            _request_dance_from_ui,
        )
        eval_play_ui.set_policy_selector_callbacks(
            lambda: list(policy_entries),
            _request_policy_from_ui,
        )
        eval_play_ui.set_window_frames_callbacks(
            lambda: int(window_frames_current),
            _request_window_frames_from_ui,
        )
        eval_play_ui.create_eval_play_ui()

    start_listener: _StartKeyListener | None = None
    wait_for_key = not bool(args_cli.auto_start)
    if wait_for_key:
        start_listener = _StartKeyListener(args_cli.start_key)
        if not start_listener.subscribe():
            print("[WARN] Keyboard unavailable; falling back to --auto_start.")
            wait_for_key = False

    rollout_wall_start: float | None = None
    idle_wall_start: float | None = None
    idle_steps_done = 0
    play_seed_counter = 0

    def _reset_idle_wall_clock() -> None:
        nonlocal idle_wall_start, idle_steps_done
        idle_wall_start = None
        idle_steps_done = 0

    def _next_play_seed(*, episode_idx: int) -> int:
        nonlocal play_seed_counter
        if args_cli.seed is not None:
            seed_episode = int(episode_idx)
            if args_cli.eval_motion_starts:
                seed_episode //= len(args_cli.eval_motion_starts)
            return (int(args_cli.seed) + seed_episode) & 0x7FFFFFFF
        play_seed_counter += 1
        seed = (time.time_ns() + int(episode_idx) * 131 + play_seed_counter * 1009) & 0x7FFFFFFF
        return int(seed if seed > 0 else 1)

    def _begin_rollout(*, episode_idx: int) -> None:
        nonlocal obs_raw, rollout_wall_start, idle_wall_start, idle_steps_done
        audio_util.stop_wav()
        _reset_idle_wall_clock()
        rollout_seed = _next_play_seed(episode_idx=episode_idx)
        random.seed(rollout_seed)
        np.random.seed(rollout_seed % (2**32 - 1))
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        if args_cli.eval_motion_starts:
            start = args_cli.eval_motion_starts[episode_idx % len(args_cli.eval_motion_starts)]
            env_cfg.events.reset_robot_joints.params["motion_start_seconds"] = start
            env_unwrapped.event_manager.get_term_cfg("reset_robot_joints").params["motion_start_seconds"] = start
        obs_raw, _ = _env_reset(wrapped, seed=rollout_seed)
        if live_anon is not None:
            live_anon.reset()
        if args_cli.record_policy_h5:
            policy_pose_records.clear()
            _capture_policy_pose()
        if jump_diagnostics is not None:
            jump_diagnostics.begin(episode_idx, buf, {
                "checkpoint": os.path.abspath(resume_path), "dance": DANCE_NAME,
                "reference": h5_path, "deployment_bundle": args_cli.deployment_bundle,
                "seed": rollout_seed, "action_smooth_alpha": smooth_alpha,
                "play_mode": bool(args_cli.play),
                "task": args_cli.task,
                "reference_variant": "v2_leg_takeoff" if getattr(env_cfg, "jump_reference_shaping", False) else "original",
                "reward_requires_full_step_no_contact": bool(getattr(env_cfg, "jump_full_step_contact", False)),
            })
        if playback.record_avi_enabled and not getattr(args_cli, "headless", False):
            _start_live_avi_recording(episode_idx=episode_idx)
        rollout_wall_start = time.perf_counter()
        if playback.wav_path and playback.audio_enabled and (not playback.record_avi_enabled):
            start_seconds = float(env_cfg.events.reset_robot_joints.params.get("motion_start_seconds") or 0.)
            audio_util.play_wav_async(playback.wav_path, start_sec=start_seconds)
        playback.playing = True
        playback.stop_requested = False
        playback.policy_step = 0
        print(
            f"[INFO] Policy rollout started (episode={episode_idx}, "
            f"steps={rollout_steps}, ref={T}, tail={post_play_steps}, seed={rollout_seed})."
        )

    def _abort_rollout(*, export_avi: bool = False) -> None:
        nonlocal obs_raw, rollout_wall_start
        audio_util.stop_wav()
        playback.playing = False
        playback.stop_requested = False
        playback.policy_step = 0
        rollout_wall_start = None
        _reset_idle_wall_clock()
        if export_avi:
            _finalize_live_avi_recording(reason="stop")
        _reset_to_standing_pose(env_unwrapped)
        if live_anon is not None:
            live_anon.reset()
        obs_raw = wrapped.get_observations()
        print("[INFO] Policy rollout stopped.")

    _reset_to_standing_pose(env_unwrapped)
    # Prime one idle step so the first rendered frame is already re-anchored.
    obs_raw, _, _, _ = _idle_hold_step(env_unwrapped, wrapped, zero_actions)
    if wait_for_key:
        assert start_listener is not None
        print(
            f"[INFO] Standing idle. Press [{start_listener.key_char}] or UI Play to start policy "
            f"({rollout_steps} steps); repeats after each rollout."
        )
    else:
        print(f"[INFO] Auto-start: policy begins on motion reset ({rollout_steps} steps).")
        _begin_rollout(episode_idx=0)

    episode = 0
    step_profiler = _EvalStepProfiler(wrapped, env_unwrapped) if args_cli.profile_render else None
    try:
        while simulation_app.is_running() and (
            int(args_cli.num_episodes) <= 0 or episode < int(args_cli.num_episodes)
        ):
            _apply_pending_selection()
            if playback.play_requested and not playback.playing:
                playback.play_requested = False
                if policy is None or obs_adapter is None or (not playback.play_enabled):
                    print("[WARN] Play ignored: no runnable policy loaded.")
                    continue
                _begin_rollout(episode_idx=episode)

            if not playback.playing:
                if start_listener is not None and start_listener.consume():
                    _begin_rollout(episode_idx=episode)
                else:
                    if idle_wall_start is None:
                        idle_wall_start = time.perf_counter()
                    idle_steps_done += 1
                    obs_raw, _, _, _ = _idle_hold_step(env_unwrapped, wrapped, zero_actions)
                    if args_cli.real_time:
                        _wait_until_wall_deadline(idle_wall_start + idle_steps_done * dt)
                continue

            err_accum = torch.zeros(env_unwrapped.num_envs, device=env_unwrapped.device)
            max_per_step = torch.zeros(env_unwrapped.num_envs, device=env_unwrapped.device)
            steps_done = 0
            smoothed_actions = None
            rollout_aborted = False
            aborted_by_done = False
            expected_timeout = False
            done_warn_msg = ""
            skip_step_tracking_error = bool(args_cli.play)
            if policy is None or obs_adapter is None:
                print("[WARN] Rollout aborted: no runnable policy.")
                playback.playing = False
                continue
            for _ in range(rollout_steps):
                if playback.stop_requested:
                    rollout_aborted = True
                    break
                with torch.inference_mode():
                    actions = policy(obs_adapter(obs_raw))
                    if smooth_alpha < 1.0:
                        if smoothed_actions is None:
                            smoothed_actions = actions.clone()
                        else:
                            smoothed_actions = (
                                smooth_alpha * actions + (1.0 - smooth_alpha) * smoothed_actions
                            )
                        actions = smoothed_actions
                if jump_diagnostics is not None:
                    from source.my_task.motion_reference import motion_steps
                    jump_diagnostics.begin_control(motion_steps(env_unwrapped))
                try:
                    obs_raw, _, dones, _ = wrapped.step(actions)
                finally:
                    if jump_diagnostics is not None:
                        jump_diagnostics.end_control()
                try:
                    if torch.any(dones > 0):
                        done_idx = int(torch.nonzero(dones > 0, as_tuple=False)[0].item())
                        term_names = _active_termination_terms_for_env(env_unwrapped, done_idx)
                        try:
                            is_timeout = bool(env_unwrapped.reset_time_outs[done_idx].item())
                        except Exception:
                            is_timeout = False
                        try:
                            is_terminated = bool(env_unwrapped.reset_terminated[done_idx].item())
                        except Exception:
                            is_terminated = False
                        cause = ", ".join(term_names) if term_names else "unknown"
                        expected_timeout = bool(is_timeout and not is_terminated and steps_done + 1 >= rollout_steps)
                        done_warn_msg = (
                            f"[WARN] Rollout interrupted by termination "
                            f"(env={done_idx}, terminated={is_terminated}, time_out={is_timeout}, terms=[{cause}]). "
                            "Stopping playback and returning to standing pose."
                        )
                        if expected_timeout:
                            done_warn_msg = "[INFO] Rollout completed at the configured time limit."
                        rollout_aborted = True
                        aborted_by_done = True
                        break
                except Exception:
                    pass
                if live_anon is not None:
                    live_anon.advance(
                        live_anon.start_frame
                        + min(steps_done + 1, rollout_steps) * live_anon.fps / control_hz
                    )
                if args_cli.record_policy_h5 and len(policy_pose_records) < rollout_steps:
                    _capture_policy_pose()
                if not skip_step_tracking_error:
                    with torch.no_grad():
                        err = joint_pos_tracking_error(env_unwrapped, h5_path, window_s)
                    rms = torch.sqrt(torch.mean(err.pow(2), dim=1))
                    err_accum += rms
                    max_per_step = torch.maximum(max_per_step, rms)
                steps_done += 1
                playback.policy_step = steps_done
                perf.on_step()
                if avi_recording_active:
                    _sync_live_avi_recording(policy_step=steps_done)
                if args_cli.real_time and rollout_wall_start is not None:
                    _wait_until_wall_deadline(rollout_wall_start + steps_done * dt)

            if rollout_aborted:
                if jump_diagnostics is not None:
                    jump_diagnostics.flush("time_limit_complete" if expected_timeout else
                                           (done_warn_msg if aborted_by_done else "user_stop"))
                playback.stop_requested = False
                if aborted_by_done:
                    print(done_warn_msg)
                _abort_rollout(export_avi=avi_recording_active)
                status = "completed" if expected_timeout else "aborted"
                reported_steps = steps_done + 1 if expected_timeout else steps_done
                print(f"[EVAL] episode={episode} {status} steps={reported_steps}/{rollout_steps}")
                perf.on_episode_end(episode)
                episode += 1
                if int(args_cli.num_episodes) > 0:
                    if episode >= int(args_cli.num_episodes):
                        break
                    if not wait_for_key:
                        _begin_rollout(episode_idx=episode)
                continue

            if jump_diagnostics is not None:
                jump_diagnostics.flush("complete")
            audio_util.stop_wav()
            if avi_recording_active:
                _finalize_live_avi_recording(reason="complete")
            if skip_step_tracking_error:
                with torch.no_grad():
                    err = joint_pos_tracking_error(env_unwrapped, h5_path, window_s)
                final_rms = torch.sqrt(torch.mean(err.pow(2), dim=1)).mean().item()
                print(
                    f"[EVAL] episode={episode} steps={steps_done} "
                    f"final_joint_rms={final_rms:.4f} rad "
                    f"(play mode: per-step tracking error skipped)"
                )
            else:
                mean_rms = (err_accum / max(steps_done, 1)).mean().item()
                max_rms = max_per_step.mean().item()
                print(
                    f"[EVAL] episode={episode} steps={steps_done} "
                    f"mean_joint_rms={mean_rms:.4f} rad   peak_joint_rms={max_rms:.4f} rad"
                )
            perf.on_episode_end(episode)
            episode += 1
            playback.playing = False
            playback.policy_step = 0
            rollout_wall_start = None
            _reset_idle_wall_clock()

            if int(args_cli.num_episodes) > 0 and episode >= int(args_cli.num_episodes):
                break

            _reset_to_standing_pose(env_unwrapped)
            if live_anon is not None:
                live_anon.reset()
            if wait_for_key:
                assert start_listener is not None
                print(
                    f"[INFO] Rollout finished. Standing idle — press [{start_listener.key_char}] "
                    "or UI Play to run again."
                )
                obs_raw = wrapped.get_observations()
            else:
                _begin_rollout(episode_idx=episode)
    finally:
        if live_anon is not None:
            live_anon.close()
        if args_cli.record_policy_h5 and policy_pose_records:
            from pathlib import Path
            from source.train_workflow.utils.format.hdf5 import Hdf5Motion, write_hdf5_motion
            out_path = Path(args_cli.record_policy_h5).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            poses = policy_pose_records[:rollout_steps]
            absolute_joints = np.stack([row[0] for row in poses]).astype(np.float32)
            root_positions = np.stack([row[1] for row in poses]).astype(np.float32)
            root_quaternions = np.stack([row[2] for row in poses]).astype(np.float32)
            root_quaternions /= np.maximum(
                np.linalg.norm(root_quaternions, axis=1, keepdims=True), 1e-8
            )
            baseline = absolute_joints[0].copy()
            anchor_pos = root_positions[0].copy()
            anchor_quat = root_quaternions[0].copy()

            def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                aw, ax, ay, az = a
                bw, bx, by, bz = b
                return np.asarray((aw*bw-ax*bx-ay*by-az*bz,
                                   aw*bx+ax*bw+ay*bz-az*by,
                                   aw*by-ax*bz+ay*bw+az*bx,
                                   aw*bz+ax*by-ay*bx+az*bw), dtype=np.float32)

            quat_delta = np.stack([
                _quat_mul_wxyz(q, anchor_quat * np.asarray((1., -1., -1., -1.), dtype=np.float32))
                for q in root_quaternions
            ])
            quat_delta /= np.maximum(np.linalg.norm(quat_delta, axis=1, keepdims=True), 1e-8)
            motion = Hdf5Motion(
                frames=np.arange(len(poses), dtype=np.int32),
                joint_names=policy_joint_names,
                joint_pos_delta=(absolute_joints - baseline).astype(np.float32),
                root_pos_delta=(root_positions - anchor_pos).astype(np.float32),
                root_quat_delta_wxyz=quat_delta.astype(np.float32),
                root_valid=np.ones(len(poses), dtype=np.uint8),
                root_rot_bone=[""] * len(poses),
                root_rpy_deg=np.zeros((len(poses), 3), dtype=np.float32),
                source_csv="",
                has_hand_data=True,
                fps=float(control_hz),
            )
            write_hdf5_motion(str(out_path), motion)
            with h5py.File(out_path, "a") as stream:
                stream.attrs["record_baseline_joint_pos"] = baseline
                stream.attrs["record_root_anchor_pos"] = anchor_pos
                stream.attrs["record_root_anchor_quat_wxyz"] = anchor_quat
                stream.attrs["retarget_mode"] = "policy"
                stream.attrs["upper_body_retarget_method"] = "trained-policy-playback"
                stream.attrs["lower_body_retarget_method"] = "trained-policy-playback"
                stream.attrs["policy_checkpoint"] = os.path.abspath(resume_path)
                stream.attrs["source_reference_h5"] = os.path.abspath(h5_path)
                stream.attrs["rollout_seed"] = int(args_cli.seed if args_cli.seed is not None else -1)
            print(f"[INFO] Exported policy pose replay: {out_path} ({len(poses)} frames)")
        if jump_diagnostics is not None:
            jump_diagnostics.close()
        if step_profiler is not None:
            step_profiler.close()
        audio_util.stop_wav()
        if avi_recording_active:
            _finalize_live_avi_recording(reason="shutdown")
        if start_listener is not None:
            start_listener.unsubscribe()

    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
