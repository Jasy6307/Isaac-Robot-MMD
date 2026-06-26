# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a trained G1 dance tracking checkpoint and report joint
tracking error against the reference HDF5 motion.

Example
-------
.. code-block:: powershell

    ./isaac_workspace/IsaacLab/isaaclab.bat -p source/train_workflow/g1_vmd_2_eval.py `
      --task Isaac-G1-Vmd-Train-C1-v0 `
      --dance IRIS_OUT `
      --window_frames 460 `
      --num_envs 16

Press ``--start_key`` (default ``P``) or the **G1 Policy Eval** UI Play button to begin
each policy rollout from a stable standing pose; Stop ends rollout and audio early.
When ``--dance`` has a companion WAV, audio starts with Play. Use ``--auto_start`` to
run immediately without waiting for Play. Enable **Record AVI** to log poses during rollout
and export a 30 fps viewport AVI afterward (with dance WAV muxed when available).
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from source.train_workflow.utils.motion.resolve import (  # noqa: E402
    resolve_dance_h5_by_name,
    resolve_training_log_root,
)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a G1 dance tracking checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-G1-Vmd-Train-C1-v0")
parser.add_argument("--num_envs", type=int, default=16)
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
    help="Explicit checkpoint .pt file. If omitted, the latest under logs/ is used.",
)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument(
    "--dance",
    type=str,
    default=None,
    help=(
        "Dance name under media/dance/ (e.g. IRIS_OUT). "
        "Resolves HDF5 and checkpoint log subfolder logs/rsl_rl/<exp>/<dance>/."
    ),
)
parser.add_argument(
    "--window_frames",
    type=int,
    default=None,
    help="Override reference motion window (control-step frame count). Full play runs this window.",
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
    help="Number of episodes to play. Use <= 0 for infinite loop playback.",
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
    default="deploy",
    help=(
        "Robot actuator PD: deploy=Unitree rl_lab FixStand (default); "
        "isaaclab=legacy G1_29DOF defaults for old checkpoints."
    ),
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
    "--record_avi",
    action="store_true",
    default=False,
    help=(
        "Record viewport AVI at 30 fps after each policy rollout. "
        "Poses are logged during rollout (no live capture); AVI is exported offline. "
        "Muxes the dance WAV when available (ffmpeg or imageio-ffmpeg)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

DANCE_NAME: str | None = None
MOTION_H5_PATH: str | None = None
if args_cli.dance:
    MOTION_H5_PATH, DANCE_NAME = resolve_dance_h5_by_name(args_cli.dance)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

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
from source.paths import DANCE_DIR, MEDIA_DIR  # noqa: E402
from source.train_workflow.utils.motion.window import (  # noqa: E402
    control_hz_from_env_cfg,
    log_window_frames_override,
    resolve_motion_window_seconds,
)
from source.train_workflow.utils.media import audio_util  # noqa: E402
from source.train_workflow.utils.media.avi_audio_mux import mux_wav_into_avi  # noqa: E402
from source.train_workflow.utils.media.policy_rollout_avi import (  # noqa: E402
    PolicyRolloutPoseLog,
    export_pose_log_to_avi,
)
from source.train_workflow.utils.media.viewport_avi_recorder import (  # noqa: E402
    ViewportAviRecorder,
    default_playback_avi_path,
)

_RECORD_AVI_FPS = 30.0

_DANCES_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "dances_config.yaml")


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
        q = asset.data.default_joint_pos[env_ids].clone()
        qd = torch.zeros_like(q)
        asset.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        asset.set_joint_position_target(q, env_ids=env_ids)
        asset.write_root_velocity_to_sim(
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


def _env_reset(wrapped):
    """Env reset must run outside ``torch.inference_mode`` (Isaac articulation buffers)."""
    with torch.no_grad():
        return wrapped.reset()


def _resolve_dance_wav_path(dance_name: str | None) -> str | None:
    """Resolve companion WAV for ``--dance`` from dances_config.yaml or media/dance/."""
    if not dance_name:
        return None
    from source.train_workflow.utils.motion.loader import load_dances_from_yaml
    from source.train_workflow.utils.motion.resolve import normalize_dance_stem

    stem = normalize_dance_stem(dance_name)
    _, wav_by_key = load_dances_from_yaml(
        _DANCES_CONFIG_PATH,
        media_dir=MEDIA_DIR,
        script_dir=_SCRIPT_DIR,
    )
    for key, path in wav_by_key.items():
        if normalize_dance_stem(key) == stem and os.path.isfile(path):
            return path
    direct = os.path.join(DANCE_DIR, f"{stem}.wav")
    if os.path.isfile(direct):
        return direct
    return None


class _PolicyPlaybackGate:
    """Shared play/stop state for keyboard, UI, and the eval main loop."""

    def __init__(self) -> None:
        self.playing = False
        self.play_requested = False
        self.stop_requested = False
        self.policy_step = 0
        self.policy_total = 0
        self.wav_path: str | None = None
        self.audio_enabled = False
        self.record_avi_enabled = False
        self.dance_title = "Policy eval"

    def request_play(self) -> None:
        self.play_requested = True

    def request_stop(self) -> None:
        self.stop_requested = True

    def ui_status(self) -> dict[str, object]:
        return {
            "dance_title": self.dance_title,
            "playing": self.playing,
            "policy_step": self.policy_step,
            "policy_total": self.policy_total,
            "has_audio": bool(self.wav_path),
            "audio_enabled": self.audio_enabled,
            "record_avi_enabled": self.record_avi_enabled,
        }


def _find_h5_window(env_cfg: ManagerBasedRLEnvCfg) -> tuple[str, float]:
    """Recover (h5_path, window_seconds) from the env cfg's reward term params."""
    tracking = env_cfg.rewards.joint_pos_tracking
    return str(tracking.params["h5_path"]), float(tracking.params["window_seconds"])


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
    if (
        MOTION_H5_PATH is None
        and args_cli.window_frames is None
        and args_cli.residual_alpha is None
        and args_cli.use_reference_residual is None
    ):
        return
    new_h5 = MOTION_H5_PATH
    new_ws = resolve_motion_window_seconds(env_cfg, window_frames=args_cli.window_frames)
    if new_ws is not None and args_cli.window_frames is not None:
        log_window_frames_override(
            int(args_cli.window_frames),
            new_ws,
            control_hz_from_env_cfg(env_cfg),
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
        if args_cli.full_window_episode:
            if "random_episode_length" in params:
                params["random_episode_length"] = False
            if "segment_seconds" in params:
                if new_ws is not None:
                    params["segment_seconds"] = float(new_ws)
                elif "window_seconds" in params:
                    params["segment_seconds"] = float(params["window_seconds"])
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

    if new_ws is not None:
        env_cfg.episode_length_s = float(new_ws)


def _apply_play_viewer(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Override Isaac Lab viewer camera for policy playback."""
    env_cfg.viewer.eye = _PLAY_VIEWER_EYE
    env_cfg.viewer.lookat = _PLAY_VIEWER_LOOKAT


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

    print(
        f"[INFO] Play mode: rewards zeroed ({zeroed} terms), obs corruption off, "
        "per-step joint_pos_tracking_error skipped during rollout."
    )


def main() -> None:
    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(  # type: ignore[assignment]
        args_cli.task, "env_cfg_entry_point"
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(  # type: ignore[assignment]
        args_cli.task, "rsl_rl_cfg_entry_point"
    )

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot = apply_pd_profile_to_scene_robot(
        env_cfg.scene.robot, args_cli.pd_profile, o6_hands=True
    )
    log_pd_profile_summary(args_cli.pd_profile, o6_hands=True)
    _apply_motion_overrides(env_cfg)
    _apply_play_viewer(env_cfg)
    if args_cli.play:
        _apply_play_mode(env_cfg)

    log_root_path = resolve_training_log_root(agent_cfg.experiment_name, DANCE_NAME)
    print(f"[INFO] Looking for checkpoints under: {log_root_path}")
    if DANCE_NAME:
        print(f"[INFO] Dance: {DANCE_NAME}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        run_dir = args_cli.load_run if args_cli.load_run is not None else agent_cfg.load_run
        try:
            resume_path = get_checkpoint_path(
                log_root_path, run_dir, agent_cfg.load_checkpoint
            )
        except ValueError as exc:
            if DANCE_NAME is None:
                hint = (
                    f"{exc}\n"
                    "[HINT] Training logs may be under logs/rsl_rl/"
                    f"{agent_cfg.experiment_name}/<DANCE_NAME>/; pass --dance <DANCE_NAME>."
                )
                raise ValueError(hint) from exc
            raise
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env_unwrapped.device)
    obs_raw = wrapped.get_observations()
    obs = _extract_policy_obs_tensor(obs_raw)
    obs_adapter = _build_policy_obs_adapter(policy, obs_raw, obs)

    if args_cli.benchmark_inference:
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
    print(f"[INFO] Reference window steps = {T}; control_hz = {control_hz:.1f}")
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

    action_dim = int(env_unwrapped.action_manager.total_action_dim)
    zero_actions = torch.zeros(env_unwrapped.num_envs, action_dim, device=env_unwrapped.device)

    playback = _PolicyPlaybackGate()
    playback.policy_total = T
    playback.dance_title = DANCE_NAME or os.path.splitext(os.path.basename(h5_path))[0]
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

    if playback.record_avi_enabled and not args_cli.real_time:
        print(
            "[INFO] Record AVI: policy rollout runs at full sim speed; "
            f"viewport AVI exported at {_RECORD_AVI_FPS:.0f} fps after rollout."
        )

    pose_log = PolicyRolloutPoseLog()
    pose_log_active = False

    def _avi_pre_render() -> None:
        try:
            env_unwrapped.sim.render()
        except Exception:
            pass

    def _export_avi_from_pose_log(*, reason: str) -> None:
        nonlocal pose_log_active
        if len(pose_log) <= 0:
            print(f"[WARN] AVI export skipped ({reason}): no pose snapshots.")
            pose_log_active = False
            return
        label = f"eval_{playback.dance_title}"
        out_path = default_playback_avi_path(media_dir=MEDIA_DIR, motion_label=label)
        print(
            f"[INFO] Exporting AVI from {len(pose_log)} pose snapshots @ {_RECORD_AVI_FPS:.0f} fps "
            f"(control_hz={control_hz:.1f}) -> {out_path}"
        )
        recorder = ViewportAviRecorder(fps=_RECORD_AVI_FPS, pre_render=_avi_pre_render)
        try:
            recorder.start(out_path)
            export_pose_log_to_avi(
                env_unwrapped,
                pose_log,
                recorder,
                export_fps=_RECORD_AVI_FPS,
                control_hz=control_hz,
                pre_render=_avi_pre_render,
            )
            frame_count = int(recorder.frame_count)
            recorder_fps = float(recorder.fps)
            out_final = recorder.stop()
        except Exception as exc:
            print(f"[ERROR] AVI export failed ({reason}): {exc}")
            pose_log.clear()
            pose_log_active = False
            return
        pose_log.clear()
        pose_log_active = False
        if not out_final:
            print(f"[WARN] AVI export produced no frames ({reason})")
            return
        if playback.wav_path and os.path.isfile(playback.wav_path):
            mux_wav_into_avi(out_final, playback.wav_path, replace_original=True)
        expected_s = float(frame_count) / max(recorder_fps, 1e-6)
        print(
            f"[INFO] AVI export complete ({reason}): {out_final} "
            f"({frame_count} frames @ {recorder_fps:.1f} fps, "
            f"expected duration {expected_s:.2f}s)"
        )

    if not getattr(args_cli, "headless", False):
        from source.train_workflow.ui import eval_play_ui

        eval_play_ui.set_play_callback(playback.request_play)
        eval_play_ui.set_stop_callback(playback.request_stop)
        eval_play_ui.set_status_provider(playback.ui_status)
        eval_play_ui.set_audio_enabled_provider(lambda: playback.audio_enabled)
        eval_play_ui.set_audio_enabled_setter(
            lambda enabled: setattr(playback, "audio_enabled", bool(enabled))
        )
        eval_play_ui.set_record_avi_callbacks(
            lambda: playback.record_avi_enabled,
            lambda enabled: setattr(playback, "record_avi_enabled", bool(enabled)),
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

    def _reset_idle_wall_clock() -> None:
        nonlocal idle_wall_start, idle_steps_done
        idle_wall_start = None
        idle_steps_done = 0

    def _begin_rollout(*, episode_idx: int) -> None:
        nonlocal obs_raw, pose_log_active, rollout_wall_start, idle_wall_start, idle_steps_done
        audio_util.stop_wav()
        _reset_idle_wall_clock()
        obs_raw, _ = _env_reset(wrapped)
        if playback.record_avi_enabled and not getattr(args_cli, "headless", False):
            pose_log.clear()
            pose_log_active = True
        else:
            pose_log_active = False
        rollout_wall_start = time.perf_counter()
        if playback.wav_path and playback.audio_enabled:
            audio_util.play_wav_async(playback.wav_path)
        playback.playing = True
        playback.stop_requested = False
        playback.policy_step = 0
        print(f"[INFO] Policy rollout started (episode={episode_idx}, steps={T}).")

    def _abort_rollout(*, export_avi: bool = False) -> None:
        nonlocal obs_raw, pose_log_active, rollout_wall_start
        audio_util.stop_wav()
        playback.playing = False
        playback.stop_requested = False
        playback.policy_step = 0
        rollout_wall_start = None
        _reset_idle_wall_clock()
        if export_avi and pose_log_active:
            _export_avi_from_pose_log(reason="stop")
        else:
            pose_log.clear()
            pose_log_active = False
        _reset_to_standing_pose(env_unwrapped)
        obs_raw = wrapped.get_observations()
        print("[INFO] Policy rollout stopped.")

    _reset_to_standing_pose(env_unwrapped)
    # Prime one idle step so the first rendered frame is already re-anchored.
    obs_raw, _, _, _ = _idle_hold_step(env_unwrapped, wrapped, zero_actions)
    if wait_for_key:
        assert start_listener is not None
        print(
            f"[INFO] Standing idle. Press [{start_listener.key_char}] or UI Play to start policy "
            f"({T} steps); repeats after each rollout."
        )
    else:
        print(f"[INFO] Auto-start: policy begins on motion reset ({T} steps).")
        _begin_rollout(episode_idx=0)

    episode = 0
    try:
        while simulation_app.is_running() and (
            int(args_cli.num_episodes) <= 0 or episode < int(args_cli.num_episodes)
        ):
            if playback.play_requested and not playback.playing:
                playback.play_requested = False
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
            skip_step_tracking_error = bool(args_cli.play)
            for _ in range(T):
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
                obs_raw, _, _, _ = wrapped.step(actions)
                if not skip_step_tracking_error:
                    with torch.no_grad():
                        err = joint_pos_tracking_error(env_unwrapped, h5_path, window_s)
                    rms = torch.sqrt(torch.mean(err.pow(2), dim=1))
                    err_accum += rms
                    max_per_step = torch.maximum(max_per_step, rms)
                steps_done += 1
                playback.policy_step = steps_done
                perf.on_step()
                if pose_log_active:
                    pose_log.append_from_env(env_unwrapped)
                if args_cli.real_time and rollout_wall_start is not None:
                    _wait_until_wall_deadline(rollout_wall_start + steps_done * dt)

            if rollout_aborted:
                playback.stop_requested = False
                _abort_rollout(export_avi=pose_log_active)
                continue

            audio_util.stop_wav()
            if pose_log_active:
                _export_avi_from_pose_log(reason="complete")
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
        audio_util.stop_wav()
        pose_log.clear()
        if start_listener is not None:
            start_listener.unsubscribe()

    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
