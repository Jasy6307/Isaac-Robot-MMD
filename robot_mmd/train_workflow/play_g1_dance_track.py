# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a trained G1 dance tracking checkpoint and report joint
tracking error against the reference HDF5 motion.

Example
-------
.. code-block:: powershell

    ./isaac_workspace/IsaacLab/isaaclab.bat -p robot_mmd/train_workflow/play_g1_dance_track.py `
      --task Isaac-G1-Dance-Track-C1-v0 --num_envs 16

Press ``--start_key`` (default ``P``) to begin each policy rollout from a stable
standing pose; when the window finishes, the robot returns to standing and waits
again. Use ``--auto_start`` to run immediately without a key press.
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

parser = argparse.ArgumentParser(description="Play a G1 dance tracking checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-G1-Dance-Track-C1-v0")
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
    "--motion_h5",
    type=str,
    default=None,
    help="Override dance HDF5 path for evaluation.",
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
parser.add_argument("--real_time", action="store_true", default=False)
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

import robot_mmd.my_task  # noqa: F401, E402

# Play-only default viewport (overrides task env_cfg.viewer).
_PLAY_VIEWER_EYE = (0.0, 4.0, 1.0)
_PLAY_VIEWER_LOOKAT = (0.0, -4.0, 0.0)

from robot_mmd.my_task.mdp.events import reset_root_to_spawn  # noqa: E402
from robot_mmd.my_task.mdp.observations import joint_pos_tracking_error  # noqa: E402
from robot_mmd.my_task.motion_reference import (  # noqa: E402
    get_or_create_motion_buffer,
    reset_motion_start_steps,
)
from robot_mmd.train_workflow.g1_deploy_actuator_cfg import (  # noqa: E402
    apply_pd_profile_to_scene_robot,
    log_pd_profile_summary,
)
from robot_mmd.train_workflow.utils.motion_window import (  # noqa: E402
    control_hz_from_env_cfg,
    log_window_frames_override,
    resolve_motion_window_seconds,
)


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
        args_cli.motion_h5 is None
        and args_cli.window_frames is None
        and args_cli.residual_alpha is None
        and args_cli.use_reference_residual is None
    ):
        return
    new_h5 = os.path.abspath(args_cli.motion_h5) if args_cli.motion_h5 is not None else None
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

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Looking for checkpoints under: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        run_dir = args_cli.load_run if args_cli.load_run is not None else agent_cfg.load_run
        resume_path = get_checkpoint_path(
            log_root_path, run_dir, agent_cfg.load_checkpoint
        )
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

    start_listener: _StartKeyListener | None = None
    wait_for_key = not bool(args_cli.auto_start)
    if wait_for_key:
        start_listener = _StartKeyListener(args_cli.start_key)
        if not start_listener.subscribe():
            print("[WARN] Keyboard unavailable; falling back to --auto_start.")
            wait_for_key = False

    _reset_to_standing_pose(env_unwrapped)
    # Prime one idle step so the first rendered frame is already re-anchored.
    obs_raw, _, _, _ = _idle_hold_step(env_unwrapped, wrapped, zero_actions)
    playing = not wait_for_key
    if wait_for_key:
        assert start_listener is not None
        print(
            f"[INFO] Standing idle. Press [{start_listener.key_char}] to start policy "
            f"({T} steps); repeats after each rollout."
        )
    else:
        print(f"[INFO] Auto-start: policy begins on motion reset ({T} steps).")
        obs_raw, _ = _env_reset(wrapped)

    episode = 0
    try:
        while simulation_app.is_running() and (
            int(args_cli.num_episodes) <= 0 or episode < int(args_cli.num_episodes)
        ):
            if not playing:
                if start_listener is not None and start_listener.consume():
                    playing = True
                    obs_raw, _ = _env_reset(wrapped)
                    print(f"[INFO] Policy rollout started (episode={episode}, steps={T}).")
                else:
                    start_time = time.time()
                    obs_raw, _, _, _ = _idle_hold_step(env_unwrapped, wrapped, zero_actions)
                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

            err_accum = torch.zeros(env_unwrapped.num_envs, device=env_unwrapped.device)
            max_per_step = torch.zeros(env_unwrapped.num_envs, device=env_unwrapped.device)
            steps_done = 0
            smoothed_actions = None
            for _ in range(T):
                start_time = time.time()
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
                with torch.no_grad():
                    err = joint_pos_tracking_error(env_unwrapped, h5_path, window_s)
                rms = torch.sqrt(torch.mean(err.pow(2), dim=1))
                err_accum += rms
                max_per_step = torch.maximum(max_per_step, rms)
                steps_done += 1
                perf.on_step()
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            mean_rms = (err_accum / max(steps_done, 1)).mean().item()
            max_rms = max_per_step.mean().item()
            print(
                f"[EVAL] episode={episode} steps={steps_done} "
                f"mean_joint_rms={mean_rms:.4f} rad   peak_joint_rms={max_rms:.4f} rad"
            )
            perf.on_episode_end(episode)
            episode += 1

            if int(args_cli.num_episodes) > 0 and episode >= int(args_cli.num_episodes):
                break

            _reset_to_standing_pose(env_unwrapped)
            obs_raw = wrapped.get_observations()
            if wait_for_key:
                playing = False
                assert start_listener is not None
                print(
                    f"[INFO] Rollout finished. Standing idle — press [{start_listener.key_char}] "
                    "to play again."
                )
            else:
                obs_raw, _ = _env_reset(wrapped)
    finally:
        if start_listener is not None:
            start_listener.unsubscribe()

    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
