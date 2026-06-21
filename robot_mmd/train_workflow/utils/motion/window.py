"""Helpers for mapping motion window CLI args (frames) to env cfg seconds."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedEnvCfg


def control_hz_from_env_cfg(env_cfg: ManagerBasedEnvCfg) -> float:
    """Sim control rate: 1 / (physics_dt * decimation)."""
    dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    if dt <= 0.0:
        raise ValueError(f"env_cfg.sim.dt must be > 0, got {dt}")
    if decimation <= 0:
        raise ValueError(f"env_cfg.decimation must be > 0, got {decimation}")
    return 1.0 / (dt * float(decimation))


def window_seconds_from_frames(window_frames: int, control_hz: float) -> float:
    """Convert frame count to window seconds for an exact control-step window.

    ``DanceMotionReferenceBuffer`` uses ``T = round(window_seconds * control_hz)``.
    With ``window_seconds = frames / control_hz``, ``T == frames`` exactly.
    """
    wf = int(window_frames)
    if wf <= 0:
        raise ValueError(f"window_frames must be > 0, got {window_frames}")
    hz = float(control_hz)
    if hz <= 0.0:
        raise ValueError(f"control_hz must be > 0, got {control_hz}")
    return float(wf) / hz


def default_window_seconds_from_env_cfg(env_cfg: ManagerBasedEnvCfg) -> float:
    """Read the motion reference window from env cfg defaults (seconds)."""
    reset_evt = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_evt is not None and hasattr(reset_evt, "params"):
        ws = reset_evt.params.get("window_seconds")
        if ws is not None:
            return float(ws)
    return float(env_cfg.episode_length_s)


def resolve_motion_window_seconds(
    env_cfg: ManagerBasedEnvCfg,
    *,
    window_frames: int | None,
) -> float | None:
    """Convert CLI ``--window_frames`` override to seconds, or return None."""
    if window_frames is None:
        return None
    control_hz = control_hz_from_env_cfg(env_cfg)
    return window_seconds_from_frames(int(window_frames), control_hz)


def log_window_frames_override(window_frames: int, window_seconds: float, control_hz: float) -> None:
    print(
        f"[INFO] --window_frames={int(window_frames)} "
        f"=> window_seconds={window_seconds:.6f} "
        f"(control_hz={control_hz:.1f}, steps={int(window_frames)})"
    )
