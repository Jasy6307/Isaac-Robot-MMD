"""HDF5 dance motion reference buffer for RL tasks.

Loads a single dance HDF5 file (compiled via ``compile_csv_motion_to_hdf5_motion``)
and exposes per-control-step reference joint targets resampled to the env control
frequency. The buffer is cached on the ``ManagerBasedRLEnv`` instance so all MDP
terms (observations / rewards / events) share a single resampling.

Key conventions
---------------
* HDF5 ``joint_pos_delta`` stores ``angle`` deltas to be **added** to the runtime
  ``default_joint_pos`` (see ``build_joint_positions_from_frame``). For joints
  missing from the HDF5 source, the delta is 0, i.e. fall back to default.
* The buffer keeps a single absolute reference table ``q_ref_abs`` of shape
  ``[T, J]`` for the runtime joint order, plus per-step ``motion_phase`` in
  ``[0, 1]``.
* The reference is clamped to a configurable window length (e.g. 10s) and
  internally held on the same device as the simulation tensors.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from robot_mmd.train_workflow.utils.hdf5_motion import Hdf5Motion, load_hdf5_motion

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ENV_BUFFER_ATTR = "_g1_dance_motion_buffers"
_ENV_START_STEPS_ATTR = "_g1_dance_motion_start_steps"


def _resolve_h5_path(h5_path: str) -> str:
    if os.path.isabs(h5_path) and os.path.exists(h5_path):
        return h5_path
    # try resolving relative to repo root (parent of robot_mmd)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    cand = os.path.normpath(os.path.join(repo_root, h5_path))
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"HDF5 motion file not found: {h5_path}")


def _ensure_motion_start_steps(env: "ManagerBasedRLEnv") -> torch.Tensor:
    steps: torch.Tensor | None = getattr(env, _ENV_START_STEPS_ATTR, None)
    num_envs = int(env.num_envs)
    if steps is None or steps.shape[0] != num_envs:
        steps = torch.zeros((num_envs,), dtype=torch.long, device=env.device)
        setattr(env, _ENV_START_STEPS_ATTR, steps)
        return steps
    if steps.device != torch.device(env.device):
        steps = steps.to(device=env.device, dtype=torch.long)
        setattr(env, _ENV_START_STEPS_ATTR, steps)
        return steps
    return steps


def set_motion_start_steps(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, start_steps: torch.Tensor
) -> None:
    """Set per-env motion start indices used by reference lookups."""
    if env_ids.numel() == 0:
        return
    steps = _ensure_motion_start_steps(env)
    env_ids_i64 = env_ids.to(device=steps.device, dtype=torch.long)
    start_i64 = start_steps.to(device=steps.device, dtype=torch.long).clamp_min(0)
    steps[env_ids_i64] = start_i64


def reset_motion_start_steps(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Reset selected env motion starts to frame 0."""
    if env_ids.numel() == 0:
        return
    steps = _ensure_motion_start_steps(env)
    env_ids_i64 = env_ids.to(device=steps.device, dtype=torch.long)
    steps[env_ids_i64] = 0


def motion_steps(env: "ManagerBasedRLEnv", offset: int = 0) -> torch.Tensor:
    """Unified reference index: ``motion_start_step + episode_length_buf + offset``."""
    starts = _ensure_motion_start_steps(env)
    episode_steps = env.episode_length_buf.to(device=starts.device, dtype=torch.long)
    return starts + episode_steps + int(offset)


class DanceMotionReferenceBuffer:
    """Resampled reference motion table for a single dance HDF5.

    The constructor performs CPU-side loading and resampling; tensors are moved
    to ``device`` lazily on first GPU access.
    """

    def __init__(
        self,
        h5_path: str,
        runtime_joint_names: list[str],
        default_joint_pos: np.ndarray,
        control_hz: float,
        window_seconds: float,
        device: torch.device | str,
    ) -> None:
        if window_seconds <= 0.0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if control_hz <= 0.0:
            raise ValueError(f"control_hz must be > 0, got {control_hz}")

        self._h5_path = _resolve_h5_path(h5_path)
        self._runtime_joint_names = list(runtime_joint_names)
        self._default_joint_pos_np = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        if self._default_joint_pos_np.shape[0] != len(self._runtime_joint_names):
            raise ValueError(
                "default_joint_pos length mismatch: "
                f"{self._default_joint_pos_np.shape[0]} vs {len(self._runtime_joint_names)}"
            )
        self._control_hz = float(control_hz)
        self._window_seconds = float(window_seconds)
        self._device = torch.device(device)

        motion: Hdf5Motion = load_hdf5_motion(self._h5_path)
        idx = motion.build_runtime_joint_index(self._runtime_joint_names)

        h5_fps = float(motion.fps)
        h5_total_frames = int(motion.joint_pos_delta.shape[0])
        if h5_total_frames <= 0:
            raise ValueError(f"Empty HDF5 motion: {self._h5_path}")

        # number of control steps in the window
        T = int(round(self._window_seconds * self._control_hz))
        if T < 2:
            T = 2

        # Per control step t -> equivalent H5 frame (linear interpolation)
        step_idx = np.arange(T, dtype=np.float64)
        frame_pos = step_idx * (h5_fps / self._control_hz)
        # Clamp to last valid frame
        last_valid = float(h5_total_frames - 1)
        frame_pos = np.clip(frame_pos, 0.0, last_valid)
        f0 = np.floor(frame_pos).astype(np.int64)
        f1 = np.minimum(f0 + 1, int(last_valid))
        alpha = (frame_pos - f0).astype(np.float32)

        J = len(self._runtime_joint_names)
        delta = np.zeros((T, J), dtype=np.float32)
        valid = idx >= 0
        if np.any(valid):
            valid_idx = idx[valid]
            d0 = motion.joint_pos_delta[f0][:, valid_idx]  # [T, Jv]
            d1 = motion.joint_pos_delta[f1][:, valid_idx]
            d_interp = d0 + (d1 - d0) * alpha[:, None]
            delta[:, valid] = d_interp.astype(np.float32, copy=False)

        # q_ref_abs[t, j] = default_joint_pos[j] + delta[t, j]
        q_ref_abs_np = self._default_joint_pos_np[None, :] + delta
        # q_ref_rel[t, j] = delta[t, j]
        q_ref_rel_np = delta

        # Root-pose reference (used by C1/C2 root tracking rewards).
        # Apply the same temporal interpolation.
        root_pos_delta_np = (
            motion.root_pos_delta[f0]
            + (motion.root_pos_delta[f1] - motion.root_pos_delta[f0]) * alpha[:, None]
        ).astype(np.float32)
        # Quaternion interpolation: simple normalize-then-lerp; OK for 30Hz adjacent frames.
        q_ref_root_quat_np = (
            motion.root_quat_delta_wxyz[f0]
            + (motion.root_quat_delta_wxyz[f1] - motion.root_quat_delta_wxyz[f0]) * alpha[:, None]
        ).astype(np.float32)
        norm = np.linalg.norm(q_ref_root_quat_np, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        q_ref_root_quat_np = q_ref_root_quat_np / norm

        # Phase in [0, 1]: 0 at first step, 1 at last step.
        if T > 1:
            phase_np = (step_idx / float(T - 1)).astype(np.float32)
        else:
            phase_np = np.zeros((1,), dtype=np.float32)

        # Stash CPU tensors; move on first access via `to(device)`.
        self._q_ref_abs = torch.as_tensor(q_ref_abs_np, dtype=torch.float32)
        self._q_ref_rel = torch.as_tensor(q_ref_rel_np, dtype=torch.float32)
        self._root_pos_delta = torch.as_tensor(root_pos_delta_np, dtype=torch.float32)
        self._root_quat_wxyz = torch.as_tensor(q_ref_root_quat_np, dtype=torch.float32)
        self._phase = torch.as_tensor(phase_np, dtype=torch.float32)
        self._on_device = False

        # Cache for missing joints log
        self._num_steps = T
        self._h5_fps = h5_fps

    def _to_device(self) -> None:
        if self._on_device:
            return
        self._q_ref_abs = self._q_ref_abs.to(self._device)
        self._q_ref_rel = self._q_ref_rel.to(self._device)
        self._root_pos_delta = self._root_pos_delta.to(self._device)
        self._root_quat_wxyz = self._root_quat_wxyz.to(self._device)
        self._phase = self._phase.to(self._device)
        self._on_device = True

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def h5_path(self) -> str:
        return self._h5_path

    @property
    def control_hz(self) -> float:
        return self._control_hz

    @property
    def h5_fps(self) -> float:
        return self._h5_fps

    def _clamp_steps(self, steps: torch.Tensor, offset: int = 0) -> torch.Tensor:
        idx = steps + offset
        return idx.clamp_(min=0, max=self._num_steps - 1)

    def q_ref_rel(self, steps: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Return reference joint deltas (vs runtime default) for given env step indices.

        Args:
            steps: int64 tensor of shape ``[num_envs]``.
            offset: integer step lookahead (e.g. 1 for next-step reference).
        Returns:
            Tensor of shape ``[num_envs, J]``.
        """
        self._to_device()
        idx = self._clamp_steps(steps.to(dtype=torch.long, device=self._device), offset)
        return self._q_ref_rel.index_select(0, idx)

    def q_ref_abs(self, steps: torch.Tensor, offset: int = 0) -> torch.Tensor:
        self._to_device()
        idx = self._clamp_steps(steps.to(dtype=torch.long, device=self._device), offset)
        return self._q_ref_abs.index_select(0, idx)

    def motion_phase(self, steps: torch.Tensor) -> torch.Tensor:
        self._to_device()
        idx = self._clamp_steps(steps.to(dtype=torch.long, device=self._device), 0)
        return self._phase.index_select(0, idx).unsqueeze(-1)

    def q_ref_abs_first(self) -> torch.Tensor:
        """Return reference joint positions at step 0 on device, shape [J]."""
        self._to_device()
        return self._q_ref_abs[0]

    def root_pos_delta(self, steps: torch.Tensor) -> torch.Tensor:
        self._to_device()
        idx = self._clamp_steps(steps.to(dtype=torch.long, device=self._device), 0)
        return self._root_pos_delta.index_select(0, idx)

    def root_quat_wxyz(self, steps: torch.Tensor) -> torch.Tensor:
        self._to_device()
        idx = self._clamp_steps(steps.to(dtype=torch.long, device=self._device), 0)
        return self._root_quat_wxyz.index_select(0, idx)


def get_or_create_motion_buffer(
    env: "ManagerBasedRLEnv",
    h5_path: str,
    window_seconds: float,
    asset_name: str = "robot",
) -> DanceMotionReferenceBuffer:
    """Fetch or lazily build the dance reference buffer cached on the env.

    Multiple HDF5 files can coexist (keyed by absolute path + window). For
    Phase 1 we only use one, but keeping a dict keeps it future-proof.
    """
    from isaaclab.assets import Articulation

    cache: dict[tuple[str, float], DanceMotionReferenceBuffer]
    cache = getattr(env, _ENV_BUFFER_ATTR, None)  # type: ignore[assignment]
    if cache is None:
        cache = {}
        setattr(env, _ENV_BUFFER_ATTR, cache)

    abs_path = _resolve_h5_path(h5_path)
    key = (abs_path, float(window_seconds))
    if key in cache:
        return cache[key]

    asset: Articulation = env.scene[asset_name]
    joint_names: list[str] = list(asset.joint_names)
    default_joint_pos = asset.data.default_joint_pos[0].detach().cpu().numpy()
    control_hz = 1.0 / float(env.step_dt)

    buf = DanceMotionReferenceBuffer(
        h5_path=abs_path,
        runtime_joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        control_hz=control_hz,
        window_seconds=float(window_seconds),
        device=env.device,
    )
    cache[key] = buf
    print(
        f"[INFO] DanceMotionReferenceBuffer ready: {os.path.basename(abs_path)} "
        f"steps={buf.num_steps} ctrl_hz={control_hz:.1f} h5_fps={buf.h5_fps:.1f}"
    )
    return buf
