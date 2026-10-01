"""Open-loop root pose tracking from HDF5 dance reference."""

from __future__ import annotations

from source import sim_compat

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation

from source.my_task.motion_reference import get_or_create_motion_buffer, motion_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    return env.scene.sim.get_clone_plan().positions


def root_reference_pose_w(
    asset: Articulation,
    env: "ManagerBasedRLEnv",
    *,
    h5_path: str,
    window_seconds: float,
    asset_name: str = "robot",
    steps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame root position and native Lab quaternion (xyzw)."""
    buf = get_or_create_motion_buffer(
        env,
        h5_path,
        window_seconds,
        asset_name=asset_name,
    )
    if steps is None:
        steps = motion_steps(env)

    env_origin = cloner_env_origins(env)
    p_anchor = asset.data.default_root_state.torch[:, 0:3]
    p_delta = buf.root_pos_delta(steps)
    target_pos = p_anchor + env_origin + p_delta

    q_anchor = math_utils.quat_unique(asset.data.default_root_state.torch[:, 3:7])
    q_delta = math_utils.quat_unique(sim_compat.wxyz_to_xyzw(buf.root_quat_wxyz(steps)))
    target_quat = math_utils.quat_unique(math_utils.quat_mul(q_delta, q_anchor))
    return target_pos, target_quat


def root_yaw_error_rad(
    env: "ManagerBasedRLEnv",
    asset: Articulation,
    *,
    h5_path: str,
    window_seconds: float,
    asset_name: str = "robot",
    steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed root yaw error (current - reference) in ``[-pi, pi]``, shape ``[num_envs]``."""
    _, q_ref_xyzw = root_reference_pose_w(
        asset,
        env,
        h5_path=h5_path,
        window_seconds=window_seconds,
        asset_name=asset_name,
        steps=steps,
    )
    q_cur_xyzw = math_utils.quat_unique(asset.data.root_quat_w.torch)
    _, _, yaw_cur = math_utils.euler_xyz_from_quat(q_cur_xyzw)
    _, _, yaw_ref = math_utils.euler_xyz_from_quat(q_ref_xyzw)
    return torch.atan2(torch.sin(yaw_cur - yaw_ref), torch.cos(yaw_cur - yaw_ref))


def write_root_reference_from_motion(
    env: "ManagerBasedRLEnv",
    asset: Articulation,
    *,
    h5_path: str,
    window_seconds: float,
    asset_name: str = "robot",
    steps: torch.Tensor | None = None,
) -> None:
    """Teleport root to the current H5 reference pose (playback-style open-loop root)."""
    target_pos, target_quat = root_reference_pose_w(
        asset,
        env,
        h5_path=h5_path,
        window_seconds=window_seconds,
        asset_name=asset_name,
        steps=steps,
    )
    root_pose = torch.cat([target_pos, target_quat], dim=-1)
    sim_compat.write_root_pose(asset, root_pose)
    sim_compat.write_root_velocity(asset,
        torch.zeros((asset.data.root_state_w.torch.shape[0], 6), device=asset.device, dtype=torch.float32)
    )
