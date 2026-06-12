"""Open-loop root pose tracking from HDF5 dance reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation

from robot_mmd.my_task.motion_reference import get_or_create_motion_buffer, motion_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cloner_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Env origins used by GridCloner (actual robot spawn grid when available)."""
    scene = env.scene
    cloner_origins = scene._default_env_origins  # noqa: SLF001 — intentional
    if cloner_origins is not None:
        return cloner_origins
    return scene.env_origins


def root_reference_pose_w(
    asset: Articulation,
    env: "ManagerBasedRLEnv",
    *,
    h5_path: str,
    window_seconds: float,
    asset_name: str = "robot",
    steps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame root position and quaternion (wxyz) from the motion buffer."""
    buf = get_or_create_motion_buffer(
        env,
        h5_path,
        window_seconds,
        asset_name=asset_name,
    )
    if steps is None:
        steps = motion_steps(env)

    env_origin = cloner_env_origins(env)
    p_anchor = asset.data.default_root_state[:, 0:3]
    p_delta = buf.root_pos_delta(steps)
    target_pos = p_anchor + env_origin + p_delta

    q_anchor = math_utils.quat_unique(asset.data.default_root_state[:, 3:7])
    q_delta = math_utils.quat_unique(buf.root_quat_wxyz(steps))
    target_quat = math_utils.quat_unique(math_utils.quat_mul(q_delta, q_anchor))
    return target_pos, target_quat


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
    asset.write_root_pose_to_sim(root_pose)
    asset.write_root_velocity_to_sim(
        torch.zeros((asset.data.root_state_w.shape[0], 6), device=asset.device, dtype=torch.float32)
    )
