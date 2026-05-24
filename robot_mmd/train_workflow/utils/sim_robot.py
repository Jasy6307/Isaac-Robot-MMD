"""Isaac Sim articulation write helpers for playback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from robot_mmd.train_workflow.utils.trans_util import coerce_quat, quat_normalize

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def robot_root_row_clone(env: Any) -> Any | None:
    """Return CPU clone of robot root_state_w row 0, or None."""
    rs = getattr(env.unwrapped.scene["robot"].data, "root_state_w", None)
    if torch.is_tensor(rs) and rs.shape[1] >= 7:
        return rs[0].detach().cpu().clone()
    return None


def apply_joint_state_instant(env: Any, joint_pos_cmd: Any, joint_ids: Any) -> bool:
    """Write joint positions directly into simulation. Returns True on success."""
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]
    joint_pos_tensor = torch.tensor(joint_pos_cmd, dtype=torch.float32, device=device).unsqueeze(0)
    joint_pos_tensor = joint_pos_tensor.repeat(num_envs, 1)
    joint_vel_tensor = torch.zeros_like(joint_pos_tensor)

    try:
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor, joint_ids=joint_ids)
        return True
    except TypeError:
        return False


def apply_root_pos_instant(
    env: Any,
    root_pos_xyz: tuple[float, float, float],
    root_quat_wxyz: Any = None,
) -> bool:
    """Write robot root pose into simulation and zero root velocities."""
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]

    root_state = robot.data.root_state_w
    fallback_wxyz = root_state[0, 3:7].detach().cpu().tolist()
    qwxyz = quat_normalize(coerce_quat(root_quat_wxyz, fallback_wxyz))

    root_pose = torch.tensor(
        [root_pos_xyz[0], root_pos_xyz[1], root_pos_xyz[2], qwxyz[0], qwxyz[1], qwxyz[2], qwxyz[3]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    root_pose = root_pose.repeat(num_envs, 1)

    state = robot.data.root_state_w.clone()
    state[:, 0:3] = root_pose[:, 0:3]
    state[:, 3:7] = root_pose[:, 3:7]
    if state[:, 3:7].abs().sum() < 1e-6:
        state[:, 3:7] = 0.0
        state[:, 6] = 1.0
    state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(state)
    return True
