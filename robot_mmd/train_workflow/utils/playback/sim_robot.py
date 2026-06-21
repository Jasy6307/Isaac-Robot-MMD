"""Isaac Sim articulation write helpers for playback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

import isaaclab.utils.math as math_utils

from robot_mmd.train_workflow.utils.math.trans_util import coerce_quat, quat_normalize

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

# Default root pose PD gains for --pd_drive (no gravity, external wrench on pelvis).
# Raised from conservative values to improve large-displacement root tracking.
# Keep kd roughly proportional to kp to avoid underdamped oscillation.
ROOT_PD_KP_POS = 420.0
ROOT_PD_KD_POS = 84.0
ROOT_PD_KP_ROT = 120.0
ROOT_PD_KD_ROT = 24.0
ROOT_PD_BODY_ID = 0


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
    except (TypeError, RuntimeError):
        return False


def apply_root_pos_instant(
    env: Any,
    root_pos_xyz: tuple[float, float, float],
    root_quat_wxyz: Any = None,
) -> bool:
    """Write robot root pose into simulation while preserving root velocities."""
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
    # Kinematic teleport: zero root velocity so the next physics substep does not
    # integrate stale angular momentum into orientation drift (blue vs cyan root debug).
    state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(state)
    return True


def _root_pd_gains(
    kp_pos: float | None,
    kd_pos: float | None,
    kp_rot: float | None,
    kd_rot: float | None,
) -> tuple[float, float, float, float]:
    return (
        float(ROOT_PD_KP_POS if kp_pos is None else kp_pos),
        float(ROOT_PD_KD_POS if kd_pos is None else kd_pos),
        float(ROOT_PD_KP_ROT if kp_rot is None else kp_rot),
        float(ROOT_PD_KD_ROT if kd_rot is None else kd_rot),
    )


def clear_root_pd_wrench(env: Any) -> None:
    """Disable external root wrench buffers (call when playback stops/resets)."""
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.root_state_w.shape[0]
    zero_f = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
    zero_t = torch.zeros_like(zero_f)
    try:
        robot.set_external_force_and_torque(
            zero_f,
            zero_t,
            body_ids=[ROOT_PD_BODY_ID],
        )
    except Exception:
        pass


def apply_root_pd_track(
    env: Any,
    root_pos_xyz: tuple[float, float, float],
    root_quat_wxyz: Any = None,
    *,
    kp_pos: float | None = None,
    kd_pos: float | None = None,
    kp_rot: float | None = None,
    kd_rot: float | None = None,
) -> bool:
    """Track root pose through physics via external wrench PD on the pelvis."""
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.root_state_w.shape[0]
    kp_p, kd_p, kp_r, kd_r = _root_pd_gains(kp_pos, kd_pos, kp_rot, kd_rot)

    root_state = robot.data.root_state_w
    cur_pos = root_state[:, 0:3]
    cur_quat = math_utils.quat_unique(root_state[:, 3:7])
    cur_lin_vel = root_state[:, 7:10]
    cur_ang_vel = root_state[:, 10:13]

    target_pos = torch.tensor(
        [[float(root_pos_xyz[0]), float(root_pos_xyz[1]), float(root_pos_xyz[2])]],
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)
    fallback_wxyz = cur_quat[0].detach().cpu().tolist()
    qwxyz = quat_normalize(coerce_quat(root_quat_wxyz, fallback_wxyz))
    target_quat = torch.tensor(
        [[qwxyz[0], qwxyz[1], qwxyz[2], qwxyz[3]]],
        dtype=torch.float32,
        device=device,
    ).repeat(num_envs, 1)
    target_quat = math_utils.quat_unique(target_quat)

    pos_err = target_pos - cur_pos
    force_w = kp_p * pos_err - kd_p * cur_lin_vel

    q_err = math_utils.quat_mul(target_quat, math_utils.quat_conjugate(cur_quat))
    q_err = math_utils.quat_unique(q_err)
    axis_angle = math_utils.axis_angle_from_quat(q_err)
    torque_w = kp_r * axis_angle - kd_r * cur_ang_vel

    force_l = math_utils.quat_apply_inverse(cur_quat, force_w)
    torque_l = math_utils.quat_apply_inverse(cur_quat, torque_w)

    try:
        robot.set_external_force_and_torque(
            force_l.unsqueeze(1),
            torque_l.unsqueeze(1),
            body_ids=[ROOT_PD_BODY_ID],
        )
        return True
    except Exception:
        lin_vel_cmd = kp_p * pos_err - kd_p * cur_lin_vel
        ang_vel_cmd = kp_r * axis_angle - kd_r * cur_ang_vel
        root_vel = torch.cat([lin_vel_cmd, ang_vel_cmd], dim=-1)
        try:
            robot.write_root_velocity_to_sim(root_vel)
            return True
        except Exception:
            return False
