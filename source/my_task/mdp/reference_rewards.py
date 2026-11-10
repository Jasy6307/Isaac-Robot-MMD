"""Unified G1 reference objectives, independent of action labels and clip time.

Reference FK uses the exact runtime joint baseline and resampled motion buffer.
Positions, orientations and velocities of feet are measured in the pelvis frame.
The policy/observation layout is unchanged; legacy rewards remain importable.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import isaaclab.utils.math as math_utils
from source import sim_compat
from source.my_task.motion_reference import get_or_create_motion_buffer, motion_steps
from source.my_task.mdp import jump
from source.my_task.mdp.root_reference import root_reference_quat_w, root_yaw_error_rad
from source.train_workflow.utils.reference_objectives import tracking_score, contact_score, exploration_strength


def _tables(env):
    cache = getattr(env, "_reference_reward_tables", None)
    if cache is not None:
        return cache
    from source.train_workflow.utils.motion.self_collision import G1CollisionModel
    robot = env.scene["robot"]
    action = env.cfg.actions.joint_pos
    buf = get_or_create_motion_buffer(env, action.motion_h5_path, action.motion_window_seconds)
    steps = torch.arange(buf.num_steps, device=env.device)
    q = buf.q_ref_abs(steps).detach().cpu()
    model = G1CollisionModel(Path(env.cfg.scene.robot.spawn.usd_path), list(robot.joint_names), "cpu")
    with torch.no_grad():
        rotations, positions = model.fk(q)
    pos = np.stack([positions[f"{side}_ankle_roll_link"].numpy() for side in ("left", "right")], axis=1)
    quat = np.stack([Rotation.from_matrix(rotations[f"{side}_ankle_roll_link"].numpy()).as_quat()
                     for side in ("left", "right")], axis=1)
    # Append an identical last frame: held targets have exactly zero velocity.
    velocity = np.diff(np.concatenate((pos, pos[-1:]), axis=0), axis=0) / env.step_dt
    weights = torch.zeros(len(robot.joint_names), device=env.device)
    for j, name in enumerate(robot.joint_names):
        if "waist_" in name or any(x in name for x in ("hip_", "knee_")):
            weights[j] = 1. if "hip_yaw" in name else 2.5
        elif "ankle_" in name:
            weights[j] = .5
    cache = dict(buf=buf, ids=[list(robot.body_names).index(f"{side}_ankle_roll_link") for side in ("left", "right")],
                 pos=torch.as_tensor(pos, device=env.device, dtype=torch.float32),
                 quat=torch.as_tensor(quat, device=env.device, dtype=torch.float32),
                 vel=torch.as_tensor(velocity, device=env.device, dtype=torch.float32), weights=weights)
    report = env.cfg.auto_jump_report
    for key in ("contact_support_confidence", "contact_air_confidence"):
        if not report.get(key):
            raise ValueError("Unified rewards require a freshly analyzed v6 reference")
        cache[key] = torch.tensor(report[key], device=env.device, dtype=torch.float32)
    env._reference_reward_tables = cache
    return cache


def metrics(env):
    key = int(env.common_step_counter)
    cached = getattr(env, "_reference_reward_metrics", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    table = _tables(env)
    robot = env.scene["robot"]
    data = robot.data
    buf = table["buf"]
    steps = motion_steps(env)
    indices = steps.long().clamp(0, len(table["pos"])-1)
    root = data.root_state_w.torch
    q = root[:, 3:7]
    foot_pose = data.body_link_pose_w.torch[:, table["ids"]]
    offset = foot_pose[:, :, :3] - root[:, None, :3]
    qfeet = q[:, None, :].expand(-1, 2, -1)
    local_pos = math_utils.quat_apply_inverse(qfeet, offset)
    relative_v = data.body_link_lin_vel_w.torch[:, table["ids"]] - data.root_lin_vel_w.torch[:, None, :]
    relative_v -= torch.cross(data.root_ang_vel_w.torch[:, None, :].expand_as(offset), offset, dim=-1)
    local_v = math_utils.quat_apply_inverse(qfeet, relative_v)
    local_q = math_utils.quat_mul(math_utils.quat_conjugate(qfeet), foot_pose[:, :, 3:7])
    local_q = local_q / local_q.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    dots = (local_q * table["quat"][indices]).sum(-1).square().clamp(0., 1.)
    origins = sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
    ref_pos = data.default_root_state.torch[:, :3] + origins + buf.root_pos_delta(steps)
    ref_v = (buf.root_pos_delta(steps+1) - buf.root_pos_delta(steps)) / env.step_dt
    ref_q = root_reference_quat_w(robot, env, h5_path=env.cfg.actions.joint_pos.motion_h5_path,
                                 window_seconds=env.cfg.actions.joint_pos.motion_window_seconds, steps=steps)
    up = lambda x: math_utils.quat_apply(x, x.new_tensor([0., 0., 1.]).expand(x.shape[0], -1))
    tilt = torch.acos((up(q) * up(ref_q)).sum(-1).clamp(-1., 1.))
    yaw = root_yaw_error_rad(env, robot, h5_path=env.cfg.actions.joint_pos.motion_h5_path,
                             window_seconds=env.cfg.actions.joint_pos.motion_window_seconds, steps=steps)
    scales = env.cfg.reference_reward_scales
    weights = table["weights"]
    joint_error = ((data.joint_pos.torch-buf.q_ref_abs(steps)).square()*weights).sum(-1)/weights.sum().clamp_min(1.)
    report = env.cfg.auto_jump_report
    ri = (steps.float()*env.step_dt*report["fps"]).long().clamp(0, len(table["contact_support_confidence"])-1)
    result = dict(
        joint=tracking_score(joint_error/scales["joint_rad"]**2),
        root_position=tracking_score(((root[:, :3]-ref_pos)/root.new_tensor(scales["root_position_m"])).square().mean(-1)),
        root_orientation=tracking_score(.5*((yaw/scales["yaw_rad"]).square()+(tilt/scales["tilt_rad"]).square())),
        root_velocity=tracking_score(((data.root_lin_vel_w.torch-ref_v)/scales["root_velocity_m_s"]).square().mean(-1)),
        foot_position=tracking_score(((local_pos-table["pos"][indices])/scales["foot_position_m"]).square().sum(-1)).mean(-1),
        foot_orientation=tracking_score(4.*(1.-dots)/scales["foot_orientation_rad"]**2).mean(-1),
        foot_velocity=tracking_score(((local_v-table["vel"][indices])/scales["foot_velocity_m_s"]).square().sum(-1)).mean(-1),
        contact=contact_score(jump.full_step_ground_contacts(env), table["contact_support_confidence"][ri], table["contact_air_confidence"][ri]),
    )
    env._reference_reward_metrics = (key, result)
    return result


def joint_tracking(env, h5_path, window_seconds):
    controller = getattr(env, "_auto_jump_controller", None)
    if controller is not None:
        controller.observe()
    return metrics(env)["joint"]


def root_position(env):
    return metrics(env)["root_position"]


def root_orientation(env):
    return metrics(env)["root_orientation"]


def root_velocity(env):
    return metrics(env)["root_velocity"]


def foot_position(env):
    return metrics(env)["foot_position"]


def foot_orientation(env):
    return metrics(env)["foot_orientation"]


def foot_velocity(env):
    return metrics(env)["foot_velocity"]


def contact_tracking(env):
    return metrics(env)["contact"]


def jump_exploration(env):
    """Small temporary aid, not a permanent flight/landing objective."""
    controller = getattr(env, "_auto_jump_controller", None)
    strength = exploration_strength(controller.progress if controller is not None else 1.)
    if strength == 0.:
        return torch.zeros(env.num_envs, device=env.device)
    state = jump.flight_state(env)
    return strength * (state["stance"] & state["rising"] & jump.phases(env)[:, 2]).float() * (state["air"].float()*env.step_dt/.12).clamp(max=1.)
