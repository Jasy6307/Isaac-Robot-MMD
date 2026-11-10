"""Generic phase rewards; no absolute dance time or reference-pose edits."""
import torch
from source.my_task.mdp import jump
from source.my_task.mdp.rewards import joint_pos_tracking_exp
from source.my_task.motion_reference import motion_steps


def _jump_gate(env, seconds):
    from source.train_workflow.utils.phase_gate import jump_teacher_gate
    cfg = env.cfg
    return jump_teacher_gate(
        seconds,
        getattr(cfg, "jump_phase_schedules", ()),
        pre_seconds=float(getattr(cfg, "jump_gate_pre_seconds", .6)),
        post_seconds=float(getattr(cfg, "jump_gate_post_seconds", .6)),
        ramp_seconds=float(getattr(cfg, "jump_gate_ramp_seconds", .3)),
    )


def tracking_metrics(env, h5_path, window_seconds):
    from source import sim_compat
    from source.my_task.motion_reference import get_or_create_motion_buffer
    from source.my_task.mdp.root_reference import root_reference_quat_w
    from source.train_workflow.utils.motion_quality import up_vector_xyzw
    key = int(env.common_step_counter)
    cached = getattr(env,"_auto_tracking_metrics",None)
    if cached is not None and cached[0]==key:
        return cached[1]
    robot = env.scene["robot"]
    ids = getattr(env,"_auto_tracking_leg_ids",None)
    if ids is None:
        ids = [i for i,name in enumerate(robot.joint_names) if any(x in name for x in ("hip_","knee_","ankle_"))]
        env._auto_tracking_leg_ids = ids
    buf = get_or_create_motion_buffer(env,h5_path,window_seconds)
    steps = motion_steps(env)
    reference = buf.q_ref_abs(steps)
    origins = sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
    reference_z = robot.data.default_root_state.torch[:,2]+origins[:,2]+buf.root_pos_delta(steps)[:,2]
    q_ref = root_reference_quat_w(robot,env,h5_path=h5_path,window_seconds=window_seconds,steps=steps)
    up = up_vector_xyzw(robot.data.root_state_w.torch[:,3:7])
    ref_up = up_vector_xyzw(q_ref)
    normal = 1.-_jump_gate(env, steps.float()*env.step_dt)
    result = dict(leg_rms_rad=(robot.data.joint_pos.torch[:,ids]-reference[:,ids]).square().mean(-1).sqrt(),
        height_error_m=(robot.data.root_state_w.torch[:,2]-reference_z).abs(),
        height_delta_m=robot.data.root_state_w.torch[:,2]-reference_z,
        tilt_error_rad=torch.acos((up*ref_up).sum(-1).clamp(-1.,1.)),normal_weight=normal)
    env._auto_tracking_metrics=(key,result)
    return result


def tracking_tail(env, h5_path, window_seconds):
    from source.train_workflow.utils.motion_quality import tracking_tail_cost
    values=tracking_metrics(env,h5_path,window_seconds)
    return tracking_tail_cost(values["leg_rms_rad"],values["height_error_m"],
                              values["tilt_error_rad"],values["normal_weight"])


def observation(env, h5_path, window_seconds):
    base = jump.jump_observation(env, h5_path, window_seconds)
    seconds = motion_steps(env).float() * env.step_dt
    progress = torch.zeros_like(seconds)
    if env.cfg.jump_phase_schedules:
        bounds = env._jump_phase_bounds[1].reshape(-1, 2)
        active = (seconds[:, None] >= bounds[None, :, 0]) & (seconds[:, None] < bounds[None, :, 1])
        progress = torch.where(active, (seconds[:, None] - bounds[None, :, 0]) /
                               (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-6), 0.).amax(dim=1)
    if getattr(env.cfg,"auto_jump_exact_residual",False):
        gate=_jump_gate(env, seconds)
        return torch.cat((base,progress[:,None],gate[:,None]),dim=-1)
    return torch.cat((base, progress[:, None]), dim=-1)


def tracking(env, h5_path, window_seconds=10., sigma=.12,
             joint_weight_by_expr=None, joint_weight_default=1.):
    controller = getattr(env, "_auto_jump_controller", None)
    if controller is not None:
        controller.observe()
    value = joint_pos_tracking_exp(env, h5_path, window_seconds, sigma,
                                   joint_weight_by_expr, joint_weight_default)
    phase = jump.phases(env)
    strength = getattr(env, "_auto_jump_strength", 1.)
    # Keep the original reference. Relax only high-confidence push/flight phases.
    return value * (1 - strength * torch.where(phase[:, 1], .7, torch.where(phase[:, 2], .5, 0.)))


def jump_leg_tracking(env, h5_path, window_seconds=10., sigma=.18,
                      joint_weight_by_expr=None, joint_weight_default=0.):
    """Strictly track hip/knee/ankle reference poses during inferred jumps.

    ``tracking`` intentionally relaxes the all-joint term during push and
    flight so that a bad early jump does not immediately destroy learning.
    That relaxation also lets the policy pass the old contact-only jump gate
    while its legs swing far away from the VMD.  This companion term restores
    a dense, reference-specific gradient for the lower body only while phases
    1--3 (push, flight, landing) are active.
    """
    phase = jump.phases(env)
    active = phase[:, 1:].any(dim=1).float()
    value = joint_pos_tracking_exp(
        env, h5_path, window_seconds, sigma,
        joint_weight_by_expr=joint_weight_by_expr,
        joint_weight_default=joint_weight_default,
    )
    return value * active


def root_tilt_tracking(env, h5_path, window_seconds=10., sigma=.35):
    """Track reference roll/pitch while preserving the reference yaw.

    This keeps the robot from sacrificing pre-takeoff balance when the jump
    branch is active; it compares gravity/up vectors, so it is not a world-
    upright constraint and still allows choreographed body lean.
    """
    from source.my_task.mdp.root_reference import root_reference_quat_w
    robot = env.scene["robot"]
    q_ref = root_reference_quat_w(
        robot, env, h5_path=h5_path, window_seconds=window_seconds,
        asset_name="robot",
    )
    q_cur = robot.data.root_quat_w.torch

    def up_vector(q):
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
        x, y, z, w = q.unbind(-1)
        return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)

    error = torch.acos((up_vector(q_cur) * up_vector(q_ref)).sum(-1).clamp(-1., 1.))
    return torch.exp(-torch.square(error) / (float(sigma) ** 2))


def upward(env):
    seconds = motion_steps(env).float() * env.step_dt
    events = [e for e in env.cfg.auto_jump_report["events"] if e["flight_reward"]]
    target = torch.zeros_like(seconds)
    if events:
        table = getattr(env, "_auto_jump_velocity_table", None)
        if table is None:
            table = torch.tensor([[*e["phases"][1], e["target_upward_velocity"]]
                                  for e in events], device=seconds.device)
            env._auto_jump_velocity_table = table
        active = (seconds[:, None] >= table[None, :, 0]) & (seconds[:, None] < table[None, :, 1])
        target = torch.where(active, table[None, :, 2], 0.).amax(dim=1)
    velocity = env.scene["robot"].data.root_lin_vel_w.torch[:, 2]
    return torch.exp(-((velocity - target) / .5).square()) * jump.phases(env)[:, 1] * gain(env)


def gain(env):
    return getattr(env, "_auto_jump_strength", 1.)


def flight(env):
    return jump.sustained_flight(env) * gain(env)


def missed(env):
    return jump.missed_flight(env) * gain(env)


def release(env):
    return jump.release_ground_load(env) * gain(env)


def slip(env):
    return jump.ground_slip(env) * gain(env)


def landing(env):
    return jump.landing_stability(env) * gain(env)


def support_metrics(env):
    """Only positive support evidence is constrained; unknown is not grounded."""
    seconds = motion_steps(env).float() * env.step_dt
    report = env.cfg.auto_jump_report
    expected = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
    if report.get("support_feet"):
        table = getattr(env, "_auto_support_table", None)
        if table is None:
            table = torch.tensor(report["support_feet"], device=env.device, dtype=torch.bool)
            env._auto_support_table = table
        indices = (seconds * report["fps"]).long().clamp(0, len(table) - 1)
        expected = table[indices]
    required = expected.any(-1)
    for window in report.get("support_windows", []):
        required |= (seconds >= window["start_s"]) & (seconds < window["end_s"])
    # Protect all inferred jump transitions, unless explicitly reclassified.
    protected = jump.phases(env).any(-1)
    expected = expected & ~protected[:, None]
    required = required & ~protected
    contacts = jump.full_step_ground_contacts(env)
    return dict(required=required, expected=expected, unexpected_air=required & ~contacts.any(-1),
                missing_feet=expected & ~contacts, contacts=contacts)


def swing_metrics(env):
    """Reference ankle-height difference; invariant to global root translation."""
    report = env.cfg.auto_jump_report
    if not report.get("swing_feet"):
        raise ValueError("Swing tracking requires a freshly analyzed reference (version 5+)")
    cached = getattr(env, "_auto_swing_tables", None)
    if cached is None:
        robot = env.scene["robot"]
        ids = [list(robot.body_names).index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        cached = (torch.tensor(report["swing_feet"], device=env.device, dtype=torch.bool),
                  torch.tensor(report["swing_gap_m"], device=env.device, dtype=torch.float32), ids)
        env._auto_swing_tables = cached
    table, gaps, ids = cached
    indices = (motion_steps(env).float() * env.step_dt * report["fps"]).long().clamp(0, len(table)-1)
    expected = table[indices] & ~jump.phases(env).any(-1)[:, None]
    heights = env.scene["robot"].data.body_link_pose_w.torch[:, ids, 2]
    actual_gap = heights - heights.flip(-1)
    error = torch.where(expected, (actual_gap - gaps[indices]).abs(), 0.).amax(-1)
    contacts = jump.full_step_ground_contacts(env)
    early = (expected & contacts).any(-1)
    return dict(required=expected.any(-1), error_m=error, early_contact=early)


def swing_height_tracking(env):
    metrics = swing_metrics(env)
    return metrics["required"].float() * torch.exp(-(metrics["error_m"] / .05).square())


def swing_early_contact(env):
    return swing_metrics(env)["early_contact"].float()


def _support_details(env, h5_path, window_seconds):
    from source import sim_compat
    from source.my_task.motion_reference import get_or_create_motion_buffer
    metrics = support_metrics(env)
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds)
    steps = motion_steps(env)
    robot = env.scene["robot"]
    origins = sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
    z = robot.data.root_state_w.torch[:, 2]
    ref = robot.data.default_root_state.torch[:, 2] + origins[:, 2] + buf.root_pos_delta(steps)[:, 2]
    ref_vz = (buf.root_pos_delta(steps + 1)[:, 2] - buf.root_pos_delta(steps)[:, 2]) / env.step_dt
    seconds = steps.float() * env.step_dt
    track_required = metrics["required"].clone()
    for event in env.cfg.auto_jump_report.get("events", []):
        if not event["flight_reward"]:
            lo, hi = event["phases"][0][0], event["phases"][-1][1]
            track_required |= (seconds >= lo) & (seconds < hi)
    excess_z = ((z - ref - .025) / .08).clamp(0., 1.)
    excess_v = ((robot.data.root_lin_vel_w.torch[:, 2] - ref_vz - .15) / .6).clamp(0., 1.)
    return metrics, (.4 * excess_z + .2 * excess_v) * track_required


def support_tracking(env, h5_path, window_seconds):
    # Preserve the cost function used by archived diagnostic bundles.
    metrics, dense = _support_details(env, h5_path, window_seconds)
    return ((metrics["unexpected_air"].float() + .25 * metrics["missing_feet"].sum(-1)) * metrics["required"] + dense)


def support_reward(env, h5_path, window_seconds):
    # Pay for the desired contact rather than make early termination a cheap
    # alternative to repeatedly paying an airborne penalty.
    metrics, dense = _support_details(env, h5_path, window_seconds)
    expected = metrics["expected"]
    count = expected.sum(-1)
    fraction = (expected & metrics["contacts"]).sum(-1) / count.clamp_min(1)
    contact = torch.where(count > 0, fraction, metrics["contacts"].any(-1).float())
    return contact * metrics["required"] - dense
