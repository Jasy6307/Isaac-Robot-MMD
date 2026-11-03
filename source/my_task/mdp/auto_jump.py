"""Generic phase rewards; no absolute dance time or reference-pose edits."""
import torch
from source.my_task.mdp import jump
from source.my_task.mdp.rewards import joint_pos_tracking_exp
from source.my_task.motion_reference import motion_steps


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
    from source.train_workflow.utils.phase_gate import jump_teacher_gate
    normal = 1.-jump_teacher_gate(steps.float()*env.step_dt,env.cfg.jump_phase_schedules)
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
        from source.train_workflow.utils.phase_gate import jump_teacher_gate
        gate=jump_teacher_gate(seconds,env.cfg.jump_phase_schedules)
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
