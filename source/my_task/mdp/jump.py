"""Phase-aware specialist terms. The reference clock is never paused."""
from __future__ import annotations

import torch
from source import sim_compat
from source.my_task.motion_reference import motion_steps, get_or_create_motion_buffer
from source.my_task.mdp.rewards import joint_pos_tracking_exp

# Reviewed open-to-close event near 17s; other events remain evaluation holdouts.
PHASES = ((16.80, 16.93), (16.93, 17.08), (17.08, 17.30), (17.30, 17.50))


def phases(env):
    seconds = motion_steps(env).float() * env.step_dt
    schedules = getattr(getattr(env, "cfg", None), "jump_phase_schedules", (PHASES,))
    if not schedules:
        return torch.zeros((len(seconds), 4), device=seconds.device, dtype=torch.bool)
    cache = getattr(env, "_jump_phase_bounds", None)
    if cache is None or cache[0] is not schedules:
        cache = (schedules, torch.tensor(schedules, device=seconds.device, dtype=seconds.dtype))
        env._jump_phase_bounds = cache
    bounds = cache[1]
    return ((seconds[:, None, None] >= bounds[None, :, :, 0]) &
            (seconds[:, None, None] < bounds[None, :, :, 1])).any(dim=1)


def ground_forces(env):
    values = []
    for side in ("left", "right"):
        matrix = env.scene[f"jump_{side}_ground"].data.normal_force_matrix_w
        if matrix is None:
            raise RuntimeError("Filtered ground contact forces unavailable; refusing unfiltered reward")
        force = sim_compat.as_torch(matrix).sum(dim=2)[:, 0]
        values.append(torch.linalg.vector_norm(force, dim=-1))
    return torch.stack(values, dim=-1)


def full_step_ground_contacts(env):
    """A foot is in contact if any physics substep in the control step touched."""
    values = []
    for side in ("left", "right"):
        history = env.scene[f"jump_{side}_ground"].data.normal_force_matrix_w_history
        if history is None:
            raise RuntimeError("Continuous-flight reward requires physics-step ground contact history")
        force = sim_compat.as_torch(history)
        if force.shape[1] < int(env.cfg.decimation):
            raise RuntimeError("Ground history is shorter than one control step")
        force = force[:, :int(env.cfg.decimation)].sum(dim=3)[:, :, 0]
        values.append((torch.linalg.vector_norm(force, dim=-1) >= 5.).any(dim=1))
    return torch.stack(values, dim=-1)


def jump_observation(env, h5_path, window_seconds):
    asset = env.scene["robot"]
    buf = get_or_create_motion_buffer(env, h5_path, window_seconds)
    origins = sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
    reference_z = asset.data.default_root_state.torch[:, 2] + origins[:, 2] + buf.root_pos_delta(motion_steps(env))[:, 2]
    return torch.cat((asset.data.root_lin_vel_w.torch[:, 2:3],
                      (ground_forces(env) >= 5.).float(), phases(env).float(),
                      (asset.data.root_state_w.torch[:, 2] - reference_z).unsqueeze(-1)), dim=-1)


def phase_joint_tracking(env, h5_path, window_seconds=10., sigma=.12,
                         joint_weight_by_expr=None, joint_weight_default=1.):
    value = joint_pos_tracking_exp(env, h5_path, window_seconds, sigma,
                                   joint_weight_by_expr, joint_weight_default)
    pushing = phases(env)[:, 1]
    return value * torch.where(pushing, .3, 1.)


def upward_velocity(env):
    vz = env.scene["robot"].data.root_lin_vel_w.torch[:, 2]
    return torch.exp(-((vz - .8) / .5) ** 2) * phases(env)[:, 1]


def flight_state(env):
    """Update once per env control step, with explicit per-episode reset."""
    key = int(env.common_step_counter)
    state = getattr(env, "_jump_flight_state", None)
    if state is None:
        state = dict(key=-1, air=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
                     stance=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
                     rising=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        env._jump_flight_state = state
    if state["key"] != key:
        fresh = env.episode_length_buf <= 1
        # A new preparation stage must not inherit takeoff history from an earlier jump.
        preparation = phases(env)[:, 0]
        previous_preparation = state.get("preparation", torch.zeros_like(preparation))
        fresh = fresh | (preparation & ~previous_preparation)
        state["preparation"] = preparation.clone()
        for name in ("air", "stance", "rising"):
            state[name][fresh] = 0
        contact = ground_forces(env) >= 5.
        if getattr(getattr(env, "cfg", None), "jump_full_step_contact", False):
            contact = full_step_ground_contacts(env)
        phase = phases(env)
        state["stance"] |= contact.all(dim=-1) & (phase[:, 0] | phase[:, 1])
        state["air"] = torch.where(~contact.any(dim=-1), state["air"] + 1, 0)
        state["rising"] |= (state["air"] >= 1) & (env.scene["robot"].data.root_lin_vel_w.torch[:, 2] > .1) & (phase[:, 1] | phase[:, 2])
        state["key"] = key
    return state


def airborne_close(env):
    state = flight_state(env)
    return ((state["air"] >= 2) & state["stance"] & state["rising"] & phases(env)[:, 2]).float()


def ground_slip(env):
    asset = env.scene["robot"]
    ids = [list(asset.body_names).index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    velocity = asset.data.body_link_lin_vel_w.torch[:, ids, :2]
    contact = ground_forces(env) >= 5.
    return (velocity.square().sum(dim=-1) * contact).sum(dim=-1) * phases(env)[:, 2]


def landing_stability(env):
    asset = env.scene["robot"]
    contact = (ground_forces(env) >= 5.).all(dim=-1)
    return torch.exp(-(asset.data.root_lin_vel_w.torch[:, 2] / .5) ** 2) * contact * phases(env)[:, 3]


def phase_joint_tracking_v2(env, h5_path, window_seconds=10., sigma=.12,
                            joint_weight_by_expr=None, joint_weight_default=1.):
    value = joint_pos_tracking_exp(env, h5_path, window_seconds, sigma,
                                   joint_weight_by_expr, joint_weight_default)
    phase = phases(env)
    scale = torch.where(phase[:, 1], .1, torch.where(phase[:, 2], .3, 1.))
    return value * scale


def sustained_flight(env):
    state = flight_state(env)
    duration = (state["air"].float() * env.step_dt / .12).clamp(max=1.)
    qualified = state["stance"] & state["rising"] & phases(env)[:, 2]
    return duration * qualified


def release_ground_load(env):
    # Dense exploration aid. Bounded; upward history and safe root height prevent
    # simply rewarding an unsupported fall as successful flight.
    state = flight_state(env)
    asset = env.scene["robot"]
    origins = sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
    safe_height = asset.data.root_state_w.torch[:, 2] - origins[:, 2] > .65
    vz = asset.data.root_lin_vel_w.torch[:, 2]
    return torch.exp(-ground_forces(env).sum(dim=-1) / 80.) * state["stance"] * (vz > -.5) * safe_height * phases(env)[:, 2]


def missed_flight(env):
    return (ground_forces(env) >= 5.).any(dim=-1).float() * phases(env)[:, 2]


def validate_ground_contacts(env):
    """Prove the ground filter produces support forces before training."""
    env.reset()
    maximum = torch.zeros(2, device=env.device)
    for _ in range(8):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        maximum = torch.maximum(maximum, ground_forces(env).amax(dim=0))
    if not bool((maximum > 5.).all()):
        raise RuntimeError(f"Ground contact preflight failed: {maximum.tolist()}")
    print(f"[JUMP] Ground-only contact preflight passed: {maximum.tolist()} N")
    env.reset()
