"""One physical knockdown impulse for interactive policy inspection."""

import math
import random

import torch

PUSH_DELTA_SPEED_MIN = .50
PUSH_DELTA_SPEED_MAX = .55


def push_character(env, actor=0, *, rng=None):
    """Apply a world-horizontal upper-body force for the next physics tick.

    Mass scaling gives a 0.50-0.55 m/s whole-body impulse equivalent,
    independent of actor mass and physics frequency. Calibrated to induce
    loss of balance rather than launch the body. This changes no poses or policies.
    Use a separate RNG so clicking cannot change the policy's random stream.
    """
    if not env.evaluation:
        raise ValueError('Character pushes are for evaluation only')
    if not 0 <= actor < env.num_envs:
        raise ValueError('Invalid push actor')
    rng = rng if rng is not None else random.SystemRandom()
    angle = rng.uniform(-math.pi, math.pi)
    delta_speed = rng.uniform(PUSH_DELTA_SPEED_MIN, PUSH_DELTA_SPEED_MAX)
    link = env.links.index('waist_pitch') if 'waist_pitch' in env.links else env.pelvis_id
    mass = float(env.robot.get_masses()[actor].sum().item())
    magnitude = mass * delta_speed * float(env.cfg['physics_hz'])
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError('Push requires finite positive mass and physics frequency')
    force = [magnitude * math.cos(angle), magnitude * math.sin(angle), 0.]
    forces = torch.zeros((env.num_envs, env.robot.max_links, 3), device=env.device)
    forces[actor, link] = torch.tensor(force, device=env.device)
    ids = env.ids[actor:actor + 1]
    env.robot.apply_forces_and_torques_at_position(forces, None, None, ids, True)
    return dict(actor=actor, link=env.links[link], force=force,
                impulse_ns=mass * delta_speed, delta_speed=delta_speed)
