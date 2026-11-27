"""One physical impulse for interactive policy inspection."""

import math
import random

PUSH_DELTA_SPEED_MIN = .50
PUSH_DELTA_SPEED_MAX = .55
MANUAL_PUSH_MIN = .05
MANUAL_PUSH_MAX = 1.25


def push_character(env, actor=0, *, rng=None, delta_speed=None, angle_degrees=None):
    """Apply a world-horizontal upper-body force for the next physics tick.

    A specified angle uses the projected pelvis heading, with positive angles
    to the character's left. An omitted angle uses a random world direction.
    Mass scaling gives a whole-body velocity increment independent of timestep.
    """
    import torch

    if not env.evaluation:
        raise ValueError('Character pushes are for evaluation only')
    if not 0 <= actor < env.num_envs:
        raise ValueError('Invalid push actor')
    rng = rng if rng is not None else random.SystemRandom()
    if angle_degrees is None:
        angle = rng.uniform(-math.pi, math.pi)
    else:
        angle_degrees = float(angle_degrees)
        if not math.isfinite(angle_degrees):
            raise ValueError('Push direction must be finite')
        x, y, z, w = env.root[actor, 3:7].tolist()
        forward_x = 2 * (z*w - x*y)
        forward_y = 2 * (x*x + z*z) - 1
        heading = math.atan2(forward_y, forward_x) if math.hypot(forward_x, forward_y) > 1e-6 else -math.pi/2
        angle = heading + math.radians(angle_degrees)
    delta_speed = (rng.uniform(PUSH_DELTA_SPEED_MIN, PUSH_DELTA_SPEED_MAX)
                   if delta_speed is None else float(delta_speed))
    if not MANUAL_PUSH_MIN <= delta_speed <= MANUAL_PUSH_MAX:
        raise ValueError('Push strength is outside the supported interval')
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
                impulse_ns=mass * delta_speed, delta_speed=delta_speed, angle_degrees=angle_degrees)


def apply_requested_push(env, request, actor=0):
    """Accept both the legacy random-push flag and the configured Inspector request."""
    if isinstance(request, dict):
        return push_character(env, actor, delta_speed=request.get('delta_speed'),
                              angle_degrees=request.get('angle_degrees'))
    return push_character(env, actor)
