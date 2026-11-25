"""Head facing and stop-recovery reward in the controller's heading frame."""

import math

import torch


REWARD_DEAD_ZONE_DEG = 3.
REWARD_DISTANCE_SCALE_DEG = 8.
REWARD_SOFTENING_DEG = 1.
LARGE_ERROR_START_DEG = 6.
LARGE_ERROR_SCALE_DEG = 28.
NECK_PITCH_DEAD_ZONE_DEG = 2.
NECK_PITCH_ERROR_SCALE_DEG = 12.
NECK_YAW_DEAD_ZONE_DEG = 3.
NECK_YAW_ERROR_SCALE_DEG = 8.
RECOVERY_RAMP_SECONDS = .35
MOVING_GAIN = .15
MEAN_BIAS_LIMIT_DEG = 10.
P95_LIMIT_DEG = 10.
RECENTER_LIMIT_DEG = 6.
RECENTER_HOLD_SECONDS = .5
STOP_COMMAND_TOLERANCE = .01
HEAD_SWAY_WINDOW_SECONDS = .40
HEAD_SWAY_ALLOWANCE_DEG = 2.5
HEAD_SWAY_SCALE_DEG = 4.
BACKWARD_HEAD_ANGLE_SCALE_DEG = 10.


def head_facing_error(rotation, heading):
    """Return signed yaw and elevation in radians from the head's local -Y axis."""
    x, y, z, w = rotation.unbind(-1)
    forward_x = 2 * (w * z - x * y)
    forward_y = 2 * (x.square() + z.square()) - 1
    forward_z = -2 * (y * z + w * x)
    yaw = torch.atan2(forward_x, -forward_y) - heading
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    pitch = torch.atan2(forward_z, torch.hypot(forward_x, forward_y))
    return torch.stack((yaw, pitch), dim=-1)


def neck_pitch_level_target(chest_rotation, heading):
    """Pitch component of the horizontal commanded look direction in the chest frame."""
    x, y, z, w = chest_rotation.unbind(-1)
    chest_up_x = 2. * (x * z + w * y)
    chest_up_y = 2. * (y * z - w * x)
    # The commanded forward vector is [sin(heading), -cos(heading), 0].
    local_vertical = heading.sin() * chest_up_x - heading.cos() * chest_up_y
    # A positive neck pitch rotates local -Y down, so elevation and pitch have opposite signs.
    pitch = -local_vertical.clamp(-1., 1.).asin()
    return torch.where(torch.isfinite(local_vertical), pitch, float('nan'))


def neck_yaw_forward_target(chest_rotation, heading):
    """Project the commanded world forward direction into the chest frame."""
    x, y, z, w = chest_rotation.unbind(-1)
    forward_x, forward_y = heading.sin(), -heading.cos()
    # Dot the commanded forward vector with the chest's local X and Y axes.
    local_x = (1. - 2. * (y.square() + z.square())) * forward_x + 2. * (x * y + w * z) * forward_y
    local_y = 2. * (x * y - w * z) * forward_x + (1. - 2. * (x.square() + z.square())) * forward_y
    valid = torch.isfinite(local_x) & torch.isfinite(local_y) & (torch.hypot(local_x, local_y) > 1.e-6)
    return torch.where(valid, torch.atan2(local_x, -local_y), float('nan'))


def _in_place_turn(requested):
    return (requested[..., :2].norm(dim=-1) < .03) & (requested[..., 2].abs() >= .05)


def neck_pitch_stop_context(previous, requested, walk_speed_limit):
    """Remember whether the last motion command was a straight walk or an in-place turn."""
    forward, lateral, yaw = requested.unbind(-1)
    stopping = requested.abs().amax(dim=-1) < STOP_COMMAND_TOLERANCE
    forward_walk = (forward >= .05) & (forward <= walk_speed_limit)
    forward_walk &= (lateral.abs() < .03) & (yaw.abs() < .05)
    return torch.where(stopping, previous, forward_walk | _in_place_turn(requested))


def neck_yaw_stop_context(previous, requested):
    """Remember an in-place turn until the next nonzero request."""
    stopping = requested.abs().amax(dim=-1) < STOP_COMMAND_TOLERANCE
    return torch.where(stopping, previous, _in_place_turn(requested))


def idle_neck_pitch_reward(neck_pitch, target, gain, weight):
    """Bounded joint-space correction using the existing stop gain."""
    return _idle_neck_joint_reward(neck_pitch, target, gain, weight,
                                   NECK_PITCH_DEAD_ZONE_DEG, NECK_PITCH_ERROR_SCALE_DEG)


def idle_neck_yaw_reward(neck_yaw, target, gain, pivot_stop, weight):
    """Correct yaw only when braking or standing after an in-place turn."""
    gain = torch.where(pivot_stop, gain, 0.)
    return _idle_neck_joint_reward(neck_yaw, target, gain, weight,
                                   NECK_YAW_DEAD_ZONE_DEG, NECK_YAW_ERROR_SCALE_DEG)


def _idle_neck_joint_reward(position, target, gain, weight, dead_zone_deg, error_scale_deg):
    valid = (gain > 0.) & torch.isfinite(position) & torch.isfinite(target)
    excess = ((position - target).abs() - math.radians(dead_zone_deg)).clamp_min(0.)
    scaled_square = (excess / math.radians(error_scale_deg)).square()
    rate = -weight * gain * (1. - 1. / (1. + scaled_square))
    return torch.where(valid, rate, 0.)


def head_sway_excursion(changes):
    """Smaller of the forward and reverse travel on each axis in the window."""
    return .5 * (changes.abs().sum(dim=0) - changes.sum(dim=0).abs()).clamp_min(0.)


def straight_backward_walk(command, walk_speed_limit):
    forward, lateral, yaw = command.unbind(-1)
    return ((forward <= -.05) & (forward >= -walk_speed_limit)
            & (lateral.abs() < .03) & (yaw.abs() < .05))


def backward_walk_head_alignment_reward(errors, active, weight):
    """Favor level forward facing during straight backward walking."""
    valid = active & torch.isfinite(errors).all(dim=-1)
    excess = (errors.abs() - math.radians(REWARD_DEAD_ZONE_DEG)).clamp_min(0.)
    scaled = excess / math.radians(BACKWARD_HEAD_ANGLE_SCALE_DEG)
    cost = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled), reduction='none').mean(dim=-1)
    return torch.where(valid, -weight * cost, 0.)


class HeadSwayPenalty:
    """Penalize repeated changes of direction relative to the planned heading."""

    def __init__(self, num_envs, dt, device, weight):
        self.weight = weight
        self.previous = torch.zeros(num_envs, 2, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        samples = max(2, round(HEAD_SWAY_WINDOW_SECONDS / dt))
        self.changes = torch.zeros(samples, num_envs, 2, device=device)
        self.cursor = 0

    def reset(self, ids):
        self.valid[ids] = False
        self.changes[:, ids] = 0.

    def step(self, errors, upright):
        valid = upright & torch.isfinite(errors).all(dim=-1)
        delta = torch.remainder(errors - self.previous + math.pi, 2. * math.pi) - math.pi
        delta = torch.where((valid & self.valid)[:, None], delta, 0.)
        self.changes.masked_fill_(~valid[None, :, None], 0.)
        self.changes[self.cursor].copy_(delta)
        self.cursor = (self.cursor + 1) % len(self.changes)
        self.previous.copy_(torch.where(valid[:, None], errors, 0.))
        self.valid.copy_(valid)
        sway = head_sway_excursion(self.changes)
        excess = (sway - math.radians(HEAD_SWAY_ALLOWANCE_DEG)).clamp_min(0.)
        scaled_square = (excess / math.radians(HEAD_SWAY_SCALE_DEG)).square().sum(dim=-1)
        return -self.weight * (scaled_square / (1. + scaled_square))


def _smoothstep(value):
    value = value.clamp(0., 1.)
    return value.square() * (3. - 2. * value)


def idle_head_reward(errors, requested, planned, velocity, stop_time, upright, dt, weight, *, large_error_penalty=0.):
    """Ramp in on a stop request while allowing motion during braking."""
    planned_speed = planned[:, :2].norm(dim=-1)
    measured_speed = velocity[:, :2].norm(dim=-1)
    linear = torch.maximum((planned_speed - .03) / (.55 - .03), (measured_speed - .12) / (.55 - .12))
    angular = torch.maximum((planned[:, 2].abs() - .05) / (.60 - .05), (velocity[:, 2].abs() - .20) / (.60 - .20))
    motion = torch.maximum(linear, angular)
    stopping = (requested.abs().amax(-1) < STOP_COMMAND_TOLERANCE) & upright
    stopping &= torch.isfinite(errors).all(-1) & torch.isfinite(motion)
    stop_time = torch.where(stopping, (stop_time + dt).clamp(max=RECOVERY_RAMP_SECONDS), 0.)
    gain = _smoothstep(stop_time / RECOVERY_RAMP_SECONDS) * (1. - (1. - MOVING_GAIN) * _smoothstep(motion))

    # Soften the edge of the dead zone. A reciprocal still distinguishes
    # large errors that all earned nearly zero from the exponential bonus.
    excess = (errors.abs() - math.radians(REWARD_DEAD_ZONE_DEG)).clamp_min(0)
    softening = math.radians(REWARD_SOFTENING_DEG)
    distance = (excess.square().sum(-1) + softening**2).sqrt() - softening
    rate = weight / (1. + distance / math.radians(REWARD_DISTANCE_SCALE_DEG))
    if large_error_penalty:
        # Increase the reward difference in the large-error range while
        # keeping both the small-error curve and the maximum bonus intact.
        large = (distance - math.radians(LARGE_ERROR_START_DEG)).clamp_min(0.)
        scaled_square = (large / math.radians(LARGE_ERROR_SCALE_DEG)).square()
        rate -= weight * large_error_penalty * (scaled_square / (1. + scaled_square))
    return stop_time, torch.where(stopping, gain, 0.), torch.where(stopping, gain * rate, 0.)
