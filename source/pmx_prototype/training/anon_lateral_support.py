"""Reward a laterally placed foot only when it establishes measured support."""

import math

import torch

from anon_lateral_followup import LoadedFollowupReward


class SideSupportReward:
    def __init__(self, env, mass, settings):
        self.dt = env.dt
        self.body_weight = mass * 9.81
        self.reach_weight = float(settings['reach_weight'])
        self.load_weight = float(settings['load_weight'])
        self.early_capture_weight = float(settings.get('early_capture_weight', 0.))
        followup_weight = float(settings.get('followup_weight', 0.))
        prelanding_brake_weight = float(settings.get('prelanding_brake_weight', 0.))
        if not all(math.isfinite(value) and value >= 0. for value in (self.reach_weight, self.load_weight,
                                                                     self.early_capture_weight, followup_weight,
                                                                     prelanding_brake_weight)):
            raise ValueError('Lateral support reward weights must be finite and nonnegative')
        if prelanding_brake_weight and not followup_weight:
            raise ValueError('Prelanding braking needs first-landing contact tracking')
        self.followup = (LoadedFollowupReward(env, self.body_weight, followup_weight, prelanding_brake_weight)
                         if followup_weight else None)
        shape = (env.num_envs, 2)
        self.start_position = torch.zeros((*shape, 2), device=env.device)
        self.swing_frames = torch.zeros(shape, dtype=torch.int32, device=env.device)
        self.loaded_frames = torch.zeros_like(self.swing_frames)
        self.has_swung = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.reach_frames = torch.zeros((), device=env.device)
        self.corrected_support_frames = torch.zeros((), device=env.device)
        if self.early_capture_weight:
            self.urgency = torch.zeros(env.num_envs, device=env.device)
            self.previous_travel = torch.zeros(shape, device=env.device)
            self.previous_speed = torch.zeros_like(self.urgency)
            self.early_reach_frames = torch.zeros((), device=env.device)
            self.braking_support_frames = torch.zeros((), device=env.device)

    def reset(self, ids):
        self.swing_frames[ids] = 0
        self.loaded_frames[ids] = 0
        self.has_swung[ids] = False
        if self.followup is not None:
            self.followup.reset(ids)
        if self.early_capture_weight:
            self.urgency[ids] = 0.
            self.previous_travel[ids] = 0.
            self.previous_speed[ids] = 0.

    def start(self, ids, feet, impulse=None):
        self.reset(ids)
        self.start_position[ids] = feet
        if self.followup is not None:
            if impulse is None:
                raise ValueError('The follow-up objective needs the actual sampled impulse')
            self.followup.start(ids, impulse)
        if self.early_capture_weight:
            if impulse is None:
                raise ValueError('The early capture objective needs the actual sampled impulse')
            self.urgency[ids] = ((impulse-.50)/.20).clamp(0., 1.)

    def reward(self, active, age, axis, feet, sole_height, force, com, velocity):
        side = active & (axis.square().sum(dim=-1) > .5)
        eligible = side & (age < (1.40 if self.early_capture_weight else 1.10))
        force = force.clamp_min(0.)
        moving_foot = (force < 12.) & (sole_height >= .013) & eligible[:, None]
        self.swing_frames = torch.where(moving_foot, self.swing_frames + 1, 0)
        self.has_swung |= self.swing_frames >= 2
        self.has_swung &= eligible[:, None]

        travel = ((feet - self.start_position) * axis[:, None]).sum(dim=-1)
        # Encourage reach while the other foot carries load, then require the
        # reaching foot to carry real ground reaction for consecutive frames.
        opposite_load = (force.flip((1,)) / (.45 * self.body_weight[:, None])).clamp(0., 1.)
        reached = ((travel - .015) / .18).clamp(0., 1.)
        through_swing = reached * opposite_load * moving_foot * (age[:, None] < .33)
        planted = (force >= .10 * self.body_weight[:, None]) & self.has_swung & (travel >= .035)
        self.loaded_frames = torch.where(planted, self.loaded_frames + 1, 0)
        supports = self.loaded_frames >= 2

        time_scale = (com[:, 2].clamp(.4, 1.2) / 9.81).sqrt()
        capture = com[:, :2] + time_scale[:, None] * velocity
        gap = ((capture[:, None] - feet) * axis[:, None]).sum(-1) - .04
        # A distant touchdown alone does little if the moving COM is still
        # outside the new footprint. The score fades continuously with that gap.
        coverage = (1. - gap.clamp_min(0.) / .36).clamp(0., 1.)
        load = (force / (.55 * self.body_weight[:, None])).clamp(0., 1.)
        second_phase = supports * load * (.30 + .70 * coverage) * (age[:, None] < 1.10)
        reach_reward = through_swing.amax(dim=-1)
        load_reward = second_phase.amax(dim=-1)
        self.reach_frames += (reach_reward > .1).sum()
        self.corrected_support_frames += (load_reward > 0.).sum()
        total = self.reach_weight * reach_reward + self.load_weight * load_reward
        if self.early_capture_weight:
            # The urgency term rewards progress before impact recovery drifts
            # away. Keep a smooth signal even beyond the old 36 cm gap cutoff.
            urgency = self.urgency * eligible
            timely = (-age.clamp_min(0.)/.63).exp()
            gain = ((travel - self.previous_travel)/.045).clamp(0., 1.)
            approach = (gain * opposite_load * moving_foot * (age[:, None] < .58)).amax(dim=-1)
            self.previous_travel.copy_(travel)
            side_speed = (velocity * axis).sum(dim=-1).abs()
            braking = ((self.previous_speed-side_speed)/.09).clamp(0., 1.)
            self.previous_speed.copy_(side_speed)
            capture_nearness = (1. + (gap.clamp_min(0.)/.24).square()).reciprocal()
            held = (supports * load * capture_nearness).amax(dim=-1)
            total += self.early_capture_weight * urgency * timely * (.45*approach + .55*held*(.60+.40*braking))
            self.early_reach_frames += ((urgency > 0.) & (approach > .1)).sum()
            self.braking_support_frames += ((urgency > 0.) & (held > .1) & (braking > .1)).sum()
        reward = self.dt * total
        if self.followup is not None:
            reward += self.followup.reward(active, age, axis, feet, sole_height, force, com, velocity, supports)
        return reward
