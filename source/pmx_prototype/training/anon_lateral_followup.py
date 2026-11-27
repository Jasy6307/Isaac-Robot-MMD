"""Reward the next lateral step when the first support cannot stop the body."""

import torch


class LoadedFollowupReward:
    def __init__(self, env, body_weight, weight, prelanding_brake_weight=0.):
        self.body_weight = body_weight
        self.dt = env.dt
        self.followup_weight = weight
        self.followup_urgency = torch.zeros(env.num_envs, device=env.device)
        self.first_foot = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.first_age = torch.zeros_like(self.followup_urgency)
        self.first_speed = torch.zeros_like(self.followup_urgency)
        self.need_followup = torch.zeros_like(self.followup_urgency)
        self.first_trailing_position = torch.zeros_like(self.followup_urgency)
        self.previous_trailing_position = torch.zeros_like(self.followup_urgency)
        self.trailing_swing = torch.zeros_like(self.first_foot)
        self.trailing_support = torch.zeros_like(self.first_foot)
        self.trailing_has_swung = torch.zeros_like(self.first_foot, dtype=torch.bool)
        self.followup_swing_frames = torch.zeros((), device=env.device)
        self.followup_braking_frames = torch.zeros((), device=env.device)
        self.prelanding_brake_weight = prelanding_brake_weight
        if prelanding_brake_weight:
            self.prelanding_intensity = torch.zeros_like(self.followup_urgency)
            self.best_prelanding_speed = torch.full_like(self.followup_urgency, torch.inf)
            self.prelanding_brake_frames = torch.zeros((), device=env.device)

    def reset(self, ids):
        self.followup_urgency[ids] = 0.
        self.first_foot[ids] = -1
        self.first_age[ids] = 0.
        self.first_speed[ids] = 0.
        self.need_followup[ids] = 0.
        self.first_trailing_position[ids] = 0.
        self.previous_trailing_position[ids] = 0.
        self.trailing_swing[ids] = 0
        self.trailing_support[ids] = 0
        self.trailing_has_swung[ids] = False
        if self.prelanding_brake_weight:
            self.prelanding_intensity[ids] = 0.
            self.best_prelanding_speed[ids] = torch.inf

    def start(self, ids, impulse):
        self.reset(ids)
        self.followup_urgency[ids] = ((impulse-.60)/.12).clamp(0., 1.)
        if self.prelanding_brake_weight:
            self.prelanding_intensity[ids] = ((impulse-.70)/.10).clamp(0., 1.)

    def reward(self, active, age, axis, feet, sole_height, force, com, velocity, support):
        hard_side = active & (self.followup_urgency > 0.)
        new_foot = torch.where(support, force, 0.).argmax(dim=1)
        first = hard_side & support.any(dim=1) & (self.first_foot < 0) & (age < 1.16)
        self.first_foot = torch.where(first, new_foot, self.first_foot)
        planted = self.first_foot.clamp_min(0).unsqueeze(1)
        trailing = 1 - planted
        foot_position = (feet * axis[:, None]).sum(-1)
        forward_speed = (velocity * axis).sum(-1).clamp_min(0.)
        time_scale = (com[:, 2].clamp(.4, 1.2)/9.81).sqrt()
        capture = (com[:, :2] * axis).sum(-1) + time_scale * forward_speed
        support_gap = capture - foot_position.gather(1, planted).squeeze(1) - .06
        self.first_age = torch.where(first, age, self.first_age)
        self.first_speed = torch.where(first, forward_speed, self.first_speed)
        need = torch.maximum((support_gap/.24).clamp(0., 1.), ((forward_speed-.18)/.60).clamp(0., 1.))
        self.need_followup = torch.where(first, need, self.need_followup)
        trailing_position = foot_position.gather(1, trailing).squeeze(1)
        self.first_trailing_position = torch.where(first, trailing_position, self.first_trailing_position)
        age_after_plant = age - self.first_age
        following = hard_side & (self.first_foot >= 0) & (age_after_plant > 0.) & (age_after_plant <= .64)

        stance_force = force.gather(1, planted).squeeze(1)
        trailing_force = force.gather(1, trailing).squeeze(1)
        trailing_height = sole_height.gather(1, trailing).squeeze(1)
        swing = following & (trailing_force < 12.) & (trailing_height >= .013)
        self.trailing_swing = torch.where(swing, self.trailing_swing + 1, 0)
        self.trailing_has_swung |= self.trailing_swing >= 2
        self.trailing_has_swung &= following
        stride = trailing_position - self.first_trailing_position
        advance = ((trailing_position-self.previous_trailing_position)/.045).clamp(0., 1.)
        self.previous_trailing_position.copy_(trailing_position)
        stance_load = (stance_force/(.42*self.body_weight)).clamp(0., 1.)
        reach = advance * stance_load * swing * (age_after_plant < .42)

        lands = self.trailing_has_swung & (stride >= .045) & (trailing_force >= .12*self.body_weight)
        self.trailing_support = torch.where(lands, self.trailing_support + 1, 0)
        loaded = (self.trailing_support >= 2) * (trailing_force/(.48*self.body_weight)).clamp(0., 1.)
        gap = (capture-trailing_position-.05).clamp_min(0.)
        nearness = (1. + (gap/.20).square()).reciprocal()
        braking = ((self.first_speed-forward_speed)/.45).clamp(0., 1.)
        catch = loaded * nearness * (.25 + .75*braking)
        urgency = self.followup_urgency * self.need_followup * following
        self.followup_swing_frames += ((urgency > .01) & (reach > .08)).sum()
        self.followup_braking_frames += ((urgency > .01) & (loaded > .1) & (braking > .15)).sum()
        timely = (-age_after_plant.clamp_min(0.)/.48).exp()
        reward = self.dt * self.followup_weight * urgency * timely * (.42*reach + .58*catch)
        if self.prelanding_brake_weight:
            reward += self.prelanding_reward(active, age, force, forward_speed)
        return reward

    def prelanding_reward(self, active, age, force, forward_speed):
        # Start after the longest force pulse. Award new velocity minima
        # while the old support is loaded and the first step is still in flight.
        eligible = (active & (self.prelanding_intensity > 0.) & (age > .10)
                    & (age < .62) & (self.first_foot < 0))
        loaded = (force.amax(dim=1)/(.36*self.body_weight)).clamp(0., 1.)
        best = self.best_prelanding_speed
        improvement = torch.where(eligible & torch.isfinite(best), ((best-forward_speed)/.12).clamp(0., 1.), 0.)
        self.best_prelanding_speed = torch.where(eligible, torch.minimum(best, forward_speed), best)
        self.prelanding_brake_frames += ((loaded >= .5) & (improvement >= .08)).sum()
        timely = (-(age-.10).clamp_min(0.)/.30).exp()
        return self.prelanding_brake_weight*self.dt*self.prelanding_intensity*timely*loaded*improvement

