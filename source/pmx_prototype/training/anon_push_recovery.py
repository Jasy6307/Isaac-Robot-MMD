"""Training-only horizontal impulses and recovery-window bookkeeping."""

import math

import torch

from anon_push_waveforms import PROFILES, horizontal_axes, profile_table
from command_plan import limit_yaw_command


class PushRecovery:
    def __init__(self, env, settings, seed):
        self.env = env
        self.settings = settings
        for name in ('initial_delay_seconds', 'repeat_delay_seconds', 'light_m_s', 'challenge_m_s'):
            low, high = settings[name]
            if not (0 < low <= high < math.inf):
                raise ValueError(f'Invalid push recovery interval: {name}')
        for name in ('episode_probability', 'challenge_probability', 'weak_command_probability', 'style_floor'):
            if not 0 <= settings[name] <= 1:
                raise ValueError(f'Invalid push recovery fraction: {name}')
        self.weak_challenge_probability = settings.get('weak_challenge_probability', settings['challenge_probability'])
        if not 0 <= self.weak_challenge_probability <= 1:
            raise ValueError('Invalid push recovery fraction: weak_challenge_probability')
        self.weak_direction_bias = float(settings.get('weak_direction_bias', 0.))
        if not 0 <= self.weak_direction_bias <= 1:
            raise ValueError('Invalid push recovery fraction: weak_direction_bias')
        self.left_step_focus_fraction = float(settings.get('left_step_focus_fraction', 0.))
        if not 0 <= self.left_step_focus_fraction <= 1:
            raise ValueError('Invalid push recovery fraction: left_step_focus_fraction')
        self.window = float(settings['recovery_window_seconds'])
        if not 0 < self.window < settings['repeat_delay_seconds'][0]:
            raise ValueError('The recovery window must end before the next push')
        self.style_window = float(settings.get('style_recovery_seconds', self.window))
        if not 0 < self.style_window <= self.window:
            raise ValueError('Style recovery must fit inside the recovery window')
        self.balance_weight = float(settings.get('recovery_balance_weight', 0.))
        self.fall_penalty = float(settings.get('recovery_fall_penalty', 0.))
        if not all(math.isfinite(value) and value >= 0 for value in (self.balance_weight, self.fall_penalty)):
            raise ValueError('Recovery reward weights must be finite and nonnegative')
        speed_scale = float(settings.get('recovery_speed_scale_m_s', .3))
        self.speed_scale_squared = speed_scale * speed_scale
        if not speed_scale > 0. or not 0. < self.speed_scale_squared < math.inf:
            raise ValueError('Recovery speed scale must be finite and positive')
        self.steps = settings.get('corrective_steps')
        self.capture_support = bool(self.steps and self.steps.get('capture_support'))
        self.standing_brake_weight = float(self.steps.get('standing_brake_weight', 0.)) if self.steps else 0.
        if (not math.isfinite(self.standing_brake_weight) or self.standing_brake_weight < 0.
                or (self.standing_brake_weight > 0. and not self.capture_support)):
            raise ValueError('Standing braking reward requires real support forces and a finite weight')
        self.forward_focus = float(settings.get('forward_focus_fraction', 0.))
        self.stand_focus = float(settings.get('stand_focus_fraction', 0.))
        if not (0 <= self.forward_focus and 0 <= self.stand_focus and self.forward_focus + self.stand_focus <= 1):
            raise ValueError('Forward and standing focus must fit within the targeted samples')
        self.phase_boost = float(settings.get('vulnerable_phase_boost', 0.))
        if not 0 <= self.phase_boost <= 1:
            raise ValueError('Invalid vulnerable-phase boost')
        self.extreme_probability = float(settings.get('extreme_probability', 0.))
        if not 0 <= self.extreme_probability <= 1:
            raise ValueError('Invalid push recovery fraction: extreme_probability')
        if self.extreme_probability:
            low, high = settings['extreme_m_s']
            if not settings['challenge_m_s'][1] <= low <= high < math.inf:
                raise ValueError('The extreme interval must start at or above the challenge interval')
        if self.steps is not None:
            if not 0 < self.steps['reference_grace_seconds'] <= self.style_window:
                raise ValueError('Reference grace must fit inside the style recovery window')
            if not 0 <= self.steps['style_floor'] <= 1:
                raise ValueError('Invalid corrective-step style floor')
            if not all(0 < self.steps[name] < math.inf for name in
                       ('minimum_impulse_m_s', 'landing_bonus', 'trunk_weight')):
                raise ValueError('Corrective-step thresholds and weights must be finite and positive')
        self.rng = torch.Generator(device=env.device).manual_seed(seed + 57211)
        masses = env.robot.get_masses().to(env.device)
        self.mass = masses.sum(dim=1)
        self.link = env.links.index('waist_pitch')
        self.forces = torch.zeros((env.num_envs, env.robot.max_links, 3), device=env.device)
        self.enabled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.delay = torch.zeros_like(self.mass)
        self.remaining = torch.zeros_like(self.mass)
        self.stats = torch.zeros(29, device=env.device)
        self.side_focus = float(settings.get('standing_side_fraction', 0.))
        self.fore_aft_rehearsal = float(settings.get('standing_fore_aft_fraction', 0.))
        self.left_fraction = float(settings.get('standing_left_fraction', .5))
        self.side_spread = math.radians(float(settings.get('standing_side_spread_deg', 14.)))
        if not (0. <= self.side_focus <= 1. and 0. <= self.fore_aft_rehearsal <= 1. - self.side_focus
                and 0. < self.left_fraction < 1. and 0. <= self.side_spread <= math.pi / 3):
            raise ValueError('Invalid standing push direction mix')
        self.side_weight = float(settings.get('standing_side_landing_weight', 0.))
        self.shape_fraction = float(settings.get('standing_shaped_fraction', 0.))
        if (not 0. <= self.shape_fraction <= 1. or not 0. <= self.side_weight < math.inf
                or (self.side_weight and (not self.side_focus or not self.capture_support))
                or (self.shape_fraction and not self.side_focus)):
            raise ValueError('Invalid standing push waveform or landing settings')
        self.side_bands = None
        if side_bands := settings.get('standing_side_bands'):
            if not self.side_focus:
                raise ValueError('Directional impulse ranges require targeted standing side pushes')
            from anon_push_bands import DirectionalPushBands
            self.side_bands = DirectionalPushBands(side_bands, env.device)
        if self.side_focus or self.fore_aft_rehearsal:
            self.protected_fore_aft = torch.zeros_like(self.enabled)
            self.side_axis = torch.zeros((env.num_envs, 2), device=env.device)
        if self.shape_fraction:
            self.waveforms = profile_table(env.device)
            self.waveform_id = torch.zeros_like(self.enabled, dtype=torch.long)
            self.waveform_tick = torch.full_like(self.waveform_id, self.waveforms.shape[1] - 1)
            self.waveform_force = torch.zeros((env.num_envs, 2), device=env.device)
        self.waveform_cdf = None
        if waveform_mix := settings.get('standing_waveform_mix'):
            names = tuple(PROFILES)[1:]
            if set(waveform_mix) != set(names) or not self.shape_fraction:
                raise ValueError('Provide weights for every non-impulse waveform in shaped training')
            weights = [float(waveform_mix[name]) for name in names]
            total = sum(weights)
            if not all(math.isfinite(weight) and weight >= 0. for weight in weights) or not 0. < total < math.inf:
                raise ValueError('Standing waveform weights must be finite, nonnegative, and have a positive sum')
            self.waveform_cdf = torch.tensor([weight/total for weight in weights], device=env.device).cumsum(0)
            self.waveform_choices = torch.arange(1, len(PROFILES), device=env.device)
            self.waveform_counts = torch.zeros(len(names), device=env.device)
        self.side_contact = None
        if contact_settings := settings.get('standing_contact_reward'):
            if not self.side_focus or not self.capture_support:
                raise ValueError('Lateral support rewards require side pushes and ankle force sensors')
            from anon_lateral_support import SideSupportReward
            self.side_contact = SideSupportReward(env, self.mass, contact_settings)
        if self.steps is not None:
            self.strong = torch.zeros_like(self.enabled)
            self.grace_used = torch.zeros_like(self.enabled)
            self.airtime = torch.zeros((env.num_envs, 2), device=env.device)
        if self.capture_support:
            self.mass_weights = (masses / self.mass[:, None])[..., None]
            self.local_com = env.robot.get_coms()[..., :3].to(env.device).clone()
            self.first_foot = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
            self.first_speed = torch.zeros_like(self.mass)
            self.second_paid = torch.zeros_like(self.enabled)
            if self.standing_brake_weight:
                self.first_landing_age = torch.zeros_like(self.mass)
        self.weak_commands = torch.tensor([[-.4, 0., 0.], [.4, 0., .4], [0., .4, 0.], [.4, 0., 0.],
                                           [0., 0., 0.]], device=env.device)

    def _sample(self, ids, interval=(0., 1.)):
        low, high = interval
        return low + (high - low) * torch.rand(len(ids), device=self.env.device, generator=self.rng)

    def reset(self, ids):
        enabled = self._sample(ids) < self.settings['episode_probability']
        self.enabled[ids] = enabled
        self.delay[ids] = self._sample(ids, self.settings['initial_delay_seconds'])
        self.remaining[ids] = 0.
        if self.side_focus or self.fore_aft_rehearsal:
            self.protected_fore_aft[ids] = False
            self.side_axis[ids] = 0.
        if self.shape_fraction:
            self.waveform_force[ids] = 0.
        if self.side_contact is not None:
            self.side_contact.reset(ids)
        if self.steps is not None:
            self.strong[ids] = False
            self.grace_used[ids] = False
            self.airtime[ids] = 0.
        self.stats[0] += len(ids)
        self.stats[1] += enabled.sum()

    def resample_weak_commands(self, ids, on_reset):
        env = self.env
        targeted = ids[self.enabled[ids] & (self._sample(ids) < self.settings['weak_command_probability'])]
        choices = (self._sample(targeted) < .5).long()
        if self.left_step_focus_fraction:
            choices[self._sample(targeted) < self.left_step_focus_fraction] = 2
        if self.forward_focus or self.stand_focus:
            fraction = self._sample(targeted)
            choices[fraction < self.forward_focus] = 3
            choices[(fraction >= self.forward_focus) & (fraction < self.forward_focus+self.stand_focus)] = 4
        commands = self.weak_commands[choices]
        env.commands[targeted] = commands
        env.unified_command_hold[targeted] = 9. + self._sample(targeted)
        if on_reset:
            env.continuous_ref_commands[targeted] = commands

    def apply(self):
        """Called once, immediately before the first substep of a control frame."""
        env = self.env
        self.delay.sub_(env.dt)
        self.remaining.sub_(env.dt).clamp_(min=0.)
        target = limit_yaw_command(env.commands, env.cfg)
        tracking = (target - env.continuous_ref_commands).abs().amax(dim=1)
        upright = (env.gravity[:, 2] < -.9) & (env.poses[:, env.pelvis_id, 2] > .75)
        steady = (env.command_age >= 1.) & (tracking < .035)
        enough_time = env.unified_command_hold - env.command_age >= self.window + env.dt
        ids = (self.enabled & (self.delay <= 0.) & steady & upright & enough_time).nonzero().flatten()
        if len(ids) == 0:
            return
        forward_speed, side_speed, yaw_rate = target[ids].unbind(-1)
        backward = forward_speed < -.2
        left_turn = (forward_speed > .2) & (yaw_rate > .2)
        weak = backward | left_turn
        standing = target[ids].abs().amax(dim=1) < .02
        self.stats[21] += standing.sum()
        hard_cases = weak
        if self.left_step_focus_fraction:
            left_step = (side_speed > .25) & (forward_speed.abs() < .12) & (yaw_rate.abs() < .12)
            hard_cases = weak | left_step
        chance = torch.where(hard_cases, self.weak_challenge_probability, self.settings['challenge_probability'])
        if self.phase_boost:
            forward_walk = (forward_speed > .2) & (forward_speed < .65) & (side_speed.abs() < .15)
            phase = env.phase[ids] % 1.
            vulnerable = (backward | forward_walk) & (phase >= .25) & (phase < .875)
            chance = (chance + self.phase_boost * vulnerable).clamp(max=1.)
        challenge = self._sample(ids) < chance
        if self.left_step_focus_fraction:
            self.stats[11] += left_step.sum()
            self.stats[12] += (left_step & challenge).sum()
        impulse = torch.where(challenge, self._sample(ids, self.settings['challenge_m_s']),
                              self._sample(ids, self.settings['light_m_s']))
        if self.extreme_probability:
            extreme = challenge & (self._sample(ids) < self.extreme_probability)
            impulse = torch.where(extreme, self._sample(ids, self.settings['extreme_m_s']), impulse)
        else:
            extreme = torch.zeros_like(challenge)
        if self.phase_boost:
            self.stats[17] += (vulnerable & challenge).sum()
        angle = self._sample(ids, (0., 2. * math.pi))
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        if self.weak_direction_bias:
            # Keep the uniform directions and replace only a subset of weak-command challenge pushes.
            direction_focus = (weak | forward_walk) if self.phase_boost else weak
            biased = direction_focus & challenge & (self._sample(ids) < self.weak_direction_bias)
            selected = ids[biased]
            # Side/front pushes destabilize retreat. Side/rear pushes destabilize the forward turn.
            center = torch.where(backward[biased], math.pi / 3, 2 * math.pi / 3)
            side = torch.where(self._sample(selected) < .5, -1., 1.)
            local_angle = center * side + self._sample(selected, (-math.pi / 4, math.pi / 4))
            forward, left = horizontal_axes(env.poses[selected, env.pelvis_id, 3:])
            direction[biased] = local_angle[:, None].cos() * forward + local_angle[:, None].sin() * left
            self.stats[10] += len(selected)
        if self.side_focus or self.fore_aft_rehearsal:
            mix = self._sample(ids)
            side = standing & challenge & (mix < self.side_focus)
            fore_aft = standing & challenge & (mix >= self.side_focus) & (mix < self.side_focus + self.fore_aft_rehearsal)
            forward, left = horizontal_axes(env.poses[ids, env.pelvis_id, 3:])
            leftward = self._sample(ids) < self.left_fraction
            angle = torch.where(leftward, math.pi/2, -math.pi/2) + self._sample(ids, (-self.side_spread, self.side_spread))
            local = angle[:, None].cos() * forward + angle[:, None].sin() * left
            direction = torch.where(side[:, None], local, direction)
            fore_sign = torch.where(self._sample(ids) < .5, 1., -1.)
            direction = torch.where(fore_aft[:, None], fore_sign[:, None] * forward, direction)
            self.protected_fore_aft[ids] = fore_aft
            self.side_axis[ids] = left * torch.where(leftward, 1., -1.)[:, None] * side[:, None]
            self.stats[24] += (side & leftward).sum()
            self.stats[25] += (side & ~leftward).sum()
            self.stats[26] += fore_aft.sum()
        if self.shape_fraction:
            shaped = side & (self._sample(ids) < self.shape_fraction)
            choice = self._sample(ids)
            if self.waveform_cdf is None:
                waveform = 1 + (choice * (len(PROFILES) - 1)).long()
            else:
                waveform = 1 + torch.searchsorted(self.waveform_cdf, choice, right=True).clamp_max(len(PROFILES) - 2)
                self.waveform_counts += ((waveform[:, None] == self.waveform_choices) & shaped[:, None]).sum(0)
            waveform = torch.where(shaped, waveform, 0)
            self.waveform_id[ids] = waveform
            self.waveform_tick[ids] = 0
            self.stats[27] += shaped.sum()
        if self.side_bands is not None:
            profile = waveform if self.shape_fraction else torch.zeros_like(ids)
            impulse, extreme = self.side_bands.choose(impulse, extreme, side, leftward, profile,
                                                       self._sample(ids), self._sample(ids))
        if self.extreme_probability or self.side_bands is not None:
            self.stats[13] += extreme.sum()
        strength = (self.mass[ids] * env.cfg['physics_hz'] * impulse)[:, None] * direction
        if self.shape_fraction:
            self.waveform_force[ids] = strength
        else:
            self._apply_single(ids, strength)
        self.delay[ids] = self._sample(ids, self.settings['repeat_delay_seconds'])
        self.remaining[ids] = self.window
        if self.steps is not None:
            strong = impulse >= self.steps['minimum_impulse_m_s']
            self.strong[ids] = strong
            self.grace_used[ids] = False
            self.airtime[ids] = 0.
            self.stats[14] += strong.sum()
            self.stats[22] += (standing & strong).sum()
        if self.capture_support:
            self.first_foot[ids] = -1
            self.second_paid[ids] = False
        if self.side_contact is not None:
            side_ids = ids[side]
            self.side_contact.start(side_ids, env.poses[side_ids][:, env.foot_ids, :2], impulse[side])
        self.stats[2] += len(ids)
        self.stats[3] += challenge.sum()
        self.stats[4] += impulse.sum()
        self.stats[5] += weak.sum()
        self.stats[7] += (weak & challenge).sum()
        self.stats[8] += backward.sum()
        self.stats[9] += left_turn.sum()

    def _apply_single(self, ids, strength):
        self.forces.zero_()
        self.forces[ids, self.link, :2] = strength
        self.env.robot.apply_forces_and_torques_at_position(self.forces, None, None, self.env.ids, True)

    def apply_substep(self):
        if self.shape_fraction:
            weight = self.waveforms[self.waveform_id, self.waveform_tick]
            self.forces[:, self.link, :2] = self.waveform_force * weight[:, None]
            self.env.robot.apply_forces_and_torques_at_position(self.forces, None, None, self.env.ids, True)
            self.waveform_tick.add_(1).clamp_(max=self.waveforms.shape[1] - 1)
            expired = (self.waveform_id == 0) | (self.waveform_tick == self.waveforms.shape[1] - 1)
            # Keep the last sample until it has been applied. The table has a trailing zero.
            self.waveform_force.masked_fill_(expired[:, None], 0.)

    def recovery_fraction(self):
        return (self.remaining - (self.window - self.style_window)).clamp_min(0.) / self.style_window

    def style_scale(self):
        floor = self.settings['style_floor']
        if self.steps is not None:
            floor = torch.where(self.strong, self.steps['style_floor'], floor)
        return 1. - (1. - floor) * self.recovery_fraction()

    def reference_failure(self, failed, physical_fall):
        if self.steps is None:
            return failed
        grace = self.strong & (self.remaining > self.window - self.steps['reference_grace_seconds'])
        rescued = failed & grace & ~physical_fall & ~self.grace_used
        self.stats[15] += rescued.sum()
        self.grace_used |= rescued
        return failed & ~grace

    def tracking_scale(self):
        if not self.capture_support:
            return 1.
        initial = (1. - (self.window - self.remaining) / .5).clamp(0., 1.)
        return 1. - .5 * initial * self.strong

    def corrective_step_reward(self, pelvis, feet, sole_height, contact, reference_velocity, fallen):
        """Credit actual corrective touchdowns and damp large torso tilt after the initial response."""
        env = self.env
        active = self.strong & (self.remaining > 0.) & ~fallen
        grounded = contact.bool()
        if self.capture_support:
            contact_force = env.recovery_contact_view.get_contact_force_matrix(1 / env.cfg['physics_hz'])
            normal_force = contact_force[env.recovery_contact_indices, 0, 2]
            grounded = normal_force > 12.
            self.stats[18] += (grounded.any(dim=1) & active).sum()
        landed = grounded & (self.airtime >= .05)
        self.airtime = torch.where((sole_height > .025) & active[:, None] & ~grounded, self.airtime + env.dt,
                                   torch.where(grounded | ~active[:, None], 0., self.airtime))
        # Project the pelvis along excess velocity in world XY. Comparing the
        # landing to the other foot credits catching the body, rather than stepping in place.
        excess_velocity = env.root_vel[:, :2] - reference_velocity[:, :2]
        capture_origin = pelvis[:, :2]
        if self.capture_support:
            from anon_locomotion_env import quat_rotate_inverse
            q = env.poses[..., 3:]
            inverse = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
            world_com = env.poses[..., :3] + quat_rotate_inverse(inverse, self.local_com)
            capture_origin = (world_com[..., :2] * self.mass_weights).sum(dim=1)
            excess_velocity = (env.link_vel[..., :2] * self.mass_weights).sum(dim=1) - reference_velocity[:, :2]
        speed = excess_velocity.norm(dim=-1)
        offset = .23 * excess_velocity / speed.clamp_min(1.)[:, None]
        capture_point = capture_origin + offset
        distance = (feet[:, :, :2] - capture_point[:, None]).norm(dim=-1)
        gain = ((distance.flip((1,)) - distance) / .20).clamp(0., 1.)
        corrective = landed & active[:, None] & (speed[:, None] > .22) & (gain > 0.)
        self.stats[16] += corrective.sum()
        quality, foot = (gain * corrective).max(dim=1)
        landing_reward = self.steps['landing_bonus'] * quality
        if self.capture_support:
            age = self.window - self.remaining
            first = (quality > 0.) & (self.first_foot < 0) & (age <= 1.0)
            next_step = (active & (self.first_foot >= 0) & ~self.second_paid
                         & landed.gather(1, (1-self.first_foot).clamp(max=1)[:, None]).squeeze(1) & (age <= 1.6))
            slowed = ((self.first_speed - speed) / .25).clamp(0., 1.)
            landing_reward *= ((1.0-age) / .65).clamp(.25, 1.)
            landing_reward += self.steps['landing_bonus'] * .6 * next_step * slowed
            self.first_foot = torch.where(first, foot, self.first_foot)
            self.first_speed = torch.where(first, speed, self.first_speed)
            self.second_paid |= next_step & (slowed > 0.)
            self.stats[19] += first.sum()
            self.stats[20] += (next_step & (slowed > 0.)).sum()
            if self.standing_brake_weight:
                self.first_landing_age = torch.where(first, age, self.first_landing_age)
                standing = env.continuous_ref_commands.abs().amax(dim=-1) < .02
                since_landing = age - self.first_landing_age
                loaded_foot = normal_force.gather(1, self.first_foot.clamp_min(0)[:, None]).squeeze(1)
                load = (loaded_foot / (self.mass * 9.81 * .18)).clamp(0., 1.)
                braking = ((self.first_speed - speed) / .35).clamp(0., 1.)
                caught = (active & standing & (self.first_foot >= 0)
                          & (since_landing > 0.) & (since_landing <= .36))
                landing_reward += self.standing_brake_weight * env.dt * load * braking * caught
                self.stats[23] += (caught & (load >= .5) & (braking > 0.)).sum()
            if self.side_weight:
                stride = ((feet[:, :, :2] - feet.flip((1,))[:, :, :2]) * self.side_axis[:, None]).sum(-1)
                reach = ((stride - .16) / .22).clamp(0., 1.)
                lateral_touchdown = (reach * corrective).amax(dim=1) * (age <= 1.15)
                landing_reward += self.side_weight * lateral_touchdown
                self.stats[28] += (lateral_touchdown > 0.).sum()
            if self.side_contact is not None:
                com_height = (world_com[..., 2:3] * self.mass_weights).sum(dim=1)
                center = torch.cat((capture_origin, com_height), dim=-1)
                landing_reward += self.side_contact.reward(active, age, self.side_axis, feet[:, :, :2],
                    sole_height, normal_force, center, excess_velocity)

        torso = env.poses[:, self.link, 3:]
        tilt_cost = ((2 * torso[:, :2].square().sum(-1) - .08) / .20).clamp(0., 1.)
        rotation_cost = ((env.link_vel[:, self.link, 3:5].norm(dim=-1) - 1.3) / 3.).clamp(0., 1.)
        settling = ((self.window - self.remaining - .20) / .45).clamp(0., 1.)
        torso_cost = self.steps['trunk_weight'] * (.7 * tilt_cost + .3 * rotation_cost) * settling
        return torch.where(active, landing_reward - torso_cost * env.dt, 0.)

    def adjust_reward(self, reward, fallen):
        if self.balance_weight:
            env = self.env
            tilt = (env.gravity[:, :2].square().sum(-1) / .16).clamp(max=1.)
            error = env.logical_vel[:, :2] - env.continuous_ref_commands[:, :2]
            speed = (error.square().sum(-1) / self.speed_scale_squared).clamp(max=1.)
            reward -= self.balance_weight * (.6*tilt + .4*speed*self.tracking_scale()) * self.remaining / self.window * env.dt
        if self.fall_penalty:
            reward -= self.fall_penalty * (fallen & (self.remaining > 0.))
        return reward

    def record_falls(self, fallen):
        self.stats[6] += (fallen & (self.remaining > 0.)).sum()

    def summary(self):
        (episodes, enabled, pushes, challenge, impulse, weak, falls, weak_challenge,
         backward, left_turn, biased, left_step, left_step_challenge, extreme,
         strong, grace, landings, vulnerable, supports, first, second, stand, stand_strong,
         stand_braking, side_left, side_right, fore_aft, shaped, side_landings) = self.stats.tolist()
        result = dict(episodes=int(episodes), push_episodes=int(enabled), pushes=int(pushes),
                    challenge_pushes=int(challenge), weak_command_pushes=int(weak),
                    weak_challenge_pushes=int(weak_challenge), backward_pushes=int(backward),
                    forward_left_turn_pushes=int(left_turn),
                    direction_biased_pushes=int(biased),
                    left_step_focus_pushes=int(left_step), left_step_focus_challenges=int(left_step_challenge),
                    extreme_pushes=int(extreme), strong_recovery_pushes=int(strong),
                    reference_grace_events=int(grace), corrective_landings=int(landings),
                    vulnerable_phase_challenges=int(vulnerable), measured_support_frames=int(supports),
                    early_corrective_steps=int(first), slowing_followup_steps=int(second),
                    standing_pushes=int(stand), strong_standing_pushes=int(stand_strong),
                    standing_braking_support_frames=int(stand_braking),
                    standing_side_left_pushes=int(side_left), standing_side_right_pushes=int(side_right),
                    standing_fore_aft_pushes=int(fore_aft), standing_shaped_pushes=int(shaped),
                    standing_side_landings=int(side_landings),
                    mean_impulse_m_s=impulse/pushes if pushes else 0.,
                    falls_in_recovery_window=int(falls))
        if self.side_contact is not None:
            result.update(side_swing_reach_frames=int(self.side_contact.reach_frames.item()),
                          side_captured_support_frames=int(self.side_contact.corrected_support_frames.item()))
            if self.side_contact.early_capture_weight:
                result.update(early_reach_frames=int(self.side_contact.early_reach_frames.item()),
                              early_braking_support_frames=int(self.side_contact.braking_support_frames.item()))
            followup = self.side_contact.followup
            if followup is not None:
                result.update(followup_swing_frames=int(followup.followup_swing_frames.item()),
                              followup_braking_frames=int(followup.followup_braking_frames.item()))
                if followup.prelanding_brake_weight:
                    result['prelanding_brake_frames'] = int(followup.prelanding_brake_frames.item())
        if self.waveform_cdf is not None:
            result['standing_shaped_profile_counts'] = dict(zip(tuple(PROFILES)[1:], map(int, self.waveform_counts.tolist())))
        if self.side_bands is not None:
            result['standing_band_counts'] = self.side_bands.summary()
        return result
