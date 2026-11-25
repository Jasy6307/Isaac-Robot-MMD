"""Verify the stop-recovery reward and the forward-facing acceptance gate."""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from test_low_speed_acceptance import item, reference_trace
from evaluate_unified_walk_run import suite
from idle_head_alignment import (
    HeadSwayPenalty, MOVING_GAIN, RECOVERY_RAMP_SECONDS, backward_walk_head_alignment_reward,
    head_facing_error, idle_head_reward,
    idle_neck_pitch_reward, idle_neck_yaw_reward, neck_pitch_level_target, neck_pitch_stop_context,
    neck_yaw_forward_target, neck_yaw_stop_context, straight_backward_walk,
)
from locomotion_acceptance import CURRENT_PROTOCOL,PROTOCOL,score_segment,specification


class IdleHeadRewardTests(unittest.TestCase):
    def test_head_forward_uses_ctrl_heading_and_world_horizontal(self):
        headings = np.array([172., -175., 65.])
        yaw_error = np.array([0., -12., 5.])
        pitch_error = np.array([0., -4., 8.])
        world_yaw = Rotation.from_euler('z', (headings + yaw_error)[:, None], degrees=True)
        elevation = Rotation.from_euler('x', -pitch_error[:, None], degrees=True)
        roll = Rotation.from_euler('y', [[10.], [-10.], [15.]], degrees=True)
        quat = torch.from_numpy((world_yaw * elevation * roll).as_quat())
        errors = head_facing_error(quat, torch.from_numpy(np.deg2rad(headings)))
        np.testing.assert_allclose(errors.rad2deg(), np.column_stack((yaw_error, pitch_error)), atol=1e-10)

    def test_stop_reward_ramps_while_braking_or_finishing_a_turn(self):
        errors = torch.tensor([[1., -2.], [-3., 3.], [0., 0.]]).deg2rad()
        requested = torch.zeros(3, 3)
        initial_motion = torch.tensor([[.55, 0., 0.], [0., 0., -.60], [0., 0., 0.]])
        upright = torch.ones(3, dtype=torch.bool)
        stop_time = torch.zeros(3)
        last_gain = torch.zeros(3)
        for motion_scale in (1., .7, .4, .1, 0.):
            remaining = motion_scale * initial_motion
            stop_time, gain, reward = idle_head_reward(
                errors, requested, remaining, remaining, stop_time, upright, RECOVERY_RAMP_SECONDS/5, 2.)
            self.assertTrue(torch.all(gain > last_gain))
            torch.testing.assert_close(reward, 2 * gain)
            last_gain = gain
        torch.testing.assert_close(gain, torch.ones(3))

        # Planned motion and actual motion each limit the gain if the other has already stopped.
        zeros = torch.zeros_like(initial_motion)
        for plan, velocity in ((initial_motion, zeros), (zeros, initial_motion)):
            _, gain, _ = idle_head_reward(errors, requested, plan, velocity, stop_time, upright, 1/30, 2.)
            torch.testing.assert_close(gain, torch.tensor([MOVING_GAIN, MOVING_GAIN, 1.]))

    def test_resumed_motion_falls_and_invalid_samples_clear_recovery(self):
        requested = torch.tensor([[.03, 0., 0.], [0., 0., -.6], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        planned = torch.zeros_like(requested)
        velocity = planned.clone()
        velocity[3, 0] = float('inf')
        errors = torch.zeros(5, 2)
        errors[4, 0] = float('nan')
        upright = torch.tensor([True, True, False, True, True])
        stop_time = torch.full((5,), RECOVERY_RAMP_SECONDS)
        stop_time, gain, reward = idle_head_reward(errors, requested, planned, velocity, stop_time, upright, 1/30, 2.)
        for value in (stop_time, gain, reward):
            torch.testing.assert_close(value, torch.zeros(5))

    def test_reward_distinguishes_large_errors_and_is_mirror_symmetric(self):
        angles = torch.tensor([3., 6., 10., 20., 30., 40., 90., 179.])
        commands = torch.zeros(len(angles), 3)
        errors = torch.stack((angles, torch.zeros_like(angles)), dim=-1).deg2rad()
        stop_time = torch.full_like(angles, RECOVERY_RAMP_SECONDS)
        upright = torch.ones_like(angles, dtype=torch.bool)
        _, _, score = idle_head_reward(errors, commands, commands, commands, stop_time, upright, 1/30, 2.)
        _, _, mirrored = idle_head_reward(-errors, commands, commands, commands, stop_time, upright, 1/30, 2.)
        torch.testing.assert_close(score, mirrored)
        self.assertEqual(float(score[0]), 2.)
        self.assertTrue(torch.all(score[:-1] > score[1:]))
        self.assertGreater(float(score[3] - score[4]), .1)
        self.assertGreater(float(score[4] - score[5]), .1)
        self.assertGreater(float(score[-1]), 0.)

    def test_large_angle_penalty_preserves_the_small_error_curve_and_boosts_correction(self):
        errors = torch.tensor([[3., 2.], [8., 0.], [20., 0.], [21., 0.], [30., 0.],
                               [31., 0.], [40., 0.], [41., 0.], [179., 90.]]).deg2rad()
        commands = torch.zeros(len(errors), 3)
        stop_time = torch.full((len(errors),), RECOVERY_RAMP_SECONDS)
        upright = torch.ones(len(errors), dtype=torch.bool)
        args = (commands, commands, commands, stop_time, upright, 1/30, 2.)
        _, _, before = idle_head_reward(errors, *args)
        _, gain, stronger = idle_head_reward(errors, *args, large_error_penalty=2.)
        _, _, mirrored = idle_head_reward(errors * torch.tensor([-1., 1.]), *args, large_error_penalty=2.)
        torch.testing.assert_close(stronger, mirrored)
        torch.testing.assert_close(gain, torch.ones_like(gain))
        torch.testing.assert_close(before[:2], stronger[:2], rtol=0, atol=0)
        for i in (2, 4, 6):
            self.assertGreater(float(stronger[i] - stronger[i+1]), 3. * float(before[i] - before[i+1]))
        self.assertTrue((stronger[:-1] > stronger[1:]).all())
        self.assertGreater(float(stronger[-1]), -4.)
        requested = commands.clone()
        requested[:, 2] = .6
        _, _, rate = idle_head_reward(errors, requested, *args[1:], large_error_penalty=2.)
        torch.testing.assert_close(rate, torch.zeros_like(rate))


class HeadSwayTests(unittest.TestCase):
    def test_backward_boost_requires_a_backward_walk_command_and_plan(self):
        requested = torch.tensor([
            [-.40, 0., 0.], [-.15, 0., 0.], [.40, 0., 0.], [0., 0., 0.],
            [-.9, 0., 0.], [-.4, .12, 0.], [-.4, 0., -.4], [-.4, 0., 0.],
        ])
        plan = requested.clone()
        plan[-1] = torch.tensor([0., 0., -.45])
        active = straight_backward_walk(requested, .55) & straight_backward_walk(plan, .55)
        torch.testing.assert_close(active, torch.tensor([True, True, False, False, False, False, False, False]))

    def test_small_sway_and_monotonic_motion_have_room(self):
        penalty = HeadSwayPenalty(3, 1/30, 'cpu', .4)
        upright = torch.ones(3, dtype=torch.bool)
        for step in range(36):
            sway = 1.5 * math.sin(2. * math.pi * 1.3 * step / 30.)
            heading = (170. + step * 2. + 180.) % 360. - 180.
            errors = torch.tensor([[sway, sway], [heading, 0.], [0., 30. - step]]).deg2rad()
            torch.testing.assert_close(penalty.step(errors, upright), torch.zeros(3))

    def test_large_reversals_penalized_and_reset_is_per_environment(self):
        penalty = HeadSwayPenalty(3, 1/30, 'cpu', .4)
        upright = torch.ones(3, dtype=torch.bool)
        for step in range(16):
            yaw = (0., 6., 0., -6.)[step % 4]
            rate = penalty.step(torch.tensor([[yaw, 0.], [0., yaw], [yaw, -yaw]]).deg2rad(), upright)
        self.assertTrue(torch.all(rate < -.2))
        self.assertTrue(torch.all(rate >= -.4))
        penalty.reset(torch.tensor([0]))
        next_rate = penalty.step(torch.tensor([[50., 0.], [float('nan'), 0.], [0., 0.]]).deg2rad(), upright)
        torch.testing.assert_close(next_rate[:2], torch.zeros(2))
        self.assertLess(float(next_rate[2]), -.2)


class BackwardWalkHeadAlignmentTests(unittest.TestCase):
    def test_dead_zone_sign_symmetry_and_larger_offsets(self):
        errors = torch.tensor([[3., -3.], [6., 0.], [15., 0.], [-15., 0.], [0., -15.],
                               [30., 0.], [15., 0.], [float('nan'), 15.]]).deg2rad()
        active = torch.tensor([True, True, True, True, True, True, False, True])
        rate = backward_walk_head_alignment_reward(errors, active, .6)
        torch.testing.assert_close(rate[[0, 6, 7]], torch.zeros(3), atol=0., rtol=0.)
        torch.testing.assert_close(rate[2:5], torch.full((3,), -.21))
        self.assertGreater(float(rate[1]), float(rate[2]))
        self.assertGreater(float(rate[2]), float(rate[5]))

    def test_first_stop_tick_hands_over_while_planner_is_still_moving(self):
        errors = torch.tensor([[12., -7.]]).deg2rad()
        moving = torch.tensor([[-.4, 0., 0.]])
        stopped = torch.zeros_like(moving)
        upright = torch.ones(1, dtype=torch.bool)
        planned_backwards = straight_backward_walk(moving, .55)
        during_walk = backward_walk_head_alignment_reward(errors, planned_backwards, .6)
        self.assertLess(float(during_walk[0]), 0.)
        active_on_stop = straight_backward_walk(stopped, .55) & planned_backwards
        on_stop = backward_walk_head_alignment_reward(errors, active_on_stop, .6)
        torch.testing.assert_close(on_stop, torch.zeros(1), atol=0., rtol=0.)
        _, gain, head_rate = idle_head_reward(errors, stopped, moving, moving, torch.zeros(1), upright, 1/30, 2.)
        self.assertGreater(float(gain[0]), 0.)
        self.assertGreater(float(head_rate[0]), 0.)


class IdleNeckPitchTargetTests(unittest.TestCase):
    def test_pitch_boost_remembers_straight_walk_and_both_in_place_turn_stops(self):
        moves = torch.tensor([
            [.3, 0., 0.], [0., 0., .6], [0., 0., -.6], [0., 0., 0.],
            [.8, 0., 0.], [-.3, 0., 0.], [.3, .15, 0.], [.3, 0., .2],
            [.3, 0., -.2],
        ])
        previous = torch.zeros(len(moves), dtype=torch.bool)
        still = torch.zeros_like(moves)
        context = neck_pitch_stop_context(previous, moves, .55)
        expected = torch.tensor([True, True, True, False, False, False, False, False, False])
        torch.testing.assert_close(context, expected)
        torch.testing.assert_close(neck_pitch_stop_context(context, still, .55), expected)
        torch.testing.assert_close(neck_pitch_stop_context(previous, still, .55), previous)
        moves[:3] = moves[4]
        torch.testing.assert_close(neck_pitch_stop_context(context, moves, .55), previous)

    def test_joint_targets_level_the_commanded_direction_and_reflect_correctly(self):
        chest = Rotation.from_euler('zyx', [[12., 8., -9.], [139., -5., 14.], [-171., 10., 19.]], degrees=True)
        heading = np.deg2rad([8., 127., 175.])
        q = torch.from_numpy(chest.as_quat())
        target = neck_pitch_level_target(q, torch.from_numpy(heading)).numpy()
        forward = np.stack((np.sin(heading), -np.cos(heading), np.zeros_like(heading)), axis=-1)
        local = chest.apply(forward, inverse=True)
        yaw = neck_yaw_forward_target(q, torch.from_numpy(heading))
        torch.testing.assert_close(yaw, torch.from_numpy(np.arctan2(local[:, 0], -local[:, 1])),
                                   rtol=0., atol=1e-12)
        realized = chest * Rotation.from_euler('z', yaw.numpy()[:, None]) * Rotation.from_euler('x', target[:, None])
        errors = head_facing_error(torch.from_numpy(realized.as_quat()), torch.from_numpy(heading))
        torch.testing.assert_close(errors, torch.zeros_like(errors), rtol=0., atol=1e-12)

        reflected = q * torch.tensor([1., -1., -1., 1.])
        reflected_pitch = neck_pitch_level_target(reflected, -torch.from_numpy(heading))
        torch.testing.assert_close(reflected_pitch, torch.from_numpy(target), rtol=0., atol=1e-12)
        reflected_yaw = neck_yaw_forward_target(reflected, -torch.from_numpy(heading))
        torch.testing.assert_close(reflected_yaw, -yaw, rtol=0., atol=1e-12)

    def test_reward_respects_stop_gain_dead_zone_and_invalid_pose(self):
        pitch = torch.tensor([1., 2., 8., 16., -16., 16., 16., 1.e21]).deg2rad()
        target = torch.zeros_like(pitch)
        target[5] = float('nan')
        gain = torch.tensor([1., 1., 1., 1., .5, 1., 0., 1.])
        rate = idle_neck_pitch_reward(pitch, target, gain, 1.)
        torch.testing.assert_close(rate[[0, 1, 5, 6]], torch.zeros(4))
        self.assertGreater(float(rate[2]), float(rate[3]))
        self.assertLess(float(rate[2]), 0.)
        torch.testing.assert_close(rate[4], .5 * rate[3])
        self.assertEqual(float(rate[-1]), -1.)


class IdleNeckYawTargetTests(unittest.TestCase):
    def test_context_remembers_only_in_place_turns_and_clears_on_other_motion(self):
        moves = torch.tensor([
            [0., 0., .6], [0., 0., -.6], [.3, 0., 0.], [0., -.3, 0.],
            [.8, 0., .4], [-.3, 0., -.2], [0., 0., 0.],
        ])
        reset = torch.zeros(len(moves), dtype=torch.bool)
        stopped = torch.zeros_like(moves)
        context = neck_yaw_stop_context(reset, moves)
        expected = torch.tensor([True, True, False, False, False, False, False])
        torch.testing.assert_close(context, expected)
        torch.testing.assert_close(neck_yaw_stop_context(context, stopped), expected)
        torch.testing.assert_close(neck_yaw_stop_context(reset, stopped), reset)
        moves[0:2] = moves[2]
        torch.testing.assert_close(neck_yaw_stop_context(context, moves), reset)

    def test_reward_is_symmetric_and_uses_pivot_context_stop_gain_and_tolerance(self):
        target = torch.zeros(10)
        target[9] = float('nan')
        yaw = torch.tensor([3., -3., 5., -5., 10., -10., 5., -5., 5., 5.]).deg2rad()
        gain = torch.ones_like(target)
        gain[6:8] = torch.tensor([.5, 0.])
        pivot_stop = torch.ones_like(target, dtype=torch.bool)
        pivot_stop[8] = False
        rate = idle_neck_yaw_reward(yaw, target, gain, pivot_stop, .5)
        torch.testing.assert_close(rate[[0, 1, 7, 8, 9]], torch.zeros(5), atol=0., rtol=0.)
        torch.testing.assert_close(rate[[2, 4]], rate[[3, 5]], atol=0., rtol=0.)
        torch.testing.assert_close(rate[6], .5 * rate[2], atol=0., rtol=0.)
        self.assertGreater(float(rate[2]), float(rate[4]))
        self.assertLess(float(rate[2]), 0.)


class IdleHeadAcceptanceTests(unittest.TestCase):
    def score(self, label, initial, seconds, yaw, pitch=None, *, protocol=PROTOCOL, falls=0):
        cfg = {'unified_idle_head_acceptance': True, 'unified_acceptance_protocol': protocol}
        actual = reference_trace(initial, [0., 0., 0.], seconds, cfg)
        requested = np.zeros_like(actual)
        angles = None if yaw is None else np.deg2rad(np.column_stack(
            (yaw, np.zeros(len(yaw)) if pitch is None else pitch)))
        return score_segment(item(label, seconds, falls=falls), actual, actual, requested, actual[:, :2], initial,
                             cfg, 1/30, head_errors=angles)

    def test_transient_recovery_passes_with_small_sway(self):
        time = np.arange(240) / 30
        for label, initial in [('stand', [0, 0, 0]), ('stop_after_walk', [.3, 0, 0]),
                               ('stop_after_pivot_left', [0, 0, .6]), ('stand_end', [0, 0, -.6])]:
            with self.subTest(label=label):
                yaw = np.where(time < 1.8, 14., 3. * np.sin(8 * time))
                result = self.score(label, initial, 8, yaw)
                self.assertTrue(result['passed'])
                self.assertGreater(result['idle_head']['recenter_confirmed_seconds'], 2.)
                self.assertLess(result['idle_head']['yaw_abs_p95_deg'], 3.1)

    def test_ten_degree_boundary_passes_both_head_limits(self):
        yaw = 10. * (-1.)**np.arange(240)
        result = self.score('stand', [0, 0, 0], 8, yaw, np.full(240, 10.))
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['idle_head']['pitch_bias_deg'], 10.)
        self.assertAlmostEqual(result['idle_head']['yaw_abs_p95_deg'], 10.)

    def test_stable_bias_and_large_alternating_deviation_each_fail(self):
        fixed_bias = self.score('stand', [0, 0, 0], 8, np.zeros(240), np.full(240, 11.))
        self.assertTrue(fixed_bias['steady']['passed'])
        self.assertFalse(fixed_bias['passed'])
        self.assertAlmostEqual(fixed_bias['idle_head']['pitch_bias_deg'], 11.)
        oscillation = self.score('stand', [0, 0, 0], 8, 11. * (-1.)**np.arange(240))
        self.assertFalse(oscillation['passed'])
        self.assertAlmostEqual(oscillation['idle_head']['yaw_bias_deg'], 0.)
        self.assertAlmostEqual(oscillation['idle_head']['yaw_abs_p95_deg'], 11.)

    def test_short_turn_stop_stays_inconclusive(self):
        result = self.score('stand_end', [0, 0, .6], 4, np.zeros(120))
        self.assertEqual(result['idle_head']['status'], 'insufficient_duration')
        self.assertFalse(result['passed'])

    def test_v3_records_head_yaw_and_pitch_without_gating(self):
        yaw = np.full(240, 12.)
        pitch = 16. * (-1.)**np.arange(240)
        self.assertEqual(specification(True, CURRENT_PROTOCOL)['idle_head']['required_segments'], ())
        for label in ('stand', 'stop', 'stop_after_walk', 'stop_after_pivot_left', 'stand_end'):
            with self.subTest(label=label):
                result = self.score(label, [0, 0, 0], 8, yaw, pitch, protocol=CURRENT_PROTOCOL)
                self.assertEqual(result['body_status'], 'passed')
                self.assertTrue(result['passed'])
                self.assertFalse(result['idle_head']['required'])
                self.assertEqual(result['idle_head']['status'], 'recorded')
                self.assertAlmostEqual(result['idle_head']['yaw_bias_deg'], 12.)
                self.assertAlmostEqual(result['idle_head']['pitch_abs_p95_deg'], 16.)
        self.assertFalse(self.score('stop', [0, 0, 0], 8, yaw)['passed'])
        self.assertFalse(self.score('stop', [0, 0, 0], 8, yaw, protocol=CURRENT_PROTOCOL, falls=1)['passed'])

    def test_v3_scores_body_with_missing_or_nonfinite_head_data(self):
        for yaw in (None, np.zeros(239), np.full(240, np.nan)):
            result = self.score('stand', [0, 0, 0], 8, yaw, protocol=CURRENT_PROTOCOL)
            self.assertTrue(result['passed'])
            self.assertEqual(result['idle_head']['status'], 'unavailable')

    def test_official_suites_cover_start_walk_stop_and_turn_stop(self):
        for protocol in (PROTOCOL, CURRENT_PROTOCOL):
            result = suite(protocol=protocol, idle_head=True)
            self.assertEqual(result[0][0], 'stand')
            self.assertEqual([row[0] for row in result[-2:]], ['pivot_right', 'stand_end'])
            labels = [row[0] for row in result]
            self.assertEqual(labels[labels.index('stop_after_walk') - 1], 'walk_before_stop')
            self.assertEqual(labels[labels.index('stop_after_pivot_left') - 1], 'pivot_left')


if __name__ == '__main__':
    unittest.main()
