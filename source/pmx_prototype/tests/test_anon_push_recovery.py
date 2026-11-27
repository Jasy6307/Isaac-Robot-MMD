"""Push recovery writes one sparse physical impulse and resets episode masks."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_push_recovery import PushRecovery
from anon_push_waveforms import PROFILES, profile_table


class PushRecoveryTests(unittest.TestCase):
    def setUp(self):
        robot = SimpleNamespace(max_links=2, get_masses=lambda: torch.tensor([[10., 10.], [10., 20.], [15., 25.]]),
                                apply_forces_and_torques_at_position=Mock())
        commands = torch.tensor([[0., 0., 0.], [-.4, 0., 0.], [.4, 0., .4]])
        self.env = SimpleNamespace(robot=robot, device='cpu', links=['pelvis', 'waist_pitch'],
                                   ids=torch.arange(3, dtype=torch.int32), num_envs=3, dt=1/30,
                                   cfg={'physics_hz': 120}, pelvis_id=0, commands=commands,
                                   continuous_ref_commands=commands.clone(), command_age=torch.full((3,), 2.),
                                   unified_command_hold=torch.full((3,), 9.),
                                   gravity=torch.tensor([[0.,0.,-1.]]*3), poses=torch.ones(3,2,7))
        self.training = PushRecovery(self.env, dict(
            episode_probability=0., challenge_probability=0., weak_command_probability=1.,
            initial_delay_seconds=(2., 2.), repeat_delay_seconds=(5., 5.), light_m_s=(.05, .05),
            challenge_m_s=(.1, .1), recovery_window_seconds=1.2, style_floor=.55), seed=42)

    def test_only_due_enabled_actor_receives_one_tick_of_force(self):
        self.training.enabled[:] = torch.tensor([False, True, True])
        self.env.command_age[2] = 0.
        self.training.apply()
        args = self.env.robot.apply_forces_and_torques_at_position.call_args.args
        force, torque, position, ids, world = args
        self.assertEqual(ids.tolist(), [0, 1, 2])
        self.assertIsNone(torque)
        self.assertIsNone(position)
        self.assertTrue(world)
        self.assertEqual(torch.count_nonzero(force[[0, 2]]), 0)
        self.assertEqual(torch.count_nonzero(force[1, 0]), 0)
        self.assertEqual(float(force[1, 1, 2]), 0.)
        self.assertAlmostEqual(float(force[1, 1].norm()), 30 * 120 * .05, places=4)
        self.assertAlmostEqual(float(self.training.recovery_fraction()[1]), 1.)
        self.training.apply()
        self.env.robot.apply_forces_and_torques_at_position.assert_called_once()

    def test_reset_clears_pending_recovery_and_excludes_clean_commands(self):
        self.training.enabled[:] = True
        self.training.remaining[:] = 1.
        self.training.reset(self.env.ids[1:2])
        self.assertEqual(self.training.enabled.tolist(), [True, False, True])
        self.assertEqual(self.training.remaining.tolist(), [1., 0., 1.])
        old = self.env.commands.clone()
        self.training.resample_weak_commands(self.env.ids, on_reset=True)
        self.assertTrue(torch.equal(old[1], self.env.commands[1]))
        self.assertTrue(torch.equal(self.env.continuous_ref_commands, self.env.commands))

    def test_challenge_strength_can_target_backward_and_forward_left_turn(self):
        self.training.weak_challenge_probability = 1.
        self.training.enabled[:] = True
        self.training.apply()
        force = self.env.robot.apply_forces_and_torques_at_position.call_args.args[0]
        actual = force[:, 1].norm(dim=-1) / (torch.tensor([20., 30., 40.]) * 120)
        torch.testing.assert_close(actual, torch.tensor([.05, .10, .10]))
        stats = self.training.summary()
        self.assertEqual(stats['weak_challenge_pushes'], 2)
        self.assertEqual(stats['backward_pushes'], 1)
        self.assertEqual(stats['forward_left_turn_pushes'], 1)

    def test_weak_challenge_direction_bias_follows_pelvis_heading(self):
        self.training.weak_challenge_probability = 1.
        self.training.weak_direction_bias = 1.
        self.training.enabled[:] = True
        self.env.poses[1, 0, 3:] = torch.tensor([0., 0., 0., 1.])
        self.env.poses[2, 0, 3:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        self.training.apply()
        forces = self.env.robot.apply_forces_and_torques_at_position.call_args.args[0]
        directions = torch.nn.functional.normalize(forces[:, 1, :2], dim=-1)
        retreat_offset = math.acos(-float(directions[1, 1]))
        turning_offset = math.acos(float(directions[2, 0]))
        self.assertLessEqual(abs(retreat_offset - math.pi / 3), math.pi / 4)
        self.assertLessEqual(abs(turning_offset - 2 * math.pi / 3), math.pi / 4)
        actual = forces[:, 1].norm(dim=-1) / (torch.tensor([20., 30., 40.]) * 120)
        torch.testing.assert_close(actual, torch.tensor([.05, .10, .10]))
        self.assertEqual(self.training.summary()['direction_biased_pushes'], 2)

    def test_left_step_focus_resamples_and_strengthens_left_steps(self):
        self.training.left_step_focus_fraction = 1.
        self.training.weak_challenge_probability = 1.
        self.training.enabled[:] = True
        self.training.resample_weak_commands(self.env.ids[:1], on_reset=True)
        torch.testing.assert_close(self.env.commands[0], torch.tensor([0., .4, 0.]))
        self.training.apply()
        forces = self.env.robot.apply_forces_and_torques_at_position.call_args.args[0]
        self.assertAlmostEqual(float(forces[0, 1].norm()), 20 * 120 * .10, places=4)
        self.assertEqual(self.training.summary()['left_step_focus_challenges'], 1)

    def test_stationary_side_push_uses_body_frame_and_leaves_moving_actor_unbiased(self):
        settings = {**self.training.settings, 'challenge_probability': 1., 'standing_side_fraction': 1.,
                    'standing_left_fraction': .99, 'standing_side_spread_deg': 0.}
        training = PushRecovery(self.env, settings, seed=42)
        training.enabled[:] = True
        self.env.poses[0, 0, 3:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        training.apply()
        force = self.env.robot.apply_forces_and_torques_at_position.call_args.args[0]
        torch.testing.assert_close(force[0, 1, :2] / (20 * 120 * .1), torch.tensor([0., 1.]), atol=1e-6, rtol=0.)
        self.assertEqual(training.summary()['standing_side_left_pushes'], 1)
        self.assertEqual(training.summary()['standing_side_right_pushes'], 0)
        torch.testing.assert_close(training.side_axis[1:], torch.zeros(2, 2))

    def test_shaped_force_applies_every_tick_and_reset_cancels_remaining_force(self):
        training = PushRecovery(self.env, {**self.training.settings, 'standing_side_fraction': 1.,
                                          'standing_shaped_fraction': 1.}, seed=42)
        training.waveform_id[:] = torch.arange(1, len(PROFILES))
        training.waveform_tick.zero_()
        training.waveform_force[:, 0] = 120.
        frames = []
        for _ in range(training.waveforms.shape[1] + 2):
            training.apply_substep()
            frames.append(self.env.robot.apply_forces_and_torques_at_position.call_args.args[0][:, 1, 0].clone())
        observed = torch.stack(frames, dim=1) / 120.
        torch.testing.assert_close(observed[:, :training.waveforms.shape[1]], profile_table('cpu')[1:])
        torch.testing.assert_close(observed.sum(dim=1), torch.ones(3))
        self.assertEqual(torch.count_nonzero(observed[:, -2:]), 0)
        training.waveform_tick.zero_()
        training.waveform_force[:, 0] = 120.
        training.reset(self.env.ids[1:2])
        training.apply_substep()
        force = self.env.robot.apply_forces_and_torques_at_position.call_args.args[0]
        self.assertEqual(torch.count_nonzero(force[1]), 0)
        self.assertGreater(float(force[0].abs().sum()), 0.)

    def test_recovery_cost_applies_only_inside_the_window(self):
        self.training.window = 2.5
        self.training.style_window = 1.2
        self.training.balance_weight = 4.
        self.training.fall_penalty = 4.
        self.training.remaining[:] = torch.tensor([0., .6, 2.5])
        self.env.logical_vel = self.env.commands + .5
        reward = self.training.adjust_reward(torch.ones(3), torch.tensor([True, True, False]))
        self.assertEqual(float(reward[0]), 1.)
        self.assertLess(float(reward[1]), -3.)
        self.assertGreater(float(reward[2]), 0.)
        torch.testing.assert_close(self.training.recovery_fraction(), torch.tensor([0., 0., 1.]))

    def test_corrective_grace_only_defers_reference_failure_for_a_strong_push(self):
        training = self.corrective_training()
        training.strong[:] = torch.tensor([False, True, True])
        training.remaining[:] = torch.tensor([1.2, 1.2, .3])
        failed = torch.ones(3, dtype=torch.bool)
        physical_fall = torch.tensor([False, True, False])
        self.assertEqual((training.reference_failure(failed, physical_fall) | physical_fall).tolist(),
                         [True, True, True])
        self.assertEqual(training.reference_failure(failed, ~failed).tolist(), [True, False, True])
        self.assertEqual(training.reference_failure(failed, ~failed).tolist(), [True, False, True])
        self.assertEqual(training.summary()['reference_grace_events'], 1)
        training.reset(self.env.ids[1:2])
        self.assertFalse(bool(training.strong[1]))

    def test_corrective_reward_requires_a_landing_toward_the_velocity_error(self):
        training = self.corrective_training()
        training.strong[:] = torch.tensor([False, True, True])
        training.remaining[:] = 1.2
        self.env.poses[:, 1, 3:] = torch.tensor([0., 0., 0., 1.])
        self.env.link_vel = torch.zeros((3, 2, 6))
        self.env.root_vel = torch.tensor([[1., 0., 0., 0., 0., 0.]] * 3)
        pelvis = torch.zeros((3, 3))
        feet = torch.tensor([[[.24, 0., 0.], [-.15, 0., 0.]]] * 3)
        feet[2] = feet[2].flip((0,))
        in_air = torch.tensor([[True, False]] * 3)
        fallen = torch.zeros(3, dtype=torch.bool)
        for _ in range(2):
            reward = training.corrective_step_reward(pelvis, feet, in_air * .06, ~in_air,
                                                     torch.zeros(3, 3), fallen)
            torch.testing.assert_close(reward, torch.zeros(3))
        reward = training.corrective_step_reward(pelvis, feet, torch.zeros(3, 2), torch.ones(3, 2),
                                                 torch.zeros(3, 3), fallen)
        torch.testing.assert_close(reward, torch.tensor([0., .38, 0.]))
        self.assertEqual(training.summary()['corrective_landings'], 1)

    def corrective_training(self):
        settings = {**self.training.settings, 'corrective_steps': dict(
            minimum_impulse_m_s=.24, reference_grace_seconds=.6, style_floor=.25,
            landing_bonus=.38, trunk_weight=1.6)}
        return PushRecovery(self.env, settings, seed=42)
