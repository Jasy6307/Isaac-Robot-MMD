"""Keep the lateral support reward tied to moving and loaded contact."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_lateral_support import SideSupportReward


class LateralSupportTests(unittest.TestCase):
    def test_requires_displacement_a_swing_and_two_loaded_frames(self):
        env = SimpleNamespace(num_envs=4, device='cpu', dt=1/30)
        reward = SideSupportReward(env, torch.full((4,), 50.), dict(reach_weight=1., load_weight=4.))
        start = torch.zeros((4, 2, 2))
        reward.start(torch.arange(4), start)
        active = torch.ones(4, dtype=torch.bool)
        axis = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
        moving = start.clone()
        moving[:, 0, 0] = torch.tensor([.17, .17, 0., .17])
        capture = torch.tensor([[.18, 0., .9], [.55, 0., .9], [.18, 0., .9], [.18, 0., .9]])
        speed = torch.zeros((4, 2))
        in_air = torch.tensor([[.06, 0.]] * 4)
        supported = torch.tensor([[0., 300.]] * 4)
        for age in (.08, .12):
            values = reward.reward(active, torch.full_like(active, age, dtype=torch.float32), axis,
                                   moving, in_air, supported, capture, speed)
            self.assertTrue((values[:2] > 0.).all())
            torch.testing.assert_close(values[2:], torch.zeros(2))
        load = torch.tensor([[250., 120.]] * 4)
        ground = torch.zeros_like(in_air)
        first = reward.reward(active, torch.full((4,), .19), axis, moving, ground, load, capture, speed)
        torch.testing.assert_close(first, torch.zeros_like(first))
        second = reward.reward(active, torch.full((4,), .23), axis, moving, ground, load, capture, speed)
        self.assertGreater(float(second[0]), float(second[1]))
        self.assertGreater(float(second[1]), 0.)
        torch.testing.assert_close(second[2:], torch.zeros(2))
        reward.reset(torch.tensor([0]))
        after_reset = reward.reward(active, torch.full((4,), .26), axis, moving, ground, load, capture, speed)
        self.assertEqual(float(after_reset[0]), 0.)

    def test_early_catch_credits_high_impulses_and_real_braking(self):
        env = SimpleNamespace(num_envs=3, device='cpu', dt=1/30)
        reward = SideSupportReward(env, torch.full((3,), 50.),
                                   dict(reach_weight=0., load_weight=0., early_capture_weight=3.))
        feet = torch.zeros((3, 2, 2))
        reward.start(torch.arange(3), feet, torch.tensor([.70, .70, .45]))
        feet[:, 0, 0] = .12
        axis = torch.tensor([[1., 0.]]*3)
        com = torch.tensor([[.15, 0., .90]]*3)
        velocity = torch.tensor([[.35, 0.]]*3)
        active = torch.ones(3, dtype=torch.bool)
        lifted = torch.tensor([[.04, 0.]]*3)
        one = torch.tensor([[0., 400.]]*3)
        reward.reward(active, torch.tensor([.12,.54,.12]), axis, feet, lifted, one, com, velocity)
        reward.reward(active, torch.tensor([.15,.57,.15]), axis, feet, lifted, one, com, velocity)
        landing = torch.tensor([[300.,140.]]*3)
        first = reward.reward(active, torch.tensor([.18,.60,.18]), axis, feet,
                              torch.zeros_like(lifted), landing, com, velocity)
        torch.testing.assert_close(first, torch.zeros(3))
        second = reward.reward(active, torch.tensor([.21,.63,.21]), axis, feet,
                               torch.zeros_like(lifted), landing, com, velocity*.4)
        self.assertGreater(float(second[0]), float(second[1]))
        self.assertGreater(float(second[1]), 0.)
        self.assertEqual(float(second[2]), 0.)
        self.assertEqual(float(reward.braking_support_frames), 2.)


if __name__ == '__main__':
    unittest.main()
