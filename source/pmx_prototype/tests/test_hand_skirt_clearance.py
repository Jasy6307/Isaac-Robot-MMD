import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.hand_skirt_clearance import HandSkirtClearance


class HandSkirtClearanceTests(unittest.TestCase):
    def setUp(self):
        self.reward = HandSkirtClearance(dict(
            weight=1.2, margin_m=.005, reference_allowance_m=.043,
            envelope_m=[[.2, .2], [.2, .2]], y_bounds_m=[-.5, .5], z_bounds_m=[-.5, .5],
            hand_points_m=[[[0, 0, 0], [0, 0, -.04]], [[0, 0, 0], [0, 0, -.04]]],
        ), 'cpu')
        self.pelvis = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
        self.hands = torch.tensor([[[.19, 0, 0, 0, 0, 0, 1.], [-.19, 0, 0, 0, 0, 0, 1.]]])

    def test_outward_motion_has_a_dead_zone_and_is_mirror_symmetric(self):
        command = torch.zeros(1, 3)
        before, gap, _, _ = self.reward(self.pelvis, self.hands, self.pelvis, self.hands, command)
        far = self.hands.clone()
        far[:, :, 0] += torch.tensor([.03, -.03])
        after, _, _, _ = self.reward(self.pelvis, far, self.pelvis, self.hands, command)
        self.assertAlmostEqual(float(gap[0, 0]), float(gap[0, 1]))
        self.assertLess(float(before), 0.)
        self.assertEqual(float(after), 0.)

    def test_reference_limits_the_target_even_inside_the_conservative_surface(self):
        reference = self.hands.clone()
        reference[:, :, 0] = torch.tensor([.07, -.07])
        current = reference.clone()
        current[:, :, 0] += torch.tensor([.044, -.044])
        rate, gap, _, _ = self.reward(self.pelvis, current, self.pelvis, reference, torch.zeros(1, 3))
        self.assertTrue((gap < 0).all())
        self.assertEqual(float(rate), 0.)

    def test_fingers_respond_to_wrist_rotation_and_running_has_zero_penalty(self):
        rotated = self.hands.clone()
        rotated[0, 0, 4] = 1.
        rotated[0, 0, 6] = 1.
        rotated[0, 0, 3:] /= 2 ** .5
        before = self.reward.surface_gap(self.pelvis, self.hands)
        rate, after, _, _ = self.reward(self.pelvis, rotated, self.pelvis, self.hands, torch.tensor([[1.2, 0., 0.]]))
        self.assertAlmostEqual(float(before[0, 0] - after[0, 0]), .04, places=6)
        self.assertEqual(float(rate), 0.)


if __name__ == '__main__':
    unittest.main()
