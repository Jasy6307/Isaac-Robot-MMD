"""Check the live preview clock and discontinuities without starting Kit."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_interpolated_display import InterpolatedDisplay


class DisplayClockTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.draws = []
        self.display = InterpolatedDisplay(lambda *args: self.draws.append(args), lambda: self.now, self.advance)

    def advance(self, seconds):
        self.now += seconds

    def push(self, frame, x, heading=0., restart=False):
        self.display.push(frame, np.array([[x, 0., 0.]]), np.array([[0., 0., 0., 1.]]),
                          np.array([x, 0., 0.]), heading, restart)

    def test_display_draws_between_solved_poses_and_moves_camera_with_pose(self):
        self.push(0, 0.)
        self.display.present_if_due()
        self.advance(1 / 30)
        self.push(1, 2., math.pi / 2)
        self.display.wait_until(self.now + 1 / 30)
        pose, _, center, heading, frame = self.draws[-1]
        self.assertAlmostEqual(frame, .5)
        np.testing.assert_allclose(pose, [[1., 0., 0.]])
        np.testing.assert_allclose(center, [1., 0., 0.])
        self.assertAlmostEqual(heading, math.pi / 4)
        self.display.finish()
        self.assertEqual(self.draws[-1][-1], 1.)
        np.testing.assert_array_equal(self.draws[-1][0], [[2., 0., 0.]])

    def test_stall_holds_latest_pose_and_restart_snaps_without_catchup(self):
        self.push(0, 0.)
        self.display.present_if_due()
        self.push(1, 2.)
        self.advance(.2)
        self.display.present_if_due()
        self.display.present_if_due()
        self.assertEqual(len(self.draws), 2)
        self.assertEqual(self.draws[-1][-1], 1)
        self.push(2, 10., -math.pi / 2, restart=True)
        self.advance(1 / 60 + .001)
        self.display.present_if_due()
        pose, _, center, heading, frame = self.draws[-1]
        self.assertEqual(frame, 2)
        self.assertAlmostEqual(heading, -math.pi / 2)
        np.testing.assert_array_equal(pose, [[10., 0., 0.]])
        np.testing.assert_array_equal(center, [10., 0., 0.])

    def test_heading_interpolation_takes_the_short_arc_across_180(self):
        self.push(0, 0., math.radians(179))
        self.display.present_if_due()
        self.advance(1 / 30)
        self.push(1, 0., math.radians(-179))
        self.display.wait_until(self.now + 1 / 30)
        self.assertAlmostEqual(self.draws[-1][-2], math.pi)


if __name__ == '__main__':
    unittest.main()
