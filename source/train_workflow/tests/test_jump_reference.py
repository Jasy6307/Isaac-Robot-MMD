"""Verify experimental reference shaping stays local and leg-only."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion.jump_reference import shape_takeoff_reference


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"{side}_{joint}_joint" for side in ("left", "right")
                      for joint in ("knee", "hip_pitch", "ankle_pitch", "hip_roll")]
        self.names += ["waist_yaw_joint", "left_shoulder_pitch_joint", "lh_thumb_joint"]
        torch.manual_seed(42)
        self.reference = torch.randn(600, len(self.names)) * .1

    def test_scope_source_and_upper_body_preserved(self):
        original = self.reference.clone()
        shaped = shape_takeoff_reference(self.reference, self.names, 30.)
        torch.testing.assert_close(self.reference, original, rtol=0, atol=0)
        torch.testing.assert_close(shaped[:, -3:], original[:, -3:], rtol=0, atol=0)
        t = torch.arange(600) / 30.
        outside = (t <= 16.86) | (t >= 17.22)
        torch.testing.assert_close(shaped[outside], original[outside], rtol=0, atol=0)
        self.assertLess(shaped[510, 0], original[510, 0])

    def test_joint_limits_respected_in_shaped_interval(self):
        reference = torch.zeros_like(self.reference)
        limits = torch.tensor([[-.2, .2]] * len(self.names))
        shaped = shape_takeoff_reference(reference, self.names, 30., limits)
        self.assertGreaterEqual(float(shaped.min()), -.200001)
        self.assertLessEqual(float(shaped.max()), .200001)

    def test_three_events_leave_ordinary_motion_untouched(self):
        reference = torch.zeros(2500, len(self.names))
        events = ((16.86,17.,17.08,17.22,16.9),(20.53,20.67,20.75,20.89,20.57),
                  (47.96,48.10,48.18,48.32,48.))
        shaped = shape_takeoff_reference(reference, self.names, 30., events=events)
        for peak in (17.,20.67,48.1):
            self.assertLess(float(shaped[round(peak*30),0]), -.4)
        torch.testing.assert_close(shaped[round(53*30)], reference[round(53*30)], rtol=0, atol=0)
        torch.testing.assert_close(shaped[:, -3:], reference[:, -3:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
