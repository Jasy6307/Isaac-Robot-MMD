"""Check inspector launch defaults and independent display/solve measurements."""

from collections import deque
import math
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import source_layout
from anon_inspector_ui import PolicyInspector


class InspectorPreviewTests(unittest.TestCase):
    def launch_args(self, *options):
        entry = SOURCE / 'inspect_anon_policy.py'
        run_wrapper = runpy.run_path
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            checkpoint.touch()
            arguments = []
            with (
                patch.object(sys, 'argv', [str(entry), '--checkpoint', str(checkpoint), *options]),
                patch.object(sys, 'path', sys.path.copy()),
                patch('runpy.run_path', side_effect=lambda *a, **k: arguments.extend(sys.argv)) as player,
            ):
                run_wrapper(str(entry), run_name='__main__')
                player.assert_called_once()
            return arguments

    def test_default_launch_and_explicit_override(self):
        defaults = self.launch_args()
        self.assertIn('--secondary-worker', defaults)
        for option, value in (('--physics-detail', 'sparse'), ('--frontend-task-threads', '4'), ('--render-fps', '60')):
            self.assertEqual(defaults[defaults.index(option) + 1], value)
        explicit = self.launch_args('--physics-detail', 'full', '--frontend-task-threads', '8', '--render-fps', '30')
        for option, value in (('--physics-detail', 'full'), ('--frontend-task-threads', '8'), ('--render-fps', '30')):
            self.assertEqual(explicit[explicit.index(option) + 1], value)

    def test_modes_without_interpolation_keep_30_hz(self):
        for options in (('--inline-secondary',), ('--no-secondary-physics',), ('--record-video', 'test.mp4')):
            with self.subTest(options=options):
                arguments = self.launch_args(*options)
                self.assertEqual(arguments[arguments.index('--render-fps') + 1], '30')
                if options[0] != '--record-video':
                    self.assertNotIn('--secondary-worker', arguments)

    def test_render_activity_does_not_mask_a_stalled_solver(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.solved_times = deque([1.2, 1.3, 1.4])
        inspector.render_times = deque([1.2, 1.3, 1.4, 2.0, 2.1])
        inspector.render_frames = 5
        inspector.last_rate_update = 0.
        inspector.render_target = 60
        inspector.solve_rate_label = SimpleNamespace(text='')
        inspector.render_rate_label = SimpleNamespace(text='')
        with patch('anon_inspector_ui.time.monotonic', return_value=2.5):
            inspector.record_render()
        self.assertEqual(inspector.solve_rate_label.text, 'Solved 0 / 30 Hz')
        self.assertEqual(inspector.render_rate_label.text, 'Render 3 / 60 FPS')
        self.assertEqual(inspector.render_frames, 6)

    def test_front_back_and_side_rotate_with_the_character(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        for view, expected_offset in ((1, (0, 3.6, .2)), (2, (-4.6, 0, .7)), (3, (4.6, 0, .7))):
            with self.subTest(view=view):
                inspector.controls = SimpleNamespace(view=view)
                np.testing.assert_allclose(inspector.camera_offset(math.pi / 2), expected_offset, atol=1e-12)

    def test_angle_keeps_its_original_world_offset_and_target(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.controls = SimpleNamespace(view=0)
        inspector.reset_camera()
        inspector.camera_target(np.zeros(3), 0., 0.)
        position = np.array([1., 2., .88])
        center, offset = inspector.camera_target(position, math.pi / 2, 1 / 60)
        np.testing.assert_array_equal(center, position)
        self.assertEqual(offset, (2.3, -4, 1.15))

    def test_tracking_damps_small_pose_changes_and_snaps_on_reset(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.controls = SimpleNamespace(view=2)
        inspector.reset_camera()
        inspector.camera_target(np.zeros(3), 0., 0.)
        position = np.array([1., 0., 0.])
        center, offset = inspector.camera_target(position, math.pi / 2, 1 / 60)
        self.assertLess(center[0], .25)
        self.assertGreater(offset[1], 4.)
        center, offset = inspector.camera_target(position, math.pi / 2, 3.)
        np.testing.assert_allclose(center, position, atol=.001)
        np.testing.assert_allclose(offset, (-4.6, 0, .7), atol=.002)
        inspector.reset_camera()
        center, offset = inspector.camera_target(np.zeros(3), 0., 3.01)
        np.testing.assert_array_equal(center, np.zeros(3))
        self.assertEqual(offset, (0., 4.6, .7))


if __name__ == '__main__':
    unittest.main()
