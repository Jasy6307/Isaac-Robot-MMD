"""Coordinate, channel preservation and source-contact contracts without USD assets."""
from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.grounding.motion import read_motion, write_motion, continuous_quaternions
from source.train_workflow.utils.grounding.ik import pelvis_rotations, source_contact_report, stance_root_feedback
from source.train_workflow.utils.grounding.pipeline import require_same_reference, prepare
from source.train_workflow.utils.grounding import pipeline
from source.train_workflow.utils.grounding.smoothing import height_bounds, sample_brackets
from source.train_workflow.utils.grounding.arms import temporal_scores
from source.train_workflow.utils.grounding import dance_optimize
from source.train_workflow.scripts.compare_dance_auto import Run


def fixture(path, count=12):
    with h5py.File(path, 'x') as f:
        f.attrs['fps'] = 30.
        f.attrs['record_baseline_joint_pos'] = np.array([.1, .2])
        f.attrs['record_root_anchor_pos'] = np.array([0., 0., .8])
        f.attrs['record_root_anchor_quat_wxyz'] = np.array([1., 0., 0., 0.])
        f.attrs['without_collision_validated'] = True
        f.create_dataset('joint_names', data=['a', 'b'], dtype=h5py.string_dtype())
        f.create_dataset('joint_pos_delta', data=np.zeros((count, 2), np.float32))
        f.create_dataset('root_pos_delta', data=np.arange(count*3).reshape(count, 3)*.001)
        f.create_dataset('root_quat_delta_wxyz', data=np.tile([1., 0., 0., 0.], (count, 1)))
        d = f.create_group('aux').create_dataset('signal', data=np.arange(count))
        d.attrs['unit'] = 'frame'


class GroundingCoreTests(unittest.TestCase):
    def test_noncommuting_pelvis_coordinate_identity(self):
        source = Rotation.from_euler('xyz', [[20, 35, 90], [80, 5, -40]], degrees=True).as_matrix()
        c = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        a = Rotation.from_euler('xyz', [5, 20, 160], degrees=True).as_matrix()
        result = pelvis_rotations(source, c, a)
        np.testing.assert_allclose(a@result@c@source.transpose(0, 2, 1), np.tile(np.eye(3), (2, 1, 1)), atol=1e-12)

    def test_quaternion_sign_continuity_across_half_turn(self):
        rotations = Rotation.from_euler('z', np.array([[170.], [179.], [181.], [190.]]), degrees=True).as_matrix()
        q = continuous_quaternions(rotations, [1, 0, 0, 0])
        self.assertTrue(np.all((q[:-1]*q[1:]).sum(1) > 0))
        np.testing.assert_allclose(Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix(), rotations, atol=1e-12)

    def test_h5_copy_clips_aux_and_preserves_root_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.h5', Path(directory)/'output.h5'
            fixture(source)
            ref = read_motion(source, 9)
            write_motion(ref, output, ref.delta)
            candidate = read_motion(output)
            self.assertTrue(require_same_reference(ref, candidate))
            with h5py.File(source) as f, h5py.File(output) as g:
                for key in ('root_pos_delta', 'root_quat_delta_wxyz'):
                    np.testing.assert_array_equal(f[key][:9], g[key][:])
                np.testing.assert_array_equal(g['aux/signal'][:], np.arange(9))
                self.assertEqual(g['aux/signal'].attrs['unit'], 'frame')
                self.assertNotIn('without_collision_validated', g.attrs)
            with self.assertRaises((FileExistsError, OSError)):
                write_motion(ref, output, ref.delta)
            with h5py.File(output, 'r+') as f:
                f['root_pos_delta'][0, 2] += .01
            self.assertFalse(require_same_reference(ref, read_motion(output)))

    def test_contact_and_flight_cannot_be_excused(self):
        desired = np.array([[0., 0.], [.04, .05]])
        report = source_contact_report(np.array([[.03, .04], [0., .05]]), desired)
        self.assertEqual(report['false_floating_on_source_stance'], 1)
        self.assertEqual(report['source_flight_frames'], 1)
        self.assertEqual(report['source_flight_retained'], 0)
        lower, upper = height_bounds(desired)
        self.assertTrue(np.all(lower < upper))
        self.assertGreater(lower[1, 0], .02)

    def test_contact_at_an_integer_frame_does_not_depend_on_the_next_frame(self):
        support = np.array([False, True, False])
        coordinates, low, high = sample_brackets(len(support))
        np.testing.assert_array_equal(coordinates[support[low] & support[high]], [1.])

    def test_bounded_root_feedback_retains_flight_and_fixes_support(self):
        desired = np.zeros((20, 2)); desired[8:12] = .06
        heights = desired + .03
        offset = stance_root_feedback(heights, desired)
        report = source_contact_report(heights+offset[:, None], desired)
        self.assertEqual(report['false_floating_on_source_stance'], 0)
        self.assertEqual(report['source_flight_retained'], 4)
        self.assertTrue(np.all(heights+offset[:, None] >= -.001-1e-7))

    def test_unknown_character_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'run'
            with self.assertRaisesRegex(ValueError, 'Only the calibrated'):
                prepare('missing', 'missing', 'missing', out, source_leg_mode='fk', character='other')
            self.assertFalse(out.exists())

    def test_failed_encoded_acceptance_never_publishes_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, run = root/'source.h5', root/'run'
            fixture(source)
            with patch.object(pipeline, 'GroundProblem') as problem, \
                 patch.object(pipeline, 'audit_motion', return_value={'passed': False}):
                problem.return_value.smooth.return_value = (read_motion(source).q, [])
                with self.assertRaisesRegex(ValueError, 'no final H5'):
                    pipeline.smooth_motion(source, source, 'unused.usd', run)
            self.assertTrue((run/'candidate.h5').is_file())
            self.assertTrue((run/'validation.json').is_file())
            self.assertFalse((run/'final.h5').exists())

    def test_wrist_detector_preserves_source_accents_but_flags_an_added_spike(self):
        target = np.zeros((36, 2, 3))
        target[18:, 0, 0] = .25
        self.assertEqual(temporal_scores(target, target, target, target).max(), 0.)
        candidate = target.copy()
        candidate[8, 1, 1] += .13
        peak = temporal_scores(candidate, candidate, target, target)
        self.assertGreater(peak[8], 1.)
        self.assertEqual(peak[18], 0.)

    def test_arm_retry_switches_seed_and_rejects_exhausted_budget(self):
        for solvable in (True, False):
            with self.subTest(solvable=solvable), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root/'baseline.h5'
                fixture(source, count=36)
                with h5py.File(source, 'r+') as stream:
                    stream['joint_pos_delta'][15, 0] = .2
                motion = read_motion(source)
                problem = SimpleNamespace(reference=motion, robot_usd='unused.usd')
                run = Run(root/'run', accept_eula=False)
                attempts = run.directory/'attempts'
                fake = Mock(columns=np.array([0]), fps=30.)
                fake.scores.side_effect = lambda q: np.where(q[:, 0] > .2, 3., 0.)
                fake.solve.side_effect = lambda seed, *_args, **_kwargs: seed.copy()
                def seeds(current, lo, hi):
                    smoothed = current[lo:hi].copy()
                    smoothed[:, 0] = .1
                    return [(0., 2.4, .6, smoothed)] if solvable else []
                fake.warm_seeds.side_effect = seeds
                mesh = dict(unexpected=[])
                with patch.object(dance_optimize, 'ArmRefinement', return_value=fake), \
                     patch.object(dance_optimize, '_local_clearance', return_value=.01), \
                     patch.object(dance_optimize, '_hard_checks', return_value={'passed': True}), \
                     patch.object(dance_optimize, 'audit_mesh', return_value=mesh) as audit:
                    if solvable:
                        result, _, _, quality = dance_optimize._arm_motion(
                            run, attempts, problem, motion, None, None, None, 1)
                        self.assertEqual(quality['flagged_frames'], [])
                        self.assertTrue(require_same_reference(motion, result))
                        np.testing.assert_array_equal(result.delta[:, 1], motion.delta[:, 1])
                        self.assertEqual(audit.call_count, 1)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'failed acceptance'):
                            dance_optimize._arm_motion(run, attempts, problem, motion, None, None, None, 1)
                        self.assertFalse(audit.called)
                self.assertEqual(fake.warm_seeds.call_count, 1)
                self.assertTrue((run.reports/'arms_01.json').is_file())
                self.assertFalse((run.directory/'deliverables').exists())


if __name__ == '__main__':
    unittest.main()
