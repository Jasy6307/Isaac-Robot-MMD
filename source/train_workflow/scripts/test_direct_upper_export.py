"""Recorder export contract: solve before any H5 write; retain legacy output."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.playback import recorder as module


class DirectExportTests(unittest.TestCase):
    def setUp(self):
        prepare=patch('source.train_workflow.utils.retarget.source_animation.prepare_source_animation',return_value=Path('source.usdc'))
        upper=patch('source.train_workflow.utils.retarget.upper_body_export.solve_upper_body_arrays',side_effect=lambda delta,*a:(delta,{'method':'test-upper'}))
        lower=patch('source.train_workflow.utils.retarget.lower_body_ik.solve_lower_body_arrays',side_effect=lambda delta,*a:(delta,{'method':'test-lower'}))
        for item in (prepare,upper,lower):item.start();self.addCleanup(item.stop)
    def recorder(self, folder):
        r=module.PlaybackH5Recorder.begin(source_csv=str(folder/'motion.csv'),
            runtime_joint_names=['left_elbow_joint'], compile_joint_names=['left_elbow_joint'],
            baseline_joint_pos=np.array([.5]),root_anchor_pos=(0,0,.76),
            root_anchor_quat_wxyz=[1,0,0,0],max_frame=1,has_hand_data=False)
        for i in range(2):
            r.record_frame(i,np.array([.5+i*.1]),root_pos=(0,0,.76+i*.01),root_quat_wxyz=[1,0,0,0])
        return r

    def test_solve_precedes_single_h5_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            folder=Path(d);r=self.recorder(folder)
            original=r.build_motion();calls=[]
            def solve(delta,names,baseline,fps,source,robot):
                self.assertFalse(list(folder.glob('*.h5')))
                np.testing.assert_array_equal(baseline,[.5])
                return delta+.25,{'method':'test'}
            real=module.write_hdf5_motion
            def write(path,motion):
                calls.append(path)
                np.testing.assert_array_equal(motion.root_pos_delta,original.root_pos_delta)
                np.testing.assert_allclose(motion.joint_pos_delta,original.joint_pos_delta+.25)
                return real(path,motion)
            with patch('source.train_workflow.utils.retarget.upper_body_export.solve_upper_body_arrays',side_effect=solve),patch.object(module,'write_hdf5_motion',side_effect=write):
                out=r.write(upper_body_source_animation='source.usdc')
            self.assertEqual(len(calls),1)
            self.assertEqual(len(list(folder.glob('*.h5'))),1)
            with h5py.File(out) as f:
                self.assertEqual(f.attrs['upper_body_export_mode'],'direct-in-memory')

    def test_failed_solver_creates_no_h5(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            folder=Path(d);r=self.recorder(folder)
            with patch('source.train_workflow.utils.retarget.upper_body_export.solve_upper_body_arrays',side_effect=ValueError('infeasible')):
                with self.assertRaisesRegex(ValueError,'infeasible'):
                    r.write(upper_body_source_animation='source.usdc')
            self.assertFalse(list(folder.glob('*.h5')))

    def test_legacy_export_does_not_invoke_solver(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            r=self.recorder(Path(d));expected=r.build_motion()
            with patch('source.train_workflow.utils.retarget.upper_body_export.solve_upper_body_arrays') as solve:
                out=r.write(retarget_mode='legacy')
                solve.assert_not_called()
            with h5py.File(out) as f:
                np.testing.assert_array_equal(f['joint_pos_delta'][:],expected.joint_pos_delta)
                self.assertNotIn('upper_body_export_mode',f.attrs)

    def test_lower_body_solve_before_first_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            folder=Path(d);r=self.recorder(folder)
            def solve(delta,names,baseline,fps,source,robot,pos,quat):
                self.assertFalse(list(folder.glob('*.h5')))
                np.testing.assert_allclose(pos[:,2],[.76,.77])
                return delta.copy(),{'method':'test-leg'}
            with patch('source.train_workflow.utils.retarget.lower_body_ik.solve_lower_body_arrays',side_effect=solve):
                out=r.write(lower_body_source_animation='source.usdc')
            self.assertEqual(len(list(folder.glob('*.h5'))),1)
            with h5py.File(out) as f:self.assertEqual(f.attrs['lower_body_export_mode'],'direct-in-memory')

    def test_lower_failure_does_not_publish_upper_only_h5(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            folder=Path(d);r=self.recorder(folder)
            with patch('source.train_workflow.utils.retarget.upper_body_export.solve_upper_body_arrays',side_effect=lambda delta,*a:(delta,{'method':'test'})),patch('source.train_workflow.utils.retarget.lower_body_ik.solve_lower_body_arrays',side_effect=ValueError('bad leg')):
                with self.assertRaisesRegex(ValueError,'bad leg'):
                    r.write(upper_body_source_animation='s.usdc',lower_body_source_animation='s.usdc')
            self.assertFalse(list(folder.glob('*.h5')))


if __name__=='__main__':unittest.main()
