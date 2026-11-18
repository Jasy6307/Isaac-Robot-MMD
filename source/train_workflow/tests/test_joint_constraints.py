"""All-joint limits and USD mimic-derived passive-joint speeds."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.motion.sync import get_default_compile_joint_names
from source.train_workflow.utils.motion.joint_constraints import constrain_joint_positions,validate_joint_positions,read_constraints

class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.names=get_default_compile_joint_names();self.usd=ROOT/'assets/g1_29dof_o6_hand_V3.usd'
    def test_valid_noop(self):
        q=np.zeros((4,len(self.names)))
        out,r=constrain_joint_positions(q,self.names,30,self.usd)
        np.testing.assert_array_equal(out,q);self.assertTrue(r['after']['passed'])
        self.assertEqual(r['after']['joint_count'],51)
        self.assertEqual(r['after']['derived_mimic_speed_limits'],10)
    def test_coupled_rate_and_angles_are_satisfied(self):
        q=np.zeros((30,len(self.names)))
        q[:,self.names.index('lh_index_mcp_pitch')]=np.linspace(0,1,30)
        q[:,self.names.index('lh_index_dip')]=np.linspace(1.8,-1,30)
        q[:,self.names.index('rh_thumb_cmc_pitch')]=1.5
        q[:,self.names.index('rh_thumb_ip')]=3.
        out,r=constrain_joint_positions(q,self.names,30,self.usd)
        self.assertFalse(r['before']['passed']);self.assertTrue(r['after']['passed'])
        for name in self.names:
            if name.startswith(('left_','right_','waist_')):
                np.testing.assert_array_equal(out[:,self.names.index(name)],q[:,self.names.index(name)])
        self.assertTrue(validate_joint_positions(out.astype(np.float32),self.names,30,self.usd)['passed'])
    def test_follower_restricts_driver_range(self):
        joints,mapping,groups,bounds=read_constraints(self.usd,self.names)
        root,a,b=mapping['rh_thumb_ip'];lo,hi,_=bounds[root]
        self.assertLessEqual(max(a*lo+b,a*hi+b),joints['rh_thumb_ip'].upper+1e-8)
        self.assertGreaterEqual(min(a*lo+b,a*hi+b),joints['rh_thumb_ip'].lower-1e-8)

if __name__=='__main__':unittest.main()
