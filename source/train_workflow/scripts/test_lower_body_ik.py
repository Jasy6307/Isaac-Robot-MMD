"""Lower-limb FK derivatives, feasibility and kinematic contact checks."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg,LegTarget,contact_mask
from source.train_workflow.utils.retarget.upper_body_ik import SolverWeights

class LegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legs=[RobotLeg(ROOT/'assets/g1_29dof_o6_hand_V3.usd',s) for s in ('left','right')]
    def test_jacobian_including_ground_penalty(self):
        for leg in self.legs:
            for height in (.4,1.2):
                q=np.clip(np.array([-.3,.1,.2,.7,-.3,.05]),leg.lower+.01,leg.upper-.01)
                s=leg.fk(q);t=LegTarget(s['elbow'],s['wrist'],s['palm'],np.eye(3),np.array([0,0,height]),True)
                _,analytic=leg.residual_jacobian(q,t,q)
                numeric=np.column_stack([(leg.residual_jacobian(q+np.eye(6)[i]*1e-6,t,q)[0]-leg.residual_jacobian(q-np.eye(6)[i]*1e-6,t,q)[0])/2e-6 for i in range(6)])
                np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-6)
    def test_reachable_pose(self):
        for leg in self.legs:
            q=np.clip(np.array([-.3,0,.1,.7,-.3,0]),leg.lower+.01,leg.upper-.01)
            s=leg.fk(q);t=LegTarget(s['elbow'],s['wrist'],s['palm'],np.eye(3),np.array([0,0,1.2]))
            solved=leg.solve(t,q+.03,seed=q,weights=SolverWeights(continuity=0))
            self.assertLess(np.linalg.norm(leg.fk(solved)['wrist']-t.wrist),1e-5)
            seq=leg.solve_sequence([t,t],[q,q],30,passes=1)
            self.assertTrue(np.all(abs(np.diff(seq,axis=0))*30<=leg.velocity+1e-6))
    def test_contact_excludes_sliding_and_airborne_feet(self):
        static=np.zeros((30,3));sliding=static.copy();sliding[:,0]=np.arange(30)/30
        self.assertTrue(contact_mask(np.zeros(30),static,30,1).all())
        self.assertFalse(contact_mask(np.zeros(30),sliding,30,1).any())
        self.assertFalse(contact_mask(np.ones(30),static,30,1).any())
    def test_contact_discards_short_touches(self):
        heights=np.ones(30);heights[10]=0
        self.assertFalse(contact_mask(heights,np.zeros((30,3)),30,1).any())
if __name__=='__main__':unittest.main()
