"""Numerical and skeleton-invariance checks for the general upper-body solver."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.retarget.upper_body_ik import (
    ArmTarget, RobotArm, SolverWeights, SourceSkeleton, read_joints,
)


class UpperBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = [RobotArm(read_joints(ROOT/'assets/g1_29dof_o6_hand_V3.usd'), side)
                    for side in ('left', 'right')]

    def test_jacobian_matches_central_differences(self):
        rng = np.random.default_rng(2718)
        for arm in self.arms:
            for _ in range(8):
                q = rng.uniform(arm.lower * .6, arm.upper * .6)
                state = arm.fk(q)
                target = ArmTarget(state['elbow'], state['wrist'], state['palm'])
                _, analytic = arm.residual_jacobian(q, target, q)
                numeric = np.zeros_like(analytic)
                for j in range(7):
                    d = np.eye(7)[j] * 1e-6
                    numeric[:,j] = (arm.residual_jacobian(q+d,target,q)[0]-arm.residual_jacobian(q-d,target,q)[0])/(2e-6)
                np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)

    def test_reachable_pose_and_joint_limits(self):
        rng = np.random.default_rng(31415)
        for arm in self.arms:
            for _ in range(10):
                q = rng.uniform(arm.lower * .5, arm.upper * .5)
                state = arm.fk(q)
                target = ArmTarget(state['elbow'], state['wrist'], state['palm'])
                solved = arm.solve(target, q + rng.normal(0,.08,7), seed=q,
                                   weights=SolverWeights(continuity=0))
                residual, _ = arm.residual_jacobian(solved,target,solved,SolverWeights(continuity=0))
                self.assertLess(np.linalg.norm(residual), 1e-4)
                self.assertTrue(np.all(solved >= arm.lower))
                self.assertTrue(np.all(solved <= arm.upper))

    def test_unreachable_target_stays_finite_and_bounded(self):
        for arm in self.arms:
            target = ArmTarget(np.array([3.,4.,5.]), np.array([8.,7.,9.]), np.eye(3))
            solved = arm.solve(target,np.zeros(7))
            self.assertTrue(np.isfinite(solved).all())
            self.assertTrue(np.all(solved >= arm.lower))
            self.assertTrue(np.all(solved <= arm.upper))

    def test_sequence_respects_authored_velocity_limits(self):
        for arm in self.arms:
            seeds = np.asarray([arm.lower*.8,arm.upper*.8,arm.lower*.7,arm.upper*.7])
            targets = [ArmTarget(s['elbow'],s['wrist'],s['palm']) for s in (arm.fk(q) for q in seeds)]
            solved = arm.solve_sequence(targets,seeds,30,passes=1)
            self.assertTrue(np.all(abs(np.diff(solved,axis=0))*30 <= arm.velocity+1e-6))

    def test_source_target_independent_of_world_heading_position_and_scale(self):
        # A synthetic source: this tests the coordinate convention independently
        # of any dance file or matching source/robot forward-kinematics code.
        source = SourceSkeleton.__new__(SourceSkeleton)
        source.chest = '上半身2'
        source.names = {'上半身2':0,'左腕':1,'左ひじ':2,'左手首':3}
        source._palm_rest = {'left':np.eye(3)}
        matrices = np.tile(np.eye(4),(4,1,1))
        matrices[1,:3,3]=[.2,0,.5]
        matrices[2,:3,3]=[.4,-.1,.4]
        matrices[3,:3,3]=[.3,-.2,.6]
        expected=source.target(matrices,'left',self.arms[0])
        global_pose=np.eye(4);global_pose[:3,:3]=Rotation.from_euler('xyz',[.4,-.2,1.3]).as_matrix();global_pose[:3,3]=[4.,-3.,2.]
        moved=matrices.copy();moved[:,:3,3]*=2.7
        actual=source.target(global_pose@ moved,'left',self.arms[0])
        for key in ('elbow','wrist','palm'):
            np.testing.assert_allclose(getattr(expected,key),getattr(actual,key),atol=1e-12)


if __name__ == '__main__':
    unittest.main()
