"""Constraint chain rules and nonlinear acceptance for gesture refinement."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.motion.constrained_trajectory import difference_matrix,local_rows,soc_rows,refine_feasible_trajectory,refine_feasible_windows
from scipy import sparse


class ConstrainedTrajectoryTests(unittest.TestCase):
    def test_sparse_derivatives_match_numpy_in_time_order(self):
        q=np.random.default_rng(5).normal(size=(9,3))
        for order in (1,2):
            np.testing.assert_allclose((difference_matrix(9,3,order)@q.ravel()).reshape(9-order,3),np.diff(q,n=order,axis=0))
        self.assertEqual(difference_matrix(2,3,2).shape,(0,6))

    def test_midpoint_jacobian_splits_between_adjacent_frames(self):
        count=4;blocks=np.tile(np.array([[[2.,3.]]]),(2*count-1,1,1))
        matrix=local_rows(blocks,np.array([0,5]),np.array([0,0]),count,2)
        x=np.arange(8).reshape(4,2)
        np.testing.assert_allclose(matrix@x.ravel(),[x[0]@[2,3],((x[1]+x[2])/2)@[2,3]])

    def test_quarter_sample_jacobians_use_correct_adjacent_frame_weights(self):
        count=4;blocks=np.tile(np.array([[[2.,3.]]]),(4*count-3,1,1));x=np.arange(8).reshape(4,2)
        rows=np.array([count+1,count+(count-1)+1,count+2*(count-1)+1])
        matrix=local_rows(blocks,rows,np.zeros(3,dtype=int),count,2,(.25,.5,.75))
        np.testing.assert_allclose(matrix@x.ravel(),[((1-a)*x[1]+a*x[2])@[2,3] for a in (.25,.5,.75)])

    def test_soc_rows_preserve_all_three_motion_components(self):
        x=np.array([[2.,3.,4.],[-1.,2.,1.]])
        a,b=soc_rows(sparse.eye(6),x,np.array([8.,8.]))
        direction=np.random.default_rng(3).normal(size=6)
        slack=(b-a@direction).reshape(2,4)
        np.testing.assert_allclose(slack[:,0],[8,8])
        np.testing.assert_allclose(slack[:,1:],x+direction.reshape(2,3))

    def test_improves_pose_without_crossing_curved_collision_or_changing_other_channels(self):
        count=9;seed=np.zeros((count,2),dtype=np.float32);seed[:,1]=np.arange(count)
        target=np.linspace(.12,.4,count)
        def sample(q):
            residual=(q[:,0]-target)[:,None]
            points=np.zeros((count,2,3))
            samples=np.r_[q[:,0],(q[:-1,0]+q[1:,0])/2]
            return residual,points,(.05-samples**2)[:,None]
        def feasible(q):
            return (sample(q)[2].min()>=.0005 and abs(np.diff(q[:,0])).max()<=.025+1e-5
                    and abs(np.diff(q[:,0],n=2)).max()<=.008+1e-5)
        result,report=refine_feasible_trajectory(seed,[0],[-1.],[1.],[.025],[.008],30,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,feasible,iterations=20)
        self.assertTrue(feasible(result));np.testing.assert_array_equal(result[:,1],seed[:,1])
        self.assertLess(report['final_objective'],report['initial_objective']*.5)
        self.assertGreater(report['accepted_steps'],0)

    def test_invalid_seed_cannot_enter_refinement(self):
        with self.assertRaisesRegex(ValueError,'feasible seed'):
            refine_feasible_trajectory(np.zeros((3,1)),[0],[-1],[1],[.1],[.1],30,
                np.ones((2,2)),np.ones((1,2)),lambda q:None,lambda q:False,iterations=1)

    def test_new_subframe_collision_is_restored_even_if_pose_error_must_increase(self):
        count=9;seed=np.full((count,1),.224,dtype=np.float32)
        def sample(q):
            midpoint=(q[:-1,0]+q[1:,0])/2
            return q-.23,np.zeros((count,2,3)),np.r_[np.ones(count),.05-midpoint**2][:,None]
        def original_gate(q):
            return abs(np.diff(q[:,0])).max()<.025 and abs(np.diff(q[:,0],n=2)).max()<.03
        result,report=refine_feasible_trajectory(seed,[0],[-1],[1],[.025],[.03],30,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,original_gate,iterations=8,restore_collisions=True)
        self.assertTrue(original_gate(result));self.assertGreaterEqual(float(sample(result)[2].min()),.0005-1e-7)
        self.assertGreater(report['final_objective'],report['initial_objective'])
        self.assertGreater(report['accepted_steps'],0)

    def test_restoration_can_take_multiple_steps_when_full_clearance_exceeds_trust_region(self):
        count=9;seed=np.full((count,1),.6,dtype=np.float32)
        def sample(q):
            middle=(q[:-1,0]+q[1:,0])/2
            return q-.7,np.zeros((count,2,3)),np.r_[np.ones(count),.05-middle**2][:,None]
        result,report=refine_feasible_trajectory(seed,[0],[-1],[1],[.025],[.03],30,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,lambda q:abs(np.diff(q[:,0])).max()<.025,
            iterations=12,restore_collisions=True)
        self.assertGreaterEqual(float(sample(result)[2].min()),.0005-1e-7)
        self.assertGreater(report['accepted_steps'],1)

    def test_frozen_valid_boundary_does_not_require_extra_optimization_reserve(self):
        count=10;seed=np.full((count,1),.222,dtype=np.float32);seed[0]=.19705
        def sample(q):
            samples=np.r_[q[:,0],(q[:-1,0]+q[1:,0])/2]
            return q-.15,np.zeros((count,2,3)),(.05-samples**2)[:,None]
        def feasible(q):
            return sample(q)[2].min()>=.0005 and abs(np.diff(q[:,0])).max()<=.025+1e-6 and abs(np.diff(q[:,0],n=2)).max()<=.03+1e-6
        result,report=refine_feasible_trajectory(seed,[0],[-1],[1],[.025],[.03],30,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,feasible,iterations=10,fixed_endpoints=(2,2))
        np.testing.assert_array_equal(result[[0,1,-2,-1]],seed[[0,1,-2,-1]])
        self.assertTrue(feasible(result));self.assertGreater(report['accepted_steps'],0)

    def test_overlapping_windows_validate_all_splice_derivatives(self):
        count=30;seed=np.zeros((count,2),dtype=np.float32);seed[:,1]=4.
        target=np.linspace(.1,.25,count)
        def sample(q,selector):
            n=len(q)
            return (q[:,0]-target[selector])[:,None],np.zeros((n,2,3)),np.ones((2*n-1,1))
        checked=[]
        def feasible(q):
            checked.append(True)
            return (abs(np.diff(q[:,0])).max()<.1+1e-6 and abs(np.diff(q[:,0],n=2)).max()<.05+1e-6)
        result,report=refine_feasible_windows(seed,[0],[-1],[1],[.1],[.05],5,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,feasible,iterations=30)
        self.assertTrue(feasible(result));self.assertGreater(len(checked),len(report['windows']))
        np.testing.assert_array_equal(result[:,1],seed[:,1])
        self.assertLess(report['final_objective'],report['initial_objective']*.5)

    def test_repair_only_keeps_safe_motion_outside_detected_window(self):
        count=40;seed=np.zeros((count,1),dtype=np.float32);seed[16:18]=.6
        def sample(q,selector):
            n=len(q);middle=(q[:-1,0]+q[1:,0])/2
            return q-.2,np.zeros((n,2,3)),(.1-np.r_[q[:,0],middle]**2)[:,None]
        result,report=refine_feasible_windows(seed,[0],[-1],[1],[1.],[1.],5,
            np.ones((count-1,2)),np.ones((count-2,2)),sample,lambda q:True,iterations=20,repair_only=True)
        np.testing.assert_array_equal(result[:8],seed[:8]);np.testing.assert_array_equal(result[-8:],seed[-8:])
        self.assertGreaterEqual(float(sample(result,slice(None))[2].min()),.0005-1e-7)


if __name__=='__main__':unittest.main()
