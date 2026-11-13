"""Pose-independent invariance and rate feasibility for gesture repair."""
from pathlib import Path
import sys,unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.motion.gesture_collision import relative_points,project_rates,temporal_derivatives,arm_temporal_terms,ArmMotionBudget

class GestureTests(unittest.TestCase):
    def test_derivatives_use_seconds_and_detect_route_reversal(self):
        for fps in (30,60):
            t=torch.arange(fps+1,dtype=torch.float64)/fps
            velocity,acceleration=temporal_derivatives((t*t)[:,None],fps)
            torch.testing.assert_close(acceleration,torch.full_like(acceleration,2.))
        q=torch.tensor([[0.],[.1],[0.]],requires_grad=True)
        hand=torch.zeros(3,2,3)
        penalty,_,ratios=arm_temporal_terms(q,hand,[0],30,torch.ones(2),torch.ones(3,2),ArmMotionBudget())
        self.assertGreater(float(ratios[2].max().detach()),1.)
        penalty.backward();self.assertTrue(torch.isfinite(q.grad).all())

    def test_near_head_budget_tightens_continuously(self):
        q=torch.zeros(4,1);hand=torch.zeros(4,2,3)
        hand[:,:,0]=torch.arange(4)[:,None]*.05
        def ratio(proximity):
            return arm_temporal_terms(q,hand,[0],30,torch.ones(2),torch.full((4,2),proximity),ArmMotionBudget())[2][1]
        torch.testing.assert_close(ratio(4.),ratio(1.)*2)

    def test_isolated_spike_is_not_diluted_by_clip_length(self):
        def penalty(count):
            q=torch.zeros(count,1);q[5]=.1
            return arm_temporal_terms(q,torch.zeros(count,2,3),[0],30,torch.ones(2),torch.ones(count,2),ArmMotionBudget())[0]
        # The peak component survives even after appending many quiet frames.
        self.assertGreater(float(penalty(2000)),.4)

    def test_landmark_error_is_invariant_to_world_heading_and_translation(self):
        rng=np.random.default_rng(123)
        rotations={n:torch.tensor(Rotation.random(5,random_state=rng).as_matrix()) for n in ('torso','head','left','right')}
        positions={n:torch.tensor(rng.normal(size=(5,3))) for n in rotations}
        offsets=torch.tensor(rng.normal(size=(2,3)))
        links=['left','right']
        global_r=torch.tensor(Rotation.from_euler('xyz',[.3,-.5,2.]).as_matrix())
        global_t=torch.tensor([3.,-2.,1.])
        changed_r={n:global_r@r for n,r in rotations.items()}
        changed_p={n:(global_r@p[...,None]).squeeze(-1)+global_t for n,p in positions.items()}
        for anchor in ('torso','head'):
            a=relative_points(rotations,positions,links,offsets,anchor)
            b=relative_points(changed_r,changed_p,links,offsets,anchor)
            torch.testing.assert_close(a,b)

    def test_rate_projection_keeps_nonadjustable_channels(self):
        rng=np.random.default_rng(42);q=rng.uniform(-1,1,(100,5));step=np.array([.1,.15])
        result=project_rates(q,[0,3],step,sweeps=1000)
        np.testing.assert_array_equal(result[:,[1,2,4]],q[:,[1,2,4]])
        self.assertTrue(np.all(abs(np.diff(result[:,[0,3]],axis=0))<=step+1e-7))
        self.assertTrue(np.all(result>=-1) and np.all(result<=1))

    def test_relative_landmarks_have_finite_gradients(self):
        position=torch.randn(3,3,requires_grad=True)
        rotations={n:torch.eye(3).repeat(3,1,1) for n in ('head','left','right')}
        positions={'head':torch.zeros(3,3),'left':position,'right':position*2}
        result=relative_points(rotations,positions,['left','right'],torch.zeros(2,3),'head')
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(position.grad).all())

if __name__=='__main__':unittest.main()
