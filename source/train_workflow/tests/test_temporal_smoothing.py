"""Temporal refinement reduces a one-frame jerk spike without losing collision feasibility."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.motion.constrained_trajectory import refine_feasible_trajectory
class TemporalSmoothingTest(unittest.TestCase):
 def test_strong_quadratic_restoration_clears_nonfeasible_collision_seed(self):
  count=9;seed=np.full((count,1),.6,dtype=np.float32)
  def sample(q):
   middle=(q[:-1,0]+q[1:,0])/2
   return q-.7,np.zeros((count,2,3)),np.r_[np.ones(count),.05-middle**2][:,None]
  result,report=refine_feasible_trajectory(seed,[0],[-1],[1],[.025],[.03],30,
      None,None,sample,lambda q:abs(np.diff(q[:,0])).max()<.025,
      iterations=16,restore_collisions=True,restoration_quadratic_weight=2e8)
  self.assertGreaterEqual(float(sample(result)[2].min()),.0005-1e-7)
  self.assertGreater(report['accepted_steps'],1)
 def test_spike_is_smoothed_and_frozen_channels_and_edges_are_preserved(self):
  n=13;seed=np.zeros((n,2),dtype=np.float32);seed[:,0]=.1;seed[6,0]=.2;seed[:,1]=4.
  def sample(q):
   values=np.r_[q[:,0],(q[:-1,0]+q[1:,0])/2]
   return q[:,:1]-seed[:,:1],np.zeros((n,2,3)),(.05-values**2)[:,None]
  def feasible(q):return sample(q)[2].min()>=.0005 and abs(np.diff(q[:,0])).max()<.5 and abs(np.diff(q[:,0],n=2)).max()<1.
  result,report=refine_feasible_trajectory(seed,[0],[0.],[.22],[.5],[1.],30,None,None,sample,feasible,iterations=12,fixed_endpoints=(2,2),temporal_weights=(0.,.001,.00001))
  self.assertTrue(feasible(result));self.assertGreater(report['accepted_steps'],0)
  np.testing.assert_array_equal(result[:,1],seed[:,1]);np.testing.assert_array_equal(result[[0,1,-2,-1]],seed[[0,1,-2,-1]])
  self.assertLess(np.linalg.norm(np.diff(result[:,0],n=3)),.25*np.linalg.norm(np.diff(seed[:,0],n=3)))
  self.assertLess(np.sqrt(np.mean((result[:,0]-seed[:,0])**2)),.035)
if __name__=='__main__':unittest.main()
