"""The slip reward must account for rolling feet and off-center COMs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import torch
import unittest
from stance_motion import point_velocity,displacement_slip_square,displacement_slip_speed


class PointVelocityTests(unittest.TestCase):
    def test_mean_speed_is_not_root_mean_square_speed(self):
        old=torch.zeros(1,1,2,3);new=old.clone()
        new[0,0,:,0]=torch.tensor([.01,.02]);near=torch.ones(1,1,2,dtype=torch.bool)
        self.assertTrue(torch.allclose(displacement_slip_speed(new,old,near,near,.1),torch.tensor([.15])))
        self.assertTrue(torch.allclose(displacement_slip_square(new,old,near,near,.1),torch.tensor([.025])))

    def test_displacement_uses_only_matching_grounded_corners(self):
        previous=torch.zeros(1,2,2,3)
        current=previous.clone();current[0,0,0,0]=.01;current[0,0,1,0]=.5
        old_near=torch.tensor([[[True,False],[False,False]]])
        near=torch.tensor([[[True,True],[False,False]]])
        self.assertTrue(torch.allclose(displacement_slip_square(current,previous,near,old_near,.1),torch.tensor([.01])))

    def test_instantaneous_rolling_point_is_stationary(self):
        linear=torch.tensor([[[1.,0.,0.]]])
        angular=torch.tensor([[[0.,0.,1.]]])
        sole=torch.tensor([[[[0.,1.,0.]]]])
        com=torch.zeros(1,1,3)
        self.assertTrue(torch.equal(point_velocity(linear,angular,sole,com),torch.zeros_like(sole)))

    def test_offset_is_measured_from_com_not_link_origin(self):
        linear=torch.zeros(1,1,3)
        angular=torch.tensor([[[0.,0.,2.]]])
        sole=torch.tensor([[[[1.,1.,0.],[2.,1.,0.]]]])
        com=torch.tensor([[[1.,1.,0.]]])
        expected=torch.tensor([[[[0.,0.,0.],[0.,2.,0.]]]])
        self.assertTrue(torch.equal(point_velocity(linear,angular,sole,com),expected))


if __name__=='__main__':unittest.main()
