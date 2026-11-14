"""Continuous gait weights and genuine whole-bank phase behavior."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import unittest
from types import SimpleNamespace
import torch
from unified_walk_run_reference import components,sample,velocity_at_offset
from full_motion_reference import FullMotionReference

class UnifiedReferenceTests(unittest.TestCase):
    def test_rotating_body_origin_and_com_velocities_are_distinct(self):
        velocity=torch.tensor([[0.,0.,0.,0.,0.,2.]])
        rotation=torch.tensor([[0.,0.,2**-.5,2**-.5]])
        result=velocity_at_offset(velocity,rotation,torch.tensor([.1,0.,0.]))
        self.assertTrue(torch.allclose(result,torch.tensor([[-.2,0.,0.,0.,0.,2.]]),atol=1e-6))
        self.assertTrue(torch.equal(velocity_at_offset(velocity,rotation,torch.zeros(3)),velocity))
    def motion(self):
        commands=torch.zeros((15,3))
        commands[2,0]=.4;commands[4,0]=-.4;commands[5,1]=.35;commands[6,1]=-.35
        commands[9,2]=.6;commands[10,2]=-.6
        commands[11,0]=1.2;commands[12,0]=-1.2;commands[13,1]=1.2;commands[14,1]=-1.2
        return SimpleNamespace(commands=commands,gait_cycles=torch.tensor([1.]*11+[4.]*4),periods=torch.tensor([1.5]*11+[3.2]*4))

    def test_all_directions_and_pivot_use_correct_motion(self):
        cmd=torch.tensor([[0.,0.,0.],[.4,0,0],[1.2,0,0],[-1.2,0,0],[0,1.2,0],[0,-1.2,0],[0,0,.6],[0,0,-.6]])
        ids,weights,frequency=components(self.motion(),cmd,{})
        selected=ids.gather(1,weights.argmax(-1)[:,None]).flatten().tolist()
        self.assertEqual(selected,[3,2,11,12,13,14,9,10])
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(len(cmd))))
        self.assertTrue(torch.isfinite(frequency).all())

    def test_transition_is_continuous_and_monotonic(self):
        speed=torch.linspace(.4,1.2,1001);cmd=torch.zeros((len(speed),3));cmd[:,0]=speed
        _,weights,_=components(self.motion(),cmd,{})
        self.assertTrue(((weights[1:,2]-weights[:-1,2])>=-1e-6).all())
        self.assertLess(float((weights[1:]-weights[:-1]).abs().max()),.005)
        self.assertEqual(float(weights[0,2]),0.);self.assertEqual(float(weights[-1,2]),1.)

    def test_long_running_passage_does_not_restart_every_stride(self):
        m=FullMotionReference.__new__(FullMotionReference);base=self.motion()
        m.commands=base.commands;m.periods=base.periods;m.gait_cycles=base.gait_cycles
        m.gait_offsets=torch.zeros(15);m.count=128;m.bank=True
        q=torch.sin(torch.arange(128)*2*torch.pi/128)[None,:,None].repeat(15,1,1)
        rotations=torch.zeros((15,128,1,4));rotations[...,3]=1.
        m.data=dict(q=q,qd=torch.zeros_like(q),positions=torch.zeros((15,128,1,3)),rotations=rotations,
                    velocities=torch.zeros((15,128,1,6)),contacts=torch.zeros((15,128,2)))
        phase=torch.tensor([0.,.9999,1.0001])
        cmd=torch.tensor([[1.2,0,0]]*3)
        state=sample(m,phase,cmd,{})
        self.assertGreater(float((state['q'][1]-state['q'][0]).abs().max()),.9)
        self.assertLess(float((state['q'][1]-state['q'][2]).abs().max()),.001)
        self.assertTrue(torch.allclose(state['rotations'].norm(dim=-1),torch.ones((3,1))))
        m.data['q'].zero_();m.data['q'][11].fill_(1.)
        cmd=torch.tensor([[.7,0,0]]);rate=torch.tensor([[.1,0,0]]);phase=torch.zeros(1)
        corrected=sample(m,phase,cmd,{'unified_reference_derivatives':True},rate)
        numerical=(sample(m,phase,cmd+.001*rate,{})['q']-sample(m,phase,cmd-.001*rate,{})['q'])/.002
        self.assertTrue(torch.allclose(corrected['qd'],numerical,atol=1e-5))
        vectorized=sample(m,phase,cmd,{'unified_reference_derivatives':True,'unified_vectorized_reference':True},rate)
        for key in corrected:self.assertTrue(torch.allclose(corrected[key],vectorized[key],atol=1e-5))

if __name__=='__main__':unittest.main()
