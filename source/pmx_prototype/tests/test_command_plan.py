"""Numerical invariants for the optional fixed-policy split planner."""
import sys
from pathlib import Path
import numpy as np
import torch
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from command_plan import command_increment


def legacy(commands,reference,dt):
    delta=(commands-reference)*(1-np.exp(-dt/.35))
    length=delta[:,:2].norm(dim=-1).clamp_min(1e-6)
    delta[:,:2]*=(.6*dt/length).clamp(max=1)[:,None]
    delta[:,2]=delta[:,2].clamp(-1.2*dt,1.2*dt)
    return delta


def test_split_invariants():
    torch.manual_seed(42)
    reference=torch.randn(2000,3)
    commands=torch.randn_like(reference)
    dt=1/30
    assert torch.equal(command_increment(commands,reference,dt,{}),legacy(commands,reference,dt))
    cfg=dict(command_plan_mode='split',command_speedup_limit=1.2,command_direction_change_limit=1.2,command_braking_limit=.6)
    stop=torch.zeros_like(commands)
    assert torch.equal(command_increment(stop,reference,dt,cfg),legacy(stop,reference,dt))
    slower=reference*.7
    assert torch.equal(command_increment(slower,reference,dt,cfg),legacy(slower,reference,dt))
    result=command_increment(commands,reference,dt,cfg)
    axis=reference[:,:2]/reference[:,:2].norm(dim=-1,keepdim=True)
    assert torch.all((result[:,:2]*axis).sum(-1)>=-.6*dt-1e-7)
    assert result[:,:2].norm(dim=-1).max()<=1.2*dt+1e-7
    assert torch.equal(result[:,2],legacy(commands,reference,dt)[:,2])
    reverse=torch.tensor([[-.9,0.,.6]])
    state=torch.tensor([[1.1,0.,-.6]])
    for _ in range(180):
        step=command_increment(reverse,state,dt,cfg)
        assert step.isfinite().all()
        assert step[0,0]<=0
        state+=step
    assert torch.allclose(state,reverse,atol=1e-5)


class CommandPlanTests(unittest.TestCase):
    def test_split_planner_invariants(self):
        test_split_invariants()


if __name__=='__main__':
    unittest.main()
