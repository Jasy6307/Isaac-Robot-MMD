"""Control-rate history, reset safety and reference-only jump protection."""
import importlib.util
from pathlib import Path
import unittest
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("turn_smoothing_test",ROOT/"source/my_task/mdp/turn_smoothing.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TurnTests(unittest.TestCase):
    def test_capture_uses_action_reference_and_cost_requires_expected_actual_support(self):
        names=["left_ankle_pitch_joint","right_ankle_pitch_joint"]
        q=torch.tensor([[1.,2.]])
        buf=NS(num_steps=100,root_quat_wxyz=lambda steps:torch.tensor([[1.,0.,0.,0.]]).repeat(len(steps),1),
               q_ref_abs=lambda steps:torch.zeros(len(steps),2))
        env=NS(num_envs=1,device="cpu",step_dt=1/30.,episode_length_buf=torch.tensor([0]),
            cfg=NS(jump_phase_schedules=(),actions=NS(joint_pos=NS(motion_h5_path="unused",motion_window_seconds=3.)),
                   auto_jump_report=dict(fps=30.,support_feet=[[1,0]]*100)))
        term=NS(_residual_joint_ids=torch.tensor([0,1]),_residual_action_cols=torch.tensor([0,1]),
                _asset=NS(joint_names=names),processed_actions=q+torch.tensor([[.1,.2]]))
        refs=NS(get_or_create_motion_buffer=lambda *a:buf,motion_steps=lambda e:e.episode_length_buf)
        with patch.dict(sys.modules,{"source.my_task.motion_reference":refs}):
            module.capture_residual(env,term,q)
        state=env._turn_smoothing_state
        self.assertTrue(torch.allclose(state.previous,torch.tensor([[.1,.2]])))
        state.rate=torch.tensor([[2.,20.]]);state.reference_steps=torch.tensor([0])
        jump=NS(full_step_ground_contacts=lambda e:torch.tensor([[True,True]]))
        with patch.dict(sys.modules,{"source.my_task.mdp.jump":jump}):
            self.assertEqual(float(module.residual_rate(env)),2.)
            jump.full_step_ground_contacts=lambda e:torch.tensor([[False,True]])
            self.assertEqual(float(module.residual_rate(env)),0.)

    def test_static_heading_has_no_turn_and_smooth_turn_has_gate(self):
        n=180;dt=1/30.
        yaw=np.zeros(n);yaw[30:90]=np.linspace(0,2,60);yaw[90:]=2
        q=np.stack((np.cos(yaw/2),np.zeros(n),np.zeros(n),np.sin(yaw/2)),axis=-1)
        gate=module.reference_turn_gate(q,np.zeros((n,2)),dt)
        self.assertEqual(float(gate[:20].max()),0.)
        self.assertGreater(float(gate[45:75].mean()),.6)
        self.assertEqual(float(gate[110:].max()),0.)

    def test_jump_margin_overrides_turn_evidence(self):
        dt=1/30.;yaw=np.arange(180)*dt
        q=np.stack((np.cos(yaw/2),yaw*0,yaw*0,np.sin(yaw/2)),axis=-1)
        schedule=(((2.,2.2),(2.2,2.4),(2.4,2.6),(2.6,2.8)),)
        gate=module.reference_turn_gate(q,np.zeros((180,2)),dt,schedule)
        t=np.arange(180)*dt
        self.assertTrue(np.all(gate[(t>=1.7)&(t<=3.1)]==0))
        self.assertGreater(gate[30],0)

    def test_rate_and_acceleration_only_after_valid_history(self):
        history=module.ResidualHistory((1,2),"cpu");dt=1/30.
        for step,values in enumerate(([0.,0.],[.15,.15],[.30,.30])):
            history.update(torch.tensor([values]),torch.tensor([step]),torch.tensor([step]),torch.ones(1),dt)
            if step==0:self.assertEqual(float(history.rate.max()),0.)
            if step==1:
                self.assertAlmostEqual(float(history.rate.mean()),1.)
                self.assertEqual(float(history.acceleration.max()),0.)
        self.assertEqual(float(history.acceleration.max()),0.)
        history.update(torch.tensor([[2.,-2.]]),torch.tensor([70]),torch.tensor([0]),torch.ones(1),dt)
        self.assertEqual(float(history.rate.max()),0.);self.assertEqual(float(history.acceleration.max()),0.)

    def test_zero_gate_and_jump_exit_cannot_penalize_launch_or_landing_changes(self):
        history=module.ResidualHistory((1,1),"cpu")
        for step,gate in enumerate((1.,1.,0.,0.,1.,1.,1.)):
            history.update(torch.tensor([[float(step%2)]]),torch.tensor([step]),torch.tensor([step]),torch.tensor([gate]),1/30.)
            if step in (2,3,4):self.assertEqual(float(history.rate.max()),0.)
            if step in (2,3,4,5):self.assertEqual(float(history.acceleration.max()),0.)
        self.assertGreater(float(history.acceleration.max()),0.)

    def test_heading_wrap_does_not_create_spurious_reverse_turn(self):
        yaw=np.linspace(2.8,3.5,150);q=np.stack((np.cos(yaw/2),yaw*0,yaw*0,np.sin(yaw/2)),axis=-1)
        gate=module.reference_turn_gate(q,np.zeros((150,2)),1/30.)
        self.assertEqual(float(gate.max()),0.)


if __name__=="__main__":unittest.main()
