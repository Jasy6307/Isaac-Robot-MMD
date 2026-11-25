import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1];ROOT=SOURCE.parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from unified_walk_run_reference import sample_commands
from command_plan import command_increment
from locomotion_acceptance import CURRENT_PROTOCOL,score_segment,specification,summarize

FIXTURES=Path(__file__).parent/'fixtures'


def load_config():
    config=json.loads((FIXTURES/'low_speed_sampling.json').read_text())
    config['full_motion_path']=str(FIXTURES/'gait_clock.npz')
    return config


def env(cfg,n=10000):
    return SimpleNamespace(cfg=cfg,num_envs=n,device='cpu',evaluation=False,iteration=5674,
        commands=torch.zeros(n,3),continuous_ref_commands=torch.zeros(n,3),command_age=torch.zeros(n))


def reference_trace(initial,target,seconds,config):
    dt=1/30;current=torch.tensor([initial],dtype=torch.float32);target=torch.tensor([target],dtype=torch.float32)
    values=[]
    for _ in range(round(seconds/dt)):
        current+=command_increment(target,current,dt,config);values.append(current[0].numpy().copy())
    return np.array(values)


def item(label,seconds,falls=0):
    return dict(label=label,seconds=seconds,falls=falls,metrics={'Imitation/keypoint_rmse_m':.02,
        'Imitation/near_hip_clearance_deficit_m':0.,'Imitation/pelvis_orientation_rmse_rad':.05},
        displacement_m=0.,pelvis_yaw_drift_rad=0.,flight_fraction=.1,force_flight_fraction=.1)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_legacy_seeded_sampling_unchanged(self):
        fixtures=Path(__file__).parent/'fixtures'
        cfg=json.loads((fixtures/'legacy_sampling.json').read_text())
        current=env(cfg);ids=torch.arange(current.num_envs)
        with np.load(fixtures/'legacy_sampling.npz') as golden:
            for step,reset in enumerate([True,False,False]):
                torch.manual_seed(123);sample_commands(current,ids,reset)
                for key in ('commands','continuous_ref_commands','unified_command_hold'):
                    np.testing.assert_array_equal(getattr(current,key).numpy(),golden[f'{step}_{key}'])

    def test_low_pools_cover_and_alternate(self):
        cfg=load_config();e=env(cfg);ids=torch.arange(e.num_envs);bucket=ids%100
        sustained=(bucket>=65)&(bucket<80);stop=(bucket>=80)&(bucket<90)
        torch.manual_seed(42);sample_commands(e,ids,True)
        speed=e.commands[:,:2].norm(dim=-1)
        self.assertTrue(((speed[sustained]>=.15-1e-6)&(speed[sustained]<=.4+1e-6)).all())
        self.assertTrue(((e.unified_command_hold[sustained]>=12)&(e.unified_command_hold[sustained]<=16)).all())
        self.assertTrue(torch.equal(e.continuous_ref_commands,e.commands))
        reference_before=e.continuous_ref_commands.clone()
        before=speed[stop]<.001
        sample_commands(e,ids,False)
        self.assertTrue(torch.equal(e.commands[stop,:2].norm(dim=-1)<.001,~before))
        self.assertTrue(torch.equal(e.continuous_ref_commands,reference_before))
        fraction=((e.commands[:,:2].norm(dim=-1)>=.14999)&(e.commands[:,:2].norm(dim=-1)<=.40001)).float().mean()
        self.assertGreater(float(fraction),.18)
        self.assertLess(float(fraction),.24)
        # Dedicated pivots remain pure and transition commands retain their bank.
        pivot=(bucket>=45)&(bucket<60)
        self.assertTrue((e.commands[pivot,:2]==0).all())
        self.assertTrue((e.commands[(bucket>=60)&(bucket<65),:2].norm(dim=-1)>=.9).all())

    def test_specialized_pools_not_overwritten(self):
        cfg=load_config();a,b=env(cfg),env(dict(cfg,unified_low_speed_env_fraction=0.,unified_low_stop_env_fraction=0.))
        ids=torch.arange(a.num_envs);protected=(ids%100<65)|(ids%100>=90)
        torch.manual_seed(42);sample_commands(a,ids,True)
        torch.manual_seed(42);sample_commands(b,ids,True)
        self.assertTrue(torch.equal(a.commands[protected],b.commands[protected]))

    def test_backward_starts_replace_only_selected_low_speed_stop_lanes(self):
        cfg = load_config()
        control = env(cfg, n=8192)
        treatment = env(dict(cfg, unified_backward_start_env_fraction=.03), n=8192)
        ids = torch.arange(control.num_envs)
        selected = (ids % 100 >= 80) & (ids % 100 < 83)
        self.assertEqual(int(selected.sum()), 246)
        for step, reset in enumerate((True, False, False, False)):
            torch.manual_seed(42 + step)
            sample_commands(control, ids, reset)
            torch.manual_seed(42 + step)
            sample_commands(treatment, ids, reset)
            for field in ('commands', 'continuous_ref_commands', 'unified_command_hold'):
                torch.testing.assert_close(getattr(control, field)[~selected],
                                           getattr(treatment, field)[~selected], rtol=0, atol=0)
            command = treatment.commands[selected]
            hold = treatment.unified_command_hold[selected]
            if step % 2 == 0:
                self.assertTrue((command == 0).all())
                self.assertTrue(((hold >= 3.) & (hold < 4.)).all())
            else:
                self.assertTrue((command[:, 1:] == 0).all())
                self.assertTrue(((command[:, 0] >= -.45) & (command[:, 0] <= -.30)).all())
                self.assertGreater(int((command[:, 0] == -.4).sum()), 75)
                self.assertTrue(((hold >= 6.) & (hold < 8.)).all())

    def test_overallocated_pools_rejected(self):
        cfg=load_config();cfg['unified_low_speed_env_fraction']=.4
        e=env(cfg)
        with self.assertRaises(ValueError):sample_commands(e,torch.arange(e.num_envs),True)

    def test_training_reset_applies_yaw_envelope_after_sampling(self):
        cfg=load_config();cfg['command_yaw_limit_by_speed']=True
        e=env(cfg,n=8192)
        torch.manual_seed(42);sample_commands(e,torch.arange(e.num_envs),True)
        speed=e.commands[:,:2].norm(dim=-1)
        slow=speed<=.55;fast=speed>=.9-1e-6
        self.assertTrue((e.commands[fast,2].abs()>.4).any())
        self.assertTrue((e.continuous_ref_commands[fast,2].abs()<=.4+1e-6).all())
        self.assertTrue(torch.equal(e.continuous_ref_commands[slow],e.commands[slow]))

    def test_backward_and_diagonal_sampling_respects_planar_cap(self):
        cfg=load_config()
        speed_cap=.859
        original=env(cfg,n=8192)
        limited=env(dict(cfg,unified_backward_diagonal_speed_cap=speed_cap),n=8192)
        ids=torch.arange(limited.num_envs)
        for reset in (True,False):
            with self.subTest(reset=reset):
                torch.manual_seed(42);sample_commands(original,ids,reset)
                torch.manual_seed(42);sample_commands(limited,ids,reset)
                old_xy=original.commands[:,:2]
                new_xy=limited.commands[:,:2]
                forward,lateral=old_xy.unbind(-1)
                axis_forward=(forward>1e-4)&(lateral.abs()<=1e-4)
                axis_side=(forward.abs()<=1e-4)&(lateral.abs()>1e-4)
                backwards=forward < -1e-4
                diagonals=(forward.abs()>1e-4)&(lateral.abs()>1e-4)
                speed=old_xy.norm(dim=-1)
                capped=(backwards|diagonals)&(speed>speed_cap)
                self.assertTrue(capped.any())
                angle=torch.atan2(lateral,forward).abs().rad2deg()
                self.assertTrue((capped&(angle>25.)&(angle<35.)).any())
                torch.testing.assert_close(new_xy[capped],old_xy[capped]*(speed_cap/speed[capped])[:,None])
                self.assertTrue(torch.equal(new_xy[~capped],old_xy[~capped]))
                self.assertGreater(float(new_xy[axis_forward].norm(dim=-1).max()),1.17)
                self.assertAlmostEqual(float(new_xy[axis_side].norm(dim=-1).max()),1.2,places=6)
                self.assertTrue(torch.equal(limited.commands[:,2],original.commands[:,2]))
                self.assertTrue(torch.equal(limited.unified_command_hold,original.unified_command_hold))
                if reset:
                    self.assertTrue(torch.equal(limited.continuous_ref_commands[:,:2],new_xy))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):self.cfg=load_config();self.dt=1/30

    def score(self,label,initial,target,seconds,actual=None,falls=0):
        planned=reference_trace(initial,target,seconds,self.cfg)
        requested=np.broadcast_to(target,planned.shape).copy()
        if actual is None:actual=planned.copy()
        return score_segment(item(label,seconds,falls),actual,planned,requested,actual[:,:2],initial,self.cfg,self.dt)

    def test_transition_can_pass_without_raw_prefix_gate(self):
        plan=reference_trace([0,1.2,0],[0,-1.2,0],8,self.cfg)
        actual=np.empty_like(plan);previous=np.array([0,1.2,0.])
        for k,row in enumerate(plan):previous=previous+(row-previous)*(1-np.exp(-self.dt/.2));actual[k]=previous
        self.assertGreater(np.linalg.norm((actual[60:150,:2]-[0,-1.2]).mean(0)),.264)
        result=self.score('run_right',[0,1.2,0],[0,-1.2,0],8,actual)
        self.assertEqual(result['status'],'passed')
        self.assertLess(result['steady']['velocity_error'],.03)

    def test_real_slow_walk_remains_failed(self):
        actual=np.zeros((240,3));actual[:,0]=.2
        result=self.score('walk',[0,0,0],[.3,0,0],8,actual)
        self.assertEqual(result['status'],'failed')
        self.assertGreater(result['steady']['velocity_error'],.09)

    def test_unchanged_hold_has_no_command_response_gate(self):
        actual=np.zeros((240,3));actual[:60,2]=.3
        result=self.score('stand',[0,0,0],[0,0,0],8,actual)
        self.assertEqual(result['response_status'],'not_applicable')
        self.assertTrue(result['passed'])

    def test_long_response_is_not_hidden_by_later_steady(self):
        actual=np.zeros((300,3));actual[90:,0]=.3
        result=self.score('walk',[0,0,0],[.3,0,0],10,actual)
        self.assertEqual(result['response_status'],'failed')
        self.assertTrue(result['steady']['passed'])
        self.assertFalse(result['passed'])

    def test_fall_and_clearance_never_excused(self):
        self.assertFalse(self.score('walk',[0,0,0],[.3,0,0],8,falls=1)['passed'])
        plan=reference_trace([0,0,0],[.3,0,0],8,self.cfg);bad=item('walk',8)
        bad['metrics']['Imitation/near_hip_clearance_deficit_m']=.02
        result=score_segment(bad,plan,plan,np.broadcast_to([.3,0,0],plan.shape),plan[:,:2],[0,0,0],self.cfg,self.dt)
        self.assertFalse(result['passed'])

    def test_short_window_is_inconclusive_not_accepted(self):
        result=self.score('run_right',[0,1.2,0],[0,-1.2,0],5)
        self.assertEqual(result['status'],'insufficient_duration')
        summary=summarize([{'response_steady':result}])
        self.assertFalse(summary['passed']);self.assertEqual(summary['inconclusive_segments'],1)

    def test_variable_lag_is_bounded(self):
        t=np.arange(360)*self.dt
        raw=np.stack([np.cos(2*np.pi*t/12),np.sin(2*np.pi*t/12),np.zeros_like(t)],-1)
        planned=np.stack([np.cos(2*np.pi*(t-.35)/12),np.sin(2*np.pi*(t-.35)/12),np.zeros_like(t)],-1)
        for lag,expected in [(.55,True),(2.,False)]:
            actual=np.stack([np.cos(2*np.pi*(t-lag)/12),np.sin(2*np.pi*(t-lag)/12),np.zeros_like(t)],-1)
            result=score_segment(item('direction_sweep',12),actual,planned,raw,actual[:,:2],[1,0,0],self.cfg,self.dt)
            self.assertEqual(result['passed'],expected)

    def test_turn_assessment_uses_reachable_steady_target(self):
        self.cfg['command_yaw_limit_by_speed']=True
        result=self.score('run_turn_right',[0,0,-.6],[0,-1.1,-.6],8)
        self.assertTrue(result['passed'])
        self.assertLess(result['steady']['yaw_error'],.001)
        self.assertEqual(result['steady']['yaw_limit'],.2)

    def test_fixed_yaw_confirmation_budget_keeps_linear_deadline_and_steady_start(self):
        self.cfg['unified_acceptance_protocol']=CURRENT_PROTOCOL
        target=[1.1,0.,-.4]
        actual=np.broadcast_to(target,(240,3))
        cases=((.1,.98,2.08333,2.,'passed'),
               (2.,2.45,2.1,3.5,'passed'),
               (2.,2.45,2.11667,3.5,'failed'))
        self.assertEqual(specification()['yaw_response_deadline_seconds'],2.1)
        for planning,linear,yaw,linear_deadline,expected in cases:
            with self.subTest(yaw=yaw), patch('locomotion_acceptance.gait_period',return_value=1.), \
                    patch('locomotion_acceptance.planning_time',return_value=planning), \
                    patch('locomotion_acceptance._response',side_effect=(linear,yaw)):
                result=self.score('run_turn_right',[0.,0.,0.],target,8,actual)
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['yaw_response_deadline_seconds'],2.1)
                self.assertEqual(result['response_deadline_seconds'],linear_deadline)
                self.assertEqual(result['steady']['start_seconds'],linear_deadline)
                self.assertTrue(result['steady']['passed'])

    def test_current_stop_yaw_confirmation_budget(self):
        self.cfg['unified_acceptance_protocol']=CURRENT_PROTOCOL
        spec=specification(protocol=CURRENT_PROTOCOL)
        self.assertEqual(spec['stop_yaw_response_deadline_seconds'],3.0)
        actual=np.zeros((240,3))
        cases=(('stop',3.0,3.0,'passed'),
               ('stop',3.0,3.03333,'failed'),
               ('stop_after_walk',3.0,2.9,'passed'),
               ('stop_after_pivot_left',3.0,2.9,'passed'),
               ('stand_end',3.0,2.9,'passed'),
               ('stand',2.1,2.9,'failed'))
        with patch('locomotion_acceptance.planning_time',return_value=1.0):
            for label,budget,yaw,expected in cases:
                with self.subTest(label=label,yaw=yaw), patch('locomotion_acceptance._response',side_effect=(1.5,yaw)):
                    result=self.score(label,[0.,0.,.6],[0.,0.,0.],8,actual)
                    self.assertEqual(result['yaw_response_deadline_seconds'],budget)
                    self.assertEqual(result['status'],expected)

    def test_missing_yaw_confirmation_uses_its_own_observation_budget(self):
        with patch('locomotion_acceptance.gait_period',return_value=1.), \
                patch('locomotion_acceptance.planning_time',return_value=2.):
            for seconds,expected in ((3.,'insufficient_duration'),(3.2,'failed')):
                with self.subTest(seconds=seconds), patch('locomotion_acceptance._response',side_effect=(2.,None)):
                    result=self.score('stop',[.3,0.,.3],[0.,0.,0.],seconds)
                    self.assertEqual(result['response_status'],expected)

    def test_backward_walk_linear_tolerance_is_four_control_frames(self):
        self.cfg['unified_acceptance_protocol']=CURRENT_PROTOCOL
        self.assertEqual(specification(protocol=CURRENT_PROTOCOL)['walk_backward_linear_response_tolerance_frames'],4)
        actual=np.broadcast_to([-.4,0.,0.],(240,3))
        cases=(('walk_backward',3.+4*self.dt,3.+4*self.dt,'passed'),
               ('walk_backward',3.+5*self.dt,3.+4*self.dt,'failed'),
               ('walk',3.+self.dt,3.,'failed'))
        for label,response,budget,expected in cases:
            with self.subTest(label=label,response=response), patch('locomotion_acceptance.gait_period',return_value=1.), \
                    patch('locomotion_acceptance.planning_time',return_value=1.5), \
                    patch('locomotion_acceptance._response',side_effect=(response,1.5)):
                result=self.score(label,[0.,0.,0.],[-.4,0.,0.],8,actual)
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['response_deadline_seconds'],3.)
                self.assertEqual(result.get('linear_response_deadline_seconds',result['response_deadline_seconds']),budget)
                self.assertEqual(result['steady']['start_seconds'],3.)
                self.assertEqual(result['yaw_response_deadline_seconds'],2.1)


if __name__=='__main__':unittest.main()
