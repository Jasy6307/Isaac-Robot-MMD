import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1];ROOT=SOURCE.parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from unified_walk_run_reference import sample_commands
from command_plan import command_increment
from locomotion_acceptance import score_segment,summarize

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

    def test_overallocated_pools_rejected(self):
        cfg=load_config();cfg['unified_low_speed_env_fraction']=.4
        e=env(cfg)
        with self.assertRaises(ValueError):sample_commands(e,torch.arange(e.num_envs),True)


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
        self.assertFalse(summary['passed']);self.assertTrue(summary['promising'])

    def test_variable_lag_is_bounded(self):
        t=np.arange(360)*self.dt
        raw=np.stack([np.cos(2*np.pi*t/12),np.sin(2*np.pi*t/12),np.zeros_like(t)],-1)
        planned=np.stack([np.cos(2*np.pi*(t-.35)/12),np.sin(2*np.pi*(t-.35)/12),np.zeros_like(t)],-1)
        for lag,expected in [(.55,True),(2.,False)]:
            actual=np.stack([np.cos(2*np.pi*(t-lag)/12),np.sin(2*np.pi*(t-lag)/12),np.zeros_like(t)],-1)
            result=score_segment(item('direction_sweep',12),actual,planned,raw,actual[:,:2],[1,0,0],self.cfg,self.dt)
            self.assertEqual(result['passed'],expected)


if __name__=='__main__':unittest.main()
