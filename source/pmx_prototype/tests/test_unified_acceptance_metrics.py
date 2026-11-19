"""Steering measurements distinguish commanded turns from natural torso sway."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluate_unified_walk_run import evaluate_low_speed,yaw_tracking_error,linear_tracking_error,suite
from anon_scratch_validation import ScratchValidation
from locomotion_acceptance import CURRENT_PROTOCOL


class LowSpeedEvaluationEnv:
    dt=1.
    num_envs=1
    ids=torch.arange(1)

    def __init__(self):
        self.cfg={'unified_substep_support_forces':True}
        self.states=[]

    def reset(self,ids,*,heading,phase):
        self.states.append((phase,heading))
        self.step_number=0

    def set_command(self,command):
        self.command=torch.tensor([command])

    def get_observations(self):
        return self.command

    def step(self,action):
        assert self.cfg['runtime_inference_only'] and not self.cfg['unified_substep_support_forces']
        self.logical_vel=action
        self.step_number+=1
        timeout=self.step_number==1
        fall=len(self.states)==8 and self.step_number==15
        return None,None,torch.tensor([timeout or fall]),{'time_outs':torch.tensor([timeout])}

class AcceptanceMetricTests(unittest.TestCase):
    def test_mean_velocity_allows_gait_oscillation_but_rejects_tracking_bias(self):
        t=np.arange(600)/30
        error=np.stack([.5*np.sin(2*np.pi*2*t)+.12,np.zeros_like(t)],-1)
        self.assertLess(linear_tracking_error(error,True,1/30),.13)
        error[:,0]+=.25
        self.assertGreater(linear_tracking_error(error,True,1/30),.36)

    def test_gait_sway_does_not_become_commanded_turn_error(self):
        errors=np.sin(np.arange(300)*2*np.pi/30)*.7
        self.assertLess(yaw_tracking_error(errors,np.zeros(300)),1e-6)
        self.assertGreater(yaw_tracking_error(errors+.4,np.full(300,.6)),.39)

    def test_changing_turn_targets_cannot_cancel_opposite_errors(self):
        targets=np.sin(np.arange(300)*2*np.pi/300)*.6
        self.assertGreater(yaw_tracking_error(-targets,targets),.3)

    def test_full_suite_includes_abrupt_forward_run_stop(self):
        cases=suite();self.assertEqual(sum(s for _,s,_ in cases),325)
        self.assertTrue(any(cases[i][0]=='run_forward' and cases[i+1][0]=='stop' for i in range(len(cases)-1)))

    def test_integrated_low_speed_samples_eight_conditions_and_gates_falls(self):
        env=LowSpeedEvaluationEnv()
        with patch('builtins.print'):
            report=evaluate_low_speed(env,lambda obs:obs)
        self.assertEqual(report['case_count'],8)
        self.assertEqual(report['simulated_seconds'],120)
        for speed,start in [(.15,0),(.4,4)]:
            cases=report['cases'][start:start+4]
            self.assertEqual({c['speed'] for c in cases},{speed})
            self.assertEqual({c['initial_phase'] for c in cases},{0.,.25,.5,.75})
            self.assertEqual({c['initial_heading_deg'] for c in cases},{-180.,-90.,0.,90.})
            self.assertLess(max(c['steady_tracking_error_m_s'] for c in cases),1e-6)
        self.assertEqual(report['falls'],1)
        self.assertEqual(report['cases'][-1]['falls_by_stage'],dict(stand=0,walk=0,stop=1))
        self.assertFalse(report['passed'])
        self.assertEqual(env.cfg,{'unified_substep_support_forces':True})

    def test_scratch_v3_uses_one_process_for_full_and_low_speed_acceptance(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            baseline=root/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json'
            baseline.parent.mkdir(parents=True)
            segments=[dict(label=label,stance_slip_geometry_m_s=.1,mean_speed=1.2)
                      for label in ('run_backward','run_left','run_right')]
            baseline.write_text(json.dumps(dict(segments=segments)),encoding='utf-8')
            validation=ScratchValidation(root/'run',root/'acceptance',500,root,target='latest_unified',
                                         protocol=CURRENT_PROTOCOL,low_speed_after_full=True)

            def run(command,**_):
                config=Path(command[command.index('--config-override')+1])
                self.assertEqual(json.loads(config.read_text())['unified_acceptance_protocol'],CURRENT_PROTOCOL)
                output=Path(command[command.index('--output')+1])
                report=dict(segments=[dict(row,stance_slip_geometry_m_s=.05) for row in segments],
                            passed_unified_gate=True,total_falls=0,acceptance_protocol=CURRENT_PROTOCOL,
                            low_speed=dict(case_count=8,passed=True,falls=0))
                (output/'report.json').write_text(json.dumps(report),encoding='utf-8')
                return SimpleNamespace(returncode=0)

            with patch('anon_scratch_validation.subprocess.run',side_effect=run) as child,patch('builtins.print'):
                result=validation.evaluate_latest_unified(SimpleNamespace(save=lambda _:None),1000,10.,force_full=True)
            child.assert_called_once()
            self.assertTrue(result)
            self.assertEqual(validation.history[0]['low_speed']['case_count'],8)

if __name__=='__main__':unittest.main()
