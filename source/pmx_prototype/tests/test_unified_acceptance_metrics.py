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
from evaluate_unified_walk_run import evaluate_low_speed,full_gate_segments,yaw_tracking_error,linear_tracking_error,suite
from diagnostics.evaluate_parallel_walk_run import plan, resolve_parallel_envs
from anon_scratch_validation import DEFAULT_ACCEPTANCE_INTERVAL, ScratchValidation
from locomotion_acceptance import (
    CURRENT_PROTOCOL, PROTOCOL, add_light_acceptance, quality_passed, score_segment,
    specification, step_body_status, summarize,
)


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
    def test_current_gate_accepts_grounded_running(self):
        config = dict(unified_acceptance_protocol=CURRENT_PROTOCOL, unified_eval_force_flight=True)
        row = dict(label='run_backward', falls=0, flight_fraction=0., force_flight_fraction=0.,
                   metrics={'Imitation/keypoint_rmse_m': .03, 'Imitation/near_hip_clearance_deficit_m': 0.})
        for label in ('run_forward', 'run_backward', 'run_left', 'run_right'):
            row['label'] = label
            self.assertTrue(quality_passed(row, config))
            self.assertFalse(quality_passed(row, dict(config, unified_acceptance_protocol=PROTOCOL)))
        self.assertFalse(specification(protocol=CURRENT_PROTOCOL)['running_flight_required'])
        self.assertEqual(step_body_status(True, 'passed', {'passed': True}), 'passed')
        self.assertEqual(step_body_status(True, 'failed', {'passed': True}), 'failed')
        row['falls'] = 1
        self.assertFalse(quality_passed(row, config))

    def test_mean_velocity_allows_gait_oscillation_but_rejects_tracking_bias(self):
        t=np.arange(600)/30
        error=np.stack([.5*np.sin(2*np.pi*2*t)+.12,np.zeros_like(t)],-1)
        self.assertLess(linear_tracking_error(error,True,1/30),.13)
        error[:,0]+=.25
        self.assertGreater(linear_tracking_error(error,True,1/30),.36)

    def test_light_acceptance_scores_the_same_motion_and_still_requires_zero_falls(self):
        dt = 1 / 30
        config = {'unified_acceptance_protocol': CURRENT_PROTOCOL}
        command = [-.4, 0., 0.]
        target = np.tile(command, (240, 1))
        actual = np.tile([-.30, 0., 0.], (240, 1))
        item = dict(label='walk_backward', seconds=8, command=command, falls=0, metrics={
            'Imitation/keypoint_rmse_m': .02, 'Imitation/near_hip_clearance_deficit_m': 0.,
        })
        strict = score_segment(item, actual, target, target, actual[:, :2], [0., 0., 0.], config, dt)
        item['response_steady'] = strict.copy()
        report = dict(acceptance_protocol=CURRENT_PROTOCOL, segments=[item], complete_scope=True,
                      total_falls=0, passed_unified_gate=False)
        add_light_acceptance(report, actual[:, None, :], dt)
        self.assertFalse(report['passed_unified_gate'])
        self.assertEqual(item['response_steady'], strict)
        self.assertTrue(report['passed_light_gate'])
        self.assertEqual(report['light_acceptance_summary']['passed_segments'], 1)
        self.assertAlmostEqual(item['light_acceptance']['steady']['linear_limit'], .1056)
        self.assertAlmostEqual(item['light_acceptance']['linear_response_deadline_seconds'],
                               strict['linear_response_deadline_seconds'] + .25)

        report['total_falls'] = 1
        add_light_acceptance(report, actual[:, None, :], dt)
        self.assertFalse(report['passed_light_gate'])

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

    def test_parallel_sequences_keep_all_cases_and_turn_stop_pairs(self):
        settings = dict(protocol=CURRENT_PROTOCOL, idle_head=True, backward_walk=True)
        cases = suite(**settings)
        previous = suite(**settings, stop_timing='compact')
        original = suite(**settings, stop_timing='original')
        groups = plan(cases, 1 / 30)
        self.assertEqual([row for group in groups for row in group.segments], cases)
        self.assertEqual([sum(row[1] for row in group.segments) for group in groups], [75.5, 84., 74.5, 87.5])
        self.assertEqual([group.warmup_steps for group in groups], [0, 120, 120, 120])
        self.assertEqual(cases[20:26], original[20:26])
        self.assertEqual(cases[:7], original[:7])
        self.assertEqual(previous[28:], original[28:])
        shortened = [(index + 1, row[1]) for index, row in enumerate(previous) if row != original[index]]
        self.assertEqual(shortened, [(index, 6.5) for index in (8, 10, 12, 14, 16, 18, 20, 28)])
        walk_stop = [(index + 1, row) for index, row in enumerate(cases) if row != previous[index]]
        self.assertEqual(walk_stop, [(34, ('stop_after_walk', 5.5, [0, 0, 0]))])
        self.assertEqual(sum(row[1] for row in previous), 323)
        self.assertEqual(sum(row[1] for row in original), 327)
        self.assertEqual([row[0] for row in groups[-1].segments[-2:]], ['pivot_right', 'stand_end'])
        recheck = plan(cases[-2:], 1 / 30, diagnostic=True)
        self.assertEqual([group.segments for group in recheck], [cases[-2:]])

    def test_preparation_stays_in_trajectory_and_does_not_count_as_a_case(self):
        cases = suite(protocol=CURRENT_PROTOCOL, idle_head=True, backward_walk=True)
        rows = [dict(label=label, seconds=seconds, response_steady=dict(status='passed'))
                for label, seconds, _ in cases]
        rows[1]['response_steady']['status'] = 'failed'
        rows[5]['response_steady']['status'] = 'failed'
        self.assertEqual(len(full_gate_segments(rows, CURRENT_PROTOCOL, {}, diagnostic=True)), 40)
        old_gate = full_gate_segments(rows, CURRENT_PROTOCOL, {'unified_eval_merge_preparation': False},
                                     diagnostic=False)
        self.assertEqual(summarize(old_gate)['passed_segments'], 38)
        self.assertFalse(summarize(old_gate)['passed'])

        gate = full_gate_segments(rows, CURRENT_PROTOCOL, {}, diagnostic=False)
        self.assertEqual([(row['label'], row['seconds']) for row in rows],
                         [(label, seconds) for label, seconds, _ in cases])
        self.assertEqual([(index, row['preparation_for_segment']) for index, row in enumerate(rows, 1)
                          if not row.get('counts_toward_gate', True)], [(2, 3), (6, 7)])
        self.assertEqual(len(gate), 38)
        self.assertEqual(summarize(gate)['passed_segments'], 38)
        self.assertTrue(summarize(gate)['passed'])
        rows[7]['response_steady']['status'] = 'failed'
        self.assertEqual(summarize(gate)['failed_segments'], 1)

    def test_parallel_opt_in_only_applies_to_matching_scratch_outputs(self):
        with TemporaryDirectory() as directory:
            acceptance = Path(directory)
            (acceptance / 'acceptance_plan.json').write_text('{"unified_eval_parallel_envs": 4}', encoding='utf-8')
            checkpoint = acceptance / 'scratch_002500.pt'
            run = acceptance / 'checkpoint_002500'
            self.assertEqual(resolve_parallel_envs(checkpoint, run / 'unified', run / 'full_config.json', {}), 4)
            self.assertEqual(resolve_parallel_envs(checkpoint, run / 'long_stand_recheck', run / 'long_stand_recheck_config.json', {}), 4)
            self.assertEqual(resolve_parallel_envs(checkpoint, run / 'unified', run / 'custom.json', {}), 1)
            self.assertEqual(resolve_parallel_envs(checkpoint, run / 'unified', run / 'full_config.json',
                                                   {'unified_eval_parallel_envs': 1}), 1)

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

    def test_resume_preserves_old_history_and_uses_current_full_cadence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / 'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json'
            baseline.parent.mkdir(parents=True)
            baseline.write_text('{"segments": []}', encoding='utf-8')
            acceptance = root / 'acceptance'
            acceptance.mkdir()
            old_history = [{'completed_iterations': 500, 'acceptance_kind': 'lite_screen',
                            'evaluation_seconds_cumulative': 21.0}]
            history_path = acceptance / 'history.json'
            history_path.write_text(json.dumps(old_history), encoding='utf-8')
            plan_path = acceptance / 'acceptance_plan.json'
            plan_path.write_text(json.dumps(dict(
                interval_iterations=500, acceptance_mode='staged', full_interval_iterations=2500,
                acceptance_protocol=CURRENT_PROTOCOL, command_yaw_limit_by_speed=True,
                unified_idle_head_acceptance=True, unified_eval_backward_walk=True,
                full_stand_hold_seconds=12, unified_eval_parallel_envs=4)), encoding='utf-8')
            validation = ScratchValidation(
                root / 'run', acceptance, DEFAULT_ACCEPTANCE_INTERVAL, root, continuation=True,
                target='latest_unified', protocol=CURRENT_PROTOCOL,
                speed_dependent_yaw=True, idle_head=True, backward_walk=True)
            self.assertEqual(json.loads(history_path.read_text(encoding='utf-8')), old_history)
            self.assertEqual(validation.validation_seconds, 21.)
            plan_settings = json.loads(plan_path.read_text(encoding='utf-8'))
            self.assertEqual(plan_settings['interval_iterations'], 1000)
            self.assertEqual(plan_settings['unified_eval_parallel_envs'], 4)
            self.assertEqual(plan_settings['full_stop_timing'], 'original')
            self.assertEqual([n for n in range(501, 2301) if validation.due(n, 2300)], [1000, 2000, 2300])

    def test_scratch_v3_does_not_require_low_speed_by_default(self):
        self.check_scratch_v3_scope(integrated_low_speed=False)

    def test_continuation_time_follows_the_selected_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'scratch_005500.pt'
            checkpoint.write_bytes(b'checkpoint')
            (root / 'training_time.json').write_text(json.dumps(dict(
                started_from_random_weights=True, completed_iterations=10000, training_seconds=45000.)),
                encoding='utf-8')
            (root / 'manifest.json').write_text(json.dumps(dict(run_dir=str(root))), encoding='utf-8')
            validation = ScratchValidation.__new__(ScratchValidation)
            validation.run = validation.output = root
            validation.history = [dict(checkpoint=str(checkpoint), completed_iterations=5500,
                                       active_training_seconds=24600.)]
            runner = SimpleNamespace(alg=SimpleNamespace(actor=torch.nn.Linear(2, 1),
                                                        optimizer=SimpleNamespace(state={1: {}})))
            validation.audit_continuation(runner, checkpoint)
            audit = json.loads((root / 'continuation_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['previous_completed_iterations'], 5500)
            self.assertEqual(audit['previous_active_training_seconds'], 24600.)

    def test_scratch_v3_can_reproduce_full_acceptance_with_low_speed(self):
        self.check_scratch_v3_scope(integrated_low_speed=True)

    def test_scratch_can_score_more_often_than_it_records(self):
        self.check_scratch_v3_scope(integrated_low_speed=False, video_interval=1000)

    def check_scratch_v3_scope(self, integrated_low_speed, video_interval=None):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            baseline=root/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json'
            baseline.parent.mkdir(parents=True)
            segments=[dict(label=label,command=command,stance_slip_geometry_m_s=.1,mean_speed=1.2)
                      for label,command in (('run_backward',[-1.2,0,0]),('run_left',[0,1.2,0]),
                                            ('run_right',[0,-1.2,0]))]
            baseline.write_text(json.dumps(dict(segments=segments)),encoding='utf-8')
            interval = 500 if video_interval else DEFAULT_ACCEPTANCE_INTERVAL
            validation=ScratchValidation(root/'run',root/'acceptance',interval,root,
                                         target='latest_unified',protocol=CURRENT_PROTOCOL,integrated_low_speed=integrated_low_speed,
                                         speed_dependent_yaw=True,idle_head=True,backward_walk=True,
                                         record_full_video=video_interval is not None,video_interval=video_interval)
            due = list(range(interval, 2300, interval)) + [2300]
            self.assertEqual([n for n in range(2301) if validation.due(n,2300)], due)
            settings=json.loads((validation.output/'acceptance_plan.json').read_text())
            self.assertEqual(settings['unified_eval_parallel_envs'],1 if integrated_low_speed else 4)
            self.assertEqual(settings.get('video_interval_iterations'),video_interval)

            def run(command,**_):
                config=Path(command[command.index('--config-override')+1])
                overrides=json.loads(config.read_text())
                self.assertEqual(overrides['unified_acceptance_protocol'],CURRENT_PROTOCOL)
                self.assertEqual(overrides['unified_eval_stand_hold_seconds'],12)
                self.assertEqual(overrides['unified_eval_stop_timing'],'compact_walk_stop')
                self.assertEqual(overrides['unified_eval_low_speed'],integrated_low_speed)
                self.assertEqual(overrides['unified_eval_backward_diagonal_run_speed'],.85)
                output=Path(command[command.index('--output')+1])
                if output.name == 'long_stand_recheck':
                    self.assertEqual(overrides['unified_eval_cases'],[
                        ['pivot_right',6,[0,0,-.6]],['stand_end',40,[0,0,0]]])
                    report=dict(segments=[dict(label='pivot_right',passed=True),dict(label='stand_end',passed=True)],
                                total_falls=0)
                else:
                    self.assertEqual(output.name,'unified')
                    self.assertIsNone(overrides['unified_eval_cases'])
                    revised=[dict(row,stance_slip_geometry_m_s=.05) for row in segments]
                    revised[0].update(command=[-.85,0,0],mean_speed=.82)
                    report=dict(segments=revised,backward_diagonal_run_speed_m_s=.85,
                                passed_unified_gate=True,total_falls=0,acceptance_protocol=CURRENT_PROTOCOL)
                    if integrated_low_speed:
                        report['low_speed']=dict(case_count=8,passed=True,falls=0)
                (output/'report.json').write_text(json.dumps(report),encoding='utf-8')
                return SimpleNamespace(returncode=0)

            runner = SimpleNamespace(save=lambda _: None)
            with (patch('anon_scratch_validation.subprocess.run',side_effect=run) as child,
                  patch('diagnostics.anon_acceptance_video.record_acceptance_video',return_value={}) as recorder,
                  patch('torch.cuda.empty_cache'),patch('builtins.print')):
                if video_interval:
                    self.assertTrue(validation.evaluate(runner,500,5.))
                    recorder.assert_not_called()
                    self.assertFalse(validation.history[0]['video_scheduled'])
                result=validation.evaluate(runner,1000,10.)
                if video_interval:
                    recorder.assert_called_once_with(validation.history[-1]['full_report'],title='Policy 1000 acceptance')
                    self.assertTrue(validation.history[-1]['video_scheduled'])
            self.assertEqual(child.call_count,4 if video_interval else 2)
            self.assertTrue(result)
            checks=validation.history[0]['slip_checks']
            self.assertAlmostEqual(checks['run_backward']['minimum_candidate_speed_m_s'],.95*.85)
            self.assertAlmostEqual(checks['run_left']['minimum_candidate_speed_m_s'],.95*1.2)
            if integrated_low_speed:
                self.assertEqual(validation.history[0]['low_speed']['case_count'],8)
            else:
                self.assertNotIn('low_speed',validation.history[0])
            self.assertTrue(validation.history[0]['long_stand_recheck']['passed'])

if __name__=='__main__':unittest.main()
