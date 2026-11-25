"""Fresh-initialization evidence and fixed, externally evaluated training gates."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_ACCEPTANCE_INTERVAL = 1000


def write_json(path, value):
    path=Path(path)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def tensor_hash(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def averages(reports):
    import numpy as np
    result={}
    keys=['Imitation/joint_rmse_rad','Imitation/keypoint_rmse_m',
          'Imitation/hand_orientation_error_rad','Imitation/arm_spacing_deficit_m',
          'Imitation/contact_accuracy']
    items=[item for report in reports for item in report['segments']]
    for key in keys:result[key]=float(np.mean([item['metrics'][key] for item in items]))
    return result


def compare_directional_slip(report, baselines):
    """Preserve the speed-retention fraction relative to each command's target."""
    current = {item['label']: item for item in report['segments']}
    checks = {}
    for label in ('run_backward', 'run_left', 'run_right'):
        baseline, candidate = baselines[label], current[label]
        baseline_target = math.hypot(*(baseline.get('requested_command') or baseline['command'])[:2])
        candidate_target = math.hypot(*(candidate.get('requested_command') or candidate['command'])[:2])
        if baseline_target <= 0 or candidate_target <= 0:
            raise ValueError(f'{label} needs nonzero target speeds for slip acceptance')
        old_slip = float(baseline['stance_slip_geometry_m_s'])
        new_slip = float(candidate['stance_slip_geometry_m_s'])
        old_speed = float(baseline['mean_speed'])
        new_speed = float(candidate['mean_speed'])
        reduction = 1. - new_slip / max(old_slip, 1e-8)
        minimum_speed = .95 * old_speed * (candidate_target / baseline_target)
        retained = new_speed >= minimum_speed
        checks[label] = dict(
            baseline_slip_m_s=old_slip, candidate_slip_m_s=new_slip, reduction_fraction=reduction,
            baseline_speed_m_s=old_speed, candidate_speed_m_s=new_speed,
            baseline_target_m_s=baseline_target, candidate_target_m_s=candidate_target,
            minimum_candidate_speed_m_s=minimum_speed, retained_speed=retained,
            passed=reduction >= .20 and retained,
        )
    return checks


class ScratchValidation:
    def __init__(self, run, output, interval, root, continuation=False,target='continuous_wrists',
                 protocol='legacy_v1',speed_dependent_yaw=False,
                 idle_head=False,backward_walk=False,low_speed_after_full=False,record_full_video=False,
                 stand_seconds=None,integrated_low_speed=False,parallel_envs=None,stop_timing=None,
                 backward_diagonal_run_speed=None,video_interval=None):
        from locomotion_acceptance import CURRENT_PROTOCOL

        self.run,self.output,self.root=Path(run).resolve(),Path(output).resolve(),Path(root)
        self.output.mkdir(parents=True,exist_ok=True);self.target=target
        self.interval=interval;self.validation_seconds=0.;self.history=[];self.training_offset=0.
        self.record_full_video=record_full_video
        self.video_interval=video_interval
        self.recording_seconds=0.
        self.protocol=protocol
        self.current_protocol=protocol==CURRENT_PROTOCOL
        self.integrated_low_speed=self.current_protocol and integrated_low_speed
        self.low_speed_after_full=low_speed_after_full and not self.current_protocol
        if continuation:
            history_path=self.output/'history.json'
            if history_path.is_file():
                self.history=json.loads(history_path.read_text(encoding='utf-8'))
            if self.history:
                self.validation_seconds=self.history[-1]['evaluation_seconds_cumulative']
            self.recording_seconds=sum(row.get('full_video',{}).get('pause_seconds',0.) for row in self.history)
        if target=='latest_unified':
            baseline_path=self.root/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json'
            self.unified_baseline={item['label']:item for item in json.loads(baseline_path.read_text(encoding='utf-8'))['segments']}
            from diagnostics.evaluate_parallel_walk_run import PARALLEL_ENVS
            from evaluate_unified_walk_run import (
                EXTENDED_STAND_SECONDS, LOW_SPEED_CASE_COUNT, LOW_SPEED_SECONDS, full_backward_diagonal_run_speed,
                full_stand_seconds, full_stop_timing, suite,
            )
            previous=json.loads((self.output/'acceptance_plan.json').read_text(encoding='utf-8')) if continuation else {}
            if continuation and stand_seconds is None:
                stand_seconds=previous.get('full_stand_hold_seconds',EXTENDED_STAND_SECONDS)
            if continuation and stop_timing is None:
                stop_timing=previous.get('full_stop_timing','original')
            self.stand_seconds=full_stand_seconds(protocol,stand_seconds)
            self.stop_timing=full_stop_timing(protocol,stop_timing)
            self.backward_diagonal_run_speed=full_backward_diagonal_run_speed(protocol,backward_diagonal_run_speed)
            duration=sum(s[1] for s in suite(protocol=protocol,idle_head=idle_head,backward_walk=backward_walk,
                                             stand_seconds=self.stand_seconds,stop_timing=self.stop_timing,
                                             backward_diagonal_run_speed=self.backward_diagonal_run_speed))
            if self.integrated_low_speed:
                duration+=LOW_SPEED_SECONDS
            plan=dict(interval_iterations=interval,baseline_report=str(baseline_path),acceptance_protocol=protocol,
                command_yaw_limit_by_speed=speed_dependent_yaw,
                unified_idle_head_acceptance=idle_head,
                unified_eval_backward_walk=backward_walk,
                low_speed_after_full=self.low_speed_after_full,
                full_stand_hold_seconds=self.stand_seconds,
                full_stop_timing=self.stop_timing,
                backward_diagonal_run_speed_m_s=self.backward_diagonal_run_speed,
                required_slip_reduction_fraction=.20,minimum_baseline_speed_retention=.95,
                speed_retention_basis='Measured-to-commanded speed ratio for each run segment.',
                official_suite=f'{duration}-second complete latest unified walk/run acceptance',
                required_slip_segments=['run_backward','run_left','run_right'],
                schedule='Complete suite at each interval boundary and on the final checkpoint.',
                timing_rule='First passing sampled checkpoint is an upper bound; acceptance pauses excluded from active training time.')
            if self.record_full_video:
                plan['video_interval_iterations'] = self.video_interval
            if parallel_envs is None:
                parallel_envs = PARALLEL_ENVS if (
                    self.current_protocol and speed_dependent_yaw and idle_head and backward_walk and not self.integrated_low_speed
                ) else 1
            plan['unified_eval_parallel_envs']=previous.get('unified_eval_parallel_envs',parallel_envs)
            if self.integrated_low_speed:
                plan['integrated_low_speed']=dict(case_count=LOW_SPEED_CASE_COUNT,simulated_seconds=LOW_SPEED_SECONDS)
            if self.current_protocol and self.stand_seconds < EXTENDED_STAND_SECONDS:
                plan['long_stand_recheck']=dict(after='first ordinary pass',pivot_seconds=6,stand_seconds=EXTENDED_STAND_SECONDS)
            if continuation:
                if (previous.get('acceptance_protocol','legacy_v1')!=protocol or
                    previous.get('command_yaw_limit_by_speed',False)!=speed_dependent_yaw or
                    previous.get('unified_idle_head_acceptance',False)!=idle_head or
                    previous.get('unified_eval_backward_walk',False)!=backward_walk):
                    raise ValueError('A new acceptance protocol requires separate versioned acceptance records')
                if previous.get('low_speed_after_full',False)!=self.low_speed_after_full:
                    raise ValueError('Scratch continuation cannot change supplemental acceptance scheduling')
            write_json(self.output/'acceptance_plan.json',plan)
            return
        base=self.root/'output/anon_locomotion/continuous_wrists'
        self.baseline=averages([json.loads((base/folder/'report.json').read_text(encoding='utf-8')) for folder in ['final_eval','boundary_eval']])
        limits={
            'Imitation/joint_rmse_rad':self.baseline['Imitation/joint_rmse_rad']+.01,
            'Imitation/keypoint_rmse_m':self.baseline['Imitation/keypoint_rmse_m']+.006,
            'Imitation/hand_orientation_error_rad':self.baseline['Imitation/hand_orientation_error_rad']+.03,
            'Imitation/arm_spacing_deficit_m':self.baseline['Imitation/arm_spacing_deficit_m']+.005,
            'Imitation/contact_accuracy':self.baseline['Imitation/contact_accuracy']-.04}
        self.limits=limits
        if continuation:
            previous=json.loads((self.output/'acceptance_plan.json').read_text(encoding='utf-8'))
            assert previous['limits']==limits,'Acceptance thresholds must not change during continuation'
            return
        write_json(self.output/'acceptance_plan.json',dict(interval_iterations=interval,baseline=self.baseline,limits=limits,
                   official_suites=['152 s main with pivots','68 s boundary'],
                   quality_rule='Upper bounds except contact_accuracy lower bound; both official suites must pass with zero falls. AMP discriminator self-scores are not comparable across runs.',
                   timing_rule='Active training time excludes checkpoint acceptance pauses. First passing sampled checkpoint is an upper bound, not a proven minimum. Visual comparison still required.'))

    @property
    def pause_seconds(self):
        return self.validation_seconds + self.recording_seconds

    def due(self, completed, final_iteration):
        return completed > 0 and (completed % self.interval == 0 or completed >= final_iteration)

    def audit_continuation(self, runner, checkpoint):
        manifest=json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))
        allowed=[Path(manifest['run_dir']).resolve()]
        allowed.extend(Path(x['run_dir']).resolve() for x in manifest.get('continuations',[]))
        checkpoint=Path(checkpoint).resolve()
        assert checkpoint.parent in allowed,'Only checkpoints from this same random-initialization lineage may be resumed'
        timing=json.loads((checkpoint.parent/'training_time.json').read_text(encoding='utf-8'))
        assert timing.get('started_from_random_weights') or timing.get('same_scratch_lineage')
        checkpoint_row = next((row for row in reversed(self.history)
                               if row.get('checkpoint') and Path(row['checkpoint']).resolve() == checkpoint
                               and row.get('active_training_seconds') is not None), None)
        if checkpoint_row is None:
            self.training_offset=timing.get('experiment_training_seconds',timing['training_seconds'])
            previous_completed=timing['completed_iterations']
        else:
            self.training_offset=checkpoint_row['active_training_seconds']
            previous_completed=checkpoint_row['completed_iterations']
        record=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    actor_sha256=tensor_hash(runner.alg.actor),previous_active_training_seconds=self.training_offset,
                    optimizer_state_entries=len(runner.alg.optimizer.state),previous_completed_iterations=previous_completed,
                    previous_initialization_audit=str(self.output/'initialization_audit.json'),
                    external_pretrained_weights_used=False)
        assert record['optimizer_state_entries']>0
        write_json(self.run/'continuation_audit.json',record)
        manifest.setdefault('continuations',[]).append(dict(run_dir=str(self.run),checkpoint=str(checkpoint),
            checkpoint_sha256=record['checkpoint_sha256'],actor_sha256=record['actor_sha256'],
            previous_active_training_seconds=self.training_offset,external_pretrained_weights_used=False))
        write_json(self.output/'manifest.json',manifest)

    def audit_initialization(self, runner, env):
        normalizers={}
        for role,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
            normalizers[role]={key:value.detach().cpu().tolist() for key,value in model.state_dict().items() if 'obs_normalizer' in key and key.endswith('count')}
        optimizers={}
        for key in ['optimizer','actor_optimizer','critic_optimizer']:
            optimizer=getattr(runner.alg,key,None)
            if optimizer is not None:optimizers[key]=len(optimizer.state)
        assert all(count==0 for count in optimizers.values()),optimizers
        assert len(env.amp.optimizer.state)==0
        assert all(float(value)==0 for values in normalizers.values() for value in values.values()),normalizers
        audit=dict(pretrained_checkpoint=None,actor_sha256=tensor_hash(runner.alg.actor),critic_sha256=tensor_hash(runner.alg.critic),
                   amp_discriminator_sha256=tensor_hash(env.amp.discriminator),ppo_optimizer_state_entries=optimizers,
                   amp_optimizer_state_entries=0,observation_normalizer_counts=normalizers,initial_iteration=runner.current_learning_iteration,
                   num_envs=env.num_envs,num_actions=env.num_actions,num_observations=int(env.get_observations()['policy'].shape[-1]),
                   note='Network weights random; physical/reference-state resets, mocap data and nominal PD are retained by design. No behavioral cloning or teacher policy is used.')
        assert runner.current_learning_iteration==0
        write_json(self.output/'initialization_audit.json',audit)
        manifest=dict(run_dir=str(self.run),continuations=[],seed=env.cfg.get('seed'),num_envs=env.num_envs,
                      initialization_audit=str(self.output/'initialization_audit.json'),
                      initialization_audit_sha256=hashlib.sha256((self.output/'initialization_audit.json').read_bytes()).hexdigest(),
                      started_from_random_weights=True)
        write_json(self.output/'manifest.json',manifest)

    def evaluate(self, runner, completed, active_seconds, *, final_checkpoint=False):
        if self.target=='latest_unified':
            if completed==0:return False
            return self.evaluate_latest_unified(runner,completed,active_seconds,final_checkpoint=final_checkpoint)
        paused=time.perf_counter()
        checkpoint=self.run/f'scratch_{completed:06d}.pt'
        runner.save(str(checkpoint))
        work=self.output/f'checkpoint_{completed:06d}';work.mkdir(parents=True,exist_ok=True)
        reports=[]
        for name,override in [('main',dict(continuous_eval_pivots=True,continuous_eval_suite='main')),
                              ('boundary',dict(continuous_eval_pivots=False,continuous_eval_suite='boundary'))]:
            override.update(simulation_device='cuda:0',runtime_inference_only=False)
            config=work/f'{name}.json';write_json(config,override)
            write_json(self.output/'experiment_status.json',dict(phase='acceptance',completed_iterations=completed,active_training_seconds=active_seconds,suite=name,checkpoint=str(checkpoint)))
            command=[sys.executable,str(self.root/'source/pmx_prototype/train_anon_locomotion.py'),
                     '--mode','eval','--checkpoint',str(checkpoint),'--num-envs','1','--headless',
                     '--config-override',str(config),'--output',str(work/name)]
            with (work/f'{name}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            path=work/name/'report.json'
            if result.returncode or not path.exists():
                raise RuntimeError(f'Scratch acceptance process failed: {work/name}; see {work/name}.log')
            reports.append(json.loads(path.read_text(encoding='utf-8')))
        metrics=averages(reports)
        quality={key:metrics[key]>=limit if key.endswith('contact_accuracy') else metrics[key]<=limit for key,limit in self.limits.items()}
        official=all(report['passed_continuous_gate'] and report['total_falls']==0 for report in reports)
        self.validation_seconds+=time.perf_counter()-paused
        item=dict(completed_iterations=completed,active_training_seconds=active_seconds,checkpoint=str(checkpoint),
                  official_passed=official,quality_passed=all(quality.values()),quality_checks=quality,metrics=metrics,
                  total_falls=sum(report['total_falls'] for report in reports),passed=official and all(quality.values()),
                  evaluation_seconds_cumulative=self.validation_seconds)
        self.history.append(item)
        write_json(self.output/'history.json',self.history)
        write_json(self.output/'experiment_status.json',dict(phase='matched_quantitative_gates' if item['passed'] else 'training',**item))
        print('SCRATCH_ACCEPTANCE '+json.dumps(item),flush=True)
        return item['passed']

    def evaluate_latest_unified(self,runner,completed,active_seconds, *, final_checkpoint=False):
        paused=time.perf_counter()
        checkpoint=self.run/f'scratch_{completed:06d}.pt';runner.save(str(checkpoint))
        base=self.output/f'checkpoint_{completed:06d}'

        def notify_stage(name):
            write_json(self.output/'experiment_status.json',dict(
                phase='acceptance',suite=name,completed_iterations=completed,
                active_training_seconds=active_seconds,checkpoint=str(checkpoint)))

        def run_suite(name,cases=None):
            notify_stage(name)
            folder = 'unified' if cases is None else name
            work=base/folder;work.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(self.root/'source/pmx_prototype/train_anon_locomotion.py'),
                     '--mode','eval','--checkpoint',str(checkpoint),'--num-envs','1','--headless',
                     '--output',str(work),'--no-acceptance-video']
            config=base/f'{name}_config.json'
            override=dict(unified_acceptance_protocol=self.protocol,
                          unified_eval_stand_hold_seconds=self.stand_seconds,
                          unified_eval_stop_timing=self.stop_timing,
                          unified_eval_backward_diagonal_run_speed=self.backward_diagonal_run_speed,
                          unified_eval_low_speed=self.integrated_low_speed,
                          unified_eval_cases=cases,unified_eval_initial_command=None)
            write_json(config,override)
            command += ['--config-override',str(config)]
            log_path=base/f'{folder}.log'
            with log_path.open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            report_path=work/'report.json'
            if result.returncode or not report_path.exists():raise RuntimeError(f'Unified scratch {name} failed: {work}; see {log_path}')
            return json.loads(report_path.read_text(encoding='utf-8'))

        report=run_suite('full')
        slip_checks=compare_directional_slip(report,self.unified_baseline)
        official=bool(report['passed_unified_gate'] and report['total_falls']==0)
        slip_passed=all(v['passed'] for v in slip_checks.values())
        low_speed=None
        if self.integrated_low_speed:
            low_speed=report['low_speed']
        elif self.low_speed_after_full and ((official and slip_passed) or final_checkpoint):
            notify_stage('low_speed')
            low_speed=self.evaluate_low_speed(checkpoint,base)
        requires_low_speed = self.integrated_low_speed or self.low_speed_after_full
        combined_passed = bool(official and slip_passed and (not requires_low_speed or (low_speed and low_speed['passed'])))
        long_stand=None
        if self.current_protocol and combined_passed:
            from evaluate_unified_walk_run import EXTENDED_STAND_SECONDS
            if self.stand_seconds < EXTENDED_STAND_SECONDS:
                check=run_suite('long_stand_recheck',[
                    ('pivot_right',6,[0,0,-.6]),('stand_end',EXTENDED_STAND_SECONDS,[0,0,0])])
                long_stand=dict(passed=bool(check['segments'][-1]['passed'] and check['total_falls']==0),
                                report=str(base/'long_stand_recheck'/'report.json'),
                                falls=check['total_falls'],standing_seconds=EXTENDED_STAND_SECONDS)
        self.validation_seconds+=time.perf_counter()-paused
        full_report=str(base/'unified'/'report.json')
        item=dict(completed_iterations=completed,active_training_seconds=active_seconds,checkpoint=str(checkpoint),
            acceptance_protocol=report.get('acceptance_protocol','legacy_v1'),
            acceptance_kind='full',full_evaluated=True,official_passed=official,
            slip_checks=slip_checks,slip_passed=slip_passed,
            total_falls=report['total_falls'],passed=official and slip_passed,
            evaluation_seconds_cumulative=self.validation_seconds,report=full_report,full_report=full_report,
            full_stand_hold_seconds=self.stand_seconds)
        item['full_stop_timing']=self.stop_timing
        item['backward_diagonal_run_speed_m_s']=report['backward_diagonal_run_speed_m_s']
        if 'passed_light_gate' in report:
            item.update(response_steady_summary=report['response_steady_summary'],
                        light_acceptance_summary=report['light_acceptance_summary'],
                        passed_light_gate=report['passed_light_gate'])
        if 'acceptance_layout' in report:
            item.update(acceptance_layout=report['acceptance_layout'],
                        acceptance_environments=report['acceptance_environments'])
        if self.integrated_low_speed or self.low_speed_after_full:
            item['low_speed']=low_speed
        if long_stand is not None:
            item['long_stand_recheck']=long_stand
            combined_passed &= long_stand['passed']
        if requires_low_speed or self.current_protocol:
            item['combined_passed']=combined_passed
        if self.record_full_video:
            item['video_scheduled'] = self.video_interval is None or completed % self.video_interval == 0
        self.history.append(item);write_json(self.output/'history.json',self.history)
        passed=item.get('combined_passed',item['passed'])
        if item.get('video_scheduled', False):
            import torch
            from diagnostics.anon_acceptance_video import record_acceptance_video
            started = time.perf_counter()
            write_json(self.output/'experiment_status.json',dict(phase='recording',**item))
            torch.cuda.empty_cache()
            print(f'ANON_ACCEPTANCE_VIDEO_START {completed}',flush=True)
            video = record_acceptance_video(item['full_report'],title=f'Policy {completed} acceptance')
            video['pause_seconds'] = time.perf_counter() - started
            self.recording_seconds += video['pause_seconds']
            item['full_video'] = video
            write_json(self.output/'history.json',self.history)
        phase='matched_latest_gates' if passed else 'full_acceptance'
        write_json(self.output/'experiment_status.json',dict(phase=phase,**item))
        print('SCRATCH_LATEST_ACCEPTANCE '+json.dumps(item),flush=True)
        return passed

    def evaluate_low_speed(self,checkpoint,base):
        models=base/'low_speed_models.json'
        config=base/'low_speed_config.json'
        output=base/'low_speed'
        write_json(models,dict(candidate=str(checkpoint)))
        write_json(config,dict(diagnostic_controller_trace=True))
        command=[sys.executable,str(resolve_source('diagnostics/probe_low_speed_tracking.py')),
            '--models',str(models),'--output',str(output),'--config-override',str(config),'--no-nominal']
        with (base/'low_speed.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        path=output/'report.json'
        if result.returncode or not path.exists():
            raise RuntimeError(f'Scratch low-speed probe failed: {base}; see low_speed.log')
        report=json.loads(path.read_text(encoding='utf-8'))
        metadata=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        speeds=[row['speed'] for row in report['cases']]
        if not report.get('checkpoint_unchanged') or speeds!=[.15,.20,.25,.30,.35,.40] or metadata['num_envs']!=16:
            raise RuntimeError(f'Incomplete scratch low-speed probe: {path}')
        falls=sum(row['falls'] for row in report['cases'])
        return dict(passed=falls==0,falls=falls,cases=len(speeds)*16,report=str(path))
