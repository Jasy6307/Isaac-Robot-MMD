"""Fresh-initialization evidence and fixed, externally evaluated training gates."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


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


class ScratchValidation:
    def __init__(self, run, output, interval, root, continuation=False):
        self.run,self.output,self.root=Path(run).resolve(),Path(output).resolve(),Path(root)
        self.output.mkdir(parents=True,exist_ok=True)
        self.interval=interval;self.validation_seconds=0.;self.history=[];self.training_offset=0.
        if continuation:
            self.history=json.loads((self.output/'history.json').read_text(encoding='utf-8'))
            self.validation_seconds=self.history[-1]['evaluation_seconds_cumulative']
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

    def audit_continuation(self, runner, checkpoint):
        manifest=json.loads((self.output/'manifest.json').read_text(encoding='utf-8'))
        allowed=[Path(manifest['run_dir']).resolve()]
        allowed.extend(Path(x['run_dir']).resolve() for x in manifest.get('continuations',[]))
        checkpoint=Path(checkpoint).resolve()
        assert checkpoint.parent in allowed,'Only checkpoints from this same random-initialization lineage may be resumed'
        timing=json.loads((checkpoint.parent/'training_time.json').read_text(encoding='utf-8'))
        assert timing.get('started_from_random_weights') or timing.get('same_scratch_lineage')
        self.training_offset=timing.get('experiment_training_seconds',timing['training_seconds'])
        record=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    actor_sha256=tensor_hash(runner.alg.actor),previous_active_training_seconds=self.training_offset,
                    optimizer_state_entries=len(runner.alg.optimizer.state),previous_completed_iterations=timing['completed_iterations'],
                    previous_initialization_audit=str(self.output/'initialization_audit.json'),
                    external_pretrained_weights_used=False)
        assert record['optimizer_state_entries']>0
        write_json(self.run/'continuation_audit.json',record)

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

    def evaluate(self, runner, completed, active_seconds):
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
