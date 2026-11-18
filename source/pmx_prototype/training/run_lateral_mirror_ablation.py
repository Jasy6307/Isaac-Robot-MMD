"""Three independent100-iteration8192 continuations with fixed full v2 gates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/lateral_mirror_ablation_20261117_8192'
BASE=ROOT/'logs/rsl_rl/anon_locomotion/scratch_balanced_v2_20261117_8192/policy_final.pt'
ITERATIONS=100
BEFORE=109335.5564666001
VARIANTS={'augmentation_only':'lateral_mirror_augmentation_only.json',
          'loss_only':'lateral_mirror_loss_only.json',
          'combined':'lateral_mirror_v1_resume.json'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(stage,args,run=None):
    write(OUT/'status.json',dict(stage=stage,training=stage.endswith('_train'),started_unix=time.time(),run_dir=str(run) if run else None))
    started=time.perf_counter()
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid,run_dir=str(run) if run else None))
        code=process.wait()
    if code:raise RuntimeError(f'{stage} failed ({code}); inspect {stage}.log')
    return time.perf_counter()-started


def full(name,checkpoint):
    folder=OUT/name/'full'
    seconds=execute(name+'_full',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
        '--num-envs','1','--checkpoint',checkpoint,'--output',folder,'--config-override',OUT/'full_config.json'])
    return read(folder/'report.json'),seconds


def paired_summary(folder):
    import numpy as np
    data=np.load(Path(folder)/'native_policy.npz');dt=1/30
    falls=data['done'].astype(bool).sum(0);valid=falls==0
    velocities=data['velocity'][round(6/dt):].mean(0);n=len(falls)//2
    result={}
    for label,sl,sign in [('left',slice(0,n),1),('right',slice(n,None),-1)]:
        speed=sign*velocities[sl,1];mask=valid[sl]
        result[label]=dict(total_states=len(speed),fall_free_states=int(mask.sum()),falls=int(falls[sl].sum()),
            speed_all_states=float(speed.mean()),speed_fall_free=float(speed[mask].mean()) if mask.any() else None,
            speed_by_state=speed.tolist(),falls_by_state=falls[sl].tolist())
    return result


def gates(report,original):
    assert report['acceptance_protocol']=='response_steady_v2' and report['complete_scope']
    assert len(report['segments'])==36 and report['total_seconds']==355
    checks={};current={x['label']:x for x in report['segments']}
    for label in ['run_backward','run_left','run_right']:
        old=original[label];new=current[label]
        reduction=1-new['stance_slip_geometry_m_s']/max(old['stance_slip_geometry_m_s'],1e-8)
        retained=new['mean_speed']>=.95*old['mean_speed']
        checks[label]=dict(reduction_fraction=reduction,retained_speed=retained,passed=reduction>=.2 and retained)
    return dict(segment_passes=sum(x['passed'] for x in report['segments']),segments=len(report['segments']),
        falls=report['total_falls'],complete_suite_passed=report['passed_unified_gate'],slip_checks=checks,
        passed=bool(report['passed_unified_gate'] and report['total_falls']==0 and all(x['passed'] for x in checks.values())))


def compare(baseline,candidate):
    """Preserve the established regression tolerances, with v2 pass retention."""
    result=[]
    for index,(old,new) in enumerate(zip(baseline['segments'],candidate['segments'])):
        assert old['label']==new['label']
        passed=new['falls']==0 and (not old['passed'] or new['passed'])
        passed&=new['velocity_error']<=old['velocity_error']+.05 and new['yaw_error']<=old['yaw_error']+.05
        passed&=new['metrics']['Imitation/keypoint_rmse_m']<=old['metrics']['Imitation/keypoint_rmse_m']+.01
        if old['label'].startswith('run_'):
            passed&=new['mean_speed']>=.95*old['mean_speed'] and new['stance_slip_geometry_m_s']<=old['stance_slip_geometry_m_s']+.03
        before=old.get('response_steady',{});after=new.get('response_steady',{})
        row=dict(index=index,label=old['label'],passed=bool(passed),baseline_passed=old['passed'],candidate_passed=new['passed'],
            baseline_falls=old['falls'],candidate_falls=new['falls'],speed_before=old['mean_speed'],speed_after=new['mean_speed'],
            velocity_error_before=old['velocity_error'],velocity_error_after=new['velocity_error'],
            yaw_error_before=old['yaw_error'],yaw_error_after=new['yaw_error'],
            keypoint_error_before=old['metrics']['Imitation/keypoint_rmse_m'],keypoint_error_after=new['metrics']['Imitation/keypoint_rmse_m'],
            steady_before=before.get('steady'),steady_after=after.get('steady'),
            linear_response_before=before.get('linear_response_seconds'),linear_response_after=after.get('linear_response_seconds'))
        result.append(row)
    return result


def audit_initial(run):
    import torch
    old=torch.load(BASE,map_location='cpu',weights_only=False)
    new=torch.load(run/'initial_preserved.pt',map_location='cpu',weights_only=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    keys=['actor_state_dict','critic_state_dict','optimizer_state_dict','amp_state']
    identity={key:equal(old[key],new[key]) for key in keys}
    assert all(identity.values()),identity
    audited=read(run/'resume_identity_audit.json')
    assert all(x['normalizer_identical'] and x['normalized_input_max_error']==0 and x['output_max_error']==0 for x in audited.values())
    return dict(state_identity=identity,input_output_identity=audited)


def main():
    if (OUT/'plan.json').exists():raise RuntimeError('Experiment already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    assert digest(BASE)=='4c2372338aad25028ceaa089cbff0b30dd2ac8fc269e3357b5277c2d55351e61'
    template=read(SOURCE/'configs/training/unified/low_speed_balanced_v2_resume.json')
    flags=['train_lateral_mirror','train_mirror_augmentation','train_mirror_loss_coeff','amp_mirror_augmentation']
    configs={}
    for name,filename in VARIANTS.items():
        config=read(SOURCE/'configs/training/unified'/filename)
        assert all(config[k]==v for k,v in template.items())
        assert set(config)-set(template)==set(flags)
        assert config['train_audit_resume_identity'] and config['train_preserve_optimizer_learning_rate']
        assert not config.get('train_leg_std') and not config.get('train_reset_optimizer') and not config.get('amp_reset_discriminator')
        write(OUT/(name+'_config.json'),config);configs[name]={k:config[k] for k in flags}
    write(OUT/'full_config.json',dict(unified_acceptance_protocol='response_steady_v2',unified_eval_lite=False,
        unified_eval_short=False,unified_record_controller=True))
    frozen=['train_anon_locomotion.py','training/anon_imitation_env.py','training/anon_locomotion_env.py',
        'training/lateral_symmetry.py','training/amp_motion_prior.py','training/command_plan.py','training/locomotion_acceptance.py',
        'motion/unified_walk_run_reference.py','diagnostics/evaluate_unified_walk_run.py','diagnostics/probe_lateral_symmetry.py']
    hashes={f:digest(SOURCE/f) for f in frozen}
    plan=dict(checkpoint=str(BASE),checkpoint_sha256=digest(BASE),base_completed_iterations=3932,iterations_each=ITERATIONS,
        final_completed_iterations_each=4032,num_envs=8192,seed=42,arms=configs,
        source_hashes=hashes,cumulative_training_seconds_before=BEFORE,base_lineage_training_seconds=20589.059673400014,
        shared_gpu_authorized=True,acceptance_protocol='response_steady_v2',complete_suite_seconds=355,
        original_slip_reduction_required=.2,original_baseline_speed_retention=.95,
        unchanged_regression_tolerances=dict(velocity_error_delta=.05,yaw_error_delta=.05,keypoint_error_delta_m=.01,
            running_speed_retention=.95,stance_slip_delta_m_s=.03,zero_falls=True,preserve_baseline_passes=True),
        scope='Three independent matched100-iteration continuations. No automatic extension or UI model promotion.',
        augmentation_bundle='PPO/AMP paired data and reward/scale symmetry; weak loss coefficient0.05 in the other factor.',
        timing_rule='Actual shared-GPU training-loop wall seconds; setup, full acceptance and paired probes recorded separately.',
        limitations='One seed and one100-iteration budget per factor; full gate and16 paired phases are distinct measurements.')
    write(OUT/'plan.json',plan)
    started=time.perf_counter();arms={};durations={};training=0.
    original={x['label']:x for x in read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
    baseline,seconds=full('baseline',BASE);durations['baseline_full']=seconds
    prior=ROOT/'output/anon_locomotion/lateral_symmetry_20261117/before'
    assert read(prior/'report.json')['checkpoint_sha256']==digest(BASE)
    baseline_paired=paired_summary(prior)
    write(OUT/'baseline.json',dict(gate=gates(baseline,original),paired=baseline_paired,
        paired_reused_from=str(prior),training_seconds=0.,cumulative_training_seconds=BEFORE))
    for name in VARIANTS:
        assert all(digest(SOURCE/f)==v for f,v in hashes.items()),'Source changed before '+name
        run=ROOT/f'logs/rsl_rl/anon_locomotion/lateral_mirror_ablation_20261117_8192_{name}'
        if run.exists():raise RuntimeError('Existing run: '+str(run))
        durations[name+'_train']=execute(name+'_train',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless',
            '--num-envs','8192','--iterations',str(ITERATIONS),'--seed','42','--checkpoint',BASE,
            '--run-dir',run,'--config-override',OUT/(name+'_config.json')],run)
        timing=read(run/'training_time.json');training+=timing['training_seconds']
        assert timing['completed_iteration']==4031,timing
        identity=audit_initial(run)
        report,seconds=full(name,run/'policy_final.pt');durations[name+'_full']=seconds
        folder=OUT/name/'paired'
        durations[name+'_paired']=execute(name+'_paired',[SOURCE/'diagnostics/probe_lateral_symmetry.py',
            '--checkpoint',run/'policy_final.pt','--output',folder])
        paired=paired_summary(folder);regression=compare(baseline,report)
        # Right running is the targeted improvement; all other segments remain
        # subject to the established no-regression limits and v2 pass retention.
        retained=[x for x in regression if x['label']!='run_right']
        old_right=next(x for x in baseline['segments'] if x['label']=='run_right')
        new_right=next(x for x in report['segments'] if x['label']=='run_right')
        item=dict(training_seconds=timing['training_seconds'],new_iterations=ITERATIONS,completed_iterations=4032,
            checkpoint=str(run/'policy_final.pt'),resume_identity=identity,gate=gates(report,original),
            regression=regression,no_other_regression=all(x['passed'] for x in retained),paired=paired,
            right_speed_before=old_right['mean_speed'],right_speed_after=new_right['mean_speed'],
            right_tracking_error_before=old_right['response_steady']['steady']['velocity_error'],
            right_tracking_error_after=new_right['response_steady']['steady']['velocity_error'],
            right_tracking_improved=new_right['response_steady']['steady']['velocity_error']<old_right['response_steady']['steady']['velocity_error'],
            lineage_training_seconds=plan['base_lineage_training_seconds']+timing['training_seconds'],
            cumulative_training_seconds=BEFORE+training,experiment_training_seconds=training,
            full_report=str(OUT/name/'full/report.json'),paired_report=str(folder/'report.json'))
        arms[name]=item
        write(OUT/(name+'_result.json'),item)
        write(OUT/'result.json',dict(stage='running',plan=plan,baseline_gate=gates(baseline,original),
            baseline_paired=baseline_paired,arms=arms,durations=durations,new_training_seconds=training,
            cumulative_training_seconds=BEFORE+training,wall_seconds=time.perf_counter()-started))
    assert all(digest(SOURCE/f)==v for f,v in hashes.items()),'Frozen source changed'
    assert digest(BASE)==plan['checkpoint_sha256'],'Base checkpoint changed'
    result=read(OUT/'result.json');result.update(stage='complete',wall_seconds=time.perf_counter()-started)
    write(OUT/'result.json',result)
    write(OUT/'status.json',dict(stage='complete',training=False,new_training_seconds=training,cumulative_training_seconds=BEFORE+training))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',training=False,error=str(error)));raise
