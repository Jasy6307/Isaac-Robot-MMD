"""One fresh AMP/PPO lineage after the fixed-policy split-planner tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/scratch_split_20261117'
TESTS=ROOT/'output/anon_locomotion/split_command_tests_20261117'
BASE_CONFIG=ROOT/'output/anon_locomotion/retention_strength_20261116/original_config.json'
RUN=ROOT/'logs/rsl_rl/anon_locomotion/scratch_split_20261117_12288'
TOTAL=7500


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def launch(stage,arguments):
    write(OUT/'controller_status.json',dict(stage=stage,started_unix=time.time()))
    start=time.perf_counter()
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid))
        code=process.wait()
    return code,time.perf_counter()-start


def train(run,config,envs,iterations,checkpoint=None):
    command=[SOURCE/'train_anon_locomotion.py','--mode','train','--headless','--num-envs',str(envs),
        '--iterations',str(iterations),'--run-dir',run,'--experiment-output',OUT/'acceptance',
        '--scratch-target','latest_unified','--acceptance-interval','500','--scratch-full-interval','2500',
        '--config-override',config,'--seed','42']
    if checkpoint:command+=['--scratch-continuation','--checkpoint',checkpoint]
    else:command+=['--from-scratch']
    return launch(run.name,command)


def extra_checks(checkpoint,tag):
    sys.path.insert(0,str(SOURCE/'training'))
    sys.path.insert(0,str(SOURCE/'diagnostics'))
    from continue_sampling_pivot import regression
    from compare_split_command_plans import full_checks
    work=OUT/tag;work.mkdir(parents=True,exist_ok=True)
    models=work/'models.json';write(models,dict(candidate=str(checkpoint)))
    code,_=launch(tag+'_pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py','--models',models,'--output',work/'pivots'])
    if code:raise RuntimeError('Fresh pivot inference failed; see log')
    pivots=read(work/'pivots/report.json')
    pivot_falls=sum(s['total_falls'] for s in pivots['models']['candidate'])
    result=dict(pivot_falls=pivot_falls)
    reports={}
    for label,override in [('lite',dict(unified_eval_lite=True)),('brakes',dict(unified_eval_cases=read(TESTS/'plan.json')['brake_cases']))]:
        config=work/(label+'_config.json');write(config,override)
        code,_=launch(tag+'_'+label,[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless','--num-envs','1',
            '--checkpoint',checkpoint,'--output',work/label,'--config-override',config])
        if code:raise RuntimeError('Fresh '+label+' inference failed; see log')
        reports[label]=read(work/label/'report.json')
    lite=reports['lite'];brakes=reports['brakes']
    checks=regression(read(TESTS/'baseline_lite/report.json'),lite)
    brake_checks=regression(read(TESTS/'baseline_brakes/report.json'),brakes)
    latest=read(OUT/'acceptance/history.json')[-1]
    full=full_checks(read(Path(latest['full_report'])),read(ROOT/'output/anon_locomotion/command_plan_tests_20261116/baseline_full/report.json'))
    targets=[s for s in lite['segments'] if s['label'] in ['run_diagonal_135','reverse_turn']]
    result.update(lite_targets=targets,lite_regression=checks,lite_falls=lite['total_falls'],
        brakes_passed=all(s['passed'] for s in brakes['segments']),brakes_falls=brakes['total_falls'],
        brake_regression=brake_checks,full_checks=full)
    result['passed']=bool(pivot_falls==0 and lite['total_falls']==0 and all(s['passed'] for s in targets) and
        all(s['passed'] for s in checks) and all(s['passed'] for s in brake_checks) and
        result['brakes_passed'] and brakes['total_falls']==0 and full['accepted'])
    write(work/'result.json',result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'experiment_plan.json').exists():raise RuntimeError('Duplicate fresh experiment')
    write(OUT/'controller_status.json',dict(stage='waiting_for_fixed_policy_tests',training=False))
    while True:
        status=read(TESTS/'status.json') if (TESTS/'status.json').exists() else {}
        if status.get('stage') in ['complete','error']:break
        time.sleep(20)
    test_result=read(TESTS/'result.json')
    chosen=test_result.get('winner')
    if not chosen:
        candidates=[name for name,row in test_result['rows'].items() if name!='baseline' and row.get('brakes_passed') and row.get('brakes_falls')==0]
        chosen=min(candidates,key=lambda n:sum(s['velocity_error'] for s in test_result['rows'][n]['targets'].values())) if candidates else 'split_120'
    config=read(BASE_CONFIG)
    removed={key:config.pop(key) for key in list(config) if key.startswith('train_retention_') or key.startswith('train_teacher_')}
    config.update(test_result['plan']['variants'][chosen])
    config.update(train_retention_weight=0.,train_leg_std=.08,scratch_anneal_iterations=300,
                  train_audit_resume_identity=True)
    for key in ['train_freeze_obs_normalization','train_reset_optimizer','train_reset_actor_head','train_normalization_rebase']:
        config.pop(key,None)
    fresh_config=OUT/'config.json';write(fresh_config,config)
    continuation=dict(config);continuation.pop('train_leg_std')
    continuation_config=OUT/'continuation_config.json';write(continuation_config,continuation)
    plan=dict(started_from_random_weights=True,external_checkpoint_used=False,external_teacher_used=False,
        source_configuration=str(BASE_CONFIG),chosen_split_plan=chosen,fixed_policy_test_report=str(TESTS/'result.json'),
        num_envs=12288,fallback_num_envs=8192,maximum_iterations=TOTAL,seed=42,
        acceptance_interval=500,full_acceptance_interval=2500,removed_teacher_settings=removed,
        acceptance='Unchanged original full325s gate, zero falls and original slip/speed thresholds; additionally64 pivot probes, rapid target switches, stop/backward diagnostics and fixed-actor lite regression checks.',
        stop_rule='Stop at first full plus extra-check pass, or7500 iterations. Same-lineage continuation only, without tuning.',
        cumulative_training_seconds_before=51378.49378980009,
        active_training_time_excludes='Scene setup, all acceptance and rendering; no video requested in this experiment.')
    write(OUT/'experiment_plan.json',plan)
    failed_training_seconds=0.;envs=12288;run=RUN
    code,wall=train(run,fresh_config,envs,TOTAL)
    if code:
        log=(OUT/(run.name+'.log')).read_text(encoding='utf-8',errors='replace')
        if 'out of memory' not in log.lower():raise RuntimeError('Fresh training failed; inspect '+run.name+'.log')
        failed_status=read(run/'status.json') if (run/'status.json').exists() else {}
        failed_training_seconds=failed_status.get('elapsed_seconds',0.)
        write(OUT/'oom_attempt.json',dict(run=str(run),wall_seconds=wall,recorded_active_training_seconds=failed_training_seconds))
        envs=8192;run=Path(str(RUN).replace('_12288','_8192'))
        code,wall=train(run,fresh_config,envs,TOTAL)
        if code:raise RuntimeError('8192 fallback failed; inspect log')
    attempts=[];extra_seconds=0.;accepted=False
    for continuation_index in range(20):
        timing=read(run/'training_time.json');history=read(OUT/'acceptance/history.json');latest=history[-1]
        completed=timing['completed_iterations']
        if latest['passed']:
            started=time.perf_counter();extra=extra_checks(run/'policy_final.pt',f'checks_{completed:06d}')
            extra_seconds+=time.perf_counter()-started
            accepted=extra['passed']
            attempts.append(dict(completed_iterations=completed,full_passed=True,extra=extra))
            write(OUT/'extra_history.json',attempts)
        if accepted or completed>=TOTAL:break
        next_run=Path(str(RUN).replace('_12288','_'+str(envs))+f'_cont{continuation_index+1}')
        code,_=train(next_run,continuation_config,envs,TOTAL-completed,run/'policy_final.pt')
        if code:raise RuntimeError('Same-lineage continuation failed; inspect log')
        run=next_run
    active=timing['experiment_training_seconds']+failed_training_seconds
    result=dict(accepted=accepted,run_dir=str(run),checkpoint=str(run/'policy_final.pt'),num_envs=envs,
        completed_iterations=completed,training_seconds=active,acceptance_seconds=timing['acceptance_seconds']+extra_seconds,
        cumulative_training_seconds=plan['cumulative_training_seconds_before']+active,started_from_random_weights=True,
        full_passed=latest['passed'],extra_attempts=attempts,configuration_unchanged_during_training=True,
        initialization_audit=str(OUT/'acceptance/initialization_audit.json'))
    write(OUT/'result.json',result)
    write(OUT/'controller_status.json',dict(stage='complete',accepted=accepted,training=False,**{k:result[k] for k in ['training_seconds','cumulative_training_seconds','checkpoint']}))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'controller_status.json',dict(stage='error',error=str(error)));raise
