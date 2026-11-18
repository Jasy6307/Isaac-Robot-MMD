"""Fresh balanced-recipe training with fixed v2 gates and timing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/scratch_balanced_v2_20261117_8192'
RUN=ROOT/'logs/rsl_rl/anon_locomotion/scratch_balanced_v2_20261117_8192'
RECIPE=SOURCE/'configs/training/unified/low_speed_balanced_v2.json'
MAXIMUM=6000
BEFORE=88746.49679320009
NUM_ENVS=8192


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def execute(stage,args,run=None):
    write(OUT/'controller_status.json',dict(stage=stage,training=stage.startswith('train'),
        started_unix=time.time(),current_run_dir=str(run) if run else None))
    started=time.perf_counter()
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=child.pid,run_dir=str(run) if run else None))
        code=child.wait()
    if code:raise RuntimeError(stage+' failed; inspect '+stage+'.log')
    return time.perf_counter()-started


def learn(run,iterations,checkpoint=None):
    config=OUT/('continuation_config.json' if checkpoint else 'config.json')
    arguments=[SOURCE/'train_anon_locomotion.py','--mode','train','--headless','--num-envs',str(NUM_ENVS),
        '--iterations',str(iterations),'--seed','42','--run-dir',run,'--experiment-output',OUT/'acceptance',
        '--scratch-target','latest_unified','--acceptance-interval','500','--scratch-full-interval','1000',
        '--config-override',config]
    arguments+=['--scratch-continuation','--checkpoint',checkpoint] if checkpoint else ['--from-scratch']
    return execute('train_'+run.name,arguments,run)


def main():
    if (OUT/'experiment_plan.json').exists() or RUN.exists():raise RuntimeError('Fresh experiment output already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    config=read(RECIPE)
    assert config['unified_acceptance_protocol']=='response_steady_v2'
    assert not any(config.get(k) for k in ['train_retention_weight','train_retention_teacher_checkpoint','train_teacher_anchor_path'])
    write(OUT/'config.json',config)
    continuation=dict(config);continuation.pop('train_leg_std',None)
    write(OUT/'continuation_config.json',continuation)
    files=['train_anon_locomotion.py','training/anon_scratch_validation.py','training/low_speed_commands.py',
        'training/locomotion_acceptance.py','training/command_plan.py','training/anon_imitation_env.py',
        'motion/unified_walk_run_reference.py','diagnostics/evaluate_unified_walk_run.py']
    hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    plan=dict(started_from_random_weights=True,pretrained_checkpoint=None,external_teacher=None,num_envs=NUM_ENVS,seed=42,
        maximum_iterations=MAXIMUM,screen_interval=500,full_interval=1000,steps_per_env_per_iteration=48,
        recipe=str(RECIPE),recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),source_hashes=hashes,
        acceptance_protocol='response_steady_v2',complete_suite_seconds=355,
        full_gate='Complete v2 suite, zero falls, unchanged original slip reduction and speed-retention checks.',
        supplemental_low_speed_check='After a full pass, fixed checkpoint walks/stops at0.15,0.20,0.25,0.30,0.35,0.40m/s across16 initial phase/heading states. Zero falls; no new speed threshold.',
        milestone_rule='Record first complete-gate pass separately from first complete+low-speed-check pass. A screen never grants acceptance.',
        stop_rule='Stop at first complete+low-speed-check pass, or6000 iterations per the prior user limit. Do not extend or retune automatically.',
        time_rule='Actual training-loop wall time including shared GPU contention; excludes initialization and all acceptance. Report acceptance and wall time separately.',
        cumulative_training_seconds_before=BEFORE,shared_gpu_authorized=True)
    write(OUT/'experiment_plan.json',plan)
    launched=time.perf_counter();run=RUN;extra_seconds=0.;attempts=[];accepted=False
    learn(run,MAXIMUM)
    for continuation_index in range(20):
        timing=read(run/'training_time.json');history=read(OUT/'acceptance/history.json');last=history[-1]
        completed=timing['completed_iterations']
        if last['passed']:
            work=OUT/f'low_speed_{completed:06d}';models=OUT/f'models_{completed:06d}.json'
            write(models,dict(candidate=str(run/'policy_final.pt')))
            check_config=OUT/'low_speed_check_config.json'
            write(check_config,dict(diagnostic_controller_trace=True))
            extra_seconds+=execute(f'low_speed_{completed:06d}',[SOURCE/'diagnostics/probe_low_speed_tracking.py',
                '--models',models,'--output',work,'--config-override',check_config,'--no-nominal'])
            report=read(work/'report.json');falls=sum(row['falls'] for row in report['cases'])
            accepted=falls==0
            attempts.append(dict(completed_iterations=completed,training_seconds=timing['experiment_training_seconds'],
                full_passed=True,low_speed_falls=falls,low_speed_passed=accepted,report=str(work/'report.json')))
            write(OUT/'supplemental_history.json',attempts)
        if accepted or completed>=MAXIMUM:break
        next_run=Path(str(RUN)+f'_cont{continuation_index+1}')
        learn(next_run,MAXIMUM-completed,run/'policy_final.pt');run=next_run
    assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==value for name,value in hashes.items()),'Frozen source changed during the experiment'
    first_full=next((row for row in history if row['passed']),None)
    first_all=next((row for row in attempts if row['low_speed_passed']),None)
    result=dict(accepted=accepted,completed_iterations=completed,num_envs=NUM_ENVS,checkpoint=str(run/'policy_final.pt'),
        first_complete_gate_pass=first_full,first_complete_and_low_speed_pass=first_all,
        training_seconds=timing['experiment_training_seconds'],acceptance_seconds=timing['acceptance_seconds']+extra_seconds,
        wall_seconds_including_setup=time.perf_counter()-launched,
        cumulative_training_seconds=BEFORE+timing['experiment_training_seconds'],
        started_from_random_weights=True,configuration_unchanged=True,acceptance_protocol='response_steady_v2',
        initialization_audit=str(OUT/'acceptance/initialization_audit.json'),supplemental_history=attempts)
    write(OUT/'result.json',result)
    write(OUT/'controller_status.json',dict(stage='complete',training=False,accepted=accepted,
        **{k:result[k] for k in ['completed_iterations','training_seconds','acceptance_seconds','cumulative_training_seconds','checkpoint']}))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'controller_status.json',dict(stage='error',training=False,error=str(error)));raise
