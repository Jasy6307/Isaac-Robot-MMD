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
RUN_TO_MAXIMUM=False


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def acceptance_durations(config):
    if str(SOURCE) not in sys.path:
        sys.path.insert(0,str(SOURCE))
    from source_layout import SOURCE_DIR
    from evaluate_unified_walk_run import LOW_SPEED_SECONDS,suite
    from locomotion_acceptance import CURRENT_PROTOCOL,RESPONSE_STEADY_PROTOCOLS
    protocol=config['unified_acceptance_protocol']
    idle_head=config.get('unified_idle_head_acceptance',protocol in RESPONSE_STEADY_PROTOCOLS)
    backward_walk=config.get('unified_eval_backward_walk',False)
    low_speed_seconds=LOW_SPEED_SECONDS if protocol==CURRENT_PROTOCOL and config.get('unified_eval_low_speed',False) else 0
    return dict(
        complete_suite_seconds=sum(seconds for _,seconds,_ in suite(
            protocol=protocol,idle_head=idle_head,backward_walk=backward_walk,
            stand_seconds=config.get('unified_eval_stand_hold_seconds'),
            stop_timing=config.get('unified_eval_stop_timing')))+low_speed_seconds)


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


def learn(run,iterations,checkpoint=None,*,latest_protocol=False):
    config=OUT/('continuation_config.json' if checkpoint else 'config.json')
    arguments=[SOURCE/'train_anon_locomotion.py','--mode','train','--headless','--num-envs',str(NUM_ENVS),
        '--iterations',str(iterations),'--seed','42','--run-dir',run,'--experiment-output',OUT/'acceptance',
        '--scratch-target','latest_unified',
        '--config-override',config]
    arguments+=['--scratch-continuation','--checkpoint',checkpoint] if checkpoint else ['--from-scratch']
    if RUN_TO_MAXIMUM:
        arguments.append('--scratch-continue-after-pass')
        if not latest_protocol:
            arguments.append('--scratch-low-speed-after-full-pass')
    return execute('train_'+run.name,arguments,run)


def main(config=None):
    if (OUT/'experiment_plan.json').exists() or RUN.exists():raise RuntimeError('Fresh experiment output already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    if config is None:
        config=read(RECIPE)
    durations=acceptance_durations(config)
    from anon_scratch_validation import DEFAULT_ACCEPTANCE_INTERVAL
    from locomotion_acceptance import CURRENT_PROTOCOL,RESPONSE_STEADY_PROTOCOLS
    protocol=config['unified_acceptance_protocol']
    assert protocol in RESPONSE_STEADY_PROTOCOLS
    latest=protocol==CURRENT_PROTOCOL
    integrated=latest and config.get('unified_eval_low_speed',False)
    assert not any(config.get(k) for k in ['train_retention_weight','train_retention_teacher_checkpoint','train_teacher_anchor_path'])
    write(OUT/'config.json',config)
    continuation=dict(config);continuation.pop('train_leg_std',None)
    write(OUT/'continuation_config.json',continuation)
    files=['train_anon_locomotion.py','training/anon_scratch_validation.py','training/low_speed_commands.py',
        'training/locomotion_acceptance.py','training/command_plan.py','training/anon_imitation_env.py',
        'motion/unified_walk_run_reference.py','diagnostics/evaluate_unified_walk_run.py','diagnostics/evaluate_parallel_walk_run.py',
        'diagnostics/anon_acceptance_video.py','diagnostics/record_gait_diagnostic.py']
    hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    plan=dict(started_from_random_weights=True,pretrained_checkpoint=None,external_teacher=None,num_envs=NUM_ENVS,seed=42,
        maximum_iterations=MAXIMUM,full_interval=DEFAULT_ACCEPTANCE_INTERVAL,steps_per_env_per_iteration=48,
        recipe=str(RECIPE),recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),source_hashes=hashes,
        asset=config.get('asset'),body_mass_profile=config.get('body_mass_profile'),
        acceptance_protocol=protocol,**durations,
        full_gate='Complete response/steady-state suite, zero falls, slip reduction and speed retention.',
        low_speed_check=('Part of full acceptance. Four starting phase/heading pairs at each speed endpoint, zero falls.'
                         if integrated else (None if latest else
                         'Separate six-speed recovery check across 16 initial phase/heading states. Zero falls.')),
        milestone_rule='Record the first complete-gate pass and the result of any required supplemental checks.',
        stop_rule=(f'Train through iteration {MAXIMUM}, even if acceptance passes earlier.' if RUN_TO_MAXIMUM
                   else f'Stop at the first acceptance pass, or iteration {MAXIMUM}.'),
        time_rule='Actual training-loop wall time including shared GPU contention; excludes initialization and all acceptance. Report acceptance and wall time separately.',
        shared_gpu_authorized=True)
    if BEFORE is not None:
        plan['cumulative_training_seconds_before']=BEFORE
    write(OUT/'experiment_plan.json',plan)
    launched=time.perf_counter();run=RUN;extra_seconds=0.;attempts=[];accepted=False;stopped=False
    video_summaries=[]
    learn(run,MAXIMUM,latest_protocol=latest)
    for continuation_index in range(20):
        timing=read(run/'training_time.json')
        history_path=OUT/'acceptance/history.json'
        history=read(history_path) if history_path.is_file() else []
        last=history[-1] if history else None
        video_summary=run/'acceptance_videos.json'
        if video_summary.is_file():video_summaries.append(video_summary)
        completed=timing['completed_iterations']
        if (run/'stop.request').is_file():
            stopped=True
            accepted=bool(last and last['completed_iterations']==completed and last.get('combined_passed',last['passed']))
            break
        if last is None:
            raise RuntimeError(f'Training ended at iteration {completed} without a complete acceptance report')
        if latest or RUN_TO_MAXIMUM:
            if ((RUN_TO_MAXIMUM and completed!=MAXIMUM) or not last.get('full_report') or
                    ((integrated or not latest) and not last.get('low_speed'))):
                raise RuntimeError(f'Run stopped at iteration {completed} without completing the fixed-budget validation')
            for row in history:
                low=row.get('low_speed')
                if low:
                    attempts.append(dict(completed_iterations=row['completed_iterations'],
                        training_seconds=row['active_training_seconds'],full_passed=row['passed'],
                        low_speed_falls=low['falls'],low_speed_passed=low['passed'],report=low.get('report',row['report'])))
            write(OUT/'supplemental_history.json',attempts)
            accepted=last.get('combined_passed',last['passed'])
            break
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
    first_all=next((row for row in attempts if row['full_passed'] and row['low_speed_passed']),None)
    result=dict(accepted=accepted,completed_iterations=completed,num_envs=NUM_ENVS,checkpoint=str(run/'policy_final.pt'),
        first_complete_gate_pass=first_full,first_complete_and_low_speed_pass=first_all,
        training_seconds=timing['experiment_training_seconds'],acceptance_seconds=timing['acceptance_seconds']+extra_seconds,
        wall_seconds_including_setup=time.perf_counter()-launched,
        started_from_random_weights=True,configuration_unchanged=True,acceptance_protocol=protocol,
        initialization_audit=str(OUT/'acceptance/initialization_audit.json'),supplemental_history=attempts,
        final_acceptance=last,recording_seconds=sum(read(p)['recording_seconds'] for p in video_summaries),
        acceptance_video_summaries=[str(p) for p in video_summaries])
    if BEFORE is not None:
        result['cumulative_training_seconds']=BEFORE+timing['experiment_training_seconds']
    write(OUT/'result.json',result)
    write(OUT/'controller_status.json',dict(stage='stopped' if stopped else 'complete',training=False,accepted=accepted,
        **{k:result[k] for k in ['completed_iterations','training_seconds','acceptance_seconds','cumulative_training_seconds','checkpoint'] if k in result}))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'controller_status.json',dict(stage='error',training=False,error=str(error)));raise
