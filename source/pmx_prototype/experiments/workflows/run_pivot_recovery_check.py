"""Bounded 300-iteration recovery, matched diagnostics, then gated full acceptance."""
from source.paths import REPO_ROOT
from source.pmx_prototype.experiments import jobs
from source.pmx_prototype.experiments.jobs import read, write
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(REPO_ROOT)
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/pivot_recovery_short_20261116'
RUN=ROOT/'logs/rsl_rl/anon_locomotion/pivot_recovery_short_20261116'
BASE=ROOT/'logs/rsl_rl/anon_locomotion/v8_scratch_unified_20261116_12288_cont1/scratch_007500.pt'




def execute(stage, arguments):
    return jobs.execute(ROOT, OUT, stage, arguments)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if RUN.exists():raise RuntimeError('Run directory exists; refusing accidental duplicate training')
    cfg=read(SOURCE/'configs/training/unified_pivot_recovery.json')
    # Preserve the 7500-iteration exploration state: do not reset leg std.
    cfg.pop('train_leg_std',None)
    cfg['train_entropy_coef']=.0005
    write(OUT/'config.json',cfg)
    plan=dict(checkpoint=str(BASE),checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
              iterations=300,num_envs=12288,expected_completed_iterations=7800,
              pivot_gate='Zero falls in all four 16-condition probes and lower total falls than matched baseline',
              regression_gate='Zero non-pivot falls; no previously passing non-pivot segment fails; speed/yaw/keypoint degradation within .05 m/s/.05 rad/s/.01 m',
              full_gate='Original 325-second unified gate, zero falls, at least 20 percent slip reduction and 95 percent speed retention versus original slip baseline',
              failure_action='Stop and report; no automatic extra training')
    write(OUT/'plan.json',plan)
    durations={}
    durations['training_job']=execute('train',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless','--no-acceptance-video',
        '--checkpoint',BASE,'--num-envs','12288','--iterations','300','--seed','42','--run-dir',RUN,
        '--config-override',OUT/'config.json'])
    candidate=RUN/'policy_final.pt'
    timing=read(RUN/'training_time.json')
    result=dict(plan=plan,checkpoint=str(candidate),training_seconds=timing['training_seconds'],
                lineage_training_seconds=42645.89903300011+timing['training_seconds'],
                cumulative_experiment_training_seconds=43541.45205180009+timing['training_seconds'],
                cumulative_note='Experiment total includes 895.553 seconds stopped at 7660, which this branch does not inherit.',
                durations=durations)
    write(OUT/'result.json',result)
    write(OUT/'models.json',dict(baseline=str(BASE),candidate=str(candidate)))
    durations['pivot_diagnostics']=execute('pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py',
        '--models',OUT/'models.json','--output',OUT/'pivots'])
    probes=read(OUT/'pivots/report.json')['models']
    totals={key:sum(item['total_falls'] for item in rows) for key,rows in probes.items()}
    result['pivot_falls']=totals
    result['pivot_gate']=totals['candidate']==0 and totals['candidate']<totals['baseline']
    for key,checkpoint in [('baseline_full',BASE),('full',candidate)]:
        durations[key]=execute(key,[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
            '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/key])
    old=read(OUT/'baseline_full/report.json');new=read(OUT/'full/report.json')
    regression=[]
    for a,b in zip(old['segments'],new['segments']):
        assert a['label']==b['label']
        if a['label'].startswith('pivot'):continue
        passed=b['falls']==0 and (not a['passed'] or b['passed'])
        for key,tolerance in [('velocity_error',.05),('yaw_error',.05)]:
            passed &= b[key]<=a[key]+tolerance
        passed &= b['metrics']['Imitation/keypoint_rmse_m']<=a['metrics']['Imitation/keypoint_rmse_m']+.01
        if a['label'].startswith('run_'):
            passed &= b['mean_speed']>=a['mean_speed']*.95
            passed &= b['stance_slip_geometry_m_s']<=a['stance_slip_geometry_m_s']+.03
        regression.append(dict(label=a['label'],passed=bool(passed),baseline=a,candidate=b))
    result['regression']=regression
    result['regression_gate']=all(item['passed'] for item in regression)
    write(OUT/'result.json',result)
    if not (result['pivot_gate'] and result['regression_gate']):
        result['outcome']='pivot_or_regression_check_failed_no_further_training'
        write(OUT/'result.json',result);write(OUT/'status.json',dict(stage=result['outcome'],result=str(OUT/'result.json')))
        return
    full=new
    original=read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')
    by_label={row['label']:row for row in original['segments']}
    checks=[]
    for row in full['segments']:
        if row['label'] in ['run_backward','run_left','run_right']:
            previous=by_label[row['label']]
            checks.append(dict(label=row['label'],passed=row['mean_speed']>=previous['mean_speed']*.95 and
                row['stance_slip_geometry_m_s']<=previous['stance_slip_geometry_m_s']*.8))
    result['full_passed']=bool(full['passed_unified_gate'] and full['total_falls']==0 and all(c['passed'] for c in checks))
    result['slip_checks']=checks
    result['outcome']='passed' if result['full_passed'] else 'full_acceptance_failed_no_further_training'
    write(OUT/'result.json',result);write(OUT/'status.json',dict(stage=result['outcome'],result=str(OUT/'result.json')))


