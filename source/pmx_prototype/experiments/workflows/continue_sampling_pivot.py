"""Continue sampling-only policy with staged regression and full acceptance."""
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
OUT=ROOT/'output/anon_locomotion/sampling_pivot_continue_20261116'
BASE=ROOT/'logs/rsl_rl/anon_locomotion/pivot_ablation_preserved_20261116_sampling_only/policy_final.pt'
PREVIOUS=ROOT/'output/anon_locomotion/pivot_ablation_preserved_20261116'
CONFIG=PREVIOUS/'sampling_only.json'
STAGE_ITERATIONS=[200,100,100]
CUMULATIVE_BASE=None
EXACT_PREFIX_CONFIG=None




def execute(stage, args):
    return jobs.execute(ROOT, OUT, stage, args)


def regression(old,new):
    result=[]
    assert len(old['segments'])==len(new['segments'])
    for a,b in zip(old['segments'],new['segments']):
        assert a['label']==b['label']
        if a['label'].startswith('pivot'):continue
        passed=b['falls']==0 and (not a['passed'] or b['passed'])
        passed &= b['velocity_error']<=a['velocity_error']+.05 and b['yaw_error']<=a['yaw_error']+.05
        passed &= b['metrics']['Imitation/keypoint_rmse_m']<=a['metrics']['Imitation/keypoint_rmse_m']+.01
        if a['label'].startswith('run_'):
            passed &= b['mean_speed']>=a['mean_speed']*.95 and b['stance_slip_geometry_m_s']<=a['stance_slip_geometry_m_s']+.03
        result.append(dict(label=a['label'],passed=bool(passed),baseline=a,candidate=b))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Refusing duplicate controller')
    cfg=read(CONFIG)
    assert cfg['unified_pivot_sampling'] and not cfg['unified_pivot_rewards']
    assert cfg['train_audit_resume_identity']
    write(OUT/'config.json',cfg)
    earlier=read(PREVIOUS/'result.json')
    plan=dict(checkpoint=str(BASE),checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),num_envs=12288,
              stage_iterations=STAGE_ITERATIONS,maximum_new_iterations=sum(STAGE_ITERATIONS),
              cumulative_training_seconds_before=earlier['cumulative_training_seconds'] if CUMULATIVE_BASE is None else CUMULATIVE_BASE,
              base_lineage_training_seconds=earlier['arms']['sampling_only']['lineage_training_seconds'],
              gates='Zero falls across 64 pivot stress conditions; no walk/run regression versus the starting 7600 checkpoint; then original full 325-second gate and original slip/speed thresholds.',
              stop_rule='Stop on a passing complete gate, any walk/run regression, or the 400-iteration cap. No reward additions.')
    write(OUT/'plan.json',plan)
    baseline=read(PREVIOUS/'sampling_only_lite/report.json')
    slip_baseline={s['label']:s for s in read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
    checkpoint=BASE;stages=[];training=0.;durations={};new_iterations=0
    write(OUT/'lite_config.json',dict(unified_eval_lite=True))
    outcome='iteration_limit_not_passed'
    for index,iterations in enumerate(plan['stage_iterations'],1):
        new_iterations+=iterations
        tag=f'stage{index}';run=ROOT/f'logs/rsl_rl/anon_locomotion/{OUT.name}_{tag}'
        if run.exists():raise RuntimeError(f'Run exists: {run}')
        durations[tag+'_train']=execute(tag+'_train',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless',
            '--checkpoint',checkpoint,'--num-envs','12288','--iterations',iterations,'--seed','42','--run-dir',run,'--config-override',OUT/'config.json'])
        timing=read(run/'training_time.json');training+=timing['training_seconds'];checkpoint=run/'policy_final.pt'
        item=dict(checkpoint=str(checkpoint),new_iterations=new_iterations,stage_training_seconds=timing['training_seconds'],
                  round_training_seconds=training,experiment_cumulative_training_seconds=plan['cumulative_training_seconds_before']+training,
                  lineage_training_seconds=plan['base_lineage_training_seconds']+training,resume_identity=read(run/'resume_identity_audit.json'))
        stages.append(item)
        write(OUT/'result.json',dict(plan=plan,stages=stages,durations=durations,outcome='validating'))
        write(OUT/(tag+'_models.json'),dict(candidate=str(checkpoint)))
        durations[tag+'_pivots']=execute(tag+'_pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py','--models',OUT/(tag+'_models.json'),'--output',OUT/tag/'pivots'])
        item['pivot_probes']=read(OUT/tag/'pivots/report.json')['models']['candidate']
        item['pivot_total_falls']=sum(r['total_falls'] for r in item['pivot_probes'])
        durations[tag+'_lite']=execute(tag+'_lite',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
            '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/tag/'lite','--config-override',OUT/'lite_config.json'])
        item['regression']=regression(baseline,read(OUT/tag/'lite/report.json'))
        item['no_walk_run_regression']=all(r['passed'] for r in item['regression'])
        if not item['no_walk_run_regression']:
            outcome='walk_run_regression_stopped';break
        prefix_passed=True
        if EXACT_PREFIX_CONFIG is not None:
            durations[tag+'_prefix']=execute(tag+'_prefix',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
                '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/tag/'prefix','--config-override',EXACT_PREFIX_CONFIG])
            prefix=read(OUT/tag/'prefix/report.json')
            reverse=next(row for row in prefix['segments'] if row['label']=='reverse_turn')
            old_reverse=next(row for row in baseline['segments'] if row['label']=='reverse_turn')
            prefix_passed=bool(prefix['total_falls']==0 and reverse['velocity_error']<=old_reverse['velocity_error']+.05 and
                               reverse['metrics']['Imitation/keypoint_rmse_m']<=old_reverse['metrics']['Imitation/keypoint_rmse_m']+.01)
            item['exact_prefix_passed']=prefix_passed;item['exact_prefix_reverse']=reverse
        if item['pivot_total_falls']==0 and prefix_passed:
            durations[tag+'_full']=execute(tag+'_full',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
                '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/tag/'full'])
            report=read(OUT/tag/'full/report.json')
            checks=[]
            for row in report['segments']:
                if row['label'] in ('run_backward','run_left','run_right'):
                    old=slip_baseline[row['label']]
                    checks.append(dict(label=row['label'],passed=row['mean_speed']>=old['mean_speed']*.95 and row['stance_slip_geometry_m_s']<=old['stance_slip_geometry_m_s']*.8))
            item['full_report']=str(OUT/tag/'full/report.json');item['slip_checks']=checks
            item['full_passed']=bool(report['passed_unified_gate'] and report['total_falls']==0 and all(c['passed'] for c in checks))
            if item['full_passed']:outcome='passed';break
        write(OUT/'result.json',dict(plan=plan,stages=stages,durations=durations,outcome='continuing'))
    write(OUT/'result.json',dict(plan=plan,stages=stages,durations=durations,outcome=outcome,
        round_training_seconds=training,cumulative_training_seconds=plan['cumulative_training_seconds_before']+training,
        lineage_training_seconds=plan['base_lineage_training_seconds']+training))
    write(OUT/'status.json',dict(stage=outcome,result=str(OUT/'result.json')))


