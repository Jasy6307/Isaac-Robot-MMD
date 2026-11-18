"""Normalization identity preflight and two bounded one-factor continuations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/pivot_ablation_preserved_20261116'
BASE=ROOT/'logs/rsl_rl/anon_locomotion/v8_scratch_unified_20261116_12288_cont1/scratch_007500.pt'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(p)


def execute(stage,arguments):
    write(OUT/'status.json',dict(stage=stage,started_unix=time.time()))
    start=time.perf_counter()
    with (OUT/f'{stage}.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid))
        code=process.wait()
    if code:raise RuntimeError(f'{stage}: exit {code}; see {OUT/stage}.log')
    return time.perf_counter()-start


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    durations={};results={};models={'baseline':str(BASE)}
    plan=dict(checkpoint=str(BASE),iterations_per_arm=100,num_envs=12288,seed=42,
              arms=['sampling_only','reward_only'],cumulative_training_seconds_before=45539.32527630008,
              base_lineage_training_seconds=42645.89903300011,
              continuation='Preserve checkpoint normalization and optimizer; both arms independently branch from 7500, same effective entropy/exploration settings.',
              scope='Matched pivot probes and lightweight walk/run regression only; no automatic longer training.',
              limitation='Single seed and short horizon; no-change training control not included.')
    write(OUT/'plan.json',plan)
    preflight=ROOT/'logs/rsl_rl/anon_locomotion/pivot_ablation_preserved_20261116_preflight'
    if preflight.exists():raise RuntimeError('Preflight already exists; refusing duplicate run')
    durations['preflight']=execute('preflight',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless',
        '--checkpoint',BASE,'--num-envs','256','--run-dir',preflight,'--resume-audit-only','--config-override',OUT/'sampling_only.json'])
    identity=read(preflight/'resume_identity_audit.json');write(OUT/'preflight_identity.json',identity)
    for role in ('actor','critic'):
        assert identity[role]['normalizer_identical'] and identity[role]['normalized_input_max_error']==0 and identity[role]['output_max_error']==0
    for arm in plan['arms']:
        run=ROOT/f'logs/rsl_rl/anon_locomotion/pivot_ablation_preserved_20261116_{arm}'
        if run.exists():raise RuntimeError(f'Run already exists: {run}')
        durations[arm+'_train']=execute(arm+'_train',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless',
            '--checkpoint',BASE,'--num-envs','12288','--iterations','100','--seed','42','--run-dir',run,
            '--config-override',OUT/(arm+'.json')])
        timing=read(run/'training_time.json');models[arm]=str(run/'policy_final.pt')
        results[arm]=dict(training_seconds=timing['training_seconds'],lineage_training_seconds=plan['base_lineage_training_seconds']+timing['training_seconds'],
                          resume_identity=read(run/'resume_identity_audit.json'))
        write(OUT/'result.json',dict(plan=plan,durations=durations,arms=results,models=models))
    write(OUT/'models.json',models)
    durations['pivot_probes']=execute('pivot_probes',[SOURCE/'diagnostics/diagnose_pivot_failure.py','--models',OUT/'models.json','--output',OUT/'pivot_probes'])
    probes=read(OUT/'pivot_probes/report.json')['models']
    write(OUT/'lite_config.json',dict(unified_eval_lite=True))
    reports={}
    for arm,checkpoint in models.items():
        durations[arm+'_lite']=execute(arm+'_lite',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
            '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/(arm+'_lite'),'--config-override',OUT/'lite_config.json'])
        reports[arm]=read(OUT/(arm+'_lite')/'report.json')
    baseline_falls=sum(row['total_falls'] for row in probes['baseline'])
    for arm,result in results.items():
        rows=probes[arm];result['pivot_probes']=rows
        result['pivot_total_falls']=sum(row['total_falls'] for row in rows)
        result['pivot_improved']=result['pivot_total_falls']<baseline_falls
        regressions=[]
        for old,new in zip(reports['baseline']['segments'],reports[arm]['segments']):
            assert old['label']==new['label']
            if old['label'].startswith('pivot'):continue
            passed=new['falls']==0 and (not old['passed'] or new['passed'])
            passed &= new['velocity_error']<=old['velocity_error']+.05 and new['yaw_error']<=old['yaw_error']+.05
            passed &= new['metrics']['Imitation/keypoint_rmse_m']<=old['metrics']['Imitation/keypoint_rmse_m']+.01
            if old['label'].startswith('run_'):
                passed &= new['mean_speed']>=old['mean_speed']*.95 and new['stance_slip_geometry_m_s']<=old['stance_slip_geometry_m_s']+.03
            regressions.append(dict(label=old['label'],passed=bool(passed),baseline=old,candidate=new))
        result['walk_run_regression']=regressions;result['no_walk_run_regression']=all(r['passed'] for r in regressions)
        result['lite_total_falls']=reports[arm]['total_falls']
        result['experiment_cumulative_training_seconds']=plan['cumulative_training_seconds_before']+sum(results[a]['training_seconds'] for a in plan['arms'][:plan['arms'].index(arm)+1])
    report=dict(plan=plan,durations=durations,arms=results,baseline_pivot_falls=baseline_falls,
                cumulative_training_seconds=plan['cumulative_training_seconds_before']+sum(row['training_seconds'] for row in results.values()),
                outcome='short_ablation_complete_no_further_training')
    write(OUT/'result.json',report);write(OUT/'status.json',dict(stage=report['outcome'],result=str(OUT/'result.json')))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',error=str(error)));raise
