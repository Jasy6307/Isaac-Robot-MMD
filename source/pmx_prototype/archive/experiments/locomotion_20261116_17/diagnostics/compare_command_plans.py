"""Fixed-policy, single-parameter command-plan comparison. Never trains."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/command_plan_tests_20261116'
CHECKPOINT=ROOT/'logs/rsl_rl/anon_locomotion/retention_strength_20261116_original/policy_final.pt'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(p)


def execute(stage,args):
    write(OUT/'status.json',dict(stage=stage,training=False))
    start=time.perf_counter()
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=child.pid))
        if child.wait():raise RuntimeError(stage+' failed; see log')
    return time.perf_counter()-start


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Duplicate experiment')
    start=time.perf_counter();before_hash=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    variants={'baseline':{},'tau_020':{'command_smoothing_seconds':.20},
              'accel_090':{'command_linear_acceleration_limit':.9},'accel_120':{'command_linear_acceleration_limit':1.2}}
    plan=dict(training=False,checkpoint=str(CHECKPOINT),checkpoint_sha256=before_hash,variants=variants,
              unchanged='Reference blend formulas, weights, physics, rewards, yaw acceleration and original acceptance thresholds.',
              baseline=dict(command_smoothing_seconds=.35,command_linear_acceleration_limit=.6,command_yaw_acceleration_limit=1.2),
              promotion='No production/default update; any winning plan is only a validated diagnostic candidate.')
    write(OUT/'plan.json',plan)
    reports={};rows={};durations={}
    sys.path.insert(0,str(SOURCE/'training'));from continue_sampling_pivot import regression
    for name,delta in variants.items():
        config=dict(unified_eval_lite=True,unified_record_controller=True,**delta)
        write(OUT/(name+'_config.json'),config)
        durations[name]=execute(name,[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless','--num-envs','1',
            '--checkpoint',CHECKPOINT,'--output',OUT/name,'--config-override',OUT/(name+'_config.json')])
        report=read(OUT/name/'report.json');reports[name]=report
        targets={s['label']:s for s in report['segments'] if s['label'] in ('run_diagonal_135','reverse_turn')}
        checks=regression(reports['baseline'],report)
        rows[name]=dict(delta=delta,total_falls=report['total_falls'],screen_passed=report['passed_lite_gate'],
                        passed_segments=sum(s['passed'] for s in report['segments']),targets=targets,
                        regression=checks,no_regression=all(s['passed'] for s in checks))
        write(OUT/'result.json',dict(plan=plan,rows=rows,durations=durations,outcome='running'))
    candidates=[name for name in variants if name!='baseline' and rows[name]['no_regression'] and rows[name]['total_falls']==0 and
                all(s['passed'] for s in rows[name]['targets'].values())]
    winner=None
    if candidates:
        winner=min(candidates,key=lambda n:sum(s['velocity_error'] for s in rows[n]['targets'].values()))
        write(OUT/'pivot_models.json',dict(candidate=str(CHECKPOINT)))
        durations['candidate_pivots']=execute('candidate_pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py',
            '--models',OUT/'pivot_models.json','--output',OUT/'candidate_pivots','--config-override',OUT/(winner+'_config.json')])
        probe=read(OUT/'candidate_pivots/report.json');rows[winner]['pivot_probe']=probe
        if sum(s['total_falls'] for s in probe['models']['candidate'])==0:
            write(OUT/'full_config.json',dict(unified_record_controller=True,**variants[winner]))
            durations['candidate_full']=execute('candidate_full',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
                '--num-envs','1','--checkpoint',CHECKPOINT,'--output',OUT/'candidate_full','--config-override',OUT/'full_config.json'])
            full=read(OUT/'candidate_full/report.json')
            old={s['label']:s for s in read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
            slip=all(s['mean_speed']>=old[s['label']]['mean_speed']*.95 and s['stance_slip_geometry_m_s']<=old[s['label']]['stance_slip_geometry_m_s']*.8
                     for s in full['segments'] if s['label'] in ['run_backward','run_left','run_right'])
            rows[winner]['full_passed']=bool(full['passed_unified_gate'] and full['total_falls']==0 and slip)
            rows[winner]['full_report']=str(OUT/'candidate_full/report.json')
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==before_hash
    write(OUT/'result.json',dict(plan=plan,rows=rows,durations=durations,winner=winner,training=False,
        checkpoint_unchanged=True,wall_seconds=time.perf_counter()-start,outcome='complete'))
    write(OUT/'status.json',dict(stage='complete',training=False))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',error=str(error)));raise
