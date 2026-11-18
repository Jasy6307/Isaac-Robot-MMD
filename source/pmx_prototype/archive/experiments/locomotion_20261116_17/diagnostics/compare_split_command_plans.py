"""Fixed-policy acceleration/turn/braking tests. No training or promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/split_command_tests_20261117'
CHECKPOINT=ROOT/'logs/rsl_rl/anon_locomotion/retention_strength_20261116_original/policy_final.pt'
PREVIOUS=ROOT/'output/anon_locomotion/command_plan_tests_20261116'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def execute(stage,args):
    write(OUT/'status.json',dict(stage=stage,training=False,started_unix=time.time()))
    start=time.perf_counter()
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid))
        if process.wait():raise RuntimeError(stage+' failed; inspect its log')
    return time.perf_counter()-start


def evaluate(name,config):
    write(OUT/(name+'_config.json'),config)
    seconds=execute(name,[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless','--num-envs','1',
        '--checkpoint',CHECKPOINT,'--output',OUT/name,'--config-override',OUT/(name+'_config.json')])
    return read(OUT/name/'report.json'),seconds


def timing(name):
    data=np.load(OUT/name/'rollout.npz');dt=float(data['dt']);result={}
    for label in ['run_diagonal_135','reverse_turn']:
        ix=np.flatnonzero(data['labels']==label);velocity=data['velocities'][ix,0,0]
        target=float(data['commands'][ix[-1],0,0]);window=round(.5/dt)
        filtered=np.convolve(velocity,np.ones(window)/window,mode='valid')
        reached=np.flatnonzero(filtered<=target*.9)
        result[label]=dict(actual_forward_90_percent_seconds=float((reached[0]+window-1)*dt) if len(reached) else None,
            filter='Trailing .5-second mean')
    return result


def full_checks(report,baseline):
    old={s['label']:s for s in read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
    checks=[]
    for segment in report['segments']:
        if segment['label'] not in ['run_backward','run_left','run_right']:continue
        a=old[segment['label']]
        speed=segment['mean_speed']/a['mean_speed'];slip=segment['stance_slip_geometry_m_s']/a['stance_slip_geometry_m_s']
        checks.append(dict(label=segment['label'],speed_ratio=speed,slip_ratio=slip,passed=speed>=.95 and slip<=.8))
    # Retain the original full gate and the prior slip/speed gate, and also
    # check the requested stop/backward regressions against this fixed actor.
    regressions=regression(baseline,report)
    protected=[s for s in regressions if s['label'] in ['stop','stand','stand_end','run_backward','reverse_turn']]
    return dict(original_full_passed=report['passed_unified_gate'],total_falls=report['total_falls'],
        passed_segments=sum(s['passed'] for s in report['segments']),slip_speed_checks=checks,
        stop_backward_checks=protected,
        accepted=bool(report['passed_unified_gate'] and report['total_falls']==0 and
                      all(s['passed'] for s in checks) and all(s['passed'] for s in protected)))


sys.path.insert(0,str(SOURCE/'training'))
from continue_sampling_pivot import regression


def main():
    if (OUT/'plan.json').exists():raise RuntimeError('Duplicate experiment; choose a new output folder')
    start=time.perf_counter();sha=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    variants={'baseline':{},'split_120':dict(command_plan_mode='split',command_speedup_limit=1.2,
        command_direction_change_limit=1.2,command_braking_limit=.6),
        'split_090':dict(command_plan_mode='split',command_speedup_limit=.9,command_direction_change_limit=.9,command_braking_limit=.6)}
    brakes=[('stand',4,[0,0,0]),('run_forward',8,[1.2,0,0]),('stop',5,[0,0,0]),
        ('run_backward',8,[-1.2,0,0]),('stop',5,[0,0,0]),('run_turn_left',8,[1.1,0,.6]),('stop',5,[0,0,0]),
        ('run_turn_right',8,[1.1,0,-.6]),('stop',5,[0,0,0]),('reverse_turn',8,[-.9,0,.6]),('stop',5,[0,0,0]),('stand_end',4,[0,0,0])]
    plan=dict(training=False,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha,variants=variants,brake_cases=brakes,
        unchanged='Actor, normalizers, reference blending, physics, rewards, yaw planner and all original acceptance thresholds.',
        promotion='No default/eval UI changes; experimental opt-in configuration only.',
        selection='Both target switches pass; zero falls; no regression versus fixed-actor baseline in all non-pivot lite cases and all brake cases. Then zero falls in 64 pivot probes, original full gate and slip/speed gate, plus stop/backward regression checks.',
        cumulative_training_seconds=51378.49378980009)
    write(OUT/'plan.json',plan);reports={};brake_reports={};rows={};durations={}
    for name,delta in variants.items():
        report,seconds=evaluate(name+'_lite',dict(unified_eval_lite=True,unified_record_controller=True,**delta))
        reports[name]=report;durations[name+'_lite']=seconds
        brake,seconds=evaluate(name+'_brakes',dict(unified_eval_cases=brakes,unified_record_controller=True,**delta))
        brake_reports[name]=brake;durations[name+'_brakes']=seconds
        targets={s['label']:s for s in report['segments'] if s['label'] in ['run_diagonal_135','reverse_turn']}
        checks=regression(reports['baseline'],report);brake_checks=regression(brake_reports['baseline'],brake)
        rows[name]=dict(targets=targets,response=timing(name+'_lite'),lite_passed=report['passed_lite_gate'],
            lite_passed_segments=sum(s['passed'] for s in report['segments']),lite_falls=report['total_falls'],
            lite_regression=checks,brakes_passed=all(s['passed'] for s in brake['segments']),brakes_falls=brake['total_falls'],brake_regression=brake_checks)
        rows[name]['eligible']=bool(name!='baseline' and report['total_falls']==0 and brake['total_falls']==0 and
            all(s['passed'] for s in targets.values()) and all(s['passed'] for s in checks) and
            all(s['passed'] for s in brake_checks) and rows[name]['brakes_passed'])
        write(OUT/'result.json',dict(plan=plan,rows=rows,durations=durations,outcome='screening',training=False))
    candidates=sorted([n for n in variants if rows[n]['eligible']],key=lambda n:sum(s['velocity_error'] for s in rows[n]['targets'].values()))
    winner=None
    baseline_full=read(PREVIOUS/'baseline_full/report.json')
    for name in candidates:
        write(OUT/'pivot_models.json',dict(candidate=str(CHECKPOINT)))
        durations[name+'_pivots']=execute(name+'_pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py',
            '--models',OUT/'pivot_models.json','--output',OUT/(name+'_pivots'),'--config-override',OUT/(name+'_lite_config.json')])
        pivots=read(OUT/(name+'_pivots/report.json'));rows[name]['pivot_probe']=pivots
        if sum(s['total_falls'] for s in pivots['models']['candidate'])!=0:continue
        full,seconds=evaluate(name+'_full',dict(unified_record_controller=True,**variants[name]));durations[name+'_full']=seconds
        assert full['evaluator_sha256']==baseline_full['evaluator_sha256']
        rows[name]['full_checks']=full_checks(full,baseline_full)
        write(OUT/'result.json',dict(plan=plan,rows=rows,durations=durations,outcome='full_validation',training=False))
        if rows[name]['full_checks']['accepted']:winner=name;break
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==sha
    write(OUT/'result.json',dict(plan=plan,rows=rows,durations=durations,winner=winner,checkpoint_unchanged=True,
        training=False,training_seconds=0,cumulative_training_seconds=plan['cumulative_training_seconds'],
        wall_seconds=time.perf_counter()-start,outcome='complete'))
    write(OUT/'status.json',dict(stage='complete',training=False,winner=winner))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',error=str(error),training=False));raise
