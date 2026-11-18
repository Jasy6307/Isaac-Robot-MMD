"""Matched retention-strength continuations; unchanged screen/regression gates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/retention_strength_20261116'
BASE=ROOT/'logs/rsl_rl/anon_locomotion/pivot_ablation_preserved_20261116_sampling_only/policy_final.pt'
WEIGHTS={'weak':.5,'medium':1.,'original':2.}


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(p)


def execute(stage,arguments):
    write(OUT/'status.json',dict(stage=stage,started_unix=time.time()))
    start=time.perf_counter()
    with (OUT/f'{stage}.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid))
        code=process.wait()
    if code:raise RuntimeError(f'{stage} failed ({code}); see log')
    return time.perf_counter()-start


def switch_metrics(folder):
    import numpy as np
    data=np.load(folder/'rollout.npz');dt=float(data['dt']);result={}
    for label in ['run_diagonal_135','reverse_turn']:
        ids=np.flatnonzero(data['labels']==label);times=np.arange(len(ids))*dt
        # Reset frames may contain the reset command; the evaluator compares
        # against the requested segment command even when a fall occurs.
        command=[-1.2/np.sqrt(2),1.2/np.sqrt(2)] if label=='run_diagonal_135' else [-.9,0.]
        target=np.broadcast_to(command,(len(ids),2));actual=data['velocities'][ids,0,:2]
        row={}
        for name,start in [('original_window',2.),('last_second',max(0.,len(ids)*dt-1.))]:
            error=actual[times>=start]-target[times>=start]
            row[name]=float(np.linalg.norm(error.mean(0)))
        row['falls']=int(data['dones'][ids].sum())
        result[label]=row
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Refusing duplicate experiment')
    started=time.perf_counter()
    template=read(ROOT/'output/anon_locomotion/transition_retention_20261116/prepared_config.json')
    assert template['unified_pivot_sampling'] and not template['unified_pivot_rewards']
    plan=dict(checkpoint=str(BASE),sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),weights=WEIGHTS,
              iterations_each=100,num_envs=12288,seed=42,teacher=template['train_retention_teacher_checkpoint'],
              cumulative_training_seconds_before=48650.871125100064,base_lineage_training_seconds=43407.43916590011,
              only_changed_config_key='train_retention_weight',retention_steps=template['train_retention_steps'],
              limitations='One seed per strength; last-second errors are diagnostic only, never replace original gates.',
              scope='Short training comparison only; no automatic additional training or promotion.')
    write(OUT/'plan.json',plan)
    durations={};arms={};models={'baseline':str(BASE)}
    for name,weight in WEIGHTS.items():
        cfg=dict(template,train_retention_weight=weight)
        assert {k for k in cfg if cfg[k]!=template[k]}<= {'train_retention_weight'}
        write(OUT/(name+'_config.json'),cfg)
        run=ROOT/f'logs/rsl_rl/anon_locomotion/retention_strength_20261116_{name}'
        if run.exists():raise RuntimeError('Existing run: '+str(run))
        durations[name+'_train']=execute(name+'_train',[SOURCE/'train_anon_locomotion.py','--mode','train','--headless',
            '--checkpoint',BASE,'--num-envs','12288','--iterations','100','--seed','42','--run-dir',run,
            '--config-override',OUT/(name+'_config.json')])
        timing=read(run/'training_time.json');identity=read(run/'resume_identity_audit.json')
        for row in identity.values():assert row['normalizer_identical'] and row['output_max_error']==0 and row['normalized_input_max_error']==0
        arms[name]=dict(weight=weight,training_seconds=timing['training_seconds'],resume_identity=identity,
                       lineage_training_seconds=plan['base_lineage_training_seconds']+timing['training_seconds'])
        models[name]=str(run/'policy_final.pt')
        write(OUT/'result.json',dict(plan=plan,arms=arms,durations=durations,models=models,outcome='running'))
    write(OUT/'models.json',models)
    durations['pivots']=execute('pivots',[SOURCE/'diagnostics/diagnose_pivot_failure.py','--models',OUT/'models.json','--output',OUT/'pivots'])
    pivots=read(OUT/'pivots/report.json')['models']
    write(OUT/'lite_config.json',dict(unified_eval_lite=True,unified_record_controller=True))
    reports={};responses={}
    for name,checkpoint in models.items():
        durations[name+'_lite']=execute(name+'_lite',[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless',
            '--checkpoint',checkpoint,'--num-envs','1','--output',OUT/(name+'_lite'),'--config-override',OUT/'lite_config.json'])
        reports[name]=read(OUT/(name+'_lite')/'report.json');responses[name]=switch_metrics(OUT/(name+'_lite'))
    # Reuse precisely the established regression tolerances and official rows.
    from continue_sampling_pivot import regression
    for name,arm in arms.items():
        arm['pivot_probes']=pivots[name];arm['pivot_total_falls']=sum(row['total_falls'] for row in pivots[name])
        arm['switch_response']=responses[name];arm['walk_run_regression']=regression(reports['baseline'],reports[name])
        arm['walk_run_regression_passes']=sum(row['passed'] for row in arm['walk_run_regression'])
        arm['no_walk_run_regression']=all(row['passed'] for row in arm['walk_run_regression'])
        arm['screen_total_falls']=reports[name]['total_falls'];arm['screen_passed']=reports[name]['passed_lite_gate']
        arm['screen_segment_passes']=sum(row['passed'] for row in reports[name]['segments'])
        arm['original_gates_passed']=bool(arm['pivot_total_falls']==0 and arm['no_walk_run_regression'] and arm['screen_passed'])
    cumulative=plan['cumulative_training_seconds_before']
    for arm in arms.values():
        cumulative+=arm['training_seconds'];arm['experiment_cumulative_training_seconds']=cumulative
    report=dict(plan=plan,arms=arms,durations=durations,baseline_pivots=pivots['baseline'],baseline_response=responses['baseline'],
                total_new_training_seconds=sum(row['training_seconds'] for row in arms.values()),
                cumulative_training_seconds=cumulative,wall_seconds=time.perf_counter()-started,
                outcome='comparison_complete_no_further_training')
    write(OUT/'result.json',report);write(OUT/'status.json',dict(stage=report['outcome'],result=str(OUT/'result.json')))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',error=str(error)));raise
