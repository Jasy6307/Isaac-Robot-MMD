"""Inference-only repeatability and steady-window tracking diagnostics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/tracking_margin_diagnosis_20261116'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def execute(stage,checkpoint,config):
    write(OUT/'status.json',dict(stage=stage,training=False))
    args=[sys.executable,'-u',str(SOURCE/'train_anon_locomotion.py'),'--mode','eval','--headless',
          '--num-envs','1','--checkpoint',checkpoint,'--output',str(OUT/stage),'--config-override',str(config)]
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=p.pid))
        if p.wait():raise RuntimeError(stage+' failed')


def main():
    started=time.perf_counter()
    sys.path.insert(0,str(SOURCE));import source_layout
    from evaluate_unified_walk_run import suite
    models=read(OUT/'models.json');rows=[]
    write(OUT/'repeat_config.json',dict(unified_eval_lite=True,unified_record_controller=True))
    execute('repeat_after',models['after'],OUT/'repeat_config.json')
    rows.append(dict(stage='repeat_after',model='after',report=read(OUT/'repeat_after/report.json')))
    for label,seconds in [('run_diagonal_135',8),('reverse_turn',10)]:
        cases=[]
        for item in suite(lite=True):
            cases.append([item[0],seconds if item[0]==label else item[1],item[2]])
            if item[0]==label:break
        config=OUT/(label+'_config.json');write(config,dict(unified_eval_cases=cases,unified_record_controller=True))
        for name,checkpoint in models.items():
            stage=label+'_'+name;execute(stage,checkpoint,config)
            rows.append(dict(stage=stage,model=name,target=label,report=read(OUT/stage/'report.json')))
            write(OUT/'partial_results.json',rows)
    write(OUT/'result.json',dict(training=False,wall_seconds=time.perf_counter()-started,results=rows))
    write(OUT/'status.json',dict(stage='complete',training=False))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as e:
        write(OUT/'status.json',dict(stage='error',error=str(e)));raise
