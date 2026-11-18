"""Inference-only repeatability and matched reverse-turn phase probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
OUT=ROOT/'output/anon_locomotion/reverse_turn_diagnosis_20261116'


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def execute(stage,arguments):
    write(OUT/'status.json',dict(stage=stage,training=False))
    with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_process.json',dict(stage=stage,pid=process.pid))
        if process.wait():raise RuntimeError(stage+' failed')


def main():
    models=read(OUT/'models.json');rows=[];started=time.perf_counter()
    for repeat,order in [(1,['before','after']),(2,['after','before'])]:
        for name in order:
            stage=f'repeat{repeat}_{name}'
            execute(stage,[SOURCE/'train_anon_locomotion.py','--mode','eval','--headless','--num-envs','1',
                '--checkpoint',models[name],'--output',OUT/stage,'--config-override',OUT/'prefix_config.json'])
            report=read(OUT/stage/'report.json');segment=next(s for s in report['segments'] if s['label']=='reverse_turn')
            rows.append(dict(repeat=repeat,model=name,segment=segment,total_falls=report['total_falls'],report=str(OUT/stage/'report.json')))
            write(OUT/'repeat_results.json',rows)
    schedule={
      'stand_reverse_left':[[2,[0,0,0]],[8,[-.9,0,.6]]],
      'stand_reverse_right':[[2,[0,0,0]],[8,[-.9,0,-.6]]],
      'right_run_to_reverse':[[2,[0,0,0]],[5,[1.1,0,-.6]],[5,[-.9,0,.6]]],
      'backward_to_reverse':[[2,[0,0,0]],[5,[-.9,0,0]],[5,[-.9,0,.6]]]
    }
    write(OUT/'schedule.json',schedule)
    execute('phase_probes',[SOURCE/'diagnostics/diagnose_pivot_failure.py','--models',OUT/'models.json',
        '--schedule',OUT/'schedule.json','--output',OUT/'phase_probes'])
    result=dict(training=False,repeats=rows,phase_probes=read(OUT/'phase_probes/report.json'),wall_seconds=time.perf_counter()-started)
    write(OUT/'result.json',result);write(OUT/'status.json',dict(stage='complete',training=False))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        write(OUT/'status.json',dict(stage='error',error=str(error)));raise
