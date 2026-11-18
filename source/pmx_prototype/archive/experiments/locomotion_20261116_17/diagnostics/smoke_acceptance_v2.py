"""Serial fixed-checkpoint integration checks; no training or promotion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/anon_locomotion/low_speed_fix_20261117'


def main():
    if (OUT/'smoke_result.json').exists():raise RuntimeError('Integration output already exists')
    models=json.loads((ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117/models.json').read_text())
    result={}
    for name,path in models.items():
        before=hashlib.sha256(Path(path).read_bytes()).hexdigest();start=time.perf_counter()
        with (OUT/(name+'_smoke.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'source/pmx_prototype/train_anon_locomotion.py'),
                '--mode','eval','--headless','--num-envs','1','--checkpoint',path,'--output',str(OUT/(name+'_smoke')),
                '--config-override',str(OUT/'smoke_config.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            (OUT/'smoke_status.json').write_text(json.dumps(dict(stage=name,training=False,pid=process.pid)))
            if process.wait():raise RuntimeError(name+' integration failed; inspect log')
        report=json.loads((OUT/(name+'_smoke')/'report.json').read_text())
        assert report['acceptance_protocol']=='response_steady_v2' and report['diagnostic'] and not report['passed_unified_gate']
        assert report['acceptance_protocol_sha256']
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before
        result[name]=dict(checkpoint_unchanged=True,seconds=time.perf_counter()-start,falls=report['total_falls'],
            legacy_passed=sum(s['legacy_passed'] for s in report['segments']),
            v2_summary=report['response_steady_summary'],
            segments=[dict(label=s['label'],legacy_passed=s['legacy_passed'],v2=s['response_steady']) for s in report['segments']])
        (OUT/'smoke_result.partial.json').write_text(json.dumps(result,indent=2))
    assert next(s for s in result['new']['segments'] if s['label']=='walk')['v2']['status']=='failed'
    (OUT/'smoke_result.json').write_text(json.dumps(dict(training=False,models=result),indent=2))
    (OUT/'smoke_status.json').write_text(json.dumps(dict(stage='complete',training=False)))


if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:main()
    except Exception as error:
        (OUT/'smoke_status.json').write_text(json.dumps(dict(stage='error',error=str(error))));raise
