"""Sum all V6 training attempts, including experiments and continued runs."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/v6_training_time.json')
args=parser.parse_args()
rows=[]
for run in sorted((ROOT/'logs/rsl_rl/anon_locomotion').glob('v6_*')):
    exact=run/'training_time.json'
    status=run/'status.json'
    if exact.exists():
        data=json.loads(exact.read_text(encoding='utf-8'))
        seconds=data['training_seconds'];estimated=False
    elif status.exists():
        data=json.loads(status.read_text(encoding='utf-8'))
        seconds=data['elapsed_seconds'];estimated=True
    else:continue
    rows.append(dict(run=run.name,seconds=seconds,estimated_from_status=estimated))
seconds=sum(row['seconds'] for row in rows)
result=dict(training_seconds=seconds,training_minutes=seconds/60,runs=rows,
            definition='All V6 training attempts; excludes preparation, scene startup, evaluation and video recording. Status-based older entries are approximate.')
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
