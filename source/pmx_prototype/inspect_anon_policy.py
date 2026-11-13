"""Start the visual keyboard inspector for a trained Anon policy."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
PRESETS={
    'run':ROOT/'logs/rsl_rl/anon_locomotion/v6_run_wrists_refine/policy_final.pt',
    'walk':ROOT/'output/anon_locomotion/release_v6_continuous_wrists/policy.pt',
}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=PRESETS,default='walk')
    p.add_argument('--checkpoint',type=Path,help='Inspect another compatible checkpoint')
    p.add_argument('--output',type=Path)
    a,extra=p.parse_known_args()
    checkpoint=(a.checkpoint or PRESETS[a.policy]).resolve()
    if not checkpoint.is_file():p.error(f'Missing checkpoint: {checkpoint}')
    output=a.output or ROOT/'output/anon_locomotion/inspector'/f'{a.policy}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    sys.argv=[str(ROOT/'source/pmx_prototype/play_anon_policy.py'),'--checkpoint',str(checkpoint),'--inspector','--output',str(output),*extra]
    sys.path.insert(0,str(ROOT/'source/pmx_prototype'))
    runpy.run_path(sys.argv[0],run_name='__main__')
