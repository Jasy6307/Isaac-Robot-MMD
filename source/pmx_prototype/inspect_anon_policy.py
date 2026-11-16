"""Start the visual keyboard inspector for a trained Anon policy."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
import argparse
from datetime import datetime
from pathlib import Path
import os
import runpy
import sys

ROOT=PROJECT_ROOT
from anon_policy_catalog import PRESETS
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=PRESETS,default='walk')
    p.add_argument('--checkpoint',type=Path,help='Inspect another compatible checkpoint')
    p.add_argument('--output',type=Path)
    p.add_argument('--body-device',choices=['cpu','cuda:0'],help='Default: CPU body/inference; all inspector models share this backend')
    p.add_argument('--physics-threads',type=int,default=16)
    p.add_argument('--inline-secondary',action='store_true',help='Use the previous in-process secondary physics path')
    a,extra=p.parse_known_args()
    checkpoint=(a.checkpoint or PRESETS[a.policy]).resolve()
    if not checkpoint.is_file():p.error(f'Missing checkpoint: {checkpoint}')
    output=a.output or ROOT/'output/anon_locomotion/inspector'/f'{a.policy}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    device=a.body_device or 'cpu'
    sys.argv=[str(SOURCE_DIR/'play_anon_policy.py'),'--checkpoint',str(checkpoint),'--inspector','--output',str(output),'--body-device',device,'--physics-threads',str(a.physics_threads),*extra]
    if not a.inline_secondary:sys.argv.append('--secondary-worker')
    sys.path.insert(0,str(SOURCE_DIR))
    runpy.run_path(sys.argv[0],run_name='__main__')
