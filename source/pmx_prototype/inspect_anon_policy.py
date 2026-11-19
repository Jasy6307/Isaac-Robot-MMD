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
from secondary_profiles import SKIRT_EDGE_STRIDES
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=PRESETS,default='walk')
    p.add_argument('--checkpoint',type=Path,help='Inspect another compatible checkpoint')
    p.add_argument('--output',type=Path)
    p.add_argument('--body-device',choices=['cpu','cuda:0'],help='Default: CPU body/inference; all inspector models share this backend')
    p.add_argument('--physics-threads',type=int,default=16)
    p.add_argument('--physics-detail',choices=SKIRT_EDGE_STRIDES,
                   help='VMD preview skirt-edge proxy density (default: sparse)')
    p.add_argument('--frontend-task-threads',type=int,default=4,help='Kit task threads for the viewer (default: 4)')
    p.add_argument('--render-fps',type=int,choices=[30,60],
                   help='60 interpolates the 30 Hz solved poses (default with the secondary worker: 60)')
    p.add_argument('--interpolate-60',action='store_true',help='Use 60 Hz display interpolation like the dance viewer')
    p.add_argument('--inline-secondary',action='store_true',help='Use the previous in-process secondary physics path')
    p.add_argument('--no-secondary-physics',action='store_true',help='Preview the policy without secondary physics')
    p.add_argument('--legacy-secondary-physics',action='store_true',help='Use the historical secondary physics profile')
    p.add_argument('--record-video',type=Path,help='Record the preview at 30 FPS')
    a,extra=p.parse_known_args()
    checkpoint=(a.checkpoint or PRESETS[a.policy]).resolve()
    if not checkpoint.is_file():p.error(f'Missing checkpoint: {checkpoint}')
    output=a.output or ROOT/'output/anon_locomotion/inspector'/f'{a.policy}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    device=a.body_device or 'cpu'
    use_secondary_worker=not (a.inline_secondary or a.no_secondary_physics)
    physics_detail=a.physics_detail or ('full' if a.no_secondary_physics or a.legacy_secondary_physics else 'sparse')
    render_fps=a.render_fps or (60 if a.interpolate_60 or (use_secondary_worker and not a.record_video) else 30)
    sys.argv=[str(SOURCE_DIR/'play_anon_policy.py'),'--checkpoint',str(checkpoint),'--inspector','--output',str(output),
              '--body-device',device,'--physics-threads',str(a.physics_threads),'--physics-detail',physics_detail,
              '--frontend-task-threads',str(a.frontend_task_threads),'--render-fps',str(render_fps),*extra]
    if use_secondary_worker:sys.argv.append('--secondary-worker')
    if a.no_secondary_physics:sys.argv.append('--no-secondary-physics')
    if a.legacy_secondary_physics:sys.argv.append('--legacy-secondary-physics')
    if a.record_video:sys.argv+=['--record-video',str(a.record_video)]
    if a.interpolate_60:sys.argv.append('--interpolate-60')
    sys.path.insert(0,str(SOURCE_DIR))
    runpy.run_path(sys.argv[0],run_name='__main__')
