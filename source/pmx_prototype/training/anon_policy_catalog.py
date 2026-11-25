"""Named inspector checkpoints and labels."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path

ROOT=PROJECT_ROOT
DEFAULT_INSPECTOR_CHECKPOINT = (
    ROOT / 'logs/rsl_rl/anon_locomotion/lab_direct_backward_start_continue_02409_to_10000_8192_20261124/scratch_010000.pt'
)
PRESETS={
    'walk':ROOT/'output/anon_locomotion/release_v7_unified_walk_run_slip_refine/policy.pt',
}
LABELS={
    'walk':'WALK + RUN | Unified 31 DOF',
}

def label_for(key,config,checkpoint):
    if Path(checkpoint).resolve() == DEFAULT_INSPECTOR_CHECKPOINT.resolve():
        return 'WALK + RUN | Iteration 10,000'
    if key in LABELS:return LABELS[key]
    if config.get('unified_walk_run'):return 'WALK + RUN | Unified 31 DOF'
    return 'CUSTOM | '+Path(checkpoint).parent.name

def identify(checkpoint):
    path=Path(checkpoint).resolve()
    return next((key for key,value in PRESETS.items() if value.resolve()==path),'custom')
