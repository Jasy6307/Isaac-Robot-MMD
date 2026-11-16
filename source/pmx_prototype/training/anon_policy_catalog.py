"""Named inspector models and the scene compatibility needed for hot switching."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path

ROOT=PROJECT_ROOT
PRESETS={
    'walk':ROOT/'output/anon_locomotion/release_v7_unified_walk_run_slip_refine/policy.pt',
}
LABELS={
    'walk':'WALK + RUN | Unified 31 DOF (default)',
}

def label_for(key,config,checkpoint):
    if key in LABELS:return LABELS[key]
    if config.get('unified_walk_run'):return 'WALK + RUN | Unified 31 DOF'
    return 'CUSTOM | '+Path(checkpoint).parent.name

def identify(checkpoint):
    path=Path(checkpoint).resolve()
    return next((key for key,value in PRESETS.items() if value.resolve()==path),'custom')

def physics_signature(config):
    """Only scene construction settings; controller/reference settings may differ."""
    defaults=dict(physics_hz=120,decimation=4,crouch=.015,simulation_device='cuda:0',
                  full_body_reference=False,articulated_wrists=False,
                  upper_drive_scale=1.,upper_damping_scale=1.,measure_ground_contact=False)
    return (str(Path(config['asset']).resolve()).lower(),)+tuple(config.get(k,v) for k,v in defaults.items())
