"""Compact continuous walk/run demonstration, shared by live playback."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

DEMO=[
    ('stand',3,[0,0,0]),('walk',4,[.3,0,0]),('ramp_up',10,None),
    ('run_forward',4,[1.2,0,0]),('run_turn',6,[1.1,0,.45]),('stop',4,[0,0,0]),
    ('backward',6,[-.9,0,0]),('stop',3,[0,0,0]),('left',6,[0,.9,0]),
    ('stop',3,[0,0,0]),('right',6,[0,-.9,0]),('stop',3,[0,0,0]),
    ('diagonal_turn',6,[.7,.7,-.3]),('stop',4,[0,0,0]),('stand',4,[0,0,0]),
]

def demo_command(t):
    for label,seconds,cmd in DEMO:
        if t<seconds:return [.3+.9*t/seconds,0,0] if label=='ramp_up' else cmd
        t-=seconds
    return [0,0,0]

def demo_seconds():return sum(s for _,s,_ in DEMO)
