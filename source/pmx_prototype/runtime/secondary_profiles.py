"""Shared defaults and native skirt-grid resolution for secondary physics."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path

DEFAULT_SECONDARY_PROFILE = resolve_source('secondary_default.json')
DEFAULT_SECONDARY_QUALITY = 'cloth'
SKIRT_EDGE_STRIDES = {'full': 1, 'reduced': 2, 'sparse': 3}
SECONDARY_MODES = ('full', 'light', 'light2', 'light3')


def crowd_secondary_modes(count, controlled, mode='full', mixed=False, controlled_mode='light'):
    """Resolve an ordered per-actor budget from the control request snapshot."""
    if count<1 or not 0<=controlled<count:raise ValueError('Invalid crowd actor selection')
    if mode not in SECONDARY_MODES:raise ValueError(f'Unknown secondary mode: {mode}')
    if controlled_mode not in SECONDARY_MODES:raise ValueError(f'Unknown controlled secondary mode: {controlled_mode}')
    return [(controlled_mode if i==controlled else 'light3') if mixed else mode for i in range(count)]


def apply_secondary_mode(settings, mode='full'):
    """Opt-in solve budget; preserve topology, collision coverage and 120 Hz."""
    if mode not in SECONDARY_MODES:raise ValueError(f'Unknown secondary mode: {mode}')
    if mode=='full':return settings
    from copy import deepcopy
    result=deepcopy(settings)
    position,hair,contacts={'light':(64,48,8),'light2':(32,16,8),'light3':(8,4,2)}[mode]
    result['native_solver_position_iterations']=position
    result['native_solver_velocity_iterations']=8
    if result.get('hair_constraint_solver'):
        result['hair_constraint_solver'].update(iterations=hair,contact_refine_iterations=contacts)
    result['secondary_mode']=mode
    return result


def resolve_skirt_grid(settings, names):
    """Resolve geometry-specific dimensions without copying another model's rig."""
    if not settings.get('skirt_max_stretch'):
        return
    grid = {tuple(map(int, n.split('_')[1:])) for n in names if n.startswith('qun_')}
    if not grid:
        raise ValueError('Skirt stretch limits require a native qun_<row>_<column> grid')
    rows = 1 + max(r for r, c in grid)
    columns = 1 + max(c for r, c in grid)
    if columns != 21 or grid != {(r, c) for r in range(rows) for c in range(columns)}:
        raise ValueError('Expected a complete 21-column skirt grid')
    for key, value in (('rows', rows), ('columns', columns)):
        if key in settings and settings[key] != value:
            raise ValueError(f'{key}={settings[key]} does not match native skirt grid {value}')
        settings[key] = value
    appearance = settings.get('appearance', {})
    waist = appearance.get('skirt_waist_support', {})
    gains = waist.get('shape_gain_by_grid_rows', {}).get(str(rows))
    if gains is not None:
        settings['appearance'] = dict(appearance, skirt_waist_support=dict(
            waist, shape_gain_by_row=list(gains)))
