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
