"""Shared defaults and native skirt-grid resolution for secondary physics."""
from pathlib import Path

DEFAULT_SECONDARY_PROFILE = Path(__file__).with_name('secondary_default.json')
DEFAULT_SECONDARY_QUALITY = 'cloth'


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
