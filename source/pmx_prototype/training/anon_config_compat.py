"""Read historical null settings, but reject removed active compensation."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source



def clean_legacy_config(config):
    result = dict(config)
    if result.pop('runtime_arm_clearance', None):
        raise ValueError(
            'Runtime arm compensation has been removed. Remove the legacy '
            'runtime_arm_clearance override; use a policy trained with spatial clearance.'
        )
    return result
