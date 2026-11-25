"""Load historical configurations with the current evaluation and runtime settings."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source



def clean_legacy_config(config):
    result = dict(config)
    result.pop('unified_eval_lite', None)
    result.pop('unified_eval_short', None)
    if result.pop('runtime_arm_clearance', None):
        raise ValueError(
            'Runtime arm compensation has been removed. Remove the legacy '
            'runtime_arm_clearance override; use a policy trained with spatial clearance.'
        )
    return result
