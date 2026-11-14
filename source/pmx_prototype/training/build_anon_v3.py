"""Create the separate 25-DOF reference-guided body. Called after Kit starts."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
from reference_asset_builder import ensure_reference_asset

ROOT = PROJECT_ROOT


def ensure_asset():
    return ensure_reference_asset(ROOT, version=3)
