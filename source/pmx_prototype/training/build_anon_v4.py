"""Separate 31-DOF model with forearm twist and two-axis wrists; keep v3 intact."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path

if __name__ == '__main__':
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

from reference_asset_builder import ensure_reference_asset

ROOT = PROJECT_ROOT


def ensure_asset(mass_profile=None):
    return ensure_reference_asset(ROOT, version=4, mass_profile=mass_profile)


if __name__ == '__main__':
    print(ensure_asset())
    app.close()
