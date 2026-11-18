"""Compatibility entry; implementation: tools.dance_preview."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dance_preview import main as _preview_main, dance_catalog, select_dance, playback_arguments


def main(argv=None):
    return _preview_main(argv, preset_only=True)

if __name__ == '__main__':
    raise SystemExit(main())
