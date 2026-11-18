"""Compatibility entry; implementation: tools.dance_preview."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dance_preview import main, model_dances

if __name__ == '__main__':
    raise SystemExit(main())
