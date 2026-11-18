"""Compatibility entry for character_cache reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.character_cache import main

if __name__ == '__main__':
    raise SystemExit(main(action='reference'))
