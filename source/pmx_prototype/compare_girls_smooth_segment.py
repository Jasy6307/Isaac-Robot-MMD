"""Compatible comparison entry; implementation shared in runtime/comparison_player.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from source.pmx_prototype.runtime.comparison_player import main

if __name__ == '__main__':
    raise SystemExit(main(preset='girls-segment'))
