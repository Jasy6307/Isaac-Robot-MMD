"""Historical controller compatibility entry; see experiments/README.md."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.pmx_prototype.experiments.compat import controller_module, legacy_main

implementation=controller_module('run_pivot_ablation_preserved')
if __name__ == '__main__':
    raise SystemExit(legacy_main(implementation))
else:
    sys.modules[__name__]=implementation
