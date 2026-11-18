"""Historical compatibility entry; prefer g1_ground_motion.py smooth for repaired IK."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.grounding.legacy_refinement import main

if __name__ == '__main__':
    raise SystemExit(main(default_method='fidelity'))
