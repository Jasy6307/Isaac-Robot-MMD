"""Compatibility entry; implementation: source.train_workflow.utils.motion.legacy_leg_full."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion.legacy_leg_full import main, foot_metrics, LegClearance

if __name__ == '__main__':
    raise SystemExit(main())
