"""Compatibility entry; implementation: source.train_workflow.utils.motion.legacy_leg_local."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion.legacy_leg_local import repair_h5_leg_spikes_main as main, foot_positions, world_metrics, load

if __name__ == '__main__':
    raise SystemExit(main())
