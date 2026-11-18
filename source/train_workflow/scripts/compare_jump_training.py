"""Compatibility entry; implementation: source.train_workflow.utils.jump_comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.jump_comparison import compare_jump_training_main as main, compare_jump_training_summary as summarize

if __name__ == '__main__':
    raise SystemExit(main())
