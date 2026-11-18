"""Compatibility entry; implementation: source.train_workflow.utils.jump_reports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.jump_reports import plot_jump_reference_change_main as main

if __name__ == '__main__':
    raise SystemExit(main())
