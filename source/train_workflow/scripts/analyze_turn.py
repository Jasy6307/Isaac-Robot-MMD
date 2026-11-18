"""Compatibility entry; implementation: source.train_workflow.utils.turn_analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.turn_analysis import analyze_turn_main as main, spectral_rms, control_rows, summarize, plot

if __name__ == '__main__':
    raise SystemExit(main())
