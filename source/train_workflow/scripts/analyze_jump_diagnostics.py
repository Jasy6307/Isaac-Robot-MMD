"""Reanalyze an existing report without launching Isaac or rerunning a policy."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.jump_analysis import export_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--window", action="append", default=[], help="label:start_seconds:end_seconds")
    args = parser.parse_args()
    arrays = np.load(args.report / "samples.npz")
    data = dict(zip(arrays["columns"], arrays["values"].T))
    windows = []
    for value in args.window:
        label, start, end = value.rsplit(":", 2)
        windows.append((label, float(start), float(end)))
    result = export_analysis(data, args.report, windows)
    print(json.dumps(result["annotated_windows"], ensure_ascii=False, indent=2))
