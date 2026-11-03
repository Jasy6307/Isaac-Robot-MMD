"""Analyze an unseen reference without reading reviewed annotation files."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.motion.auto_jump import analyze_h5

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--robot_usd", type=Path, default=ROOT / "assets/g1_29dof_o6_hand_V3.usd")
    p.add_argument("--frames", type=int)
    args = p.parse_args()
    report = analyze_h5(args.source, args.robot_usd, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
