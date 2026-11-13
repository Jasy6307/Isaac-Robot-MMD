"""CPU background worker for the replay UI's collision-reference export."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw H5 captured by replay")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot_usd", default=str(ROOT / "assets/g1_29dof_o6_hand_V3.usd"))
    parser.add_argument("--status", type=Path)
    parser.add_argument("--iterations", type=int, default=600,
                        help='Gesture-mode iterations; local mode uses at most eight restoration trials per window')
    parser.add_argument('--collision_mode',choices=('local','gesture'),default='local',
                        help='local: verified avoidance then bounded pose restoration; gesture: expensive full-motion refinement')
    parser.add_argument('--max_seconds',type=float,default=60.,help='Local pose-restoration budget; collision validation is always required')
    parser.add_argument('--legacy_collision',action='store_true',help='Comparison only: omit Cartesian gesture preservation')
    args = parser.parse_args()
    from source.paths import resolve_dance_asset
    if args.input:
        args.input = str(resolve_dance_asset(args.input))
    if args.output:
        args.output = str(resolve_dance_asset(args.output))
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:
        parser.error('--max_seconds must be finite and positive')

    def publish(state, stage, fraction=0.0, detail=""):
        phases = {"geometry": (0, .02), "collision": (.02, .60), "polish": (.62, .13), "spikes": (.75, .20), "gestures":(.95,.04), "validate": (.99, .009), "done": (1, 0)}
        start, span = phases.get(stage, (0, 0))
        payload = {"state": state, "stage": stage, "progress": start+span*fraction, "message": detail or stage, "output": str(Path(args.output).resolve()), "report": str(Path(args.output).with_suffix(".collision.json").resolve())}
        if args.status:
            args.status.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.status.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, args.status)
        print(f"[COLLISION H5] {stage} {100*payload['progress']:.1f}% {detail}", flush=True)

    publish("running", "geometry")
    try:
        from source.train_workflow.utils.motion.collision_reference import repair_reference_h5
        repair_reference_h5(args.input, args.output, args.robot_usd, iterations=args.iterations,
                             preserve_gestures=not args.legacy_collision,
                             collision_mode=args.collision_mode,max_seconds=args.max_seconds,
                             progress=lambda s, f, d: publish("running", s, f, d))
    except Exception as exc:
        traceback.print_exc()
        publish("failed", "failed", detail=str(exc))
        return 1
    publish("done", "done", 1, "Validated reference saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
