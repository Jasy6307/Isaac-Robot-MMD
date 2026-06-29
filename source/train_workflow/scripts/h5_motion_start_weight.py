"""Compute offline motion-start weights from a dance H5 file.

Example
-------
python source/train_workflow/scripts/h5_motion_start_weight.py `
  --h5 media/dance/IRIS_OUT_z_editted.h5 `
  --target_steps 920 `
  --lookahead_seconds 3.0 `
  --top_ratio 0.25 `
  --save_csv logs/weights.csv `
  --save_json logs/weights.json
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from source.train_workflow.utils.motion.resolve import resolve_dance_h5_by_name
from source.train_workflow.utils.motion.start_weight import (
    compute_motion_start_weights_from_h5,
    save_weights_csv,
    save_weights_json,
    summarize_result,
)


def _resolve_h5_arg(value: str) -> str:
    v = str(value).strip()
    if os.path.isfile(v):
        return os.path.abspath(v)
    if os.path.isfile(os.path.join(_WORKSPACE_ROOT, v)):
        return os.path.abspath(os.path.join(_WORKSPACE_ROOT, v))
    h5_path, _ = resolve_dance_h5_by_name(v)
    return h5_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute motion-start sampling weights from H5.")
    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="H5 path or dance name (e.g. IRIS_OUT).",
    )
    parser.add_argument(
        "--target_steps",
        type=int,
        default=None,
        help="Output weight length. Use training window_frames for matching start-step space.",
    )
    parser.add_argument(
        "--lookahead_seconds",
        type=float,
        default=3.0,
        help="Future horizon to score start-step difficulty.",
    )
    parser.add_argument(
        "--top_ratio",
        type=float,
        default=0.25,
        help="Top-ratio used for high-weight range extraction.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional output CSV path: step,weight.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional output JSON summary + full weights.",
    )
    args = parser.parse_args()

    h5_path = _resolve_h5_arg(args.h5)
    result = compute_motion_start_weights_from_h5(
        h5_path,
        target_steps=args.target_steps,
        lookahead_seconds=args.lookahead_seconds,
        top_ratio=args.top_ratio,
    )

    print(f"[INFO] Input H5: {h5_path}")
    print(summarize_result(result))

    if args.save_csv:
        out_csv = os.path.abspath(args.save_csv)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        save_weights_csv(out_csv, result.weights)
        print(f"[INFO] Saved CSV: {out_csv}")
    if args.save_json:
        out_json = os.path.abspath(args.save_json)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        save_weights_json(out_json, result)
        print(f"[INFO] Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

