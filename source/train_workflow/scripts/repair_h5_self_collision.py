"""Export a separate G1 H5 with offline capsule/box self-collision repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from source.train_workflow.utils.motion.self_collision import G1CollisionModel, playback_baseline, repair_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial_h5", type=Path, help="Warm start from an earlier repair; deviations are still measured from --input")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4, help="CPU tensor worker threads")
    parser.add_argument("--margin", type=float, default=0.004)
    args = parser.parse_args()
    from source.paths import resolve_dance_asset
    if args.input:
        args.input = resolve_dance_asset(args.input)
    if args.output:
        args.output = resolve_dance_asset(args.output)
    if args.initial_h5:
        args.initial_h5 = resolve_dance_asset(args.initial_h5)
    source, output = args.input.resolve(), args.output.resolve()
    if source == output:
        parser.error("Input and output must be different; original H5 is preserved.")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if args.iterations < 1 or args.threads < 1 or not np.isfinite(args.margin) or args.margin < 0:
        parser.error("iterations must be positive and margin must be finite/nonnegative")
    torch.set_num_threads(args.threads)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with h5py.File(source, "r") as stream:
        names = list(stream["joint_names"].asstr()[:])
        delta = stream["joint_pos_delta"][:]
        fps = float(stream.attrs["fps"])
    if len(delta) < 3 or not np.isfinite(delta).all():
        raise ValueError("At least 3 finite reference frames are required")
    baseline = playback_baseline(names, ROOT / "source/my_task/g1_replay_env_cfg.py")
    initial_pose = None
    if args.initial_h5:
        with h5py.File(args.initial_h5, "r") as stream:
            if list(stream["joint_names"].asstr()[:]) != names or float(stream.attrs["fps"]) != fps:
                parser.error("Initial H5 joint order and FPS must match input")
            initial_pose = stream["joint_pos_delta"][:] + baseline
    print(f"[REPAIR] Reading G1 USD geometry; frames={len(delta)}, device={args.device}", flush=True)
    model = G1CollisionModel(ROOT / "assets/g1_29dof_o6_hand_V3.usd", names, args.device)
    print(f"[REPAIR] Model ready: {len(model.shapes)} envelopes, {len(model.pairs)} body pairs", flush=True)
    corrected, report = repair_poses(model, delta + baseline, iterations=args.iterations, margin=args.margin, initial_pose=initial_pose)
    new_delta = (corrected - baseline).astype(delta.dtype)
    if not np.isfinite(new_delta).all():
        raise ValueError("Non-finite corrected poses; refusing to export")
    # Clone all datasets and attributes, preserving debug data, root motion,
    # hand angles, frame indexing and the original schema.
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with h5py.File(output, "r+") as stream:
        stream["joint_pos_delta"][:] = new_delta
        stream.attrs["self_collision_repair"] = "usd_capsule_box_v1"
        stream.attrs["repair_source_h5"] = str(source)
        stream.attrs["repair_source_sha256"] = source_hash
        stream.attrs["repair_margin_m"] = args.margin
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Source changed during repair")
    report.update(input=str(source), output=str(output), source_sha256=source_hash, frames=len(delta), fps=fps, duration_seconds=len(delta) / fps)
    report_path = output.with_suffix(".repair.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[REPAIR] Exported: {output}")
    print(f"[REPAIR] Report: {report_path}")
    print(f"[REPAIR] Before/after: {report['before']['penetrating_frames']} / {report['after']['penetrating_frames']} penetrating frames (>1mm, envelopes)")


if __name__ == "__main__":
    main()
