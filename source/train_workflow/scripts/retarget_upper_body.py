"""Fit a PMX animation's upper limbs to G1, preserving all other H5 channels.

Run in the Isaac Python runtime (USD, SciPy, h5py); no simulator is launched.
Input animation is made by source/pmx_prototype/load_vmd.py for ANY VMD.

Example (from the repository root, in the Isaac Python environment)::

    python source/train_workflow/scripts/retarget_upper_body.py \
        --source-animation path/to/baked_pmx.usdc \
        --motion-h5 path/to/raw_g1.h5 --output path/to/new_upper_body.h5

Use the same VMD and source PMX skeleton for the two inputs. The solver only
changes the fourteen shoulder/elbow/wrist channels; the original fingers,
waist, legs, root trajectory and timing are preserved. No collision avoidance
or dynamics/torque feasibility is implied. Standard MMD Japanese landmarks
and the converted PMX coordinate convention are required; missing landmarks
fail explicitly. Existing training/replay assets are not replaced.

--samples validates independent poses distributed across a whole clip.
--window-start / --window-frames validates a continuous trajectory segment.
Neither validation mode publishes a partial motion H5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from source.train_workflow.utils.retarget.upper_body_export import solve_upper_body_arrays
from source.train_workflow.utils.motion.self_collision import playback_baseline


def run(args):
    started = time.perf_counter()
    if args.output.resolve() == args.motion_h5.resolve() or args.output.exists():
        raise ValueError("Choose a new output path; source motions are never overwritten")
    with h5py.File(args.motion_h5) as f:
        names = list(f['joint_names'].asstr()[:])
        delta = f['joint_pos_delta'][:]
        fps = float(f.attrs['fps'])
        frames = f['frames'][:]
        baseline = f.attrs.get('record_baseline_joint_pos')
    if not np.isfinite(fps) or fps <= 0 or not len(frames) or len(set(names)) != len(names):
        raise ValueError("Invalid FPS, empty motion or duplicate joint names")
    if baseline is None:
        baseline = playback_baseline(names, ROOT / 'source/my_task/g1_replay_env_cfg.py')
    baseline = np.asarray(baseline, dtype=float)
    if len(baseline) != len(names) or not np.isfinite(delta).all():
        raise ValueError("Invalid motion baseline or joint channels")
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Expected dense H5 frame indices starting at zero")
    # Validation can sample the entire clip, but sample outputs are not playable H5s.
    validation_only = bool(args.samples or args.window_frames)
    indices = (np.unique(np.linspace(0, len(frames) - 1, args.samples, dtype=int))
               if args.samples else frames[args.window_start:args.window_start+args.window_frames]
               if args.window_frames else frames)
    if not len(indices):
        raise ValueError('Empty validation window')
    result, report = solve_upper_body_arrays(delta, names, baseline, fps,
        args.source_animation, args.robot_usd, indices=indices, independent=bool(args.samples))
    report['source_h5'] = str(args.motion_h5.resolve())
    report['source_h5_sha256'] = hashlib.sha256(args.motion_h5.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not validation_only:
        temp = args.output.with_suffix('.tmp.h5')
        shutil.copyfile(args.motion_h5, temp)
        with h5py.File(temp, 'r+') as f:
            f['joint_pos_delta'][:] = result
            # Existing collision certificates no longer describe this pose sequence.
            for key in list(f.attrs):
                if key.startswith('collision_') or key == 'without_collision_validated':
                    del f.attrs[key]
            f.attrs['upper_body_retarget_method'] = report['method']
            f.attrs['upper_body_source_animation'] = str(args.source_animation.resolve())
        temp.replace(args.output)
        # Verify all non-arm datasets and metadata survive byte-for-byte at array level.
        cols = [j for j,n in enumerate(names) if n not in sum([v['joint_names'] for v in report['arms'].values()], [])]
        with h5py.File(args.motion_h5) as old, h5py.File(args.output) as new:
            def verify(key, dataset):
                if not isinstance(dataset, h5py.Dataset):
                    return
                if key == 'joint_pos_delta':
                    assert np.array_equal(old[key][:,cols], new[key][:,cols]), key
                else:
                    assert np.array_equal(old[key][()], new[key][()]), key
            old.visititems(verify)
        report['non_arm_channels_unchanged'] = True
    report['wall_seconds'] = time.perf_counter() - started
    args.output.with_suffix('.upper_body.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('UPPER_IK_PASS ' + json.dumps({s:{k:v for k,v in r.items() if k not in ('per_frame','joint_names')} for s,r in report['arms'].items()}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-animation', type=Path, required=True)
    parser.add_argument('--motion-h5', type=Path, required=True)
    parser.add_argument('--robot-usd', type=Path, default=ROOT/'assets/g1_29dof_o6_hand_V3.usd')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=0, help='Validation only: sample N frames over entire clip; no H5 written')
    parser.add_argument('--window-start', type=int, default=0, help='Validation only: contiguous sequence start frame')
    parser.add_argument('--window-frames', type=int, default=0, help='Validate a contiguous trajectory; no H5 written')
    args = parser.parse_args()
    from source.paths import resolve_dance_asset
    args.motion_h5 = resolve_dance_asset(args.motion_h5)
    if args.output:
        args.output = resolve_dance_asset(args.output)
    if args.samples and args.samples < 2:
        parser.error('--samples must be at least two')
    if args.window_start < 0 or args.window_frames < 0 or (args.samples and args.window_frames):
        parser.error('Invalid validation window or mixed sampling modes')
    run(args)
