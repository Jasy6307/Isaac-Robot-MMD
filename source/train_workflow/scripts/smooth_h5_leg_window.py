"""Smooth a selected repaired H5 leg trajectory while checking self collisions."""
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
from scipy.ndimage import gaussian_filter1d
import torch

from source.train_workflow.utils.motion.self_collision import G1CollisionModel, playback_baseline


def metrics(model, q, legs):
    with torch.no_grad():
        _, p = model.fk(q)
        feet = torch.stack([p[n] for n in ('left_ankle_roll_link', 'right_ankle_roll_link')], dim=1)
        return {
            'foot_max_step_m': float(torch.linalg.vector_norm(feet[1:] - feet[:-1], dim=-1).max()),
            'foot_max_second_difference_m': float(torch.linalg.vector_norm(feet[2:] - 2 * feet[1:-1] + feet[:-2], dim=-1).max()),
            'leg_max_step_deg': float((q[1:, legs] - q[:-1, legs]).abs().max() * 180 / np.pi),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=920)
    parser.add_argument('--end', type=int, default=990, help='Exclusive frame index')
    parser.add_argument('--iterations', type=int, default=1400)
    parser.add_argument('--uncross', action='store_true', help='Reduce this window to a consistent left/right leg path instead of preserving a deep crossing')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; input H5 files are preserved')
    torch.set_num_threads(2)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.input, args.reference)}
    with h5py.File(args.input) as f:
        names = list(f['joint_names'].asstr()[:])
        delta = f['joint_pos_delta'][:]
        fps = float(f.attrs['fps'])
    with h5py.File(args.reference) as f:
        if names != list(f['joint_names'].asstr()[:]) or fps != float(f.attrs['fps']):
            parser.error('Reference joint order and FPS must match input')
        reference_delta = f['joint_pos_delta'][:]
    if reference_delta.shape != delta.shape or not np.isfinite(delta).all() or not np.isfinite(reference_delta).all():
        parser.error('Input and reference must contain matching finite trajectories')
    lo, hi = args.start, args.end
    if not 0 <= lo < hi <= len(delta) or hi - lo < 20 or args.iterations < 1:
        parser.error('Require at least 20 valid window frames and positive iterations')
    baseline = playback_baseline(names, ROOT / 'source/my_task/g1_replay_env_cfg.py')
    model = G1CollisionModel(ROOT / 'assets/g1_29dof_o6_hand_V3.usd', names, 'cpu')
    legs = [i for i, n in enumerate(names) if any(part in n for part in ('hip_', 'knee_', 'ankle_'))]
    current = model._tensor(delta[lo:hi] + baseline)
    reference = model._tensor(reference_delta[lo:hi] + baseline)
    fixed = 6
    seed = gaussian_filter1d(current[:, legs].numpy(), sigma=2, axis=0)
    if args.uncross:
        # A consistent initialization avoids converging to intersecting segment
        # centerlines, where a distance-only penalty has an ambiguous direction.
        t = np.linspace(0, 1, len(seed))[:, None]
        bridge = current[fixed, legs].numpy() * (1-t) + current[-fixed-1, legs].numpy() * t
        seed = .2 * seed + .8 * bridge
    variable = torch.nn.Parameter(model._tensor(seed))
    with torch.no_grad():
        variable[:fixed] = current[:fixed, legs]
        variable[-fixed:] = current[-fixed:, legs]
        _, p = model.fk(reference)
        foot_reference = torch.stack([p[n] for n in ('left_ankle_roll_link', 'right_ankle_roll_link')], dim=1)
        foot_target = model._tensor(gaussian_filter1d(foot_reference.numpy(), sigma=1.5, axis=0))
        if args.uncross:
            center_y = foot_target[:, :, 1].mean(1)
            half_gap = ((foot_target[:, 0, 1] - foot_target[:, 1, 1]).abs() / 2).clamp_min(.09)
            foot_target[:, 0, 1] = center_y + half_gap
            foot_target[:, 1, 1] = center_y - half_gap
    optimizer = torch.optim.Adam([variable], lr=.008)
    history = []
    before = metrics(model, current, legs)
    print('BEFORE', before, flush=True)
    for step in range(args.iterations):
        progress = step / max(1, args.iterations - 1)
        optimizer.param_groups[0]['lr'] = .008 * (.02 + .98 * (1 + np.cos(np.pi * progress)) / 2)
        optimizer.zero_grad(set_to_none=True)
        q = current.clone()
        q[:, legs] = variable
        _, p = model.fk(q)
        feet = torch.stack([p[n] for n in ('left_ankle_roll_link', 'right_ankle_roll_link')], dim=1)
        velocities = feet[1:] - feet[:-1]
        accelerations = velocities[1:] - velocities[:-1]
        samples = torch.cat([q] + [q[:-1] * (1-t) + q[1:] * t for t in (.25, .5, .75)])
        distances = model.distances(*model.fk(samples))
        collision = torch.relu(.004 - distances).square().sum(-1).mean()
        tracking = (feet - foot_target).square().sum(-1).mean()
        foot_acceleration = accelerations.square().sum(-1).mean()
        foot_speed = torch.relu(torch.linalg.vector_norm(velocities, dim=-1) - .045).square().mean()
        joint_acceleration = (variable[2:] - 2*variable[1:-1] + variable[:-2]).square().sum(-1).mean()
        joint_speed = torch.relu((variable[1:] - variable[:-1]).abs() - .12).square().sum(-1).mean()
        deviation = (variable - reference[:, legs]).square().sum(-1).mean()
        ordering = torch.zeros((), dtype=q.dtype)
        if args.uncross:
            # Fade this directional constraint at the fixed boundaries.
            fade = torch.sin(torch.linspace(0, np.pi, len(q))).square()
            knee_gap = p['left_knee_link'][:, 1] - p['right_knee_link'][:, 1]
            ankle_gap = feet[:, 0, 1] - feet[:, 1, 1]
            ordering = (fade * (torch.relu(.16-knee_gap).square() + torch.relu(.18-ankle_gap).square())).mean()
        loss = 250000 * collision + 2000 * tracking + 200000 * foot_acceleration + 1000000 * foot_speed + 100 * joint_acceleration + 500 * joint_speed + .5 * deviation + 50000 * ordering
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(model.lower[legs], model.upper[legs])
            variable[:fixed] = current[:fixed, legs]
            variable[-fixed:] = current[-fixed:, legs]
        if step % 200 == 0 or step == args.iterations - 1:
            item = {'iteration':step, 'loss':float(loss.detach()), 'worst_penetration_m':max(0.,-float(distances.detach().min())), 'foot_max_step_m':float(torch.linalg.vector_norm(velocities.detach(),dim=-1).max())}
            history.append(item)
            print(item, flush=True)
    with torch.no_grad():
        result = current.clone()
        result[:, legs] = variable
        after = metrics(model, result, legs)
        collision_report = [model.report(result, .004)] + [model.report(result[:-1]*(1-t)+result[1:]*t,.004) for t in (.25,.5,.75)]
        input_frame_report = model.report(current, .004)
        input_midpoint_report = model.report((current[:-1]+current[1:])/2, .004)
    for output_report, input_report in ((collision_report[0], input_frame_report), (collision_report[2], input_midpoint_report)):
        if output_report['worst_penetration_m'] > max(.001, input_report['worst_penetration_m']) + .0001:
            raise RuntimeError('Smoothing introduced collision penetration; refusing to export')
    if after['foot_max_step_m'] >= before['foot_max_step_m']:
        raise RuntimeError('Foot trajectory did not improve; refusing to export')
    repaired = delta.copy()
    repaired[lo:hi, legs] = (result[:, legs].numpy() - baseline[legs]).astype(delta.dtype)
    assert np.isfinite(repaired).all()
    for p, digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.input,args.output)
    with h5py.File(args.output,'r+') as f:
        f['joint_pos_delta'][:] = repaired
        f.attrs['leg_trajectory_smoothing'] = f'{lo}:{hi}; cartesian acceleration and speed; quarter-frame collision constraints'
    report = {'input':str(args.input.resolve()),'reference':str(args.reference.resolve()), 'output':str(args.output.resolve()),'input_sha256':hashes,'window':[lo,hi], 'uncross':args.uncross, 'fps':fps, 'before':before,'after':after,'collision_samples_after':collision_report,'history':history}
    args.output.with_suffix('.smooth.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('EXPORTED',args.output,'AFTER',after,flush=True)


if __name__ == '__main__':
    main()
