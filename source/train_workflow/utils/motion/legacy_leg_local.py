"""Historical window smoothing and isolated spike repair strategies."""
from __future__ import annotations
from source.train_workflow.utils.motion.legacy_leg_full import ROOT, foot_positions, world_metrics, load_motion as load, export_joint_positions, trajectory_metrics
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from source.train_workflow.utils.motion.self_collision import G1CollisionModel, playback_baseline
from scipy.spatial.transform import Rotation

def metrics(model, q, legs):
    return trajectory_metrics(model, q, legs)[0]

def smooth_h5_leg_window_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=920)
    parser.add_argument('--end', type=int, default=990, help='Exclusive frame index')
    parser.add_argument('--iterations', type=int, default=1400)
    parser.add_argument('--uncross', action='store_true', help='Reduce this window to a consistent left/right leg path instead of preserving a deep crossing')
    args = parser.parse_args(argv)
    from source.paths import resolve_dance_asset
    if args.input:
        args.input = resolve_dance_asset(args.input)
    if args.reference:
        args.reference = resolve_dance_asset(args.reference)
    if args.output:
        args.output = resolve_dance_asset(args.output)
    if args.output.exists():
        parser.error('Output already exists; input H5 files are preserved')
    if args.output.with_suffix('.smooth.json').exists():
        raise FileExistsError(args.output.with_suffix('.smooth.json'))
    torch.set_num_threads(2)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.input, args.reference)}
    names, delta, fps = load(args.input)
    reference_names, reference_delta, reference_fps = load(args.reference)
    if names != reference_names or fps != reference_fps:
        parser.error('Reference joint order and FPS must match input')
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
    export_joint_positions(args.input, args.output, repaired, dict(
        leg_trajectory_smoothing=f'{lo}:{hi}; cartesian acceleration and speed; quarter-frame collision constraints'))
    report = {'input':str(args.input.resolve()),'reference':str(args.reference.resolve()), 'output':str(args.output.resolve()),'input_sha256':hashes,'window':[lo,hi], 'uncross':args.uncross, 'fps':fps, 'before':before,'after':after,'collision_samples_after':collision_report,'history':history}
    args.output.with_suffix('.smooth.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('EXPORTED',args.output,'AFTER',after,flush=True)

FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')

def repair_h5_leg_spikes_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Collision-repaired motion before global smoothing')
    parser.add_argument('--original', required=True, type=Path, help='Original H5 used to distinguish correction artifacts from intentional fast motion')
    parser.add_argument('--smooth_seed', required=True, type=Path, help='Stable repaired poses used only at detected spikes')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=1200)
    args = parser.parse_args(argv)
    from source.paths import resolve_dance_asset
    if args.input:
        args.input = resolve_dance_asset(args.input)
    if args.original:
        args.original = resolve_dance_asset(args.original)
    if args.smooth_seed:
        args.smooth_seed = resolve_dance_asset(args.smooth_seed)
    if args.output:
        args.output = resolve_dance_asset(args.output)
    if args.output.exists():
        parser.error('Output already exists; all source H5 files are preserved')
    if args.output.with_suffix('.spikes.json').exists():
        raise FileExistsError(args.output.with_suffix('.spikes.json'))
    if args.iterations < 1:
        parser.error('Iterations must be positive')
    torch.set_num_threads(2)
    hashes = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.input,args.original,args.smooth_seed)}
    names, delta, fps = load(args.input)
    original_names, raw, original_fps = load(args.original)
    seed_names, stable, seed_fps = load(args.smooth_seed)
    if original_names != names or seed_names != names or original_fps != fps or seed_fps != fps or raw.shape != delta.shape or stable.shape != delta.shape:
        parser.error('H5 joint order, FPS and trajectory dimensions must match')
    if not all(np.isfinite(q).all() for q in (delta,raw,stable)):
        parser.error('All trajectories must be finite')
    baseline = playback_baseline(names,ROOT/'source/my_task/g1_replay_env_cfg.py')
    model = G1CollisionModel(ROOT/'assets/g1_29dof_o6_hand_V3.usd',names,'cpu')
    legs = [i for i,n in enumerate(names) if any(part in n for part in ('hip_','knee_','ankle_'))]
    with torch.no_grad():
        original_feet = foot_positions(model,model._tensor(raw+baseline))
        repaired_feet = foot_positions(model,model._tensor(delta+baseline))
        velocities = [torch.linalg.vector_norm(p[1:]-p[:-1],dim=-1).amax(-1).numpy() for p in (original_feet,repaired_feet)]
        accelerations = [torch.linalg.vector_norm(p[2:]-2*p[1:-1]+p[:-2],dim=-1).amax(-1).numpy() for p in (original_feet,repaired_feet)]
    # A high speed alone is not evidence of a defect: retain fast source steps.
    bad = set(np.flatnonzero((velocities[1]>.12)&(velocities[1]>2.5*velocities[0])).tolist())
    bad.update((np.flatnonzero((accelerations[1]>.15)&(accelerations[1]>2.5*accelerations[0]))+1).tolist())
    cores = []
    for frame in sorted(bad):
        lo,hi = max(0,frame-2),min(len(delta),frame+4)
        if cores and lo <= cores[-1][1]:
            cores[-1][1] = max(hi,cores[-1][1])
        else:
            cores.append([lo,hi])
    windows = []
    alpha = np.zeros(len(delta),dtype=np.float32)
    for lo,hi in cores:
        first,last = max(0,lo-16),min(len(delta),hi+16)
        alpha[lo:hi] = 1
        for begin,end,rising in ((first,lo,True),(hi,last,False)):
            t = np.linspace(0,1,end-begin,endpoint=not rising)
            smooth = t*t*(3-2*t)
            alpha[begin:end] = np.maximum(alpha[begin:end],smooth if rising else 1-smooth)
        if windows and first <= windows[-1][1]:
            windows[-1][1] = max(last,windows[-1][1])
        else:
            windows.append([first,last])
    print('DETECTED SPIKES',sorted(bad),'CORES',cores,'WINDOWS',windows,flush=True)
    result = delta.copy()
    blended = delta.copy()
    blended[:,legs] += alpha[:,None]*(stable[:,legs]-delta[:,legs])
    with h5py.File(args.input) as f:
        root_quat = f['root_quat_delta_wxyz'][:]
        root_pos = f['root_pos_delta'][:]
    root_rotation = model._tensor((Rotation.from_quat(root_quat[:,[1,2,3,0]])*Rotation.from_euler('z',90,degrees=True)).as_matrix())
    root_position = model._tensor(root_pos)
    local_reports = []
    for lo,hi in windows:
        desired = model._tensor(blended[lo:hi]+baseline)
        current = model._tensor(delta[lo:hi]+baseline)
        with torch.no_grad():
            feet_target = foot_positions(model,desired)
            world_target = (root_rotation[lo:hi,None]@feet_target[...,None]).squeeze(-1)+root_position[lo:hi,None]
            target_velocity = world_target[1:]-world_target[:-1]
            target_acceleration = target_velocity[1:]-target_velocity[:-1]
        variable = torch.nn.Parameter(desired[:,legs].clone())
        optimizer = torch.optim.Adam([variable],lr=.004)
        weight = model._tensor(1+30*(1-alpha[lo:hi])**2)
        history = []
        for step in range(args.iterations):
            progress = step/max(1,args.iterations-1)
            optimizer.param_groups[0]['lr'] = .004*(.02+.98*(1+np.cos(np.pi*progress))/2)
            optimizer.zero_grad(set_to_none=True)
            q = current.clone()
            q[:,legs] = variable
            feet = foot_positions(model,q)
            world_feet = (root_rotation[lo:hi,None]@feet[...,None]).squeeze(-1)+root_position[lo:hi,None]
            velocity = world_feet[1:]-world_feet[:-1]
            acceleration = velocity[1:]-velocity[:-1]
            samples = torch.cat([q]+[q[:-1]*(1-t)+q[1:]*t for t in (.25,.5,.75)])
            distances = model.distances(*model.fk(samples))
            collision = torch.relu(.004-distances).square().sum(-1).mean()
            tracking = (weight[:,None]*(world_feet-world_target).square().sum(-1)).mean()
            joint_deviation = (weight*(variable-desired[:,legs]).square().sum(-1)).mean()
            # Match the desired motion's derivatives, rather than suppressing
            # velocities or imposing a uniform speed cap on normal dancing.
            velocity_error = (velocity-target_velocity).square().sum(-1).mean()
            acceleration_error = (acceleration-target_acceleration).square().sum(-1).mean()
            correction = variable-desired[:,legs]
            correction_curvature = (correction[2:]-2*correction[1:-1]+correction[:-2]).square().sum(-1).mean()
            loss = 1000000*collision+10000*tracking+10*joint_deviation+40000*velocity_error+40000*acceleration_error+100*correction_curvature
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                variable.clamp_(model.lower[legs],model.upper[legs])
                variable[:2] = current[:2,legs]
                variable[-2:] = current[-2:,legs]
            if step%300 == 0 or step == args.iterations-1:
                item = {'iteration':step,'loss':float(loss.detach()),'worst_penetration_m':max(0.,-float(distances.detach().min()))}
                history.append(item)
                print('WINDOW',lo,hi,item,flush=True)
        q = current.clone()
        q[:,legs] = variable.detach()
        result[lo:hi,legs] = q[:,legs].numpy()-baseline[legs]
        with torch.no_grad():
            world_before = (root_rotation[lo:hi,None]@foot_positions(model,current)[...,None]).squeeze(-1)+root_position[lo:hi,None]
            world_after = (root_rotation[lo:hi,None]@foot_positions(model,q)[...,None]).squeeze(-1)+root_position[lo:hi,None]
            reports = [model.report(q,.004),model.report((q[:-1]+q[1:])/2,.004)]
        if any(r['penetrating_frames'] for r in reports):
            raise RuntimeError(f'Window {lo}:{hi} still collides; refusing to export')
        local_reports.append({'window':[lo,hi],'before':world_metrics(world_before),'after':world_metrics(world_after),'collision_after':reports,'history':history})
    allowed = np.zeros(delta.shape,dtype=bool)
    for lo,hi in windows:
        allowed[lo:hi,legs] = True
    assert np.array_equal(delta[~allowed],result[~allowed])
    q = model._tensor(result+baseline)
    full_reports = [model.report(q,.004),model.report((q[:-1]+q[1:])/2,.004)]
    if any(r['penetrating_frames'] for r in full_reports):
        raise RuntimeError('Full-song collision check failed; refusing to export')
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert np.isfinite(result).all()
    changed_frames = int(np.any(result!=delta,axis=1).sum())
    report = {'input':str(args.input.resolve()),'original':str(args.original.resolve()),'smooth_seed':str(args.smooth_seed.resolve()),'source_sha256':hashes,'frames':len(delta),'fps':fps,'spike_frames':sorted(bad),'cores':cores,'windows':windows,'changed_frames':changed_frames,'unchanged_frames':len(delta)-changed_frames,'window_reports':local_reports,'full_song_collision_after':full_reports}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    export_joint_positions(args.input, args.output, result, dict(
        leg_spike_repair=json.dumps({'windows':windows,'source':str(args.input.resolve()),'mode':'isolated correction-induced spikes; other frames preserved'})))
    args.output.with_suffix('.spikes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('EXPORTED',args.output,'CHANGED',changed_frames,'UNCHANGED',len(delta)-changed_frames,flush=True)
