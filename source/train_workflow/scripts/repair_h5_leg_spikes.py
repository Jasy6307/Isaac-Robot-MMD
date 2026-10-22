"""Repair isolated, correction-induced leg spikes while retaining other frames."""
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
from scipy.spatial.transform import Rotation
import torch

from source.train_workflow.utils.motion.self_collision import G1CollisionModel, playback_baseline


FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')


def load(path):
    with h5py.File(path) as f:
        return list(f['joint_names'].asstr()[:]), f['joint_pos_delta'][:], float(f.attrs['fps'])


def foot_positions(model, q):
    _, p = model.fk(q)
    return torch.stack([p[n] for n in FEET], dim=1)


def world_metrics(feet):
    velocity = torch.linalg.vector_norm(feet[1:]-feet[:-1], dim=-1)
    acceleration = torch.linalg.vector_norm(feet[2:]-2*feet[1:-1]+feet[:-2], dim=-1)
    return {'max_step_m':float(velocity.max()), 'max_second_difference_m':float(acceleration.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Collision-repaired motion before global smoothing')
    parser.add_argument('--original', required=True, type=Path, help='Original H5 used to distinguish correction artifacts from intentional fast motion')
    parser.add_argument('--smooth_seed', required=True, type=Path, help='Stable repaired poses used only at detected spikes')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; all source H5 files are preserved')
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
    shutil.copy2(args.input,args.output)
    with h5py.File(args.output,'r+') as f:
        f['joint_pos_delta'][:] = result
        f.attrs['leg_spike_repair'] = json.dumps({'windows':windows,'source':str(args.input.resolve()),'mode':'isolated correction-induced spikes; other frames preserved'})
    args.output.with_suffix('.spikes.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('EXPORTED',args.output,'CHANGED',changed_frames,'UNCHANGED',len(delta)-changed_frames,flush=True)


if __name__ == '__main__':
    main()
