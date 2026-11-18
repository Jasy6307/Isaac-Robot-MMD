"""Optimize a whole H5 leg trajectory for continuous, collision-aware playback."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from source.paths import REPO_ROOT
ROOT = Path(REPO_ROOT)


import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import torch

from source.train_workflow.utils.motion.self_collision import G1CollisionModel, playback_baseline


FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')


def load_motion(path):
    with h5py.File(path) as stream:
        return list(stream['joint_names'].asstr()[:]), stream['joint_pos_delta'][:], float(stream.attrs['fps'])


def foot_positions(model, q):
    _, positions = model.fk(q)
    return torch.stack([positions[name] for name in FEET], dim=1)


def world_metrics(feet):
    velocity = torch.linalg.vector_norm(feet[1:]-feet[:-1], dim=-1)
    acceleration = torch.linalg.vector_norm(feet[2:]-2*feet[1:-1]+feet[:-2], dim=-1)
    return dict(max_step_m=float(velocity.max()), max_second_difference_m=float(acceleration.max()))


def trajectory_metrics(model, q, legs):
    with torch.no_grad():
        feet = foot_positions(model, q)
        measured = world_metrics(feet)
        return dict(foot_max_step_m=measured['max_step_m'],
            foot_max_second_difference_m=measured['max_second_difference_m'],
            leg_max_step_deg=float((q[1:, legs]-q[:-1, legs]).abs().max()*180/np.pi)), feet


def export_joint_positions(source, output, values, attributes):
    """Preserve every other dataset/attribute and refuse prior H5 results."""
    source, output = Path(source), Path(output)
    if output.exists() or source.resolve() == output.resolve():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with h5py.File(output, 'r+') as stream:
        stream['joint_pos_delta'][:] = values
        for key, value in attributes.items():
            stream.attrs[key] = value


def foot_metrics(model, q, legs):
    with torch.no_grad():
        basic, feet = trajectory_metrics(model, q, legs)
        velocity = torch.linalg.vector_norm(feet[1:]-feet[:-1], dim=-1)
        acceleration = torch.linalg.vector_norm(feet[2:]-2*feet[1:-1]+feet[:-2], dim=-1)
        return {**basic,
                'steps_over_6cm':int((velocity.amax(-1)>.06).sum()),
                'second_differences_over_35mm':int((acceleration.amax(-1)>.035).sum())}, feet


class LegClearance:
    """Leg FK and nine leg pairs, plus nearby arm/leg pairs with fixed arms.

    Broad phase uses a 30 cm envelope around the input motion; full-body checks
    after optimization independently check every pair without this selection.
    """
    def __init__(self, full, samples):
        self.full = full
        self.lean = copy.copy(full)
        self.links = [f'{s}_{part}_link' for s in ('left','right') for part in ('hip_yaw','knee','ankle_roll')]
        by_child = {j.child:j for j in full.joints}
        needed = set(self.links)
        for name in list(needed):
            while name in by_child:
                name = by_child[name].parent
                needed.add(name)
        self.lean.joints = [j for j in full.joints if j.child in needed]
        shapes = {s.name:s for s in full.shapes}
        self.a = full._tensor(np.stack([shapes[n].a for n in self.links]))
        self.b = full._tensor(np.stack([shapes[n].b for n in self.links]))
        self.radius = full._tensor([shapes[n].radius for n in self.links])
        self.pair_a = torch.tensor([i for i in range(3) for _ in range(3)])
        self.pair_b = torch.tensor([j for _ in range(3) for j in range(3,6)])
        with torch.no_grad():
            rotations, positions = full.fk(samples)
            leg_a,leg_b = self.endpoints(rotations,positions)
            cap_names = [n for side in full.arm_names.values() for n in side if shapes[n].kind=='capsule']
            ar = torch.stack([rotations[n] for n in cap_names],dim=1)
            ap = torch.stack([positions[n] for n in cap_names],dim=1)
            ca = (ar @ full._tensor(np.stack([shapes[n].a for n in cap_names]))[...,None]).squeeze(-1)+ap
            cb = (ar @ full._tensor(np.stack([shapes[n].b for n in cap_names]))[...,None]).squeeze(-1)+ap
            cr = full._tensor([shapes[n].radius for n in cap_names])
            distances = full._segment_distance(leg_a[:,:,None],leg_b[:,:,None],ca[:,None],cb[:,None])-self.radius[None,:,None]-cr[None,None,:]
            self.cap_frame,self.cap_leg,self.cap_arm = torch.where(distances < .30)
            self.ca = ca[self.cap_frame,self.cap_arm]
            self.cb = cb[self.cap_frame,self.cap_arm]
            self.cr = cr[self.cap_arm]
            hand_names = ['lh_hand_base_link','rh_hand_base_link']
            hr = torch.stack([rotations[n] for n in hand_names],dim=1)
            hp = torch.stack([positions[n] for n in hand_names],dim=1)
            hc = full._tensor(np.stack([shapes[n].a for n in hand_names]))
            hh = full._tensor(np.stack([shapes[n].b for n in hand_names]))
            self.t = torch.linspace(0,1,17)
            lengths = torch.linalg.vector_norm(self.b-self.a,dim=-1)
            self.sample_radius = torch.sqrt(self.radius.square()+(lengths/32).square())
            points=leg_a[:,:,None]+self.t[None,None,:,None]*(leg_b-leg_a)[:,:,None]
            local=(points[:,:,None]-hp[:,None,:,None]) @ hr[:,None]
            delta=(local-hc[None,None,:,None]).abs()-hh[None,None,:,None]
            signed=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)
            distances=signed.amin(-1)-self.sample_radius[None,:,None]
            self.box_frame,self.box_leg,self.box_hand=torch.where(distances < .30)
            self.hr=hr[self.box_frame,self.box_hand]
            self.hp=hp[self.box_frame,self.box_hand]
            self.hc=hc[self.box_hand]
            self.hh=hh[self.box_hand]
        print('COLLISION CANDIDATES',len(self.cap_frame),len(self.box_frame),flush=True)

    def endpoints(self, r, p):
        rotation=torch.stack([r[n] for n in self.links],dim=1)
        position=torch.stack([p[n] for n in self.links],dim=1)
        return ((rotation@self.a[...,None]).squeeze(-1)+position,
                (rotation@self.b[...,None]).squeeze(-1)+position)

    def loss(self,q):
        rotations,positions=self.lean.fk(q)
        a,b=self.endpoints(rotations,positions)
        leg=self.full._segment_distance(a[:,self.pair_a],b[:,self.pair_a],a[:,self.pair_b],b[:,self.pair_b])-self.radius[self.pair_a]-self.radius[self.pair_b]
        cap=self.full._segment_distance(a[self.cap_frame,self.cap_leg],b[self.cap_frame,self.cap_leg],self.ca,self.cb)-self.radius[self.cap_leg]-self.cr
        ba,bb=a[self.box_frame,self.box_leg],b[self.box_frame,self.box_leg]
        points=ba[:,None]+self.t[None,:,None]*(bb-ba)[:,None]
        local=(points-self.hp[:,None])@self.hr
        delta=(local-self.hc[:,None]).abs()-self.hh[:,None]
        signed=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.amax(-1).clamp_max(0)
        box=signed.amin(-1)-self.sample_radius[self.box_leg]
        penalty=(torch.relu(.004-leg).square().sum()+torch.relu(.004-cap).square().sum()+torch.relu(.004-box).square().sum())/len(q)
        worst=leg.min()
        if cap.numel():worst=torch.minimum(worst,cap.min())
        if box.numel():worst=torch.minimum(worst,box.min())
        return penalty,worst


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=1400)
    parser.add_argument('--resume_npz',type=Path)
    parser.add_argument('--export_state',action='store_true',help='Validate and export a saved solver state without further optimization')
    args=parser.parse_args(argv)
    from source.paths import resolve_dance_asset
    if args.input:
        args.input = resolve_dance_asset(args.input)
    if args.output:
        args.output = resolve_dance_asset(args.output)
    if args.resume_npz:
        args.resume_npz = resolve_dance_asset(args.resume_npz)
    if args.output.exists():parser.error('Output exists; earlier H5 files are preserved')
    if args.iterations<1:parser.error('Iterations must be positive')
    if args.export_state and not args.resume_npz:parser.error('--export_state requires --resume_npz')
    torch.set_num_threads(4)
    source=args.input.resolve();output=args.output.resolve()
    if not args.resume_npz and any(output.with_suffix(suffix).exists() for suffix in ('.state.npz', '.smooth.json')):
        raise FileExistsError('Solver state/report already exists; choose a new output or explicitly resume it')
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    with h5py.File(source) as f:
        names=list(f['joint_names'].asstr()[:]);delta=f['joint_pos_delta'][:];fps=float(f.attrs['fps'])
        root_quat=f['root_quat_delta_wxyz'][:]
        root_position=f['root_pos_delta'][:]
        root_valid=f['root_valid'][:]
    if len(delta)<20 or not np.isfinite(delta).all():raise ValueError('Need at least 20 finite frames')
    baseline=playback_baseline(names,ROOT/'source/my_task/g1_replay_env_cfg.py')
    model=G1CollisionModel(ROOT/'assets/g1_29dof_o6_hand_V3.usd',names,'cpu')
    # Replay applies delta rotation before its initial 90-degree yaw anchor.
    for frame in range(len(root_valid)):
        if not root_valid[frame]:
            root_quat[frame]=root_quat[frame-1] if frame else [1,0,0,0]
            root_position[frame]=root_position[frame-1] if frame else [0,0,0]
    root_rotation=model._tensor((Rotation.from_quat(root_quat[:,[1,2,3,0]])*Rotation.from_euler('z',90,degrees=True)).as_matrix())
    root_position=model._tensor(root_position)
    legs=[i for i,n in enumerate(names) if any(part in n for part in ('hip_','knee_','ankle_'))]
    current=model._tensor(delta+baseline)
    before,original_feet=foot_metrics(model,current,legs)
    print('BEFORE',before,flush=True)
    feet=original_feet.numpy()
    speed=np.linalg.norm(np.diff(feet,axis=0),axis=-1).max(1)
    acceleration=np.linalg.norm(np.diff(feet,n=2,axis=0),axis=-1).max(1)
    bad=set(np.flatnonzero(speed>.12).tolist())|set((np.flatnonzero(acceleration>.15)+1).tolist())
    windows=[]
    for frame in sorted(bad):
        lo,hi=max(0,frame-32),min(len(delta),frame+34)
        if windows and lo<=windows[-1][1]:windows[-1][1]=max(hi,windows[-1][1])
        else:windows.append([lo,hi])
    print('SEVERE CROSSING WINDOWS',windows,flush=True)
    seed=gaussian_filter1d(current[:,legs].numpy(),sigma=.7,axis=0)
    fade=np.zeros(len(delta),dtype=np.float32)
    target=gaussian_filter1d(feet,sigma=1.1,axis=0)
    for lo,hi in windows:
        t=np.linspace(0,1,hi-lo)[:,None]
        bridge=current[lo,legs].numpy()*(1-t)+current[hi-1,legs].numpy()*t
        strength=np.sin(np.pi*t)**2
        seed[lo:hi]=seed[lo:hi]*(1-.85*strength)+bridge*(.85*strength)
        fade[lo:hi]=np.maximum(fade[lo:hi],np.sin(np.linspace(0,np.pi,hi-lo))**2)
        center=target[lo:hi,:,1].mean(1)
        gap=np.maximum(np.abs(target[lo:hi,0,1]-target[lo:hi,1,1])/2,.09)
        target[lo:hi,0,1]=center+gap
        target[lo:hi,1,1]=center-gap
    if args.resume_npz:
        saved=np.load(args.resume_npz)
        if str(saved['source_sha256'])!=source_hash:raise ValueError('Resume state belongs to another input')
        seed=saved['leg_q']
        if seed.shape!=(len(delta),len(legs)) or not np.isfinite(seed).all():raise ValueError('Invalid solver state')
    variable=torch.nn.Parameter(model._tensor(seed))
    foot_target=model._tensor(target);ordering_fade=model._tensor(fade)
    fractions=(.25,.5,.75)
    def samples(q):return torch.cat([q]+[q[:-1]*(1-t)+q[1:]*t for t in fractions])
    clearance=LegClearance(model,samples(current))
    optimizer=torch.optim.Adam([variable],lr=.006)
    history=[];start_time=time.perf_counter()
    if args.export_state and output.with_suffix('.smooth.json').exists():
        history=json.loads(output.with_suffix('.smooth.json').read_text(encoding='utf-8')).get('history',[])
    for step in range(0 if args.export_state else args.iterations):
        progress=step/max(1,args.iterations-1)
        optimizer.param_groups[0]['lr']=.006*(.02+.98*(1+np.cos(np.pi*progress))/2)
        optimizer.zero_grad(set_to_none=True)
        q=current.clone();q[:,legs]=variable
        rotations,p=clearance.lean.fk(q)
        feet=torch.stack([p[n] for n in FEET],dim=1)
        velocities=feet[1:]-feet[:-1];accelerations=velocities[1:]-velocities[:-1]
        world_feet=(root_rotation[:,None]@feet[...,None]).squeeze(-1)+root_position[:,None]
        world_velocity=world_feet[1:]-world_feet[:-1]
        world_acceleration=world_velocity[1:]-world_velocity[:-1]
        world_foot_rotation=root_rotation[:,None]@torch.stack([rotations[n] for n in FEET],dim=1)
        rotation_velocity=world_foot_rotation[1:]-world_foot_rotation[:-1]
        rotation_acceleration=rotation_velocity[1:]-rotation_velocity[:-1]
        collision,worst=clearance.loss(samples(q))
        tracking=(feet-foot_target).square().sum(-1).mean()
        foot_speed=torch.relu(torch.linalg.vector_norm(velocities,dim=-1)-.045).square().mean()
        foot_acc=torch.relu(torch.linalg.vector_norm(accelerations,dim=-1)-.030).square().mean()
        world_speed=torch.relu(torch.linalg.vector_norm(world_velocity,dim=-1)-.050).square().mean()
        world_acc=torch.relu(torch.linalg.vector_norm(world_acceleration,dim=-1)-.018).square().mean()
        rotation_speed=torch.relu(torch.linalg.matrix_norm(rotation_velocity)-.37).square().mean()
        joint_speed=torch.relu((variable[1:]-variable[:-1]).abs()-.20).square().sum(-1).mean()
        joint_acc=(variable[2:]-2*variable[1:-1]+variable[:-2]).square().sum(-1).mean()
        deviation=(variable-current[:,legs]).square().sum(-1).mean()
        knee_gap=p['left_knee_link'][:,1]-p['right_knee_link'][:,1]
        ankle_gap=feet[:,0,1]-feet[:,1,1]
        ordering=(ordering_fade*(torch.relu(.16-knee_gap).square()+torch.relu(.18-ankle_gap).square())).mean()
        loss=500000*collision+1500*tracking+3000000*foot_speed+500000*foot_acc+10000*accelerations.square().sum(-1).mean()+3000000*world_speed+2000000*world_acc+10000*joint_speed+800*joint_acc+1500*rotation_acceleration.square().sum((-1,-2)).mean()+5000*rotation_speed+2*deviation+50000*ordering
        loss.backward();optimizer.step()
        with torch.no_grad():
            variable.clamp_(model.lower[legs],model.upper[legs])
            variable[:3]=current[:3,legs];variable[-3:]=current[-3:,legs]
        if step%100==0 or step==args.iterations-1:
            entry={'iteration':step,'loss':float(loss.detach()),'foot_max_step_m':float(torch.linalg.vector_norm(velocities.detach(),dim=-1).max()),'foot_max_second_difference_m':float(torch.linalg.vector_norm(accelerations.detach(),dim=-1).max()),'worst_selected_penetration_m':max(0.,-float(worst.detach())),'elapsed_seconds':time.perf_counter()-start_time}
            history.append(entry);print(entry,flush=True)
            np.savez(output.with_suffix('.state.npz'),leg_q=variable.detach().numpy(),source_sha256=source_hash)
    with torch.no_grad():
        result=current.clone();result[:,legs]=variable
        after,final_feet=foot_metrics(model,result,legs)
        world_feet=(root_rotation[:,None]@final_feet[...,None]).squeeze(-1)+root_position[:,None]
        after['world_foot_max_step_m']=float(torch.linalg.vector_norm(world_feet[1:]-world_feet[:-1],dim=-1).max())
        after['world_foot_max_second_difference_m']=float(torch.linalg.vector_norm(world_feet[2:]-2*world_feet[1:-1]+world_feet[:-2],dim=-1).max())
        after['world_steps_over_7cm']=int((torch.linalg.vector_norm(world_feet[1:]-world_feet[:-1],dim=-1).amax(-1)>.07).sum())
        after['world_second_differences_over_35mm']=int((torch.linalg.vector_norm(world_feet[2:]-2*world_feet[1:-1]+world_feet[:-2],dim=-1).amax(-1)>.035).sum())
        rotations,_=clearance.lean.fk(result)
        foot_rotation=root_rotation[:,None]@torch.stack([rotations[n] for n in FEET],dim=1)
        relative=foot_rotation[:-1].transpose(-1,-2)@foot_rotation[1:]
        cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
        after['world_foot_max_rotation_step_deg']=float(torch.acos(cosine).max()*180/np.pi)
        full_reports=[model.report(result,.004)]+[model.report(result[:-1]*(1-t)+result[1:]*t,.004) for t in fractions]
    report={'input':str(source),'output':str(output),'source_sha256':source_hash,'frames':len(delta),'fps':fps,'severe_windows':windows,'before':before,'after':after,'full_body_collision_after':full_reports,'history':history}
    output.with_suffix('.smooth.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AFTER',after,'FULL BODY COUNTS',[r['penetrating_frames'] for r in full_reports],flush=True)
    # Body-relative acceleration may exceed world acceleration while the legs
    # compensate a fast pelvis turn; judge visible jumps in world coordinates.
    if after['foot_max_step_m']>.060 or after['foot_max_second_difference_m']>.055 or after['leg_max_step_deg']>15:
        raise RuntimeError('Trajectory still has discontinuities; solver state saved, H5 not exported')
    if after['world_foot_max_step_m']>.070 or after['world_foot_max_second_difference_m']>.040:
        raise RuntimeError('World-space foot trajectory still has discontinuities; solver state saved, H5 not exported')
    if after['world_foot_max_rotation_step_deg']>20:
        raise RuntimeError('Foot rotation is discontinuous; solver state saved, H5 not exported')
    if full_reports[0]['penetrating_frames'] or full_reports[2]['penetrating_frames']:
        raise RuntimeError('Full-body collision check failed; solver state saved, H5 not exported')
    repaired=delta.copy();repaired[:,legs]=(result[:,legs].numpy()-baseline[legs]).astype(delta.dtype)
    assert np.isfinite(repaired).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    export_joint_positions(source, output, repaired, dict(
        leg_trajectory_smoothing='full sequence; Cartesian speed/acceleration; joint continuity; collision constraints',
        leg_smoothing_input_sha256=source_hash))
    print('EXPORTED',output,flush=True)


