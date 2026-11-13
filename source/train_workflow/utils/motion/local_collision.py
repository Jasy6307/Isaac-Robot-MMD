"""Bounded local geometric repair with authored joint-rate constraints.

A verified conservative pose is restored toward the source in bounded windows.
Only collision-involved limbs are changed first; a broader conservative pose
is used if required. Fingers and root channels are preserved. This certifies
geometry and joint continuity, not the optional Cartesian gesture fit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch


FRACTIONS = (.25, .5, .75)
CLEARANCE = .0005


def sampled_poses(q):
    return np.concatenate([q, *[q[:-1]*(1-a)+q[1:]*a for a in FRACTIONS]]) if len(q)>1 else q


def collision_frames(distances, count, threshold=CLEARANCE-1e-7):
    rows=np.flatnonzero(np.min(distances,axis=1)<threshold)
    if count==1:return rows
    frames=set(rows[rows<count].tolist())
    for row in rows[rows>=count]:
        frame=int((row-count)%(count-1));frames.update((frame,frame+1))
    return np.array(sorted(frames),dtype=int)


def affected_columns(model, pairs):
    names=[model.pair_labels[int(i)] for i in pairs];columns=[]
    for side,hand in [('left','lh_'),('right','rh_')]:
        arm=any(hand in n or any(f'{side}_{part}' in n for part in ('shoulder','elbow','wrist')) for n in names)
        leg=any(f'{side}_' in n for n in names) and not any(any(t in n for t in ('lh_','rh_','shoulder','elbow','wrist')) for n in names)
        for col,name in enumerate(model.names):
            if name.startswith(side+'_') and ((arm and any(t in name for t in ('shoulder','elbow','wrist'))) or
                                               (leg and any(t in name for t in ('hip_','knee','ankle')))):
                columns.append(col)
    return sorted(set(columns))


def repair_local_arrays(model, raw, fps, robot_usd, *, max_seconds=60., max_passes=2,
                        window_steps=8, progress=None):
    """Start safe, then restore source motion only through verified steps.

    A time budget bounds pose restoration, never the obligation to remove
    collisions. The last accepted collision-free path is always retained.
    """
    from .self_collision import playback_baseline
    from .joint_constraints import validate_joint_positions
    if not np.isfinite(max_seconds) or max_seconds<=0 or max_passes<1 or window_steps<1:
        raise ValueError('Local repair budgets must be positive')
    raw=np.asarray(raw,dtype=np.float32);started=time.monotonic();deadline=started+max_seconds
    source_limits=validate_joint_positions(raw,model.names,fps,robot_usd)
    if not source_limits['passed']:raise ValueError('Local repair requires a joint-limit/rate/mimic-valid source H5')
    def sample(q):
        poses=sampled_poses(q);values=[]
        with torch.no_grad():
            for lo in range(0,len(poses),128):
                values.append(model.distances(*model.fk(model._tensor(poses[lo:lo+128]))).cpu().numpy())
        return np.concatenate(values)
    initial=sample(raw);before=collision_frames(initial,len(raw));windows=[];current=raw.copy()
    acceleration_caps=np.maximum(60.,abs(np.diff(raw,n=2,axis=0))*fps**2+.001)
    # Parse authored constraints once for fast window/splice validation.
    from .joint_constraints import read_constraints
    joints,mapping,groups,constraints=read_constraints(robot_usd,model.names)
    lower=np.array([joints[n].lower for n in model.names]);upper=np.array([joints[n].upper for n in model.names])
    speed=np.array([min(joints[n].velocity,abs(mapping[n][1])*constraints[mapping[n][0]][2]) for n in model.names])
    def fast_temporal(q,lo=0):
        if np.any(q<lower-1e-6) or np.any(q>upper+1e-6):return False
        if len(q)>1 and np.any(abs(np.diff(q,axis=0))*fps>speed+1e-4):return False
        if len(q)>2 and np.any(abs(np.diff(q,n=2,axis=0))*fps**2>acceleration_caps[lo:lo+len(q)-2]+.01):return False
        return True
    if len(before):
        pairs=np.flatnonzero(np.any(initial<CLEARANCE-1e-7,axis=0));cols=affected_columns(model,pairs)
        baseline=playback_baseline(model.names,Path(__file__).resolve().parents[4]/'source/my_task/g1_replay_env_cfg.py')
        # Use the project's neutral outward arms / separated-leg avoidance pose.
        # Escalation may change more posture, but never accepts an intersecting path.
        seeds=[cols,
               [i for i,n in enumerate(model.names) if n.startswith(('left_','right_'))],
               [i for i,n in enumerate(model.names) if n.startswith(('left_','right_','waist_'))]]
        for attempt,columns in enumerate(seeds):
            candidate=raw.copy();candidate[:,columns]=baseline[columns]
            for name,value in [('left_hip_roll_joint',.125),('right_hip_roll_joint',-.125)]:
                col=model.names.index(name)
                if col in columns:candidate[:,col]=value
            if fast_temporal(candidate) and float(sample(candidate).min())>=.0005:
                current=candidate;break
        else:
            raise RuntimeError('No verified collision-free avoidance seed; source was preserved')
        if progress:progress('collision',0,'Verified collision-free seed; restoring original motion locally')
        # Coarse, uniform restoration also retains derivative feasibility.
        low,high=0.,1.;seed=current.copy()
        for _ in range(min(6,window_steps)):
            if time.monotonic()>=deadline:break
            alpha=(low+high)/2;candidate=seed+alpha*(raw-seed)
            if fast_temporal(candidate) and float(sample(candidate).min())>=.0005:
                low=alpha;current=candidate
            else:high=alpha
        # Restore more motion in overlapping short windows. Two frozen samples
        # at each seam make all cross-window velocities/accelerations explicit.
        length=min(len(raw),max(8,round(1.5*fps)));stride=max(1,length//2)
        starts=list(range(0,max(1,len(raw)-length+1),stride))
        if starts[-1]!=len(raw)-length:starts.append(len(raw)-length)
        for sweep in range(max_passes):
            for index,lo in enumerate(starts if sweep%2==0 else list(reversed(starts))):
                if time.monotonic()>=deadline:break
                hi=lo+length;local=current[lo:hi].copy();target=raw[lo:hi]
                if np.array_equal(local,target):continue
                weight=np.sin(np.linspace(0,np.pi,length))**2
                if lo==0:weight[:length//2]=1.
                else:weight[:2]=0.
                if hi==len(raw):weight[length//2:]=1.
                else:weight[-2:]=0.
                direction=(target-local)*weight[:,None];low,high=0.,1.;best=local
                for step in range(window_steps):
                    if time.monotonic()>=deadline:break
                    alpha=1. if step==0 else (low+high)/2
                    candidate=local+alpha*direction
                    if fast_temporal(candidate,lo) and float(sample(candidate).min())>=.0005:
                        low=alpha;best=candidate
                        if alpha==1.:break
                    else:high=alpha
                current[lo:hi]=best
                windows.append(dict(frames=[lo,hi],pass_index=sweep,restored_fraction=low))
                if progress:progress('collision',(sweep+(index+1)/len(starts))/max_passes,
                    f'Safe local restoration {sweep+1}/{max_passes}, window {index+1}/{len(starts)}, {time.monotonic()-started:.1f}s')
    after=sample(current);remaining=collision_frames(after,len(raw))
    joint_report=validate_joint_positions(current,model.names,fps,robot_usd)
    continuity=fast_temporal(current)
    changed=np.flatnonzero(np.any(current!=raw,axis=1));passed=not len(remaining) and joint_report['passed'] and continuity
    report=dict(method='verified-safe-local-restoration-v1',mode='local',passed=passed,
        frames=len(raw),fps=fps,elapsed_seconds=time.monotonic()-started,max_seconds=max_seconds,
        fidelity_budget_exhausted=time.monotonic()>=deadline,window_steps=window_steps,max_passes=max_passes,
        initial_repair={'skipped':'input already passes geometry and limits'} if not len(before) else {'method':'verified-safe-seed'},
        before={'penetrating_frames':len(before),'minimum_clearance_m':float(initial.min())},
        after={'penetrating_frames':len(remaining),'minimum_clearance_m':float(after.min())},
        remaining_frames=remaining.tolist(),windows=windows,changed_frames=changed.tolist(),
        joint_constraints=joint_report,continuity_passed=continuity,
        joint_rms_change_deg=float(np.degrees(np.sqrt(np.mean((current-raw)**2)))),
        validation='selected articulated USD envelopes at frames and quarter-frame samples; clearance >=0.5mm; authored joint limits/rates/mimic; joint acceleration <= max(source,60rad/s2)',
        gesture_preservation=None)
    return current,report


def repair_local_reference_h5(source, output, robot_usd, progress=None, *, max_seconds=60.,window_steps=8):
    import h5py
    from .self_collision import G1CollisionModel,playback_baseline
    from .joint_constraints import validate_joint_positions
    root=Path(__file__).resolve().parents[4]
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or output.exists():raise FileExistsError(f'Output already exists: {output}')
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();torch.set_num_threads(4)
    with h5py.File(source) as f:
        names=list(f['joint_names'].asstr()[:]);delta=f['joint_pos_delta'][:];fps=float(f.attrs['fps'])
        baseline=np.asarray(f.attrs.get('record_baseline_joint_pos',playback_baseline(names,root/'source/my_task/g1_replay_env_cfg.py')))
        rq=f['root_quat_delta_wxyz'][:];rp=f['root_pos_delta'][:];valid=f['root_valid'][:].astype(bool)
    if (delta.ndim!=2 or not len(delta) or delta.shape[1]!=len(names) or len(set(names))!=len(names) or
        baseline.shape!=(len(names),) or not np.isfinite(delta).all() or not np.isfinite(baseline).all() or not np.isfinite(fps) or fps<=0):
        raise ValueError('Invalid recorded reference trajectory')
    if rq.shape!=(len(delta),4) or rp.shape!=(len(delta),3) or valid.shape!=(len(delta),):raise ValueError('Invalid root dimensions')
    if not np.isfinite(rq[valid]).all() or not np.isfinite(rp[valid]).all() or np.any(np.linalg.norm(rq[valid],axis=1)<1e-8):raise ValueError('Invalid recorded root transforms')
    if progress:progress('geometry',0,'Reading articulated collision geometry')
    model=G1CollisionModel(Path(robot_usd),names,'cpu',articulated_hands=True)
    raw=delta+baseline
    q,report=repair_local_arrays(model,raw,fps,robot_usd,max_seconds=max_seconds,window_steps=window_steps,progress=progress)
    # Preserve unmodified floating-point channels exactly, including source fingers.
    result=delta.copy();changed=q!=raw;result[changed]=(q-baseline)[changed]
    encoded=result+baseline
    with torch.no_grad():
        minimum=min(float(model.distances(*model.fk(model._tensor(poses))).min())
                    for poses in np.array_split(sampled_poses(encoded),max(1,int(np.ceil((4*len(q)-3)/128)))))
    joint_report=validate_joint_positions(encoded,names,fps,robot_usd)
    caps=np.maximum(60.,abs(np.diff(raw,n=2,axis=0))*fps**2+.001)
    continuity=bool(len(q)<3 or np.all(abs(np.diff(encoded,n=2,axis=0))*fps**2<=caps+.01))
    report.update(source_sha256=source_hash,hand_geometry_mode=model.hand_geometry_mode,
                  encoded_minimum_clearance_m=minimum,joint_constraints=joint_report,
                  continuity_passed=continuity)
    report['passed']=bool(report['passed'] and minimum>=CLEARANCE-1e-7 and joint_report['passed'] and continuity)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.collision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:
        np.savez_compressed(output.with_suffix('.candidate.npz'),joint_pos=q,baseline=baseline)
        raise RuntimeError(f'Local collision repair did not pass within its budget; see {output.with_suffix(".collision.json")}')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Source changed during repair')
    fd,temporary=tempfile.mkstemp(prefix='.collision-local-',suffix='.h5',dir=output.parent);os.close(fd)
    try:
        shutil.copy2(source,temporary)
        with h5py.File(temporary,'r+') as f:
            for key in list(f.attrs):
                if key.startswith('collision_') or key in ('without_collision_validated','arm_motion_validated'):del f.attrs[key]
            f['joint_pos_delta'][:]=result
            f.attrs['without_collision_validated']=True
            f.attrs['collision_repair_mode']='local'
            f.attrs['collision_continuity_validated']=True
            f.attrs['collision_continuity_validation']='authored joint rates; joint acceleration <= max(source,60rad/s2)'
            f.attrs['collision_hand_geometry_mode']=model.hand_geometry_mode
            f.attrs['collision_validation']=report['validation']
            f.attrs['collision_reference_source_sha256']=source_hash
            f.attrs['all_joint_constraints_validated']=True;f.attrs['mimic_constraints_validated']=True
            # The optional global Cartesian gesture/absolute-motion budget was not run.
            f.attrs['arm_motion_validated']=False
        # Never expose an empty .h5 reservation: the UI can open it while
        # polling and prevent replacement on Windows. Publish the closed,
        # complete file atomically, with no overwrite on either platform.
        if os.name=='nt':
            os.rename(temporary,output)  # Windows rename fails if target exists.
        else:
            os.link(temporary,output)    # POSIX rename would overwrite a target.
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    if progress:progress('done',1,f'Saved {output.name} in {report["elapsed_seconds"]:.1f}s')
    return report
