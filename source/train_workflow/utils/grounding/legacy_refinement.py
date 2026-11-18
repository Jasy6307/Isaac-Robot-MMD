"""Compatibility profiles for pre-repaired-IK Cartesian refinement.

These profiles retain their original floor-only constraint and numerical knobs.
They are deliberately distinct from the stronger repaired-sole publication gate.
"""
from pathlib import Path
import argparse,json,time,hashlib,shutil
ROOT=Path(__file__).resolve().parents[4]


def build_parser(default_method='smooth'):
    p=argparse.ArgumentParser(description='Historical Cartesian refinement; new repaired-IK work should use smooth.')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--range-start',type=int,default=0);p.add_argument('--range-end',type=int,default=0)
    p.add_argument('--accel-weight',type=float,default=None);p.add_argument('--jerk-weight',type=float,default=None)
    p.add_argument('--resume-checkpoint',type=Path,help='Resume from a previously validated absolute-joint checkpoint')
    p.add_argument('--device',default='cuda:0');p.add_argument('--window-frames',type=int,default=60);p.add_argument('--stride',type=int,default=30)
    p.add_argument('--sweeps',type=int,default=2);p.add_argument('--window-limit',type=int,default=0)
    p.add_argument('--method',choices=('fidelity','smooth'),default=default_method)
    return p


def main(argv=None, *, default_method='smooth'):
    a=build_parser(default_method).parse_args(argv)
    if a.accel_weight is None:a.accel_weight=0. if a.method=='fidelity' else .002
    if a.jerk_weight is None:a.jerk_weight=0. if a.method=='fidelity' else .000003
    run(a)
    return 0


def run(a):
    import numpy as np,h5py,torch
    from scipy.spatial.transform import Rotation,Slerp
    from source.train_workflow.utils.motion.self_collision import G1CollisionModel
    from source.train_workflow.utils.motion.joint_constraints import read_constraints,validate_joint_positions
    from source.train_workflow.utils.motion.constrained_trajectory import refine_feasible_trajectory
    from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg
    from .motion import read_motion
    from .pipeline import require_same_reference
    a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    def read(path):
     motion=read_motion(path)
     return motion.names,motion.q,motion.baseline,motion.fps,motion.root_pos,motion.root_rot
    reference=read_motion(a.reference);seed_motion=read_motion(a.seed)
    if not require_same_reference(reference,seed_motion):
     raise ValueError('Historical refinement seed must preserve reference root channels')
    names,raw,base,fps,rp,rw=read(a.reference);sn,current,*_=read(a.seed)
    assert names==sn and current.shape==raw.shape
    seed=current.copy()
    if a.resume_checkpoint:
     current=np.load(a.resume_checkpoint)['q'].astype(np.float32);assert current.shape==raw.shape
    count=len(raw);usd=ROOT/'assets/g1_29dof_o6_hand_V3.usd'
    model=G1CollisionModel(usd,names,a.device,articulated_hands=True);columns=np.asarray(model.adjustable)
    joints,mapping,groups,constraints=read_constraints(usd,names)
    lower=np.array([joints[n].lower for n in names]);upper=np.array([joints[n].upper for n in names])
    step=np.array([min(joints[n].velocity,abs(mapping[n][1])*constraints[mapping[n][0]][2])/fps for n in names])
    accel=np.maximum(60.,abs(np.diff(raw,n=2,axis=0)).max(0)*fps**2+.01)
    accel=np.maximum(accel,abs(np.diff(seed,n=2,axis=0)).max(0)*fps**2+.01)
    links=['lh_hand_base_link','rh_hand_base_link','left_wrist_roll_link','right_wrist_roll_link','left_elbow_link','right_elbow_link','left_hip_roll_link','right_hip_roll_link','left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link']
    weights=np.sqrt(np.array([8000,8000,4000,4000,1800,1800,200,200,2000,2000,12000,12000]))
    feet=links[-2:];sole=np.stack([RobotLeg(usd,s).sole for s in ['left','right']])
    with torch.no_grad():
     rr,pp=model.fk(model._tensor(raw));sr,sp=model.fk(model._tensor(seed))
     target_p={n:pp[n].cpu().numpy() for n in links};target_r={n:rr[n].cpu().numpy() for n in [*links[:2],*feet]}
     # Floor bounds permit no additional sinking relative to the validated seed or original animation.
     def heights(r,p,lo,hi):
      fr=np.stack([r[n].cpu().numpy() for n in feet],1);fp=np.stack([p[n].cpu().numpy() for n in feet],1)
      corners=np.einsum('nfij,fkj->nfki',fr,sole)+fp[:,:,None]
      return (np.einsum('nij,nfkj->nfki',rw[lo:hi],corners)+rp[lo:hi,None,None])[...,2].min((1,2))
     floor_limit=np.minimum(-.001,np.minimum(heights(rr,pp,0,count),heights(sr,sp,0,count)))-1e-6

    fractions=[.25,.5,.75]
    coordinates=np.concatenate([np.arange(count),*[np.arange(count-1)+f for f in fractions]])
    world_rot=Slerp(np.arange(count),Rotation.from_matrix(rw))(coordinates).as_matrix()
    low=np.floor(coordinates).astype(int);high=np.minimum(low+1,count-1);alpha=coordinates-low
    world_pos=rp[low]*(1-alpha[:,None])+rp[high]*alpha[:,None]
    def floor_samples(q):
     values=np.concatenate([q,*[q[:-1]*(1-f)+q[1:]*f for f in fractions]])
     with torch.no_grad():
      r,p=model.fk(model._tensor(values))
      fr=np.stack([r[n].cpu().numpy() for n in feet],1);fp=np.stack([p[n].cpu().numpy() for n in feet],1)
      corners=np.einsum('nfij,fkj->nfki',fr,sole)+fp[:,:,None]
      return (np.einsum('nij,nfkj->nfki',world_rot,corners)+world_pos[:,None,None])[...,2].reshape(len(values),-1)
    floor_sample_limit=np.minimum(-.001,np.minimum(floor_samples(raw).min(1),floor_samples(seed).min(1)))-1e-6

    def distances(q):
     values=np.concatenate([q,*[q[:-1]*(1-f)+q[1:]*f for f in [.25,.5,.75]]])
     result=[]
     with torch.no_grad():
      for chunk in np.array_split(values,max(1,int(np.ceil(len(values)/128)))):
       result.append(model.distances(*model.fk(model._tensor(chunk))).cpu().numpy())
     return np.concatenate(result)

    def metrics(q):
     with torch.no_grad():
      r,p=model.fk(model._tensor(q))
      delta=q-raw
      return {'joint_rms_deg':float(np.degrees(np.sqrt(np.mean(delta**2)))),
       'hand_error_mean_m':float(np.mean([np.linalg.norm(p[n].cpu().numpy()-target_p[n],axis=1).mean() for n in links[:2]])),
       'elbow_error_mean_m':float(np.mean([np.linalg.norm(p[n].cpu().numpy()-target_p[n],axis=1).mean() for n in links[4:6]])),
       'knee_error_mean_m':float(np.mean([np.linalg.norm(p[n].cpu().numpy()-target_p[n],axis=1).mean() for n in links[8:10]])),
       'foot_error_mean_m':float(np.mean([np.linalg.norm(p[n].cpu().numpy()-target_p[n],axis=1).mean() for n in feet]))}

    reports=[];started=time.monotonic();length=min(count,a.window_frames);stride=a.stride
    range_end=a.range_end or count
    starts=list(range(a.range_start,max(a.range_start+1,range_end-length+1),stride))
    if starts[-1]!=range_end-length:starts.append(range_end-length)
    if a.window_limit:starts=starts[:a.window_limit]
    print('FIDELITY_START',len(starts),'windows',len(columns),'joints',metrics(seed),flush=True)
    for sweep in range(a.sweeps):
     for index,lo in enumerate(starts if sweep%2==0 else list(reversed(starts))):
      hi=lo+length
      root_indices=np.concatenate([np.arange(lo,hi),*[count+i*(count-1)+np.arange(lo,hi-1) for i in range(3)]])
      def batched_sample(qb):
       batch,n,channels=qb.shape
       with torch.no_grad():
        r,p=model.fk(model._tensor(qb.reshape(-1,channels)))
        position=np.concatenate([(p[name].cpu().numpy().reshape(batch,n,3)-target_p[name][None,lo:hi])*weights[i] for i,name in enumerate(links)],axis=2)
        orientation=np.concatenate([(r[name].cpu().numpy().reshape(batch,n,3,3)-target_r[name][None,lo:hi]).reshape(batch,n,-1)*np.sqrt(20 if name in feet else 60) for name in target_r],axis=2)
        pair=((p[links[0]]-p[links[1]]).cpu().numpy().reshape(batch,n,3)-(target_p[links[0]][None,lo:hi]-target_p[links[1]][None,lo:hi]))*np.sqrt(8000)
        residual=np.concatenate([position,orientation,pair,(qb-raw[None,lo:hi])[:,:,columns]*2],axis=2)
        hands=np.stack([p[name].cpu().numpy().reshape(batch,n,3) for name in links[:2]],axis=2)
        samples=np.concatenate([qb,*[qb[:,:-1]*(1-f)+qb[:,1:]*f for f in [.25,.5,.75]]],axis=1)
        sr,sp=model.fk(model._tensor(samples.reshape(-1,channels)))
        dc=model.distances(sr,sp).cpu().numpy().reshape(batch,4*n-3,-1)
        fr=np.stack([sr[name].cpu().numpy().reshape(batch,4*n-3,3,3) for name in feet],2)
        fp=np.stack([sp[name].cpu().numpy().reshape(batch,4*n-3,3) for name in feet],2)
        corners=np.einsum('bnfij,fkj->bnfki',fr,sole)+fp[:,:,:,None]
        floor=(np.einsum('nij,bnfkj->bnfki',world_rot[root_indices],corners)+world_pos[None,root_indices,None,None])[...,2].reshape(batch,4*n-3,-1)
        floor=floor-floor_sample_limit[None,root_indices,None]+.0005
        dc=np.concatenate([dc,floor],axis=-1)
        return residual.astype(float),hands.astype(float),dc.astype(float)
      def sample(q):return tuple(x[0] for x in batched_sample(q[None]))
      def jacobian(q,cols,epsilon):
       qb=np.repeat(q[None],len(cols)*2,axis=0)
       for i,c in enumerate(cols):qb[2*i,:,c]+=epsilon;qb[2*i+1,:,c]-=epsilon
       values=batched_sample(qb)
       return tuple(np.moveaxis((v[::2]-v[1::2])/(2*epsilon),0,-1) for v in values)
      sample.jacobian=jacobian
      def feasible(q):
       if not np.isfinite(q).all() or np.any(q<lower-1e-6) or np.any(q>upper+1e-6):return False
       if np.any(abs(np.diff(q,axis=0))>step+1e-5):return False
       if len(q)>2 and np.any(abs(np.diff(q,n=2,axis=0))*fps**2>accel+.02):return False
       if distances(q).min()<.0005-1e-7:return False
       with torch.no_grad():
        r,p=model.fk(model._tensor(q))
        if np.any(heights(r,p,lo,hi)<floor_limit[lo:hi]-1e-6):return False
       return True
      result,report=refine_feasible_trajectory(current[lo:hi],columns,lower[columns],upper[columns],step[columns],accel[columns]/fps**2,
          fps,(np.full((length-1,2),1e5) if a.method=='fidelity' else None),
          (np.full((length-2,2),1e5) if a.method=='fidelity' else None),sample,feasible,iterations=a.iterations,
          fixed_endpoints=((2 if a.method=='fidelity' else 4) if lo else 0,(2 if a.method=='fidelity' else 4) if hi<count else 0),collision_fractions=(.25,.5,.75),temporal_weights=(0.,a.accel_weight,a.jerk_weight))
      current[lo:hi]=result;report['frames']=[lo,hi];report['sweep']=sweep;reports.append(report)
      status={'sweep':sweep+1,'window':index+1,'windows':len(starts),'elapsed_seconds':time.monotonic()-started,'accepted_steps':sum(v['accepted_steps'] for v in reports)}
      a.output.with_suffix('.status.json').write_text(json.dumps(status))
      print('FIDELITY_WINDOW',status,report['initial_objective'],report['final_objective'],flush=True)
      if index%(20 if a.method=='fidelity' else 5)==0:np.savez_compressed(a.output.with_suffix('.checkpoint.npz'),q=current)
    minimum=float(distances(current).min());limits=validate_joint_positions(current,names,fps,usd)
    assert minimum>=.0005-1e-7 and limits['passed'],(minimum,limits)
    result=raw-base
    with h5py.File(a.reference) as f:result=f['joint_pos_delta'][:]
    changed=current!=raw;result[changed]=(current-base)[changed]
    encoded=result+base;encoded_clearance=float(distances(encoded).min())
    assert encoded_clearance>=.0005-1e-7
    summary={'method':('full-body-cartesian-feasible-sqp' if a.method=='fidelity' else 'local-temporally-smoothed-cartesian-feasible-sqp'),'frames':count,'fps':fps,'passed':True,'seed':metrics(seed),'after':metrics(encoded),
     'minimum_clearance_m':encoded_clearance,'joint_constraints':limits,'joint_acceleration_caps_rad_s2':accel.tolist(),
     'floor_validation':'no additional per-frame sinking relative to source and safe seed','reference':str(a.reference.resolve()),'seed_file':str(a.seed.resolve()),
     'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'seed_sha256':hashlib.sha256(a.seed.read_bytes()).hexdigest(),'temporal_weights':[0.,a.accel_weight,a.jerk_weight],
     'elapsed_seconds':time.monotonic()-started,'accepted_steps':sum(r['accepted_steps'] for r in reports),'windows':reports}
    a.output.with_suffix('.collision.json').write_text(json.dumps(summary,indent=2))
    with a.output.open('xb'):pass
    shutil.copyfile(a.reference,a.output)
    with h5py.File(a.output,'r+') as f:
     f['joint_pos_delta'][:]=result
     for k in list(f.attrs):
      if k.startswith('collision_') or k in ['without_collision_validated','arm_motion_validated']:del f.attrs[k]
     f.attrs['without_collision_validated']=True;f.attrs['collision_repair_mode']='cartesian-fidelity'
     f.attrs['collision_validation']='selected articulated USD envelopes at frames and 1/4,1/2,3/4 samples; clearance >=0.5mm'
     f.attrs['collision_continuity_validated']=True;f.attrs['collision_hand_geometry_mode']=model.hand_geometry_mode
     f.attrs['collision_reference_source_sha256']=summary['reference_sha256'];f.attrs['all_joint_constraints_validated']=True
    print('FIDELITY_COMPLETE',summary['seed'],summary['after'],minimum,flush=True)

    return summary
