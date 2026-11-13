"""Extend corrected running references to 31 DOF with spatial arm and wrist IK.

Keep the previous pelvis, trunk, legs, contacts and clock. Add arm clearance in
the reference and restore captured forearm/hand orientations with rest-basis
calibration. No runtime correction is involved.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R,Slerp
from retarget_100style import Retarget,read_bvh

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/anon_locomotion/running_wrists';OUT.mkdir(exist_ok=True,parents=True)
old=Retarget();model=Retarget(ROOT/'output/anon_physical_v4/manifest.json')
motion=np.load(ROOT/'output/anon_locomotion/running_mocap/corrected/long_balanced.npz')
nodes,sp,sr,dt=read_bvh(ROOT/'media/mocap/100style/Neutral_FR.bvh');ids={n['name']:i for i,n in enumerate(nodes)}
count=len(motion['q']);frames=2550+np.arange(count)*189/count
source_rotation=lambda name:Slerp(np.arange(2550,2740),R.from_matrix(sr[2550:2740,ids[name]]))(frames).as_matrix()
root_id=list(motion['link_names']).index('pelvis');trunk_id=list(motion['link_names']).index('waist_pitch')
rp=motion['positions'][:,root_id];rr=R.from_quat(motion['rotations'][:,root_id]).as_matrix()
body_align=rr@source_rotation('Hips').transpose(0,2,1)
q=np.zeros((count,len(model.names)))
oldmap=[model.names.index(n) for n in old.names];q[:,oldmap]=motion['q']
report=dict(method=__doc__,source_frames=[2550,2739],source='100STYLE Neutral_FR.bvh',arms={})
for side,sign in [('left',1.),('right',-1.)]:
    cap=side.title();joint_names=[side+'_'+n+'_joint' for n in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex','forearm_twist','wrist_pitch','wrist_deviation']]
    ji=[model.names.index(n) for n in joint_names]
    along=model.origins[side+'_hand']-model.origins[side+'_elbow_flex'];along/=np.linalg.norm(along)
    # BVH forearm and hand point along signed X in their rest basis.
    correction=R.align_vectors([[sign,0,0]],[along])[0].as_matrix()
    forearm_rot=body_align@source_rotation(cap+'Elbow')@correction
    hand_rot=body_align@source_rotation(cap+'Wrist')@correction
    trunk_r=R.from_quat(motion['rotations'][:,trunk_id]).as_matrix()
    elbow=motion['positions'][:,list(motion['link_names']).index(side+'_elbow_flex')].copy()
    wrist=motion['positions'][:,list(motion['link_names']).index(side+'_hand')].copy()
    torso=motion['positions'][:,trunk_id]
    # Modest soft widening, preserving original fore/aft paths and heights.
    e_local=np.einsum('nij,nj->ni',trunk_r.transpose(0,2,1),elbow-torso)
    w_local=np.einsum('nij,nj->ni',trunk_r.transpose(0,2,1),wrist-torso)
    e_shift=np.full(count,.025)
    w_shift=np.full(count,.045)
    e_local[:,0]+=sign*e_shift;w_local[:,0]+=sign*w_shift
    elbow_target=torso+np.einsum('nij,nj->ni',trunk_r,e_local)
    hand_target=torso+np.einsum('nij,nj->ni',trunk_r,w_local)
    initial=q[:,ji].copy()
    for k in range(count):
        seed=q[k,ji] if k==0 else q[k-1,ji]
        def error(x):
            temp=q[k].copy();temp[ji]=x;p,r=model.fk(temp,rp[k],rr[k])
            return np.r_[35*(p[side+'_elbow_flex']-elbow_target[k]),
                         45*(p[side+'_hand']-hand_target[k]),
                         1.0*R.from_matrix(forearm_rot[k].T@r[side+'_forearm_twist']).as_rotvec(),
                         1.5*R.from_matrix(hand_rot[k].T@r[side+'_hand']).as_rotvec(),
                         .04*(x-seed)]
        fit=least_squares(error,seed,bounds=(model.limits[ji,0]+.01,model.limits[ji,1]-.01),max_nfev=65)
        q[k,ji]=fit.x
    # Keep the loop bridge already present in the base passage; bridge only the
    # new arm correction, then use local cleanup (no cycle averaging).
    delta=q[:,ji]-initial
    delta=savgol_filter(delta,5,2,axis=0)
    n=8;start=count-n;u=delta[start].copy();v=delta[0].copy()
    du=(delta[start]-delta[start-1])*n;dv=(delta[1]-delta[0])*n
    for j in range(n):
        s=j/n;delta[start+j]=(2*s**3-3*s*s+1)*u+(s**3-2*s*s+s)*du+(-2*s**3+3*s*s)*v+(s**3-s*s)*dv
    q[:,ji]=initial+delta
    q[:,ji]=np.clip(q[:,ji],model.limits[ji,0]+.01,model.limits[ji,1]-.01)
    report['arms'][side]=dict(requested_elbow_shift_cm=(100*e_shift).tolist(),requested_hand_shift_cm=(100*w_shift).tolist())

arm_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['shoulder','elbow','forearm','wrist'])]
arm_links=[n for n in model.links if n.startswith(('left_shoulder','right_shoulder','left_elbow','right_elbow','left_forearm','right_forearm','left_wrist','right_wrist')) or n in ['left_hand','right_hand']]

def extend(source,run_q=None):
    data={k:source[k].copy() for k in source.files};bank=data['q'].ndim==3
    periods=data['periods'] if bank else [float(data['period'])]
    qs=data['q'] if bank else data['q'][None]
    newq=np.zeros(qs.shape[:-1]+(len(model.names),));newq[...,oldmap]=qs
    if run_q is not None:newq[2 if bank else 0]=run_q
    qd=np.zeros_like(newq);oldqd=data['qd'] if bank else data['qd'][None];qd[...,oldmap]=oldqd
    oldlinks=list(data['link_names']);newlinks=model.links
    oldp=data['positions'] if bank else data['positions'][None]
    oldr=data['rotations'] if bank else data['rotations'][None]
    oldv=data['velocities'] if bank else data['velocities'][None]
    pos=np.zeros(newq.shape[:2]+(len(newlinks),3));rot=np.zeros(newq.shape[:2]+(len(newlinks),4));vel=np.zeros(newq.shape[:2]+(len(newlinks),6))
    for c,period in enumerate(periods):
        step=period/newq.shape[1];deriv=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*step)
        qd[c][:,arm_ids]=deriv(newq[c])[:,arm_ids]
        roots=oldp[c,:,oldlinks.index('pelvis')];rots=R.from_quat(oldr[c,:,oldlinks.index('pelvis')]).as_matrix()
        for k in range(len(roots)):
            p,r=model.fk(newq[c,k],roots[k],rots[k])
            for j,name in enumerate(newlinks):pos[c,k,j]=p[name];rot[c,k,j]=R.from_matrix(r[name]).as_quat()
        for k in range(1,len(roots)):rot[c,k][(rot[c,k]*rot[c,k-1]).sum(-1)<0]*=-1
        meanv=np.mean(oldv[c,:,:,:3]-deriv(oldp[c]),axis=(0,1))
        vel[c,:,:,:3]=deriv(pos[c])+meanv
        matrices=R.from_quat(rot[c].reshape(-1,4)).as_matrix().reshape(len(roots),len(newlinks),3,3)
        vel[c,:,:,3:]=R.from_matrix((np.roll(matrices,-1,axis=0)@np.roll(matrices,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(len(roots),len(newlinks),3)/(2*step)
        # Byte-preserve the base dynamics outside the arms.
        for j,name in enumerate(newlinks):
            if name in oldlinks and name not in arm_links:
                oi=oldlinks.index(name);pos[c,:,j]=oldp[c,:,oi];rot[c,:,j]=oldr[c,:,oi];vel[c,:,j]=oldv[c,:,oi]
    for key,value in [('q',newq),('qd',qd),('positions',pos),('rotations',rot),('velocities',vel)]:data[key]=(value if bank else value[0]).astype('float32')
    data['joint_names']=np.array(model.names);data['link_names']=np.array(newlinks)
    return data

ref=extend(motion,q);np.savez_compressed(OUT/'running_reference.npz',**ref)
bank=extend(np.load(ROOT/'output/anon_locomotion/running_mocap/corrected/full_motion.npz'),q)
np.savez_compressed(OUT/'full_motion.npz',**bank)
for side,sign in [('left',1.),('right',-1.)]:
    j=model.links.index(side+'_hand');ej=model.links.index(side+'_elbow_flex');tr=model.links.index('waist_pitch')
    inv=R.from_quat(ref['rotations'][:,tr]).inv()
    report['arms'][side].update(hand_lateral_cm=np.percentile(sign*inv.apply(ref['positions'][:,j]-ref['positions'][:,tr])[:,0]*100,[10,50,90]).tolist(),elbow_lateral_cm=np.percentile(sign*inv.apply(ref['positions'][:,ej]-ref['positions'][:,tr])[:,0]*100,[10,50,90]).tolist())
report.update(new_dofs=len(model.names),max_joint_speed=float(np.abs(ref['qd']).max()),non_arm_joint_positions_unchanged=bool(np.array_equal(ref['q'][:,[model.names.index(n) for n in old.names if not any(s in n for s in ['shoulder','elbow'])]],motion['q'][:,[i for i,n in enumerate(old.names) if not any(s in n for s in ['shoulder','elbow'])]])))
(OUT/'reference_processing.json').write_text(json.dumps(report,indent=2))
print(json.dumps({s:{k:v for k,v in report['arms'][s].items() if not k.startswith('requested')} for s in ['left','right']},indent=2));print('Max qd',report['max_joint_speed'])
