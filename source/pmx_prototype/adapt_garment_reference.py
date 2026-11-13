"""Garment-aware arm IK retargeting of mocap, never a runtime correction.

Keep pelvis, trunk, legs, contacts and timing byte-identical. Move wrists a few
centimetres laterally only around the hip crossing; preserve fore/aft swing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from retarget_100style import Retarget

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--release',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--idle-only',action='store_true',help='Adapt only idle in an already adapted bank; keep AMP experts unchanged')
p.add_argument('--input-motion',type=Path,help='Adapt one non-bank chronological motion file')
p.add_argument('--motion-output',type=Path)
p.add_argument('--max-joint-change-deg',type=float,default=8.)
p.add_argument('--height-gate',action='store_true',help='Only enforce skirt clearance below the waist region')
p.add_argument('--radial-envelope',action='store_true',help='Use a waist-height radial skirt envelope for bent running arms')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cfg=json.loads((a.release/'env_config.json').read_text());model=Retarget()

def adapt(source,destination,clip=None):
    z=np.load(source);data={k:z[k].copy() for k in z.files};q=(data['q'][clip] if clip is not None else data['q']).astype(np.float64)
    positions=data['positions'][clip].copy() if clip is not None else data['positions'].copy()
    rotations=data['rotations'][clip].copy() if clip is not None else data['rotations'].copy()
    names=list(data['joint_names']);links=list(data['link_names']);assert names==model.names
    pelvis=links.index('pelvis');root=positions[:,pelvis];root_r=R.from_quat(rotations[:,pelvis]).as_matrix();old_q=q.copy();arm_ids=[];peak=0.
    def fk(k,x):
        p,r=model.fk(x,root[k],root_r[k]);return np.array([p[n] for n in links]),np.array([r[n] for n in links])
    for side,sign in [('left',1.),('right',-1.)]:
        ids=[names.index(side+'_'+j+'_joint') for j in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex']];arm_ids+=ids
        hand=links.index(side+'_hand');elbow=links.index(side+'_elbow_flex')
        local=R.from_matrix(root_r).inv().apply(positions[:,hand]-root)
        s=np.clip((np.abs(local[:,1])-.035)/.13,0,1);gate=1-s*s*(3-2*s)
        if clip==3:gate=np.ones_like(gate)
        if a.height_gate:
            h=np.clip((local[:,2]-.02)/.12,0,1);gate*=1-h*h*(3-2*h)
        desired_lateral=np.full(len(local),.184)
        if a.radial_envelope:
            radius=.14+.2*np.clip(-local[:,2],0,.25)
            desired_lateral=np.sqrt(np.maximum(radius**2-local[:,1]**2,0))
        delta=np.maximum(desired_lateral-sign*local[:,0],0)*gate;peak=max(peak,float(delta.max()))
        desired=local.copy();desired[:,0]+=sign*delta;target=root+np.einsum('nij,nj->ni',root_r,desired)
        solved=[]
        for k in range(len(q)):
            def error(x):
                temp=q[k].copy();temp[ids]=x;pp,_=fk(k,temp)
                return np.r_[(pp[hand]-target[k])*100,(pp[elbow]-positions[k,elbow])*8,(x-old_q[k,ids])*.35]
            fit=least_squares(error,q[k,ids],bounds=(model.limits[ids,0]+.01,model.limits[ids,1]-.01),max_nfev=28)
            solved.append(fit.x)
        correction=np.array(solved)-q[:,ids]
        if 'gait_sin' in data:
            from scipy.signal import savgol_filter
            q[:,ids]+=savgol_filter(correction,9,3,axis=0,mode='wrap')
        else:
            spectrum=np.fft.rfft(correction,axis=0);spectrum[9:]=0
            q[:,ids]+=np.fft.irfft(spectrum,n=len(q),axis=0)
    pp=[];rr=[]
    for k in range(len(q)):
        x,y=fk(k,q[k]);pp.append(x);rr.append(y)
    pp=np.array(pp);rr=np.array(rr);quat=R.from_matrix(rr.reshape(-1,3,3)).as_quat().reshape(len(q),len(links),4)
    for k in range(1,len(q)):quat[k][(quat[k]*quat[k-1]).sum(-1)<0]*=-1
    period=float(data['periods'][clip] if clip is not None else data['period']);dt=period/len(q)
    derivative=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
    # Only arm links change. Preserve every other link's state exactly.
    changed=[links.index(side+'_'+j) for side in ['left','right'] for j in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex','hand']]
    fields={k:data[k][clip].copy() if clip is not None else data[k].copy() for k in ['q','qd','positions','rotations','velocities']}
    fields['q'][:,arm_ids]=q[:,arm_ids];fields['qd'][:,arm_ids]=derivative(q)[:,arm_ids]
    old_vel=fields['velocities'];mean_velocity=np.mean(old_vel[:,:,:3]-derivative(positions),axis=(0,1))
    fields['positions'][:,changed]=pp[:,changed];fields['rotations'][:,changed]=quat[:,changed]
    fields['velocities'][:,changed,:3]=(derivative(pp)+mean_velocity)[:,changed]
    omega=R.from_matrix((np.roll(rr,-1,axis=0)@np.roll(rr,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(len(q),len(links),3)/(2*dt)
    fields['velocities'][:,changed,3:]=omega[:,changed]
    for key,val in fields.items():
        if clip is None:data[key]=val.astype(data[key].dtype)
        else:data[key][clip]=val.astype(data[key].dtype)
    np.savez_compressed(destination,**data)
    deficits=[];gates=[]
    for side,sign in [('left',1.),('right',-1.)]:
        local=R.from_matrix(root_r).inv().apply(pp[:,links.index(side+'_hand')]-root);s=np.clip((np.abs(local[:,1])-.035)/.13,0,1);g=1-s*s*(3-2*s)
        if a.height_gate:
            h=np.clip((local[:,2]-.02)/.12,0,1);g*=1-h*h*(3-2*h)
        desired_lateral=np.full(len(local),.175)
        if a.radial_envelope:
            radius=.12+.2*np.clip(-local[:,2],0,.25)
            desired_lateral=np.sqrt(np.maximum(radius**2-local[:,1]**2,0))
        deficits.append(np.maximum(desired_lateral-sign*local[:,0],0)*g);gates.append(g)
    result=dict(source=str(Path(source).resolve()),destination=str(destination.resolve()),clip=clip,peak_requested_wrist_shift_m=peak,max_joint_change_deg=float(np.degrees(np.abs(q-old_q)).max()),adapted_gated_mean_deficit_m=float(np.sum(deficits)/np.maximum(np.sum(gates),1e-6)),non_arm_joint_states_unchanged=bool(np.array_equal(old_q[:,[i for i in range(len(names)) if i not in arm_ids]],q[:,[i for i in range(len(names)) if i not in arm_ids]])))
    assert result['non_arm_joint_states_unchanged']
    assert result['max_joint_change_deg']<a.max_joint_change_deg,'Retargeting change is too large'
    assert result['adapted_gated_mean_deficit_m']<.001,'Adapted reference still conflicts with clearance'
    return result

if a.input_motion:
    if not a.motion_output:p.error('--motion-output required with --input-motion')
    report=adapt(a.input_motion,a.motion_output)
    a.motion_output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(0)
if a.idle_only:
    report=adapt(a.output/'full_motion.npz',a.output/'full_motion_with_idle.npz',3)
    (a.output/'idle_processing.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    override=dict(full_motion_path=str((a.output/'full_motion_with_idle.npz').resolve()),amp_reset_discriminator=False,
                  imitation_motion_probabilities=[0,0,.65,.35,0,0,0,0,0,0,0],idle_yaw_weight=2.,idle_upper_pose_weight=2.,diagnostic_controller_trace=False)
    (a.output/'idle_training_override.json').write_text(json.dumps(override,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));raise SystemExit(0)
reports=[adapt(cfg['full_motion_path'],a.output/'full_motion.npz',2)]
paths=[]
for i,source in enumerate(cfg['amp_expert_paths']):
    dest=a.output/f'amp_expert_{i}.npz';reports.append(adapt(source,dest));paths.append(str(dest.resolve()))
(a.output/'processing.json').write_text(json.dumps(dict(method=__doc__,reports=reports),indent=2),encoding='utf-8')
override=dict(full_motion_path=str((a.output/'full_motion.npz').resolve()),amp_expert_paths=paths,amp_reset_discriminator=True,train_upper_std_cap=.08,train_entropy_coef=.001,diagnostic_controller_trace=False)
(a.output/'training_override.json').write_text(json.dumps(override,indent=2),encoding='utf-8');print(json.dumps(reports,indent=2))
