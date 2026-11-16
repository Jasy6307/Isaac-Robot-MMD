"""Retarget a contiguous 100STYLE passage without averaging its gait cycles.

Remove low-frequency trajectory heading (the capture walks around the room),
adapt contacts to straight 0.4 m/s progression, and bridge only the final 0.25 s.
Interior samples and unequal step intervals remain in chronological order.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks,savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from retarget_100style import Retarget,read_bvh
from full_body_retarget import head_calibration,retarget_upper
from plant_mocap_feet import plant_feet

ROOT=PROJECT_ROOT
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--first',type=int,default=4250)
p.add_argument('--last',type=int,default=4742)
p.add_argument('--speed',type=float,default=.4)
p.add_argument('--clip',default='Neutral_FW.bvh')
p.add_argument('--running',action='store_true')
p.add_argument('--direction',choices=['forward','backward','left'],default='forward')
p.add_argument('--trajectory-cutoff',type=float,default=.35,help='Planar path removal cutoff; independent of heading smoothing')
p.add_argument('--phase-source',choices=['toe-height','ankle-direction'],default='toe-height')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
model=Retarget();nodes,sp,sr,source_dt=read_bvh(ROOT/'media/mocap/100style'/a.clip);ids={n['name']:i for i,n in enumerate(nodes)}
assert 0<a.first<a.last<len(sp)
low=lambda x:sosfiltfilt(butter(2,.35,fs=1/source_dt,output='sos'),x,axis=0)
yaw=np.unwrap(R.from_matrix(sr[:,0]).as_euler('ZYX')[:,0]);heading=low(yaw)
canonical=R.from_euler('Z',-heading[:,None]).as_matrix();aligned=canonical@sr[:,0]
trajectory=sosfiltfilt(butter(2,a.trajectory_cutoff,fs=1/source_dt,output='sos'),sp[:,0],axis=0)
local_velocity=np.einsum('nij,nj->ni',canonical,np.gradient(trajectory,source_dt,axis=0))
axis=0 if a.direction=='left' else 1
sign=-1. if a.direction=='forward' else 1.
native_speed=float(np.median(np.maximum(sign*local_velocity[a.first:a.last,axis],.2))*.9132680182298859)
ratio=(a.speed/native_speed)**(.15 if a.running else .35);stride=(a.speed/native_speed)/ratio
native_duration=(a.last-a.first)*source_dt;duration=native_duration/ratio
count=round(duration*30);dt=duration/count
frames=a.first+np.arange(count)*(a.last-a.first)/count
interp=lambda x:np.stack([np.interp(frames,np.arange(len(x)),x.reshape(len(x),-1)[:,k]) for k in range(int(np.prod(x.shape[1:])))],-1).reshape((count,)+x.shape[1:])
idn,idp,idr,_=read_bvh(ROOT/'media/mocap/100style/Neutral_ID.bvh');idids={n['name']:i for i,n in enumerate(idn)}
idframes=np.load(ROOT/'output/anon_locomotion/100style_reference/ID.npz')['source_frames'];offset=head_calibration(idr,idids,idframes)
seed=np.zeros(25);seed[model.names.index('left_shoulder_roll_joint')]=.9;seed[model.names.index('right_shoulder_roll_joint')]=-.9
source_frames=np.unique(np.r_[np.floor(frames).astype(int),np.ceil(frames).astype(int)]);qraw=[]
for k,f in enumerate(source_frames):
    seed=model.frame(sp[f],sr[f],ids,seed)
    seed=retarget_upper(model,sp[f],sr[f],ids,seed,offset)
    qraw.append(seed.copy())
    if k%60==0:print('CONTIGUOUS_RETARGET',k,len(source_frames),flush=True)
q=np.stack([np.interp(frames,source_frames,np.array(qraw)[:,j]) for j in range(25)],-1)
# Short local noise cleanup, never a per-cycle average or low harmonic template.
q=savgol_filter(q,5,2,axis=0)
scale=.9132680182298859
floor=float(np.percentile(sp[a.first:a.last,[ids['LeftToe'],ids['RightToe']],2],15))*scale
hip_offset=model.origins['pelvis'][2]-(model.origins['left_hip_yaw'][2]+model.origins['right_hip_yaw'][2])/2
root=np.zeros((count,3));sway=sp[:,0]-trajectory;root[:,:2]=interp(np.einsum('nij,nj->ni',canonical,sway))[:,:2]*scale
root[:,axis]*=stride;root[:,2]=interp(sp[:,0])[:,2]*scale+hip_offset-floor
root[:,:2]-=root[:,:2].mean(0)
euler=R.from_matrix(aligned).as_euler('ZYX');euler[:,0]=np.unwrap(euler[:,0]);root_angles=interp(euler)
velocity=np.gradient(sp,source_dt,axis=0);ankles=[];foot_angles=[];contacts=[]
for side in ['Left','Right']:
    ankle=ids[side+'Ankle'];toe=ids[side+'Toe']
    relative=np.einsum('nij,nj->ni',canonical,sp[:,ankle]-sp[:,0])*scale
    relative=interp(relative)
    baseline=model.origins[side.lower()+'_hip_yaw'][axis] if axis==0 else 0.
    relative[:,axis]=baseline+(relative[:,axis]-baseline)*stride;target=root+relative
    # Pelvis origin is above the anatomical hip center. This offset belongs
    # only to the pelvis, not the absolute captured ankle height.
    target[:,2]-=hip_offset
    angle=R.from_matrix(canonical@sr[:,ankle]).as_euler('ZYX');angle[:,0]=np.unwrap(angle[:,0]);angle=interp(angle)
    stable=((np.linalg.norm(velocity[:,toe,:2],axis=-1)<.3)&(sp[:,toe,2]<floor/scale+.035))|((np.linalg.norm(velocity[:,ankle,:2],axis=-1)<.25)&(sp[:,ankle,2]<floor/scale+.15))
    contact=np.interp(frames,np.arange(len(sp)),stable.astype(float))>.5
    # Hold stance foot yaw after removing path curvature; keep captured toe/heel roll.
    edges=np.diff(np.r_[False,contact,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    anchor={}
    for first,last in zip(starts,ends):
        value=float(np.median(angle[first:last,0]));angle[first:last,0]=value;anchor[first]=value;anchor[last-1]=value
    if anchor:
        ix=np.array(sorted(anchor));angle[:,0]=np.interp(np.arange(count),ix,[anchor[i] for i in ix])
    ankles.append(target);foot_angles.append(angle);contacts.append(contact)
ankles=np.stack(ankles,1);angles=np.stack(foot_angles,1);contact=np.stack(contacts,1)

def seam(x):
    n=max(3,round(.25/dt));start=len(x)-n;length=n*dt;y=x.copy()
    u=x[start].copy();v=x[0].copy();du=(x[start]-x[start-1])/dt;dv=(x[1]-x[0])/dt
    for k in range(n):
        s=k/n;y[start+k]=(2*s**3-3*s**2+1)*u+(s**3-2*s**2+s)*length*du+(-2*s**3+3*s**2)*v+(s**3-s**2)*length*dv
    return y
q=seam(q);root=seam(root);root_angles=seam(root_angles);ankles=seam(ankles);angles=seam(angles)
rotation=R.from_euler('ZYX',root_angles).as_matrix();foot_rotation=R.from_euler('ZYX',angles.reshape(-1,3)).as_matrix().reshape(count,2,3,3)
baseline=np.load(ROOT/'output/anon_locomotion/full_motion/forward_040_aligned.npz');corners=baseline['foot_corners'];feet=[model.links.index(s+'_ankle_roll') for s in ['left','right']]
leg_ids=[i for i,n in enumerate(model.names) if any(t in n for t in ['hip_','knee_','ankle_'])]
mean_velocity=np.zeros(3);mean_velocity[axis]=sign*a.speed
for k in range(count):
    for f in range(2):
        sole=(foot_rotation[k,f]@corners[f].T).T[:,2].min();minimum=.002-sole
        ankles[k,f,2]=minimum if contact[k,f] else max(minimum,ankles[k,f,2])
    original=q[k].copy();height=root[k,2]
    def error(x):
        temp=original.copy();temp[leg_ids]=x[:-1];origin=root[k].copy();origin[2]=x[-1];pp,rr=model.fk(temp,origin,rotation[k]);out=[]
        for f,side in enumerate(['left','right']):
            name=side+'_ankle_roll';out.extend((pp[name]-ankles[k,f])/.012);out.extend(((rr[name]-foot_rotation[k,f])/.4).ravel())
        return np.r_[out,(x[:-1]-original[leg_ids])/.7,(x[-1]-height)/.025]
    lo=np.r_[model.limits[leg_ids,0]+.005,.80];hi=np.r_[model.limits[leg_ids,1]-.005,1.04]
    fit=least_squares(error,np.clip(np.r_[q[k,leg_ids],height],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=35)
    q[k,leg_ids]=fit.x[:-1];root[k,2]=fit.x[-1]
    if k%60==0:print('CONTIGUOUS_LEG_IK',k,count,flush=True)
def fk_all():
    pp=[];rr=[]
    for k in range(count):
        p,r=model.fk(q[k],root[k],rotation[k]);pp.append([p[n] for n in model.links]);rr.append([r[n] for n in model.links])
    return np.array(pp),np.array(rr)
pp,rr=fk_all();desired=plant_feet(pp[:,feet],rr[:,feet],corners,contact,mean_velocity,dt)
for k in range(count):
    original=q[k].copy();height=root[k,2];orientation=rr[k,feet].copy()
    def error(x):
        temp=original.copy();temp[leg_ids]=x[:-1];origin=root[k].copy();origin[2]=x[-1];p,r=model.fk(temp,origin,rotation[k]);out=[]
        for f,side in enumerate(['left','right']):
            name=side+'_ankle_roll';out.extend((p[name]-desired[k,f])/.008);out.extend(((r[name]-orientation[f])/.25).ravel())
        return np.r_[out,(x[:-1]-original[leg_ids])/.8,(x[-1]-height)/.03]
    lo=np.r_[model.limits[leg_ids,0]+.005,.80];hi=np.r_[model.limits[leg_ids,1]-.005,1.04]
    fit=least_squares(error,np.clip(np.r_[q[k,leg_ids],height],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=35)
    q[k,leg_ids]=fit.x[:-1];root[k,2]=fit.x[-1]
    if k%60==0:print('CONTIGUOUS_CONTACT_IK',k,count,flush=True)
# Only a 5-sample local cleanup, retaining slower stride differences.
q=savgol_filter(q,5,2,axis=0,mode='wrap');q=np.clip(q,model.limits[:,0]+.005,model.limits[:,1]-.005)
pp,rr=fk_all();sole=np.einsum('nfij,fkj->nfki',rr[:,feet],corners)+pp[:,feet,None,:]
shift=savgol_filter(.002-sole[:,:,:,2].min((1,2)),5,2,mode='wrap')
if a.running:shift=np.where(contact.any(-1),shift,np.maximum(shift,0))
root[:,2]+=shift;pp,rr=fk_all()
deriv=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
quat=R.from_matrix(rr.reshape(-1,3,3)).as_quat().reshape(count,len(model.links),4)
for k in range(1,count):quat[k][(quat[k-1]*quat[k]).sum(-1)<0]*=-1
omega=R.from_matrix((np.roll(rr,-1,axis=0)@np.roll(rr,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(count,len(model.links),3)/(2*dt)
if a.phase_source=='ankle-direction':
    ankle_local=np.einsum('nji,nj->ni',sr[:,0],sp[:,ids['LeftAnkle']]-sp[:,0])
    peaks,_=find_peaks(savgol_filter(ankle_local[:,axis],7,2),distance=30 if a.running else 55,prominence=.1)
else:peaks,_=find_peaks(sp[:,ids['LeftToe'],2],distance=25 if a.running else 55,prominence=.035)
peaks=peaks[(peaks>=a.first)&(peaks<=a.last)]
assert len(peaks)>=4
gait=np.interp(frames,peaks,np.arange(len(peaks)))
phase=.9921875+gait
sole=np.einsum('nfij,fkj->nfki',rr[:,feet],corners)+pp[:,feet,None,:];bottom=sole[:,:,:,2].min(-1);near=sole[:,:,:,2]<bottom[:,:,None]+.003;sv=deriv(sole)+mean_velocity
slip=(np.linalg.norm(sv[:,:,:,:2],axis=-1)*near).sum(-1)/near.sum(-1)
np.savez_compressed(a.output,q=q.astype('float32'),qd=deriv(q).astype('float32'),positions=pp.astype('float32'),rotations=quat.astype('float32'),velocities=np.concatenate([deriv(pp)+mean_velocity,omega],-1).astype('float32'),contacts=contact.astype('float32'),joint_names=model.names,link_names=model.links,idle=baseline['idle'],period=duration,speed=a.speed,command=[-mean_velocity[1],mean_velocity[0],0],foot_corners=corners,gait_sin=np.sin(2*np.pi*phase).astype('float32'),gait_cos=np.cos(2*np.pi*phase).astype('float32'),expert_valid_range=[.3/duration,1-.3/duration],source_frames=frames)
report=dict(source='100STYLE Neutral_FW.bvh, Ian Mason et al., CC BY 4.0',source_first=a.first,source_last=a.last,source_duration_seconds=native_duration,retimed_duration_seconds=duration,frames=count,source_left_swing_peaks=peaks.tolist(),consecutive_gait_cycles=len(peaks)-1,cycle_averaging=False,loop_bridge_seconds=.25,source_trajectory_heading_removed_deg=float(np.degrees(np.ptp(heading[a.first:a.last]))),kinematic_stance_slip_m_s=float(slip[contact].mean()),max_joint_velocity_rad_s=float(np.abs(deriv(q)).max()),max_joint_acceleration_rad_s2=float(np.abs(deriv(deriv(q))).max()),processing=['Chronological anatomical retargeting','Remove low-frequency path heading, retain pelvis gait rotation','Uniform speed retiming; unequal step intervals preserved','Straight-line contact-consistent leg IK','Local 5-sample cleanup','Only final 0.25 seconds bridged for playback loop'])
report.update(source=f'100STYLE {a.clip}, Ian Mason et al., CC BY 4.0',running=a.running,source_contact_fraction=contact.mean(0).tolist(),source_flight_fraction=float((~contact.any(-1)).mean()),trajectory_cutoff_hz=a.trajectory_cutoff,phase_source=a.phase_source,source_cycle_peaks=peaks.tolist())
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
