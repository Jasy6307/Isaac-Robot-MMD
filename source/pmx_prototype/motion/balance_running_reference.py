"""Mass-aware running reference adaptation with horizontal flight constraints.

This changes the offline reference only. No forces or root corrections are
applied to a policy rollout. Captured arm/trunk angles and contact timing stay.
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
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from retarget_100style import Retarget

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=dict(np.load(a.input));model=Retarget();links=list(d['link_names']);orig=model.origins;center={n:v.copy() for n,v in orig.items()};rest=np.array(model.manifest['rest_positions']);bones=model.manifest['bone_names']
center['waist_pitch']=(orig['waist_pitch']+orig['head'])/2;center['head']=(orig['head']+rest[bones.index('頭')]+[0,0,.12])/2
corners=d['foot_corners']
for k,s in enumerate(['left','right']):
    center[s+'_hip_pitch']=(orig[s+'_hip_pitch']+orig[s+'_knee_pitch'])/2;center[s+'_knee_pitch']=(orig[s+'_knee_pitch']+orig[s+'_ankle_roll'])/2;center[s+'_ankle_roll']=orig[s+'_ankle_roll']+corners[k].mean(0)
    center[s+'_shoulder_yaw']=(orig[s+'_shoulder_yaw']+orig[s+'_elbow_flex'])/2;center[s+'_elbow_flex']=(orig[s+'_elbow_flex']+orig[s+'_hand'])/2;center[s+'_hand']=orig[s+'_hand']+.11*(orig[s+'_hand']-orig[s+'_elbow_flex'])
offset=np.array([center[n]-orig[n] for n in links]);mass=np.array([model.manifest['links'][n]['mass'] for n in links]);q=d['q'].astype(float);positions=d['positions'].astype(float);rot=R.from_quat(d['rotations'].reshape(-1,4)).as_matrix().reshape(len(q),len(links),3,3)
n=len(q);dt=float(d['period'])/n;pelvis=links.index('pelvis');feet=[links.index(s+'_ankle_roll') for s in ['left','right']];contact=d['contacts']>.5;air=~contact.any(-1);initial=positions[:,pelvis].copy();targetfeet=positions[:,feet].copy();targetrot=rot[:,feet].copy();original_q=q.copy()
eye=np.eye(n);second=(np.roll(eye,-1,axis=1)-2*eye+np.roll(eye,1,axis=1))/dt**2
com=lambda p,r:np.average(p+np.einsum('nbij,bj->nbi',r,offset),axis=1,weights=mass)
leg=[i for i,name in enumerate(model.names) if any(s in name for s in ['hip_','knee_','ankle_'])]
for iteration in range(3):
    c=com(positions,rot);sole=np.einsum('nfij,fkj->nfki',rot[:,feet],corners)+positions[:,feet,None,:];near=sole[:,:,:,2]<sole[:,:,:,2].min(-1)[:,:,None]+.003;cop=(sole*near[:,:,:,None]).sum(2)/near.sum(2)[:,:,None]
    support=np.sum(cop[:,:,:2]*contact[:,:,None],axis=1)/np.maximum(contact.sum(-1),1)[:,None]
    g_eff=np.clip(9.81+second@c[:,2],4.,25.);alpha=c[:,2]/g_eff
    matrix=eye-alpha[:,None]*second;matrix[air]=second[air]*.06;support[air]=0.
    mat=np.vstack([matrix,eye*.12]);rhs=np.vstack([support,c[:,:2]*.12]);desired=np.linalg.lstsq(mat,rhs,rcond=None)[0]
    correction=(desired-c[:,:2])/.85;length=np.linalg.norm(correction,axis=-1);correction*=np.minimum(1.,.12/np.maximum(length,1e-6))[:,None]
    root=positions[:,pelvis].copy();root[:,:2]+=correction
    for k in range(n):
        original=q[k].copy();height=root[k,2];root_r=rot[k,pelvis].copy()
        def error(x):
            temp=original.copy();temp[leg]=x[:-1];origin=root[k].copy();origin[2]=x[-1];pp,rr=model.fk(temp,origin,root_r);out=[]
            for f,side in enumerate(['left','right']):
                name=side+'_ankle_roll';out.extend((pp[name]-targetfeet[k,f])/.008);out.extend(((rr[name]-targetrot[k,f])/.3).ravel())
            return np.r_[out,(x[:-1]-original[leg])/.9,(x[-1]-height)/.02]
        lo=np.r_[model.limits[leg,0]+.005,.78];hi=np.r_[model.limits[leg,1]-.005,1.04];fit=least_squares(error,np.clip(np.r_[q[k,leg],height],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=35)
        q[k,leg]=fit.x[:-1];root[k,2]=fit.x[-1];pp,rr=model.fk(q[k],root[k],root_r);positions[k]=[pp[name] for name in links];rot[k]=[rr[name] for name in links]
    print('RUNNING_BALANCE',iteration,flush=True)
deriv=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
velocity=deriv(positions)+np.array([d['command'][1],-d['command'][0],0]);omega=R.from_matrix((np.roll(rot,-1,axis=0)@np.roll(rot,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(n,len(links),3)/(2*dt)
d.update(q=q.astype('float32'),qd=deriv(q).astype('float32'),positions=positions.astype('float32'),rotations=R.from_matrix(rot.reshape(-1,3,3)).as_quat().reshape(n,len(links),4).astype('float32'),velocities=np.concatenate([velocity,omega],-1).astype('float32'))
np.savez_compressed(a.output,**d)
report=dict(source=str(a.input.resolve()),method=__doc__,max_root_planar_shift_m=float(np.linalg.norm(positions[:,pelvis,:2]-initial[:,:2],axis=-1).max()),max_leg_joint_change_deg=float(np.degrees(np.abs(q-original_q)).max()),max_joint_velocity_rad_s=float(np.abs(d['qd']).max()),arm_trunk_angles_unchanged=bool(np.array_equal(q[:,[i for i in range(25) if i not in leg]],original_q[:,[i for i in range(25) if i not in leg]])))
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
