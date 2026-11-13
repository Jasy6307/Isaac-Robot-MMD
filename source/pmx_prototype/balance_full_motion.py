"""Adapt captured pelvis translation to this body's mass distribution and support."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from retarget_100style import Retarget

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();data=dict(np.load(args.input));model=Retarget();links=list(data['link_names'])
orig=model.origins;center={n:v.copy() for n,v in orig.items()}
rest=np.array(model.manifest['rest_positions']);bones=model.manifest['bone_names']
center['waist_pitch']=(orig['waist_pitch']+orig['head'])/2
center['head']=(orig['head']+rest[bones.index('頭')]+[0,0,.12])/2
corners=np.load(Path(__file__).resolve().parents[2]/'output/anon_locomotion/full_motion/forward.npz')['foot_corners']
for k,s in enumerate(['left','right']):
    center[s+'_hip_pitch']=(orig[s+'_hip_pitch']+orig[s+'_knee_pitch'])/2
    center[s+'_knee_pitch']=(orig[s+'_knee_pitch']+orig[s+'_ankle_roll'])/2
    center[s+'_ankle_roll']=orig[s+'_ankle_roll']+corners[k].mean(0)
    center[s+'_shoulder_yaw']=(orig[s+'_shoulder_yaw']+orig[s+'_elbow_flex'])/2
    center[s+'_elbow_flex']=(orig[s+'_elbow_flex']+orig[s+'_hand'])/2
    center[s+'_hand']=orig[s+'_hand']+.11*(orig[s+'_hand']-orig[s+'_elbow_flex'])
offset=np.array([center[n]-orig[n] for n in links]);mass=np.array([model.manifest['links'][n]['mass'] for n in links])
p=data['positions'].copy();r=R.from_quat(data['rotations'].reshape(-1,4)).as_matrix().reshape(len(p),len(links),3,3)
q=data['q'].copy();count=len(q);dt=float(data['period'])/count;pelvis=links.index('pelvis')
feet=[links.index(s+'_ankle_roll') for s in ['left','right']]
target_feet=p[:,feet].copy();target_rotation=r[:,feet].copy();original_root=p[:,pelvis].copy()
com=lambda pp,rr:np.average(pp+np.einsum('nbij,bj->nbi',rr,offset),axis=1,weights=mass)
initial_com=com(p,r);height=float(initial_com[:,2].mean());frequency=2*np.pi*np.fft.rfftfreq(count,dt)
sole=np.einsum('nfij,fkj->nfki',r[:,feet],corners)+p[:,feet,None,:]
near=sole[:,:,:,2]<sole[:,:,:,2].min(-1)[:,:,None]+.003
cop=(sole*near[:,:,:,None]).sum(2)/near.sum(2)[:,:,None]
support=[]
for k,contact in enumerate(data['contacts']>.5):
    if contact.sum()==1:support.append(cop[k,contact][0,:2])
    else:
        delta=cop[k,1,:2]-cop[k,0,:2]
        a=np.clip(np.dot(initial_com[k,:2]-cop[k,0,:2],delta)/max(np.dot(delta,delta),1e-6),0,1)
        support.append(cop[k,0,:2]+a*delta)
support=np.array(support)
spectrum=np.fft.rfft(support,axis=0)/(1+(height/9.81)*frequency[:,None]**2);spectrum[7:]=0
desired_com=np.fft.irfft(spectrum,n=count,axis=0)
leg_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['hip_','knee_','ankle_'])]
for iteration in range(2):
    correction=(desired_com-com(p,r)[:,:2])/.8
    length=np.linalg.norm(correction,axis=-1);correction*=np.minimum(1,.15/np.maximum(length,1e-6))[:,None]
    desired_root=p[:,pelvis].copy();desired_root[:,:2]+=correction
    seed=q.copy();previous=q[0,leg_ids].copy();previous_delta=np.zeros_like(previous)
    for tick in range(3*count):
        k=tick%count
        root=desired_root[k].copy();root_r=r[k,pelvis]
        original=seed[k].copy();height0=root[2]
        def error(x):
            temp=original.copy();temp[leg_ids]=x[:-1];origin=root.copy();origin[2]=x[-1]
            pp,rr=model.fk(temp,origin,root_r);result=[]
            for f,s in enumerate(['left','right']):
                name=s+'_ankle_roll';result.extend((pp[name]-target_feet[k,f])/.008)
                result.extend(((rr[name]-target_rotation[k,f])/.25).ravel())
            result.extend((x[:-1]-original[leg_ids])/.9);result.append((x[-1]-height0)/.025)
            result.extend((x[:-1]-previous-previous_delta)/.25)
            return result
        lo=np.r_[model.limits[leg_ids,0]+.005,.8];hi=np.r_[model.limits[leg_ids,1]-.005,1.03]
        for j,index in enumerate(leg_ids):
            if 'knee_pitch' in model.names[index]:lo[j]=max(lo[j],-.02)
        fit=least_squares(error,np.clip(np.r_[previous+previous_delta,height0],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=35)
        previous_delta=np.clip(fit.x[:-1]-previous,-.12,.12);previous=fit.x[:-1].copy()
        q[k,leg_ids]=fit.x[:-1];root[2]=fit.x[-1]
        pp,rr=model.fk(q[k],root,root_r);p[k]=np.array([pp[n] for n in links]);r[k]=np.array([rr[n] for n in links])
derivative=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
command=data['command'];angular=np.array([0.,0.,command[2]])
velocity=derivative(p)+np.array([command[1],-command[0],0])+np.cross(angular,p)
omega=R.from_matrix((np.roll(r,-1,axis=0)@np.roll(r,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(count,len(links),3)/(2*dt)+angular
data.update(q=q.astype('float32'),qd=derivative(q).astype('float32'),positions=p.astype('float32'),
            rotations=R.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(count,len(links),4).astype('float32'),
            velocities=np.concatenate([velocity,omega],-1).astype('float32'))
np.savez_compressed(args.output,**data)
report=dict(source=str(args.input),method='Mass-aware support/pelvis retarget; captured contacts and foot paths retained',
            max_pelvis_planar_correction_m=float(np.linalg.norm(p[:,pelvis,:2]-original_root[:,:2],axis=-1).max()),
            com_tracking_rmse_m=float(np.sqrt(np.mean((com(p,r)[:,:2]-desired_com)**2))),
            max_leg_speed_rad_s=float(np.abs(data['qd'][:,leg_ids]).max()))
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
