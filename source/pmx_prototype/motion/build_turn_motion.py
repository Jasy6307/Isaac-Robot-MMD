"""Contact-preserving steering retarget of a complete captured walking cycle."""

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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from retarget_100style import Retarget
from plant_mocap_feet import cleanup_world_feet

ROOT=PROJECT_ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--forward',type=float,default=.15)
parser.add_argument('--yaw',type=float,default=.6)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
base=np.load(ROOT/'output/anon_locomotion/full_motion/forward_040_aligned.npz');model=Retarget()
count=len(base['q']);period=float(base['period']);dt=period/count;total=3*count
time=np.arange(total)*dt;theta=args.yaw*time
heading=R.from_euler('Z',theta[:,None]).as_matrix()
if abs(args.yaw)>1e-6:
    path=np.stack([args.forward/args.yaw*(1-np.cos(theta)),-args.forward/args.yaw*np.sin(theta),np.zeros(total)],-1)
else:path=np.stack([np.zeros(total),-args.forward*time,np.zeros(total)],-1)
links=list(base['link_names']);feet=[links.index(s+'_ankle_roll') for s in ['left','right']];pelvis=links.index('pelvis')
positions=np.tile(base['positions'],(3,1,1));rotations=np.tile(R.from_quat(base['rotations'].reshape(-1,4)).as_matrix().reshape(count,len(links),3,3),(3,1,1,1))
relative=positions[:,feet]-positions[:,pelvis,None,:]
average=relative[:,:,1].mean(0)
relative[:,:,1]=average[None]+(relative[:,:,1]-average[None])*(args.forward/.4)
positions[:,pelvis,1]*=args.forward/.4
foot_local=positions[:,pelvis,None,:]+relative
foot_world=np.einsum('nij,nfj->nfi',heading,foot_local)+path[:,None,:]
foot_world_rotation=heading[:,None]@rotations[:,feet]
contact=np.tile(base['contacts']>.5,(3,1));corners=base['foot_corners']
foot_angles=R.from_matrix(foot_world_rotation.reshape(-1,3,3)).as_euler('ZYX').reshape(total,2,3)
air=gaussian_filter1d((~contact).astype(float),2.,axis=0)
factor=.55+.45*min(1.,abs(args.forward)/.4)
foot_angles[:,:,1:]*=(1-(1-factor)*air)[:,:,None]
for foot in range(2):
    yaw=np.unwrap(foot_angles[:,foot,0])
    edges=np.diff(np.r_[False,contact[:,foot],False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    for a,b in zip(starts,ends):yaw[a:b]=yaw[a]
    for end,start in zip(ends[:-1],starts[1:]):
        a=end-1;b=start
        if b<=a:continue
        s=np.arange(b-a+1)/(b-a);w=10*s**3-15*s**4+6*s**5
        yaw[a:b+1]=yaw[a]*(1-w)+yaw[b]*w
    foot_angles[:,foot,0]=yaw
foot_world_rotation=R.from_euler('ZYX',foot_angles.reshape(-1,3)).as_matrix().reshape(total,2,3,3)
desired=cleanup_world_feet(foot_world,foot_world_rotation,corners,contact,dt)
target_position=np.einsum('nji,nfj->nfi',heading,desired-path[:,None,:])[count:2*count]
target_rotation=(heading.transpose(0,2,1)[:,None]@foot_world_rotation)[count:2*count]
root_positions=positions[count:2*count,pelvis].copy()
root_rotation=rotations[count:2*count,pelvis].copy()
q=base['q'].copy();leg_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['hip_','knee_','ankle_'])]
previous=q[0,leg_ids].copy();previous_delta=np.zeros_like(previous)
for tick in range(3*count):
    k=tick%count
    original=base['q'][k].copy();height=root_positions[k,2]
    def error(x):
        temp=original.copy();temp[leg_ids]=x[:-1];origin=root_positions[k].copy();origin[2]=x[-1]
        p,r=model.fk(temp,origin,root_rotation[k]);result=[]
        for foot,side in enumerate(['left','right']):
            name=side+'_ankle_roll';result.extend((p[name]-target_position[k,foot])/.008)
            result.extend(((r[name]-target_rotation[k,foot])/.2).ravel())
        result.extend((x[:-1]-original[leg_ids])/.9);result.append((x[-1]-height)/.03)
        result.extend((x[:-1]-previous-previous_delta)/.20)
        return result
    lo=np.r_[model.limits[leg_ids,0]+.005,.8];hi=np.r_[model.limits[leg_ids,1]-.005,1.03]
    fit=least_squares(error,np.clip(np.r_[previous+previous_delta,height],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=45)
    previous_delta=np.clip(fit.x[:-1]-previous,-.12,.12);previous=fit.x[:-1].copy()
    q[k,leg_ids]=fit.x[:-1];root_positions[k,2]=fit.x[-1]
pp=[];rr=[]
for k in range(count):
    p,r=model.fk(q[k],root_positions[k],root_rotation[k]);pp.append([p[n] for n in links]);rr.append([r[n] for n in links])
p=np.array(pp);r=np.array(rr);derivative=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
angular=np.array([0.,0.,args.yaw])
v=derivative(p)+np.array([0.,-args.forward,0.])+np.cross(angular,p)
omega=R.from_matrix((np.roll(r,-1,axis=0)@np.roll(r,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(count,len(links),3)/(2*dt)+angular
data={k:base[k] for k in base.files}
data.update(q=q.astype('float32'),qd=derivative(q).astype('float32'),positions=p.astype('float32'),
            rotations=R.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(count,len(links),4).astype('float32'),
            velocities=np.concatenate([v,omega],-1).astype('float32'),command=[args.forward,0.,args.yaw],speed=abs(args.forward))
np.savez_compressed(args.output,**data)
report=dict(source='100STYLE Neutral FW complete cycle; contact-preserving steering retarget',forward=args.forward,yaw=args.yaw,
            period_seconds=period,pelvis_vertical_range_m=float(np.ptp(p[:,pelvis,2])),max_leg_speed_rad_s=float(np.max(np.abs(derivative(q)[:,leg_ids]))))
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
