"""Bound retargeted joint speed/acceleration while preserving captured poses."""

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
from scipy.optimize import minimize,LinearConstraint
from scipy.spatial.transform import Rotation as R
from retarget_100style import Retarget

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();data=dict(np.load(args.input));model=Retarget()
q=data['q'].astype(float);original=q.copy();count=len(q);dt=float(data['period'])/count
eye=np.eye(count);first=np.roll(eye,-1,axis=1)-eye;second=np.roll(eye,-1,axis=1)-2*eye+np.roll(eye,1,axis=1)
constraint=LinearConstraint(np.vstack([first,second]),np.r_[np.full(count,-8*dt),np.full(count,-200*dt**2)],np.r_[np.full(count,8*dt),np.full(count,200*dt**2)])
leg_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['hip_','knee_','ankle_'])]
for j in leg_ids:
    target=q[:,j].copy()
    def objective(x):
        delta=x-target;curvature=second@x
        return float(delta@delta+.2*(curvature@curvature)),2*delta+.4*second.T@curvature
    fit=minimize(objective,target,jac=True,method='SLSQP',constraints=constraint,
                 bounds=[tuple(model.limits[j])]*count,options={'maxiter':100,'ftol':1e-9})
    if not fit.success:raise RuntimeError(f'{model.names[j]} smoothing failed: {fit.message}')
    q[:,j]=fit.x
links=list(data['link_names']);pelvis=links.index('pelvis');p=[];r=[]
for k in range(count):
    pp,rr=model.fk(q[k],data['positions'][k,pelvis],R.from_quat(data['rotations'][k,pelvis]).as_matrix())
    p.append([pp[n] for n in links]);r.append([rr[n] for n in links])
p=np.array(p);r=np.array(r);feet=[links.index(s+'_ankle_roll') for s in ['left','right']]
corners=data['foot_corners'];sole=np.einsum('nfij,fkj->nfki',r[:,feet],corners)+p[:,feet,None,:]
shift=.002-sole[:,:,:,2].min((1,2));spectrum=np.fft.rfft(shift);spectrum[7:]=0
p[:,:,2]+=np.fft.irfft(spectrum,n=count)[:,None]
derivative=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
command=data['command'];angular=np.array([0.,0.,command[2]])
v=derivative(p)+np.array([command[1],-command[0],0])+np.cross(angular,p)
omega=R.from_matrix((np.roll(r,-1,axis=0)@np.roll(r,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(count,len(links),3)/(2*dt)+angular
data.update(q=q.astype('float32'),qd=derivative(q).astype('float32'),positions=p.astype('float32'),
            rotations=R.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(count,len(links),4).astype('float32'),
            velocities=np.concatenate([v,omega],-1).astype('float32'))
np.savez_compressed(args.output,**data)
report=dict(source=str(args.input),method='Periodic constrained trajectory fit; <=8 rad/s and <=200 rad/s^2',
            maximum_joint_change_deg=float(np.rad2deg(np.abs(q-original).max())),max_leg_speed_rad_s=float(np.abs(data['qd'][:,leg_ids]).max()))
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
