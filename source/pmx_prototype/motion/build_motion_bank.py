"""Combine full retargeted mocap states and captured idle into a motion bank."""

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
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from retarget_100style import Retarget

ROOT=PROJECT_ROOT
folder=ROOT/'output/anon_locomotion/full_motion'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clips',nargs='+',default=['forward','forward_060','forward_040','idle'])
parser.add_argument('--output',type=Path,default=folder/'forward_bank.npz')
args=parser.parse_args()
model=Retarget()
if 'idle' in args.clips:
    raw=ROOT/'output/anon_locomotion/100style_reference'
    d=np.load(raw/'ID.npz');q=np.load(folder/(args.clips[0]+'.npz'))['idle'].copy()
    angles=Rotation.from_matrix(d['root_rotations']).as_euler('ZYX').mean(0);angles[0]=0
    rot=Rotation.from_euler('ZYX',angles).as_matrix()
    root=np.zeros(3);p,r=model.fk(q,root,rot)
    corners=np.load(folder/'forward.npz')['foot_corners']
    minimum=min(((r[s+'_ankle_roll']@corners[k].T).T+p[s+'_ankle_roll'])[:,2].min() for k,s in enumerate(['left','right']))
    root[2]=.002-minimum;p,r=model.fk(q,root,rot)
    feet=[s+'_ankle_roll' for s in ['left','right']]
    targets=np.array([p[n] for n in feet]);foot_rotation=[]
    for k,n in enumerate(feet):
        yaw=Rotation.from_matrix(r[n]).as_euler('ZYX')[0]
        orientation=Rotation.from_euler('Z',yaw).as_matrix();foot_rotation.append(orientation)
        targets[k,2]=.002-(orientation@corners[k].T).T[:,2].min()
    leg_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['hip_','knee_','ankle_'])]
    original=q.copy();initial_height=root[2]
    def error(x):
        temp=q.copy();temp[leg_ids]=x[:-1];origin=root.copy();origin[2]=x[-1]
        pp,rr=model.fk(temp,origin,rot);result=[]
        for k,n in enumerate(feet):
            result.extend((pp[n]-targets[k])/.008);result.extend(((rr[n]-foot_rotation[k])/.15).ravel())
        result.extend((x[:-1]-original[leg_ids])/.7);result.append((x[-1]-initial_height)/.025)
        return result
    lo=np.r_[model.limits[leg_ids,0]+.005,.82];hi=np.r_[model.limits[leg_ids,1]-.005,1.02]
    fit=least_squares(error,np.clip(np.r_[q[leg_ids],root[2]],lo+1e-5,hi-1e-5),bounds=(lo,hi),max_nfev=60)
    q[leg_ids]=fit.x[:-1];root[2]=fit.x[-1];p,r=model.fk(q,root,rot)
    pose=np.array([p[n] for n in model.links]);quat=Rotation.from_matrix(np.array([r[n] for n in model.links])).as_quat()
    np.savez_compressed(folder/'idle.npz',q=np.tile(q,(128,1)).astype('float32'),qd=np.zeros((128,25),np.float32),
                        positions=np.tile(pose,(128,1,1)).astype('float32'),rotations=np.tile(quat,(128,1,1)).astype('float32'),
                        velocities=np.zeros((128,len(model.links),6),np.float32),contacts=np.ones((128,2),np.float32),
                        joint_names=model.names,link_names=model.links,period=1.5,speed=0.,command=[0,0,0],idle=q)
records=[np.load(folder/(name+'.npz')) for name in args.clips]
data={key:np.stack([record[key] for record in records]) for key in ['q','qd','positions','rotations','velocities','contacts']}
np.savez_compressed(args.output,**data,joint_names=records[0]['joint_names'],link_names=records[0]['link_names'],
                    idle=records[0]['idle'],periods=[float(d['period']) for d in records],speeds=[float(d['speed']) for d in records],
                    commands=[d['command'].tolist() if 'command' in d else [float(d['speed']),0,0] for d in records],clip_names=args.clips)
args.output.with_suffix('.json').write_text(json.dumps(dict(clips=args.clips,source='100STYLE Neutral; full-body retargeted states'),indent=2),encoding='utf-8')
print(args.output)
