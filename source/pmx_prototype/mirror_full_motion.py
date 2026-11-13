"""Mirror a complete mocap reference, then recompute FK on the real body."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from retarget_100style import Retarget

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args()
data=dict(np.load(args.input));model=Retarget();names=list(data['joint_names']);links=list(data['link_names'])
opposite=lambda n:n.replace('left_','SIDE_').replace('right_','left_').replace('SIDE_','right_')
indices=[names.index(opposite(n)) for n in names]
sign=np.array([-1 if any(s in n for s in ['yaw','roll','elbow']) else 1 for n in names])
q=data['q'][:,indices]*sign
shift=int(q[:,names.index('left_knee_pitch_joint')].argmax())
q=np.roll(q,-shift,axis=0)
source_p=np.roll(data['positions'],-shift,axis=0);source_r=np.roll(data['rotations'],-shift,axis=0)
pelvis=links.index('pelvis');reflection=np.diag([-1.,1.,1.]);poses=[];rotations=[]
for k in range(len(q)):
    p=source_p[k,pelvis].copy();p[0]*=-1
    r=reflection@R.from_quat(source_r[k,pelvis]).as_matrix()@reflection
    pp,rr=model.fk(q[k],p,r);poses.append([pp[n] for n in links]);rotations.append([rr[n] for n in links])
poses=np.array(poses);rotations=np.array(rotations);dt=float(data['period'])/len(q)
derivative=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
command=data['command'].copy();command[1:]*=-1
angular=np.array([0.,0.,command[2]])
velocity=derivative(poses)+np.array([command[1],-command[0],0])+np.cross(angular,poses)
omega=R.from_matrix((np.roll(rotations,-1,axis=0)@np.roll(rotations,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(len(q),len(links),3)/(2*dt)+angular
data.update(q=q.astype('float32'),qd=derivative(q).astype('float32'),positions=poses.astype('float32'),
            rotations=R.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(len(q),len(links),4).astype('float32'),
            velocities=np.concatenate([velocity,omega],-1).astype('float32'),command=command,
            contacts=np.roll(data['contacts'][:,::-1],-shift,axis=0))
np.savez_compressed(args.output,**data)
from pathlib import Path
Path(args.output).with_suffix('.json').write_text(json.dumps(dict(source=args.input,operation='mirror and anatomical FK',left_knee_phase_shift=shift),indent=2),encoding='utf-8')
print(args.output)
