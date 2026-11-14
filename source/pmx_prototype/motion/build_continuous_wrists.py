"""31-DOF full-direction bank: preserve accepted legs/trunk and retarget arms."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.optimize import least_squares
from retarget_100style import Retarget,read_bvh
ROOT=PROJECT_ROOT;OUT=ROOT/'output/anon_locomotion/continuous_wrists';OUT.mkdir(exist_ok=True,parents=True)
base=np.load(ROOT/'output/anon_locomotion/release_v6_continuous/full_motion.npz')
m=Retarget(ROOT/'output/anon_physical_v4/manifest.json');old_names=list(base['joint_names']);old_links=list(base['link_names']);mapping=[m.names.index(n) for n in old_names]
C,N=base['q'].shape[:2];q=np.zeros((C,N,31));q[...,mapping]=base['q'];data={k:base[k].copy() for k in base.files}
library=json.load(open(ROOT/'output/anon_locomotion/100style_reference/library.json'))
cache={}
for kind in ['FW','BW','SW','ID']:
    nodes,p,r,dt=read_bvh(ROOT/f'media/mocap/100style/Neutral_{kind}.bvh');ids={n['name']:i for i,n in enumerate(nodes)}
    frames=np.load(ROOT/f'output/anon_locomotion/100style_reference/{kind}.npz')['source_frames']
    cyc=next((x['cycles'][0] for x in library['cycles'] if x['clip']==kind),None)
    first,last=(int(frames[cyc['first']]),int(frames[cyc['last']])) if cyc else (int(frames[0]),int(frames[min(len(frames)-1,100)]))
    t=first+np.arange(N)/N*(last-first)
    cache[kind]={name:Slerp(np.arange(first,last+1),R.from_matrix(r[first:last+1,j]))(t).as_matrix() for name,j in ids.items() if name in ['Hips','LeftElbow','RightElbow','LeftWrist','RightWrist']}
arm_j=[i for i,n in enumerate(m.names) if any(k in n for k in ['shoulder','elbow','forearm','wrist'])]
arm_l=[n for n in m.links if any(k in n for k in ['shoulder','elbow','forearm','wrist','hand'])]
reports=[]
for c in range(C):
    kind='BW' if c==4 else 'SW' if c in [5,6] else 'ID' if c in [3,9,10] else 'FW'
    source=cache[kind];root=base['positions'][c,:,old_links.index('pelvis')];rr=R.from_quat(base['rotations'][c,:,old_links.index('pelvis')]).as_matrix()
    tr=base['positions'][c,:,old_links.index('waist_pitch')];trr=R.from_quat(base['rotations'][c,:,old_links.index('waist_pitch')]).as_matrix()
    for side,sign in [('left',1.),('right',-1.)]:
        cap=side.title();j=[m.names.index(side+'_'+x+'_joint') for x in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex','forearm_twist','wrist_pitch','wrist_deviation']]
        along=m.origins[side+'_hand']-m.origins[side+'_elbow_flex'];along/=np.linalg.norm(along)
        correction=R.align_vectors([[sign,0,0]],[along])[0].as_matrix()
        sr=source[cap+'Wrist'];se=source[cap+'Elbow'];hips=source['Hips']
        if c==6:
            mirror=np.diag([-1.,1.,1.]);other='Right' if side=='left' else 'Left'
            sr=mirror@source[other+'Wrist']@mirror;se=mirror@source[other+'Elbow']@mirror;hips=mirror@hips@mirror
        desired=rr@hips.transpose(0,2,1)@sr@correction
        forearm=rr@hips.transpose(0,2,1)@se@correction
        elbow=base['positions'][c,:,old_links.index(side+'_elbow_flex')].copy();hand=base['positions'][c,:,old_links.index(side+'_hand')].copy()
        # Same gentle offset as the accepted running variant; idle is unchanged.
        if c!=3:
            elbow+=trr[:,:,0]*sign*.025;hand+=trr[:,:,0]*sign*.045
        for k in range(N):
            seed=q[c,k,j].copy() if k==0 else q[c,k-1,j].copy()
            def error(x):
                temp=q[c,k].copy();temp[j]=x;p,r=m.fk(temp,root[k],rr[k])
                return np.r_[35*(p[side+'_elbow_flex']-elbow[k]),45*(p[side+'_hand']-hand[k]),.25*R.from_matrix(forearm[k].T@r[side+'_forearm_twist']).as_rotvec(),1.5*R.from_matrix(desired[k].T@r[side+'_hand']).as_rotvec(),.05*(x-seed)]
            q[c,k,j]=least_squares(error,seed,bounds=(m.limits[j,0]+.01,m.limits[j,1]-.01),max_nfev=40).x
        # Low-pass only arm corrections; retain the base gait and its contacts.
        original=np.zeros((N,31));original[:,mapping]=base['q'][c]
        delta=q[c,:,j].T-original[:,j]
        spec=np.fft.rfft(delta,axis=0);spec[7:]=0;q[c,:,j]= (original[:,j]+np.fft.irfft(spec,n=N,axis=0)).T
    print('CONTINUOUS_WRISTS_REFERENCE',c,str(base['clip_names'][c]),flush=True)
pos=np.zeros((C,N,len(m.links),3));rot=np.zeros((C,N,len(m.links),4));vel=np.zeros((C,N,len(m.links),6));qd=np.zeros_like(q);qd[...,mapping]=base['qd']
for c in range(C):
    dt=float(base['periods'][c])/N;deriv=lambda x:(np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)
    q[c]=np.clip(q[c],m.limits[:,0]+.001,m.limits[:,1]-.001);qd[c][:,arm_j]=deriv(q[c])[:,arm_j]
    for k in range(N):
        p,r=m.fk(q[c,k],base['positions'][c,k,old_links.index('pelvis')],R.from_quat(base['rotations'][c,k,old_links.index('pelvis')]).as_matrix())
        for j,n in enumerate(m.links):pos[c,k,j]=p[n];rot[c,k,j]=R.from_matrix(r[n]).as_quat()
    for k in range(1,N):rot[c,k][(rot[c,k]*rot[c,k-1]).sum(-1)<0]*=-1
    mean=np.mean(base['velocities'][c,:,:,:3]-deriv(base['positions'][c]),axis=(0,1));vel[c,:,:,:3]=deriv(pos[c])+mean
    matrices=R.from_quat(rot[c].reshape(-1,4)).as_matrix().reshape(N,len(m.links),3,3)
    vel[c,:,:,3:]=R.from_matrix((np.roll(matrices,-1,0)@np.roll(matrices,1,0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(N,len(m.links),3)/(2*dt)
    for j,n in enumerate(m.links):
        if n in old_links and n not in arm_l:
            oi=old_links.index(n);pos[c,:,j]=base['positions'][c,:,oi];rot[c,:,j]=base['rotations'][c,:,oi];vel[c,:,j]=base['velocities'][c,:,oi]
for key,value in [('q',q),('qd',qd),('positions',pos),('rotations',rot),('velocities',vel)]:data[key]=value.astype('float32')
data['joint_names']=np.array(m.names);data['link_names']=np.array(m.links);data['idle']=data['q'][3].mean(0)
np.savez_compressed(OUT/'full_motion.npz',**data)
experts=[]
for c in [2,4,5,6,9,10]:
    ex={k:data[k][c] for k in ['q','qd','positions','rotations','velocities','contacts']};ex.update(joint_names=data['joint_names'],link_names=data['link_names'],idle=data['idle'],period=base['periods'][c],speed=base['speeds'][c],command=base['commands'][c])
    path=OUT/f'expert_{c}.npz';np.savez_compressed(path,**ex);experts.append(str(path.resolve()))
cfg=json.load(open(ROOT/'output/anon_locomotion/release_v6_continuous/env_config.json'))
cfg.update(articulated_wrists=True,full_motion_path=str((OUT/'full_motion.npz').resolve()),amp_expert_paths=experts,amp_enabled=True,amp_reset_discriminator=True,amp_style_weight=2.,amp_upper_reference_mix=1.,amp_phase_upper_weight=1.,amp_phase_velocity_weight=.2,amp_phase_std_floor=.06,upper_drive_scale=.65,upper_damping_scale=.8,hand_orientation_weight=1.5,running_arm_spacing_weight=1.,reference_velocity_feedforward=True,reference_distal_arm_velocity_feedforward=True,continuous_curriculum_start=0,amp_forward_only=False,require_running_flight=False,continuous_tracking_weight=4.,motion_stage='continuous_wrists_040',train_entropy_coef=.001,train_upper_std_cap=.08,migration_old_dof_names=json.load(open(ROOT/'output/anon_physical_v3/manifest.json',encoding='utf-8'))['joints'],learn_clearance_height_gate=True,clearance_radial_envelope=True,spatial_clearance_only=True)
# PhysX's tensor DOF order is saved in the accepted walking rollout.
old=np.load(ROOT/'output/anon_locomotion/running_wrists/old_policy/rollout.npz')
# Both v3 policies use the same tensor joint order.
cfg['migration_old_dof_names']=list(old['dof_names'])
(OUT/'training.json').write_text(json.dumps(cfg,indent=2))
(OUT/'reference_report.json').write_text(json.dumps(dict(dofs=31,max_joint_speed=float(np.abs(qd).max()),method=__doc__,experts=experts),indent=2))
