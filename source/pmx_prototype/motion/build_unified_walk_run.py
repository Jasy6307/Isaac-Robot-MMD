"""Build 31-DOF genuine BR/SR references and one continuous walk/run bank."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R,Slerp
from retarget_100style import Retarget,read_bvh

ROOT=PROJECT_ROOT
FIELDS=['q','qd','positions','rotations','velocities','contacts']

def extend(path,clip,output):
    data=dict(np.load(path));old_names=list(data['joint_names']);old_links=list(data['link_names'])
    model=Retarget(ROOT/'output/anon_physical_v4/manifest.json');count=len(data['q'])
    q=np.zeros((count,len(model.names)));q[:,[model.names.index(n) for n in old_names]]=data['q']
    frames=data['source_frames'];nodes,_,rot,_=read_bvh(ROOT/'media/mocap/100style'/clip);ids={v['name']:i for i,v in enumerate(nodes)}
    first=int(np.floor(frames.min()));last=int(np.ceil(frames.max()))+1
    source=lambda name:Slerp(np.arange(first,last+1),R.from_matrix(rot[first:last+1,ids[name]]))(frames).as_matrix()
    root=data['positions'][:,old_links.index('pelvis')];root_r=R.from_quat(data['rotations'][:,old_links.index('pelvis')]).as_matrix()
    trunk_r=R.from_quat(data['rotations'][:,old_links.index('waist_pitch')]).as_matrix()
    alignment=root_r@source('Hips').transpose(0,2,1)
    arm_ids=[i for i,n in enumerate(model.names) if any(s in n for s in ['shoulder','elbow','forearm','wrist'])]
    for side,sign in [('left',1.),('right',-1.)]:
        ji=[model.names.index(side+'_'+s+'_joint') for s in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex','forearm_twist','wrist_pitch','wrist_deviation']]
        along=model.origins[side+'_hand']-model.origins[side+'_elbow_flex'];along/=np.linalg.norm(along)
        correction=R.align_vectors([[sign,0,0]],[along])[0].as_matrix()
        forearm=alignment@source(side.title()+'Elbow')@correction
        hand=alignment@source(side.title()+'Wrist')@correction
        elbow_target=data['positions'][:,old_links.index(side+'_elbow_flex')]+trunk_r[:,:,0]*sign*.025
        hand_target=data['positions'][:,old_links.index(side+'_hand')]+trunk_r[:,:,0]*sign*.045
        initial=q[:,ji].copy()
        for k in range(count):
            seed=q[k,ji] if k==0 else q[k-1,ji]
            def error(x):
                temp=q[k].copy();temp[ji]=x;p,r=model.fk(temp,root[k],root_r[k])
                return np.r_[35*(p[side+'_elbow_flex']-elbow_target[k]),45*(p[side+'_hand']-hand_target[k]),
                             .5*R.from_matrix(forearm[k].T@r[side+'_forearm_twist']).as_rotvec(),
                             1.5*R.from_matrix(hand[k].T@r[side+'_hand']).as_rotvec(),.04*(x-seed)]
            q[k,ji]=least_squares(error,seed,bounds=(model.limits[ji,0]+.01,model.limits[ji,1]-.01),max_nfev=50).x
        delta=savgol_filter(q[:,ji]-initial,5,2,axis=0,mode='wrap');q[:,ji]=initial+delta
        print('UNIFIED_WRISTS',clip,side,count,flush=True)
    return save_fk(data,q,model,root,root_r,output)

def save_fk(data,q,model,root,root_r,output):
    data=dict(data);n=len(q);dt=float(data['period'])/n
    derivative=lambda x:(np.roll(x,-1,0)-np.roll(x,1,0))/(2*dt)
    q=np.clip(q,model.limits[:,0]+.005,model.limits[:,1]-.005);positions=[];rotations=[]
    for k in range(n):
        p,r=model.fk(q[k],root[k],root_r[k]);positions.append([p[v] for v in model.links]);rotations.append([r[v] for v in model.links])
    positions=np.array(positions);rotations=np.array(rotations)
    angular=np.array([0.,0.,data['command'][2]])
    velocity=derivative(positions)+np.array([data['command'][1],-data['command'][0],0.])+np.cross(angular,positions)
    omega=R.from_matrix((np.roll(rotations,-1,0)@np.roll(rotations,1,0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(n,len(model.links),3)/(2*dt)+angular
    data.update(q=q.astype('float32'),qd=derivative(q).astype('float32'),positions=positions.astype('float32'),rotations=R.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(n,len(model.links),4).astype('float32'),velocities=np.concatenate([velocity,omega],-1).astype('float32'),joint_names=np.array(model.names),link_names=np.array(model.links),idle=np.zeros(len(model.names),dtype='float32'))
    np.savez_compressed(output,**data);return data

def mirror(data,output):
    model=Retarget(ROOT/'output/anon_physical_v4/manifest.json');m=np.diag([-1.,1.,1.])
    opposite=lambda n:n.replace('left_','SIDE_').replace('right_','left_').replace('SIDE_','right_')
    indices=[model.names.index(opposite(n)) for n in model.names];signs=[]
    for i,j in enumerate(indices):
        axis=lambda k:np.array(model.joints[k].get('axis_vector',np.eye(3)['XYZ'.index(model.joints[k]['axis'])]))
        signs.append(float(np.sign(axis(i)@(-m@axis(j)))))
    q=data['q'][:,indices]*signs
    root=data['positions'][:,model.links.index('pelvis')].copy();root[:,0]*=-1
    root_r=m@R.from_quat(data['rotations'][:,model.links.index('pelvis')]).as_matrix()@m
    other=dict(data);other['command']=np.array(data['command'])*[1,-1,-1];other['contacts']=data['contacts'][:,::-1]
    return save_fk(other,q,model,root,root_r,output)

def resample(data,count):
    n=len(data['q']);x=np.arange(count)*n/count;a=x.astype(int)%n;b=(a+1)%n;weight=x-a
    result={}
    for key in FIELDS:
        u=data[key][a];v=data[key][b].copy()
        if key=='rotations':v[(u*v).sum(-1)<0]*=-1
        w=weight.reshape((count,)+(1,)*(u.ndim-1));result[key]=u*(1-w)+v*w
        if key=='rotations':result[key]/=np.linalg.norm(result[key],axis=-1,keepdims=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/unified_walk_run');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for name,clip in [('backward','Neutral_BR.bvh'),('left','Neutral_SR.bvh')]:
        destination=out/(name+'_31dof.npz')
        if not destination.exists():extend(out/(name+'_balanced.npz'),clip,destination)
        runs.append(dict(np.load(destination)))
    runs.insert(0,dict(np.load(ROOT/'output/anon_locomotion/running_wrists/running_reference.npz')))
    runs.append(mirror(runs[2],out/'right_31dof.npz'))
    old=dict(np.load(ROOT/'output/anon_locomotion/release_v6_scratch_31dof/full_motion.npz'))
    clips=[{k:old[k][c] for k in FIELDS} for c in range(11)]+runs
    count=384;tables={k:[] for k in FIELDS}
    for c,data in enumerate(clips):
        if c>=11:
            assert list(data['joint_names'])==list(old['joint_names'])
            assert list(data['link_names'])==list(old['link_names'])
        values=resample(data,count);center=values['positions'][:,list(old['link_names']).index('pelvis'),:2].mean(0)
        values['positions'][...,:2]-=center
        for k in FIELDS:tables[k].append(values[k])
    result={k:np.asarray(v,dtype='float32') for k,v in tables.items()}
    result.update(joint_names=old['joint_names'],link_names=old['link_names'],idle=old['idle'],
                  periods=np.r_[old['periods'],[float(r['period']) for r in runs]],speeds=np.r_[old['speeds'],[float(r['speed']) for r in runs]],
                  commands=np.concatenate([old['commands'],[r['command'] for r in runs]]),clip_names=np.r_[old['clip_names'],['run_forward','run_backward','run_left','run_right']],gait_cycles=np.array([1.]*11+[4.]*4))
    np.savez_compressed(out/'full_motion.npz',**result)
    experts=[]
    for c in [2,3,4,5,6,9,10,11,12,13,14]:
        ex={k:result[k][c] for k in FIELDS};ex.update(joint_names=result['joint_names'],link_names=result['link_names'],idle=result['idle'],period=result['periods'][c],speed=result['speeds'][c],command=result['commands'][c])
        path=out/f'expert_{c}.npz';np.savez_compressed(path,**ex);experts.append(str(path))
    cfg=json.loads((ROOT/'output/anon_locomotion/release_v6_scratch_31dof/env_config.json').read_text(encoding='utf-8'))
    cfg.update(full_motion_path=str(out/'full_motion.npz'),amp_expert_paths=experts,amp_reset_discriminator=True,continuous_phase_alignment=False,unified_walk_run=True,
               max_speed=1.2,max_yaw=.6,walk_run_blend_start=.55,walk_run_blend_end=.9,motion_stage='unified_walk_run',
               unified_initial_max_speed=.4,unified_curriculum_start=1000,unified_curriculum_iterations=400,
               imitation_command_scope='unified_120_060',episode_seconds=45.,imitation_normalizer_std_floor=.06,
               continuous_tracking_weight=12.,imitation_contact_weight=4.,amp_style_weight=2.,train_entropy_coef=.001,
               idle_yaw_weight=6.,idle_upper_pose_weight=3.,unified_idle_velocity_weight=12.,imitation_velocity_error_scale=.12,
               imitation_contact_hold_seconds=12.,imitation_low_speed_pose_weight=3.,runtime_inference_only=False)
    (out/'training.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'reference_report.json').write_text(json.dumps(dict(source='100STYLE Neutral FW/BW/SW/ID/FR/BR/SR; CC BY 4.0; Mason, Starke, Komura',running_sources=['Neutral_FR.bvh','Neutral_BR.bvh','Neutral_SR.bvh','mirrored Neutral_SR.bvh'],running_source_cycles=4,cycle_averaging=False,walk_run_transition_m_s=[.55,.9],max_speed=1.2,max_yaw=.6,experts=experts),indent=2),encoding='utf-8')
    print('UNIFIED_REFERENCE_READY',out,flush=True)

if __name__=='__main__':main()
