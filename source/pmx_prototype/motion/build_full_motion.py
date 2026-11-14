"""Prepare a complete, cyclic 100STYLE reference for physical imitation.

Keeps captured joint coordination, pelvis sway/orientation and timing. Ground
alignment and a least-squares pelvis correction reduce retargeting foot slip.
No procedural swing trajectory is used.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from retarget_100style import Retarget,read_bvh
from scipy.optimize import least_squares
from full_body_retarget import head_calibration,retarget_upper

ROOT=PROJECT_ROOT


def periodic_filter(x,harmonics=6):
    spectrum=np.fft.rfft(x,axis=0); spectrum[harmonics+1:]=0
    return np.fft.irfft(spectrum,n=len(x),axis=0)


def derivative(x,dt):
    return (np.roll(x,-1,axis=0)-np.roll(x,1,axis=0))/(2*dt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['FW','BW','SW'],default='FW')
    parser.add_argument('--speed',type=float)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--anatomical-upper',action='store_true')
    parser.add_argument('--source-cycle',nargs=2,type=int,metavar=('FIRST','LAST'))
    parser.add_argument('--plant-feet',action='store_true')
    args=parser.parse_args()
    folder=ROOT/'output/anon_locomotion/100style_reference'
    out=ROOT/'output/anon_locomotion/full_motion';out.mkdir(parents=True,exist_ok=True)
    model=Retarget(); d=np.load(folder/(args.kind+'.npz'))
    nodes,source_p,source_r,source_dt=read_bvh(ROOT/'media/mocap/100style'/('Neutral_'+args.kind+'.bvh'))
    source_ids={n['name']:i for i,n in enumerate(nodes)}
    source_v=np.gradient(source_p,source_dt,axis=0)
    if args.source_cycle:
        first,last=args.source_cycle;frames=np.arange(first,last+1)
        previous=np.zeros(25);previous[model.names.index('left_shoulder_roll_joint')]=.9
        previous[model.names.index('right_shoulder_roll_joint')]=-.9
        poses=[]
        for frame in frames:
            previous=model.frame(source_p[frame],source_r[frame],source_ids,previous);poses.append(previous.copy())
        d=dict(q=np.array(poses),source_frames=frames,root_rotations=source_r[frames,0],dt=source_dt)
    scale=.9132680182298859
    source_floor=float(np.percentile(source_p[d['source_frames'][0]:d['source_frames'][-1],[source_ids['LeftToe'],source_ids['RightToe']],2],15))*scale
    idle=np.load(folder/'ID.npz')['q'].mean(0)
    bias=np.zeros(25);bias[:5]=idle[:5];bias[2]-=.04
    idle_reference=idle-bias
    if args.anatomical_upper:
        id_nodes,id_p,id_r,_=read_bvh(ROOT/'media/mocap/100style/Neutral_ID.bvh')
        id_ids={n['name']:i for i,n in enumerate(id_nodes)}
        id_frames=np.load(folder/'ID.npz')['source_frames']
        head_offset=head_calibration(id_r,id_ids,id_frames)
        id_mean_rotation=np.array([R.from_matrix(id_r[id_frames,j]).mean().as_matrix() for j in range(len(id_nodes))])
        idle_reference=retarget_upper(model,id_p[id_frames].mean(0),id_mean_rotation,id_ids,idle,head_offset)
        # Correct every original 30 Hz sample before extracting a periodic cycle.
        corrected=[]
        for k,frame in enumerate(d['source_frames']):
            corrected.append(retarget_upper(model,source_p[frame],source_r[frame],source_ids,d['q'][k],head_offset))
        d=dict(d);d['q']=np.array(corrected)
        bias[:5]=0.
    count=128; tables=[]; roots=[]; angles=[]; periods=[]; speeds=[];ankles=[];foot_angles=[];contact_tables=[]
    manifest=json.loads((folder/'library.json').read_text(encoding='utf-8'))
    cycles=[(c['first'],c['last']) for item in manifest['cycles'] if item['clip']==args.kind for c in item['cycles']]
    if args.source_cycle:cycles=[(0,len(d['source_frames'])-1)]
    for a,b in cycles:
        phase=np.arange(count)/count; t=a+phase*(b-a)
        interpolate=lambda array:np.stack([np.interp(t,np.arange(len(array)),array[:,k]) for k in range(array.shape[1])],-1)
        q=interpolate(d['q'])-bias
        source_a=int(d['source_frames'][a]);source_b=int(d['source_frames'][b]);source_t=source_a+phase*(source_b-source_a)
        interp_source=lambda array:np.stack([np.interp(source_t,np.arange(len(array)),array[:,k]) for k in range(array.shape[1])],-1)
        position=interp_source(source_p[:,0])*scale
        displacement=(source_p[source_b,0]-source_p[source_a,0])*scale
        if args.kind=='FW':heading=np.arctan2(displacement[0],-displacement[1])
        elif args.kind=='BW':heading=np.arctan2(-displacement[0],displacement[1])
        else:heading=np.arctan2(displacement[1],displacement[0])
        align=R.from_euler('Z',-heading)
        planar_origin=source_p[source_a,0,:2]*scale
        position[:,:2]-=planar_origin+phase[:,None]*displacement[None,:2]
        hip_offset=model.origins['pelvis'][2]-(model.origins['left_hip_yaw'][2]+model.origins['right_hip_yaw'][2])/2
        position[:,2]+=hip_offset-source_floor
        position=align.apply(position)
        euler=R.from_matrix(align.as_matrix()@d['root_rotations']).as_euler('ZYX')
        euler[:,0]=np.unwrap(euler[:,0])
        angular=interpolate(euler)
        angular[:,0]-=phase*(euler[b,0]-euler[a,0])
        angular[:,0]-=2*np.pi*round(float(angular[:,0].mean())/(2*np.pi))
        tables.append(q);roots.append(position);angles.append(angular)
        ap=[];ar=[];ac=[]
        for side in ['Left','Right']:
            index=source_ids[side+'Ankle'];toe=source_ids[side+'Toe']
            ankle=interp_source(source_p[:,index])*scale
            ankle[:,:2]-=planar_origin+phase[:,None]*displacement[None,:2]
            ankle[:,2]-=source_floor
            ap.append(align.apply(ankle))
            orientation=R.from_matrix(align.as_matrix()@source_r[:,index]).as_euler('ZYX')
            orientation[:,0]=np.unwrap(orientation[:,0]);foot_angle=interp_source(orientation)
            foot_angle[:,0]-=2*np.pi*round(float(foot_angle[:,0].mean())/(2*np.pi));ar.append(foot_angle)
            stable=((np.linalg.norm(source_v[:,toe,:2],axis=-1)<.3)&(source_p[:,toe,2]<source_floor/scale+.035)) | ((np.linalg.norm(source_v[:,index,:2],axis=-1)<.25)&(source_p[:,index,2]<source_floor/scale+.15))
            ac.append(np.interp(source_t,np.arange(len(stable)),stable.astype(float)))
        ankles.append(np.stack(ap,1));foot_angles.append(np.stack(ar,1));contact_tables.append(np.stack(ac,1))
        periods.append((b-a)*float(d['dt']));speeds.append(np.linalg.norm(displacement[:2])/periods[-1])
    q=periodic_filter(np.mean(tables,axis=0))
    q=np.clip(q,model.limits[:,0]+.005,model.limits[:,1]-.005)
    root=periodic_filter(np.mean(roots,axis=0));planar_center=root[:,:2].mean(0);root[:,:2]-=planar_center
    rotation=R.from_euler('ZYX',periodic_filter(np.mean(angles,axis=0))).as_matrix()
    native_period=float(np.mean(periods));native_speed=float(np.mean(speeds))
    speed=args.speed if args.speed is not None else native_speed
    frequency_ratio=(speed/native_speed)**.35;stride_ratio=(speed/native_speed)**.65
    period=native_period/frequency_ratio;dt=period/count
    mean_velocity=np.array([speed if args.kind=='SW' else 0.,speed if args.kind=='BW' else -speed if args.kind=='FW' else 0.,0.])
    ankle_target=periodic_filter(np.mean(ankles,axis=0),8)
    ankle_target[:,:,:2]-=planar_center
    if args.speed is not None:
        axis=0 if args.kind=='SW' else 1
        relative=ankle_target-root[:,None,:]
        baseline=np.array([model.origins['left_hip_yaw'][0],model.origins['right_hip_yaw'][0]]) if axis==0 else np.zeros(2)
        relative[:,:,axis]=baseline[None]+(relative[:,:,axis]-baseline[None])*stride_ratio
        root[:,axis]*=stride_ratio
        ankle_target=root[:,None,:]+relative
    foot_rotation=R.from_euler('ZYX',periodic_filter(np.mean(foot_angles,axis=0),6).reshape(-1,3)).as_matrix().reshape(count,2,3,3)
    contact=(np.mean(contact_tables,axis=0)>.5)
    corners=[]
    rest=np.array(model.manifest['rest_positions']);bones=model.manifest['bone_names']
    for side,jp in [('left','左'),('right','右')]:
        ankle=model.origins[side+'_ankle_roll'];toe=rest[bones.index(jp+'つま先')]
        center=np.array([ankle[0],(ankle[1]+toe[1])/2,.035])-ankle
        corners.append(np.array([[x,y,z] for x in [-.0475,.0475] for y in [-.115,.115] for z in [-.035,.035]])+center)
    feet=[model.links.index(s+'_ankle_roll') for s in ['left','right']]
    # Solve complete captured foot placements on this body's actual limb lengths.
    # Contact labels come from source foot speeds/heights, not retargeted box height.
    leg_ids=[i for i,n in enumerate(model.names) if any(token in n for token in ['hip_','knee_','ankle_'])]
    for k in range(count):
        for foot in range(2):
            sole_offset=(foot_rotation[k,foot]@corners[foot].T).T[:,2].min()
            minimum=.002-sole_offset
            if contact[k,foot]:ankle_target[k,foot,2]=minimum
            else:ankle_target[k,foot,2]=max(minimum,ankle_target[k,foot,2])
        captured=q[k].copy();height=root[k,2]
        def error(x):
            temp=captured.copy();temp[leg_ids]=x[:-1]
            pelvis=root[k].copy();pelvis[2]=x[-1]
            pp,rr=model.fk(temp,pelvis,rotation[k]);result=[]
            for foot,side in enumerate(['left','right']):
                name=side+'_ankle_roll'
                result.extend((pp[name]-ankle_target[k,foot])/.012)
                result.extend(((rr[name]-foot_rotation[k,foot])/.4).ravel())
            result.extend((x[:-1]-captured[leg_ids])/.7)
            result.append((x[-1]-height)/.025)
            return result
        lower=np.r_[model.limits[leg_ids,0]+.005,.82];upper=np.r_[model.limits[leg_ids,1]-.005,1.02]
        fit=least_squares(error,np.clip(np.r_[q[k,leg_ids],height],lower+1e-5,upper-1e-5),bounds=(lower,upper),max_nfev=35)
        q[k,leg_ids]=fit.x[:-1];root[k,2]=fit.x[-1]
        if k%32==0:print('FULL_MOTION_IK',k,count,flush=True)
    def fk_all():
        pp=[];rr=[]
        for k in range(count):
            p,r=model.fk(q[k],root[k],rotation[k]);pp.append([p[n] for n in model.links]);rr.append([r[n] for n in model.links])
        return np.array(pp),np.array(rr)
    p,r=fk_all()
    if args.plant_feet:
        from plant_mocap_feet import plant_feet
        desired=plant_feet(p[:,feet],r[:,feet],np.array(corners),contact,mean_velocity,dt)
        for k in range(count):
            original=q[k].copy();height=root[k,2];target_r=r[k,feet].copy()
            def contact_error(x):
                temp=original.copy();temp[leg_ids]=x[:-1];origin=root[k].copy();origin[2]=x[-1]
                pp,rr=model.fk(temp,origin,rotation[k]);result=[]
                for foot,side in enumerate(['left','right']):
                    name=side+'_ankle_roll';result.extend((pp[name]-desired[k,foot])/.008)
                    result.extend(((rr[name]-target_r[foot])/.25).ravel())
                result.extend((x[:-1]-original[leg_ids])/.8);result.append((x[-1]-height)/.03)
                return result
            lower=np.r_[model.limits[leg_ids,0]+.005,.8];upper=np.r_[model.limits[leg_ids,1]-.005,1.03]
            fit=least_squares(contact_error,np.clip(np.r_[q[k,leg_ids],height],lower+1e-5,upper-1e-5),bounds=(lower,upper),max_nfev=40)
            q[k,leg_ids]=fit.x[:-1];root[k,2]=fit.x[-1]
        p,r=fk_all()
    print('ANKLE_TARGET_ERROR',np.sqrt(np.mean((p[:,feet]-ankle_target)**2)),
          'TARGET_STANCE_SPEED',np.mean(np.linalg.norm((derivative(ankle_target,dt)+mean_velocity)[:,:,:2],axis=-1)[contact]),flush=True)
    print('FOOT_ROTATION_ERROR_DEG',np.rad2deg(R.from_matrix((r[:,feet]@foot_rotation.transpose(0,1,3,2)).reshape(-1,3,3)).magnitude()).reshape(count,2).mean(0),flush=True)
    sole=np.einsum('nfij,fkj->nfki',r[:,feet],np.array(corners))+p[:,feet,None,:]
    root[:,2]+=periodic_filter(.002-sole[:,:,:,2].min((1,2)))
    p,r=fk_all()
    sole=np.einsum('nfij,fkj->nfki',r[:,feet],np.array(corners))+p[:,feet,None,:]
    bottom=sole[:,:,:,2].min(-1)
    # Fit captured pelvis sway to planted soles, preserving its measured shape.
    lowest=(sole[:,:,:,2]<bottom[:,:,None]+.003).astype(float)
    local=sole[:,:,:,:2]-root[:,None,None,:2]
    rows=[];targets=[]
    for k in range(count):
        nxt=(k+1)%count
        for foot in range(2):
            if contact[k,foot] and contact[nxt,foot]:
                row=np.zeros(count);row[k]=-1/dt;row[nxt]=1/dt
                rows.append(row)
                corner_velocity=(local[nxt,foot]-local[k,foot])/dt
                fixed_corner_velocity=(corner_velocity*lowest[k,foot,:,None]).sum(0)/lowest[k,foot].sum()
                targets.append(-fixed_corner_velocity-mean_velocity[:2])
    matrix=np.vstack([rows,np.eye(count)*3]);target=np.vstack([targets,root[:,:2]*3])
    fitted=np.linalg.lstsq(matrix,target,rcond=None)[0]
    root[:,:2]=periodic_filter(fitted,4);root[:,:2]-=root[:,:2].mean(0)
    p,r=fk_all()
    quat=R.from_matrix(r.reshape(-1,3,3)).as_quat().reshape(count,len(model.links),4)
    # Continuous quaternion hemisphere for interpolation.
    for k in range(1,count):quat[k][(quat[k-1]*quat[k]).sum(-1)<0]*=-1
    v=derivative(p,dt)+mean_velocity
    omega=R.from_matrix((np.roll(r,-1,axis=0)@np.roll(r,1,axis=0).transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(count,len(model.links),3)/(2*dt)
    qd=derivative(q,dt)
    sole=np.einsum('nfij,fkj->nfki',r[:,feet],np.array(corners))+p[:,feet,None,:]
    near=(sole[:,:,:,2]<sole[:,:,:,2].min(-1)[:,:,None]+.003).astype(float)
    sole_vel=derivative(sole,dt)+mean_velocity
    slip=((np.linalg.norm(sole_vel[:,:,:,:2],axis=-1)*near).sum(-1)/near.sum(-1))[contact]
    destination=args.output or out/'forward.npz'
    np.savez_compressed(destination,q=q.astype('float32'),qd=qd.astype('float32'),
                        positions=p.astype('float32'),rotations=quat.astype('float32'),velocities=np.concatenate([v,omega],-1).astype('float32'),
                        contacts=contact.astype('float32'),joint_names=model.names,link_names=model.links,
                        period=period,speed=speed,command=[-mean_velocity[1],mean_velocity[0],0.],idle=idle_reference.astype('float32'),foot_corners=corners)
    report=dict(source='100STYLE Neutral '+args.kind+' / Ian Mason et al. / CC BY 4.0',cycles=cycles,frames=count,
                native_period_seconds=native_period,native_speed_m_s=native_speed,period_seconds=period,speed_m_s=speed,contact_fraction=contact.mean(0).tolist(),
                kinematic_sole_slip_m_s=float(np.mean(slip)),
                pelvis_vertical_range_m=float(np.ptp(root[:,2])),
                anatomical_upper=args.anatomical_upper,planted_contact_cleanup=args.plant_feet,processing=['full joint and pelvis motion','neutral calibration','periodic smoothing',
                            'ground alignment','contact-consistent pelvis sway fit'])
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
