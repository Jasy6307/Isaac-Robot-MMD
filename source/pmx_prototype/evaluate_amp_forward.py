"""Forward-only control and full-body coordination measurements."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter


def coordination(data):
    q=data['joints'][:,0];p=data['poses'][:,0];cmd=data['commands'][:,0]
    names=list(data['dof_names']);links=list(data['link_names']);dt=float(data['dt'])
    time=np.arange(len(q))*dt;mask=(time>=8)&(time<30)&(cmd[:,0]>.1)
    q=q[mask];p=p[mask];t=time[mask]
    pelvis=links.index('pelvis');chest=links.index('waist_pitch')
    def yaw(i):
        a=np.unwrap(Rotation.from_quat(p[:,i,3:]).as_euler('ZYX')[:,0])
        return a-np.polyval(np.polyfit(t,a,1),t)
    yp,yc=yaw(pelvis),yaw(chest)
    hands=np.stack([Rotation.from_quat(p[:,pelvis,3:]).inv().apply(p[:,links.index(side+'_hand'),:3]-p[:,pelvis,:3]) for side in ['left','right']],1)
    def span(x):return float(np.ptp(np.quantile(x,[.05,.95])))
    def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-6 else 0.
    upper=[i for i,n in enumerate(names) if any(k in n for k in ['waist','shoulder','elbow'])]
    smoothed=savgol_filter(q[:,upper],7,3,axis=0)
    acceleration=np.diff(smoothed,n=2,axis=0)/dt**2
    frequency=np.fft.rfftfreq(len(q),dt)
    spectrum=np.abs(np.fft.rfft((q[:,upper]-q[:,upper].mean(0))*np.hanning(len(q))[:,None],axis=0))**2
    velocity_power=spectrum*(2*np.pi*frequency[:,None])**2
    high_frequency=float(velocity_power[frequency>4].sum()/max(velocity_power[frequency>.2].sum(),1e-12))
    return dict(waist_yaw_range_deg=float(np.degrees(span(q[:,names.index('waist_yaw_joint')]))),
        pelvis_yaw_range_deg=float(np.degrees(span(yp))),chest_yaw_range_deg=float(np.degrees(span(yc))),
        pelvis_chest_yaw_correlation=corr(yp,yc),arm_contralateral_correlation=corr(hands[:,0,1],hands[:,1,1]),
        arm_foreaft_ranges_cm=[100*span(hands[:,i,1]) for i in range(2)],
        waist_arm_coordination=corr(q[:,names.index('waist_yaw_joint')],hands[:,0,1]-hands[:,1,1]),
        upper_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),upper_high_frequency_power_fraction=high_frequency)


def evaluate(env,runner,output,checkpoint):
    run_only=env.cfg.get('evaluation_run_only',False)
    env.max_episode_length=round(600/env.dt);env.cfg['imitation_evaluation_initial_clip']=2 if run_only else 3;env.reset(env.ids)
    policy=runner.get_inference_policy();traces={k:[] for k in ['poses','joints','commands','velocities','phases','dones','root_velocities','joint_velocities']};labels=[];items=[]
    forward_seconds=env.cfg.get('amp_evaluation_forward_seconds',30.)
    forward_speed=env.cfg.get('amp_forward_speed',.4)
    if env.cfg.get('diagnostic_controller_trace',False):
        traces.update(targets=[],references=[],actions=[])
    if hasattr(env,'ground_contact_view'):traces['ground_forces']=[]
    with torch.inference_mode():
        schedule=[('forward',forward_seconds,[forward_speed,0,0])] if run_only else [('stand',3.,[0,0,0]),('forward',forward_seconds,[forward_speed,0,0]),('stop',4.,[0,0,0])]
        for label,seconds,cmd in schedule:
            falls=0;vel=[];logs={}
            for step in range(round(seconds/env.dt)):
                env.set_command(cmd);_,_,done,extras=env.step(policy(env.get_observations()));p,q,v,phase=env.recorded_state
                values=[p,q,env.commands,v,phase,done,env.root_vel,env.qd]
                for key,value in zip(list(traces)[:8],values):traces[key].append(value.cpu().numpy())
                if env.cfg.get('diagnostic_controller_trace',False):
                    for key,value in [('targets',env.diagnostic_target),('references',env.diagnostic_reference),('actions',env.actions)]:
                        traces[key].append(value.cpu().numpy())
                if hasattr(env,'ground_contact_view'):
                    force=env.ground_contact_view.get_contact_force_matrix(1/env.cfg['physics_hz'])[:,0,:]
                    traces['ground_forces'].append(force.cpu().numpy())
                labels.append(label);falls+=int(done.sum())
                if step>=(90 if label=='forward' else 30):
                    vel.append([-float(env.root_vel[0,1]),float(env.root_vel[0,0]),float(env.root_vel[0,5])])
                    for key,value in extras['log'].items():
                        if key.startswith('Imitation/'):logs.setdefault(key,[]).append(float(value))
            item=dict(label=label,falls=falls,mean_velocity=np.mean(vel,axis=0).tolist(),metrics={k:float(np.mean(x)) for k,x in logs.items()});items.append(item)
            print('AMP_FORWARD_SEGMENT '+json.dumps(item),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    data={k:np.asarray(v) for k,v in traces.items()};data.update(labels=labels,dt=env.dt,link_names=env.links,dof_names=env.names)
    np.savez_compressed(output/'rollout.npz',**data)
    measured=coordination(data);forward=next(x for x in items if x['label']=='forward');stop=items[-1]
    passed=sum(x['falls'] for x in items)==0 and abs(forward['mean_velocity'][0]-forward_speed)<env.cfg.get('evaluation_forward_tolerance',.06) and abs(forward['mean_velocity'][1])<env.cfg.get('evaluation_lateral_tolerance',.04) and abs(forward['mean_velocity'][2])<.1
    if not run_only:passed &= all(np.linalg.norm(item['mean_velocity'][:2])<.06 and abs(item['mean_velocity'][2])<.06 for item in [items[0],stop])
    if env.cfg.get('learn_arm_clearance'):passed &= forward['metrics']['Imitation/near_hip_clearance_deficit_m']<.012
    gait={}
    if env.cfg.get('require_running_flight'):
        pose=data['poses'][:,0,env.foot_ids];rot=Rotation.from_quat(pose[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(pose),2,3,3)
        corners=env.foot_corners.cpu().numpy();sole=np.einsum('nfij,fkj->nfki',rot,corners)+pose[:,:,:3,None].transpose(0,1,3,2)
        bottom=sole[:,:,:,2].min(-1);moving=(np.arange(len(pose))*env.dt>=8)&(np.array(labels)=='forward')
        flight=(bottom>.015).all(-1);gait=dict(measured_flight_fraction=float(flight[moving].mean()),measured_double_support_fraction=float((bottom[moving]<.012).all(-1).mean()),flight_method='Both physical sole boxes more than 15mm above ground')
        if 'ground_forces' in data:
            loaded=np.abs(data['ground_forces'][:,:,2])>5.
            assert np.abs(data['ground_forces'][:,:,2]).sum(-1).max()>50,'Ground-force measurement returned no load'
            gait.update(force_measured_flight_fraction=float((~loaded[moving].any(-1)).mean()),force_measured_double_support_fraction=float(loaded[moving].all(-1).mean()),force_contact_threshold_n=5.)
        flight_fraction=gait.get('force_measured_flight_fraction',gait['measured_flight_fraction'])
        passed &= flight_fraction>env.cfg.get('minimum_flight_fraction',.015)
    report=dict(checkpoint=str(checkpoint),total_seconds=forward_seconds+(0 if run_only else 7),total_falls=sum(x['falls'] for x in items),segments=items,coordination=measured,env_config=env.cfg,
        passed_forward_gate=bool(passed and not run_only),runtime_compensation_disabled=True,runtime_compensation_removed=True,method='Forward-only AMP full-body prior and spatial clearance' if env.cfg.get('amp_enabled') else 'Baseline forward walking')
    if run_only:report.update(passed_run_only_gate=bool(passed),evaluation_mode='Reference-initialized running only; does not validate stand/start/stop')
    if gait:report['gait_diagnostics']=gait
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('AMP_FORWARD_REPORT '+json.dumps(report),flush=True)
