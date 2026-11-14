"""Fixed full-scope acceptance for one speed-conditioned walk/run policy."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
import numpy as np
import torch


def suite(short=False):
    if short:return [('stand',5,[0,0,0]),('ramp_up',12,None),('run_forward',6,[1.2,0,0]),('ramp_down',12,None),('stand_end',6,[0,0,0])]
    result=[('stand',40,[0,0,0]),('walk',6,[.3,0,0]),('ramp_up_from_walk',14,None),('run_forward',8,[1.2,0,0]),('ramp_down',14,None),('stop',5,[0,0,0])]
    for name,cmd in [('run_backward',[-1.2,0,0]),('run_left',[0,1.2,0]),('run_right',[0,-1.2,0])]:
        result += [(name,8,cmd),('stop',5,[0,0,0])]
    for degrees in [45,135,225,315]:
        a=np.deg2rad(degrees);result += [(f'run_diagonal_{degrees}',7,[1.2*np.cos(a),1.2*np.sin(a),0]),('stop',5,[0,0,0])]
    result += [('direction_sweep',24,None),('stop',5,[0,0,0])]
    for name,cmd in [('run_turn_left',[1.1,0,.6]),('run_turn_right',[1.1,0,-.6]),('reverse_turn',[-.9,0,.6]),('side_turn',[0,.9,-.6])]:
        result += [(name,8,cmd),('stop',5,[0,0,0])]
    result += [('stop',5,[0,0,0]),('pivot_left',6,[0,0,.6]),('pivot_right',6,[0,0,-.6]),('stand_end',40,[0,0,0])]
    return result


def command(label,t,seconds,fixed=None):
    if fixed is not None:return fixed
    if label=='ramp_up':return [1.2*t/seconds,0,0]
    if label=='ramp_up_from_walk':return [.3+.9*t/seconds,0,0]
    if label=='ramp_down':return [1.2*(1-t/seconds),0,0]
    if label=='direction_sweep':return [np.cos(2*np.pi*t/seconds),np.sin(2*np.pi*t/seconds),0]
    raise ValueError(label)


def evaluate(env,runner,output,checkpoint):
    env.max_episode_length=round(2000/env.dt);env.reset(env.ids)
    infer=runner.get_inference_policy();short=env.cfg.get('unified_eval_short',False)
    foot_com=env.robot.get_coms()[0,env.foot_ids,:3].to(env.device)
    diagnostic=bool(env.cfg.get('unified_eval_cases'))
    if env.cfg.get('unified_eval_initial_command') is not None:
        if not diagnostic:raise ValueError('Reference-state start is only allowed for diagnostic subsets')
        env.reset(env.ids,command=env.cfg['unified_eval_initial_command'],phase=env.cfg.get('unified_eval_initial_phase',0.))
        env.refresh()
    cases=env.cfg['unified_eval_cases'] if diagnostic else suite(short)
    traces={k:[] for k in ['poses','joints','commands','velocities','phases','dones']};labels=[];items=[]
    record_controller=bool(env.cfg.get('unified_record_controller'))
    if record_controller:traces.update(observations=[],actions=[],root=[],root_velocities=[])
    with torch.inference_mode():
        for label,seconds,fixed in cases:
            errors=[];geometric_errors=[];yaw_errors=[];speeds=[];metrics={};falls=0;flight=[];slip=[];geometric_slip=[];pelvis_yaws=[]
            position=env.root[:,:2].clone()
            previous_position=env.root[:,:3].clone()
            previous_sole=None;previous_near=None
            for step in range(round(seconds/env.dt)):
                cmd=command(label,step*env.dt,seconds,fixed);env.set_command(cmd)
                observation=env.get_observations();action=infer(observation)
                if record_controller:
                    for k,value in [('observations',observation['policy']),('actions',action),('root',env.root),('root_velocities',env.root_vel)]:traces[k].append(value.cpu().numpy().copy())
                _,_,done,extras=env.step(action)
                p,q,v,phase=env.recorded_state
                from anon_locomotion_env import quat_rotate_inverse
                actual_delta=(env.root[:,:3]-previous_position)/env.dt;previous_position=env.root[:,:3].clone()
                geometric=quat_rotate_inverse(env.root[:,3:],actual_delta)
                geometric=torch.stack([-geometric[:,1],geometric[:,0]],-1)
                pq=p[0,env.pelvis_id,3:];x,y,z,w=pq
                pelvis_yaws.append(float(torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))))
                for k,value in [('poses',p),('joints',q),('commands',env.commands),('velocities',v),('phases',phase),('dones',done)]:traces[k].append(value.cpu().numpy().copy())
                labels.append(label);falls+=int((done&~extras['time_outs']).sum())
                if step<60:continue
                target=torch.tensor(cmd,device=env.device)
                errors.append((v[:,:2]-target[:2]).mean(0).cpu().numpy());yaw_errors.append(float((v[:,2]-target[2]).mean()));speeds.append(float(v[:,:2].norm(dim=-1).mean()))
                geometric_errors.append((geometric-target[:2]).mean(0).cpu().numpy())
                for key,value in extras['log'].items():
                    if key.startswith('Imitation/'):metrics.setdefault(key,[]).append(float(value))
                feet=env.poses[:,env.foot_ids];quat=feet[:,:,None,3:].expand(-1,-1,8,-1)
                conjugate=torch.cat([-quat[...,:3],quat[...,3:]],-1)
                offset=quat_rotate_inverse(conjugate,env.foot_corners[None].expand(env.num_envs,-1,-1,-1))
                height=feet[:,:,None,2]+offset[:,:,:,2];bottom=height.amin(-1);contact=bottom<.012
                flight.append(float((~contact.any(-1)).float().mean()))
                com_world=quat_rotate_inverse(conjugate,foot_com[None,:,None].expand_as(offset))
                velocity=env.link_vel[:,env.foot_ids];point_velocity=velocity[:,:,None,:3]+torch.cross(velocity[:,:,None,3:].expand_as(offset),offset-com_world,dim=-1)
                near=(height<bottom[:,:,None]+.003)&contact[:,:,None]
                slip.append(float((point_velocity[...,:2].norm(dim=-1)*near).sum()/near.sum().clamp_min(1)))
                sole=feet[:,:,None,:3]+offset
                if previous_sole is not None:
                    stable=near&previous_near&~done[:,None,None]
                    movement=(sole-previous_sole)[...,:2].norm(dim=-1)/env.dt
                    geometric_slip.append(float((movement*stable).sum()/stable.sum().clamp_min(1)))
                previous_sole=sole.clone();previous_near=near.clone()
            variable=label.startswith('ramp_') or label=='direction_sweep';error=np.array(errors);speed=float(np.mean(speeds))
            bias=float(np.mean(np.linalg.norm(error,axis=-1))) if variable else float(np.linalg.norm(error.mean(0)))
            geometric_error=np.asarray(geometric_errors)
            geometric_bias=float(np.mean(np.linalg.norm(geometric_error,axis=-1))) if variable else float(np.linalg.norm(geometric_error.mean(0)))
            yaw=float(np.mean(np.abs(yaw_errors))) if variable else float(abs(np.mean(yaw_errors)))
            target_speed=0 if fixed is None else float(np.linalg.norm(fixed[:2]));linear_limit=.2 if variable else max(.06,target_speed*.22)
            average={k:float(np.mean(v)) for k,v in metrics.items()}
            stationary=label in ['stand','stand_end','stop'];displacement=float((env.root[:,:2]-position).norm(dim=-1).max())
            passed=falls==0 and bias<linear_limit and geometric_bias<linear_limit and yaw<(.1 if stationary else .2) and average['Imitation/keypoint_rmse_m']<.075
            if stationary:passed &= speed<.08
            pelvis_yaws=np.unwrap(pelvis_yaws);yaw_drift=float(np.max(np.abs(pelvis_yaws-pelvis_yaws[0])))
            if label in ['stand','stand_end']:passed &= displacement<.25 and yaw_drift<.35
            if label in ['run_forward','run_backward','run_left','run_right']:passed &= np.mean(flight)>.01
            passed &= average.get('Imitation/near_hip_clearance_deficit_m',1)<.015
            item=dict(label=label,seconds=seconds,command=fixed,falls=falls,velocity_error=bias,velocity_limit=linear_limit,yaw_error=yaw,mean_speed=speed,
                      geometric_velocity_error=geometric_bias,pelvis_yaw_drift_rad=yaw_drift,
                      displacement_m=displacement,flight_fraction=float(np.mean(flight)),stance_slip_proxy_m_s=float(np.mean(slip)),stance_slip_geometry_m_s=float(np.mean(geometric_slip)),metrics=average,passed=bool(passed))
            items.append(item);print('UNIFIED_SEGMENT '+json.dumps(item),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'rollout.npz',**{k:np.asarray(v) for k,v in traces.items()},labels=labels,dt=env.dt,link_names=env.links,dof_names=env.names)
    report=dict(checkpoint=str(checkpoint),env_config=env.cfg,total_seconds=sum(s for _,s,_ in cases),total_falls=sum(i['falls'] for i in items),
                segments=items,passed_unified_gate=all(i['passed'] for i in items) and not short and not diagnostic,passed_short_gate=all(i['passed'] for i in items),
                complete_scope=not short and not diagnostic,diagnostic=diagnostic,runtime_compensation_removed=True,method='One actor and normalizer; continuous speed-conditioned whole-body walk/run reference; physical residual PPO + AMP; no root assistance')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('UNIFIED_REPORT '+json.dumps(dict(passed=report['passed_unified_gate'],short_passed=report['passed_short_gate'],falls=report['total_falls'])),flush=True)
