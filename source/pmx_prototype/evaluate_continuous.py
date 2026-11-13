"""Continuous-state evaluation: no segment resets or runtime arm compensation."""
import json
import numpy as np
import torch
from continuous_command_suite import segments,command


def evaluate(env,runner,output,checkpoint):
    env.max_episode_length=round(1000/env.dt);env.reset(env.ids)
    policy=runner.get_inference_policy();items=[]
    suite=segments(boundary=env.cfg.get('continuous_eval_suite')=='boundary')
    if env.cfg.get('continuous_eval_pivots'):
        suite=suite[:-1]+[('pivot_pos',6),('pivot_neg',6)]+suite[-1:]
    diagnostic=bool(env.cfg.get('continuous_eval_cases'))
    if diagnostic:suite=env.cfg['continuous_eval_cases']
    traces={k:[] for k in ['poses','joints','commands','velocities','phases','dones']};labels=[]
    if env.cfg.get('diagnostic_controller_trace'):traces.update(actions=[],raw_actions=[],joint_velocities=[],root_velocities=[])
    with torch.inference_mode():
        for label,seconds in suite:
            velocities=[];commands=[];yaw=[];logs={};falls=0
            for step in range(round(seconds/env.dt)):
                cmd=[0.,0.,.6 if label=='pivot_pos' else -.6] if label.startswith('pivot_') else command(label,step*env.dt,seconds)
                env.set_command(cmd)
                actions=policy(env.get_observations())
                _,_,done,extras=env.step(actions)
                p,q,v,phase=env.recorded_state
                for key,value in [('poses',p),('joints',q),('commands',env.commands),('velocities',v),('phases',phase),('dones',done)]:traces[key].append(value.cpu().numpy())
                if env.cfg.get('diagnostic_controller_trace'):
                    for key,value in [('actions',env.actions),('raw_actions',actions),('joint_velocities',env.qd),('root_velocities',env.root_vel)]:traces[key].append(value.cpu().numpy())
                labels.append(label);falls+=int(done.sum())
                if step<45:continue
                a=env.reference_heading;c=torch.cos(a);s=torch.sin(a);vx,vy=env.root_vel[:,0],env.root_vel[:,1]
                velocities.append([float((s*vx-c*vy).mean()),float((c*vx+s*vy).mean())]);commands.append(cmd);yaw.append(float(env.root_vel[:,5].mean()))
                for key,value in extras['log'].items():
                    if key.startswith('Imitation/'):logs.setdefault(key,[]).append(float(value))
            velocity=np.asarray(velocities);cmds=np.asarray(commands);omega=np.asarray(yaw)
            variable=label in ['speed_ramp','direction_sweep','combined_turn']
            if variable:
                window=min(30,len(velocity));kernel=np.ones(window)/window
                error=np.column_stack([np.convolve(velocity[:,k]-cmds[:,k],kernel,'valid') for k in range(2)])
                bias=float(np.linalg.norm(error,axis=-1).mean());yb=float(np.abs(np.convolve(omega-cmds[:,2],kernel,'valid')).mean());limit=.15
            else:
                bias=float(np.linalg.norm(velocity.mean(0)-cmds[:,:2].mean(0)));yb=float(abs(omega.mean()-cmds[:,2].mean()));limit=max(.045,.3*np.linalg.norm(cmds[:,:2].mean(0)))
            metrics={k:float(np.mean(x)) for k,x in logs.items()}
            yaw_limit=.06 if label in ['stop','stand'] else .2
            passed=falls==0 and bias<limit and yb<yaw_limit and metrics['Imitation/keypoint_rmse_m']<.075
            item=dict(label=label,falls=falls,mean_velocity=velocity.mean(0).tolist(),velocity_error=bias,velocity_error_limit=limit,yaw_error=yb,metrics=metrics,passed=bool(passed));items.append(item)
            print('CONTINUOUS_SEGMENT '+json.dumps(item),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'rollout.npz',**{k:np.asarray(v) for k,v in traces.items()},labels=labels,dt=env.dt,link_names=env.links,dof_names=env.names)
    clearance=all(i['metrics']['Imitation/near_hip_clearance_deficit_m']<.012 for i in items if i['label']=='speed_0.3')
    passed=all(i['passed'] for i in items) and clearance
    report=dict(checkpoint=str(checkpoint),total_falls=sum(i['falls'] for i in items),total_seconds=sum(s for _,s in suite),segments=items,env_config=env.cfg,
                passed_bank_gate=bool(passed and not diagnostic),passed_continuous_gate=bool(passed and not diagnostic),passed_arm_clearance_gate=bool(clearance),runtime_compensation_disabled=True,runtime_compensation_removed=True,
                method='Complete mocap state interpolation with continuous commands; learned arm clearance; no runtime compensation',evaluation_mode='continuous_command_sequence')
    if diagnostic:report.update(passed_diagnostic_gate=bool(passed),evaluation_mode='diagnostic_subset_not_full_acceptance')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CONTINUOUS_REPORT '+json.dumps(dict(passed=report['passed_continuous_gate'],diagnostic_passed=report.get('passed_diagnostic_gate'),falls=report['total_falls'])),flush=True)
