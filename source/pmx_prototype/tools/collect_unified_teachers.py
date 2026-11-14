"""Collect physical teacher trajectories for offline single-actor distillation.

No teacher or action conversion is used by the released controller. The offline
labels express the old teacher's PD torque in the unified reference coordinates.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT
import argparse,json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--student',type=Path,required=True)
p.add_argument('--teacher',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--run-seconds',type=float,default=120.)
args=p.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True,'disable_viewport_updates':True,'limit_cpu_threads':2})
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from anon_imitation_env import AnonImitationEnv

def load(path,owner=None):
    cfg=json.loads((path.parent/'env_config.json').read_text())
    cfg.update(simulation_device='cpu',amp_reward_diagnostics=False,runtime_inference_only=True)
    env=AnonImitationEnv(1,42,app,evaluation=True,config=cfg,shared_physics=owner)
    env.max_episode_length=round(3600/env.dt)
    runner=OnPolicyRunner(env,json.loads((path.parent/'agent_config.json').read_text()),device='cpu')
    runner.load(str(path),map_location='cpu')
    return env,runner,runner.get_inference_policy()

def targets(env,ref,action):
    qref=ref['q'][:,env.q_order]
    scale=torch.full_like(action,env.cfg.get('imitation_residual_scale',.3))
    nominal=qref.clone()
    if env.cfg.get('amp_enabled'):
        scale[:,env.upper_ids]=.45
        mix=env.cfg.get('amp_upper_reference_mix',.35)
        nominal[:,env.upper_ids]=env.amp_neutral[env.upper_ids]+mix*(qref[:,env.upper_ids]-env.amp_neutral[env.upper_ids])
    position=(nominal+scale*action.clamp(-2,2)).clamp(env.limits[:,0]+.01,env.limits[:,1]-.01)
    velocity=torch.zeros_like(action)
    if env.cfg.get('reference_velocity_feedforward'):
        qd=ref['qd'][:,env.q_order]
        if env.cfg.get('reference_full_body_velocity_feedforward'):velocity[:]=qd
        velocity[:,env.leg_ids]=qd[:,env.leg_ids]*env.cfg.get('velocity_feedforward_scale',1.)
        if env.cfg.get('reference_distal_arm_velocity_feedforward'):
            distal=[i for i,n in enumerate(env.names) if any(k in n for k in ['forearm','wrist'])]
            velocity[:,distal]=qd[:,distal]
    return nominal,scale,position,velocity.clamp(-10,10)

try:
    torch.set_num_threads(2)
    teacher,tr,teach=load(args.teacher)
    student,sr,study=load(args.student,teacher)
    teacher.reset(teacher.ids,command=[1.2,0,0],phase=0.);teacher.refresh()
    stiffness=teacher.robot.get_dof_stiffnesses().clone()
    damping=teacher.robot.get_dof_dampings().clone()
    if not torch.allclose(stiffness,student.robot.get_dof_stiffnesses()):raise ValueError('Teacher drive gains differ')
    obs=[];actions=[];groups=[];falls=0;invalid=0;torque_error=0.;speeds=[]
    def save(x,a,group):
        obs.append(x['policy'].numpy().copy());actions.append(a.numpy().copy());groups.append(group)
    with torch.inference_mode():
        for step in range(round(args.run_seconds/teacher.dt)):
            teacher.set_command([1.2,0,0]);teacher.refresh()
            student.commands[:]=torch.tensor([1.2,0,0]);student.continuous_ref_commands[:]=student.commands
            student.continuous_ref_rate.zero_();student.reference_heading[:]=teacher.reference_heading
            student.phase[:]=teacher.phase*4-student.motion.gait_offsets[11]
            student.refresh();x=student.get_observations();a=teach(teacher.get_observations())
            old=teacher.sample_reference((teacher.phase+teacher.dt*teacher.frequency())%1)
            new=student.sample_reference(student.phase+student.dt*student.frequency())
            _,_,pt,vt=targets(teacher,old,a)
            nominal,scale,_,vs=targets(student,new,torch.zeros_like(a))
            mapped=(pt+damping/stiffness*(vt-vs)-nominal)/scale
            _,_,ps,vs=targets(student,new,mapped)
            torque_error=max(torque_error,float((stiffness*(ps-pt)+damping*(vs-vt)).abs().max()))
            valid=bool((mapped.abs()<=1.95).all())
            if valid:save(x,mapped,1)
            else:invalid+=1
            student.previous_actions.copy_(student.actions);student.actions.copy_(mapped.clamp(-2,2))
            _,_,done,_=teacher.step(a);falls+=int(done.sum());speeds.append(float(teacher.logical_vel[0,0]))
            if done.any():raise RuntimeError('Running teacher fell during collection')
            if step%300==0:print('TEACHER_COLLECTION',step,len(obs),float(mapped.abs().max()),flush=True)
        # Preserve actual low-speed behaviour and the full idle duration. The
        # target is this experiment's current actor, not a runtime ensemble.
        student.physics_time=teacher.physics_time
        low=[(40,[0,0,0]),(8,[.4,0,0]),(6,[0,0,0]),(8,[-.35,0,0]),(8,[0,.35,0]),(8,[0,-.35,0]),
             (8,[.25,.25,.4]),(8,[-.25,-.25,-.4]),(6,[0,0,.6]),(6,[0,0,-.6]),(8,[.2,0,.6]),(8,[.2,0,-.6])]
        for seconds,cmd in low:
            student.reset(student.ids,command=cmd,phase=float(torch.rand(()))*4);student.refresh()
            for step in range(round(seconds/student.dt)):
                student.set_command(cmd);x=student.get_observations();a=study(x);save(x,a,0)
                _,_,done,_=student.step(a)
                if done.any():raise RuntimeError(f'Low-speed teacher fell: {cmd}')
            print('LOW_TEACHER_COLLECTION',cmd,len(obs),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'teachers.npz',observations=np.concatenate(obs),actions=np.concatenate(actions),groups=groups)
    report=dict(student=str(args.student.resolve()),running_teacher=str(args.teacher.resolve()),samples=len(obs),run_samples=int(sum(groups)),
                low_samples=len(obs)-sum(groups),teacher_falls=falls,rejected_unrepresentable_labels=invalid,max_equivalent_drive_torque_error_nm=torque_error,
                mean_running_teacher_forward_speed=float(np.mean(speeds)),runtime_teacher_required=False,offline_label_mapping_only=True)
    (args.output/'collection.json').write_text(json.dumps(report,indent=2));print('TEACHERS_READY',json.dumps(report),flush=True)
    teacher.close()
finally:app.close()
