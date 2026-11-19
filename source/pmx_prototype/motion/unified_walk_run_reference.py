"""One continuous full-body reference clock for walking and directional running.

Phase is measured in gait cycles, not whole-file loops. Long captured passages
therefore retain successive steps while walk/run clips stay contact aligned.
Only reference data is blended; this module never writes the simulated body.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import torch
import numpy as np


def heading_velocity(velocity,heading):
    """Forward/left progression in the command controller's heading frame."""
    c=heading.cos();s=heading.sin()
    return torch.stack([s*velocity[:,0]-c*velocity[:,1],c*velocity[:,0]+s*velocity[:,1]],-1)


def velocity_at_offset(velocity,rotation,offset):
    """Move a world-space origin velocity to a body-local COM offset."""
    local=offset.expand_as(velocity[...,:3]);xyz=rotation[...,:3]
    t=2*torch.cross(xyz,local,dim=-1)
    world=local+rotation[...,3:]*t+torch.cross(xyz,t,dim=-1)
    return torch.cat([velocity[...,:3]+torch.cross(velocity[...,3:],world,dim=-1),velocity[...,3:]],-1)


def prepare(motion):
    if len(motion.clip_names)!=15:raise ValueError('Unified reference needs the 15-clip walk/run bank')
    motion.gait_cycles=torch.tensor(motion.metadata_cycles,device=motion.commands.device,dtype=torch.float32)
    phases=np.arange(256)/256
    contacts=motion.data['contacts'].cpu().numpy()
    reference=np.stack([np.interp(phases*motion.count,np.arange(motion.count+1),np.r_[contacts[2,:,k],contacts[2,0,k]]) for k in range(2)],-1)
    offsets=[]
    for c,cycles in enumerate(motion.metadata_cycles):
        errors=[]
        for shift in np.arange(128)/128:
            x=((phases+shift)/cycles%1)*motion.count
            sample=np.stack([np.interp(x,np.arange(motion.count+1),np.r_[contacts[c,:,k],contacts[c,0,k]]) for k in range(2)],-1)
            errors.append(np.mean((sample-reference)**2))
        offsets.append(float(np.argmin(errors))/128)
    motion.gait_offsets=torch.tensor(offsets,device=motion.commands.device,dtype=torch.float32)
    return offsets


def components(motion,commands,config):
    f,l,y=commands.unbind(-1);speed=commands[:,:2].norm(dim=-1)
    lower=config.get('walk_run_blend_start',.55);upper=config.get('walk_run_blend_end',.9)
    t=((speed-lower)/(upper-lower)).clamp(0,1);run=t*t*(3-2*t)
    activity=(speed/.4).clamp(0,1)
    direction=torch.stack([f.abs(),l.abs()],-1)
    direction=direction/direction.sum(-1,keepdim=True).clamp_min(1e-6)
    walk=direction*(activity*(1-run))[:,None];running=direction*(activity*run)[:,None]
    pivot=(1-activity)*(y.abs()/config.get('max_yaw',.6)).clamp(0,1)
    weights=torch.cat([walk,running,((1-activity)-pivot)[:,None],pivot[:,None]],-1)
    ids=torch.stack([torch.where(f>=0,2,4),torch.where(l>=0,5,6),
                     torch.where(f>=0,11,12),torch.where(l>=0,13,14),
                     torch.full_like(f,3,dtype=torch.long),torch.where(y>=0,9,10)],-1)
    # Match each directional combination's stance speed. At low speed the pose
    # amplitude fades toward idle while cadence remains finite.
    native=(weights[:,:,None]*motion.commands[ids][...,:2]).sum(1).norm(dim=-1)
    ratio=torch.where(speed>.001,speed/native.clamp_min(.05),torch.ones_like(speed)).clamp(.25,2.5)
    frequency=(weights*motion.gait_cycles[ids]/motion.periods[ids]).sum(-1)*ratio
    return ids,weights,frequency


def sample(motion,phase,commands,config,command_rate=None):
    if config.get('unified_vectorized_reference'):
        return sample_vectorized(motion,phase,commands,config,command_rate)
    ids,weights,frequency=components(motion,commands,config)
    rates=None
    if command_rate is not None and config.get('unified_reference_derivatives'):
        epsilon=.001
        plus=components(motion,commands+epsilon*command_rate,config)[1]
        minus=components(motion,commands-epsilon*command_rate,config)[1]
        rates=(plus-minus)/(2*epsilon)
    result={};anchor=None
    blend_qd=blend_velocity=blend_quaternion_rate=None
    for k in range(weights.shape[1]):
        clip=ids[:,k];cycles=motion.gait_cycles[clip]
        state=motion.sample((phase+motion.gait_offsets[clip])/cycles,clip)
        scale=frequency*motion.periods[clip]/cycles
        state.pop('gait_sin',None);state.pop('gait_cos',None)
        state['qd']=state['qd']*scale[:,None]
        source=motion.commands[clip];v=state['velocities'].clone()
        mean=torch.stack([source[:,1],-source[:,0],torch.zeros_like(scale)],-1)
        omega=torch.zeros_like(mean);omega[:,2]=source[:,2]
        v[:,:,:3]-=mean[:,None]+torch.cross(omega[:,None].expand_as(state['positions']),state['positions'],dim=-1)
        v[:,:,3:]-=omega[:,None];state['velocities']=v*scale[:,None,None]
        if anchor is None:anchor=state['rotations']
        state['rotations']=torch.where(((anchor*state['rotations']).sum(-1)<0)[...,None],-state['rotations'],state['rotations'])
        if rates is not None:
            dw=rates[:,k,None]
            blend_qd=dw*state['q'] if blend_qd is None else blend_qd+dw*state['q']
            pv=dw[:,:,None]*state['positions'];qr=dw[:,:,None]*state['rotations']
            blend_velocity=pv if blend_velocity is None else blend_velocity+pv
            blend_quaternion_rate=qr if blend_quaternion_rate is None else blend_quaternion_rate+qr
        for name,value in state.items():
            w=weights[:,k].reshape((len(phase),)+(1,)*(value.ndim-1))
            result[name]=result.get(name,0)+w*value
    quaternion_norm=result['rotations'].norm(dim=-1,keepdim=True).clamp_min(1e-6)
    result['rotations']=result['rotations']/quaternion_norm
    if rates is not None:
        result['qd']+=blend_qd
        result['velocities'][...,:3]+=blend_velocity
        q=result['rotations'];dq=(blend_quaternion_rate-q*(q*blend_quaternion_rate).sum(-1,keepdim=True))/quaternion_norm
        angular=2*(dq[...,:3]*q[...,3:]-dq[...,3:]*q[...,:3]-torch.cross(dq[...,:3],q[...,:3],dim=-1))
        result['velocities'][...,3:]+=angular
    mean=torch.stack([commands[:,1],-commands[:,0],torch.zeros_like(frequency)],-1)
    omega=torch.zeros_like(mean);omega[:,2]=commands[:,2]
    result['velocities'][:,:,:3]+=mean[:,None]+torch.cross(omega[:,None].expand_as(result['positions']),result['positions'],dim=-1)
    result['velocities'][:,:,3:]+=omega[:,None]
    result['gait_sin']=torch.sin(2*torch.pi*phase);result['gait_cos']=torch.cos(2*torch.pi*phase)
    return result


def sample_vectorized(motion,phase,commands,config,command_rate=None):
    """Identical reference math with the six clip reads in one tensor batch."""
    ids,weights,frequency=components(motion,commands,config)
    n,k=ids.shape;cycles=motion.gait_cycles[ids]
    phases=(phase[:,None]+motion.gait_offsets[ids])/cycles
    raw=motion.sample(phases.flatten(),ids.flatten())
    raw.pop('gait_sin',None);raw.pop('gait_cos',None)
    states={name:value.reshape((n,k)+value.shape[1:]) for name,value in raw.items()}
    scale=frequency[:,None]*motion.periods[ids]/cycles
    states['qd']=states['qd']*scale[:,:,None]
    source=motion.commands[ids];mean=torch.stack([source[...,1],-source[...,0],torch.zeros_like(scale)],-1)
    omega=torch.zeros_like(mean);omega[...,2]=source[...,2]
    velocity=states['velocities'].clone()
    velocity[...,:3]-=mean[:,:,None]+torch.cross(omega[:,:,None].expand_as(states['positions']),states['positions'],dim=-1)
    velocity[...,3:]-=omega[:,:,None];states['velocities']=velocity*scale[:,:,None,None]
    rotation=states['rotations'];anchor=rotation[:,:1]
    states['rotations']=torch.where(((anchor*rotation).sum(-1)<0)[...,None],-rotation,rotation)
    result={name:(weights.reshape((n,k)+(1,)*(value.ndim-2))*value).sum(1) for name,value in states.items()}
    norm=result['rotations'].norm(dim=-1,keepdim=True).clamp_min(1e-6)
    result['rotations']=result['rotations']/norm
    if command_rate is not None and config.get('unified_reference_derivatives'):
        epsilon=.001
        rates=(components(motion,commands+epsilon*command_rate,config)[1]-components(motion,commands-epsilon*command_rate,config)[1])/(2*epsilon)
        result['qd']+=(rates[:,:,None]*states['q']).sum(1)
        result['velocities'][...,:3]+=(rates[:,:,None,None]*states['positions']).sum(1)
        qr=(rates[:,:,None,None]*states['rotations']).sum(1);q=result['rotations']
        dq=(qr-q*(q*qr).sum(-1,keepdim=True))/norm
        result['velocities'][...,3:]+=2*(dq[...,:3]*q[...,3:]-dq[...,3:]*q[...,:3]-torch.cross(dq[...,:3],q[...,:3],dim=-1))
    mean=torch.stack([commands[:,1],-commands[:,0],torch.zeros_like(frequency)],-1)
    omega=torch.zeros_like(mean);omega[:,2]=commands[:,2]
    result['velocities'][...,:3]+=mean[:,None]+torch.cross(omega[:,None].expand_as(result['positions']),result['positions'],dim=-1)
    result['velocities'][...,3:]+=omega[:,None]
    result['gait_sin']=torch.sin(2*torch.pi*phase);result['gait_cos']=torch.cos(2*torch.pi*phase)
    return result


def sample_commands(env,ids,on_reset=False):
    n=len(ids);device=env.device;cfg=env.cfg
    progress=min(1.,max(0.,(getattr(env,'iteration',0)-cfg.get('unified_curriculum_start',0))/cfg.get('unified_curriculum_iterations',500)))
    maximum=cfg.get('unified_initial_max_speed',.4)+(cfg['max_speed']-cfg.get('unified_initial_max_speed',.4))*progress
    angle=torch.rand(n,device=device)*2*torch.pi;speed=torch.rand(n,device=device)*maximum
    choose=torch.rand(n,device=device)
    # Frequent examples near the walk/run transition, plus boundary coverage.
    transition=(choose>.45)&(choose<.7)
    speed[transition]=(.5+torch.rand(n,device=device)[transition]*.5).clamp(max=maximum)
    speed[choose>.85]=maximum
    cardinal=torch.randint(8,(n,),device=device)*torch.pi/4
    angle=torch.where(torch.rand(n,device=device)<.5,cardinal,angle)
    cmd=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),(torch.rand(n,device=device)*2-1)*cfg['max_yaw']],-1)
    cmd[choose<.2]=0.;cmd[(choose>=.2)&(choose<.3),:2]=0.
    cmd[(choose>=.3)&(choose<.45),2]=0.
    if cfg.get('unified_high_speed_focus'):
        speed=.15+torch.rand(n,device=device)*max(0.,maximum-.15)
        fast=torch.rand(n,device=device)<cfg.get('unified_high_speed_fraction',.65)
        lower=min(maximum,cfg.get('unified_high_speed_start',.85))
        speed[fast]=lower+torch.rand(n,device=device)[fast]*(maximum-lower)
        direction_choice=torch.rand(n,device=device);forward_fraction=cfg.get('unified_forward_fraction',.25)
        backward_fraction=cfg.get('unified_backward_fraction',0.)
        angle=torch.where(direction_choice<forward_fraction,torch.zeros_like(angle),angle)
        angle=torch.where((direction_choice>=forward_fraction)&(direction_choice<forward_fraction+backward_fraction),torch.full_like(angle,torch.pi),angle)
        cmd=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),(torch.rand(n,device=device)*2-1)*cfg['max_yaw']],-1)
        cmd[torch.rand(n,device=device)<.5,2]=0.
        idle=cfg.get('unified_idle_fraction',.08);pivot=cfg.get('unified_pivot_fraction',.07)
        cmd[choose<idle]=0.;cmd[(choose>=idle)&(choose<idle+pivot),:2]=0.
        if not on_reset and cfg.get('unified_transition_stop_fraction',0.):
            stopping=(env.commands[ids,:2].norm(dim=-1)>.8)&(torch.rand(n,device=device)<cfg['unified_transition_stop_fraction'])
            cmd[stopping]=0.
    # Reserve environments for the difficult skill: random task selection on
    # every failure otherwise gives short-lived tasks very little rollout time.
    if cfg.get('unified_backward_env_fraction',0.) and not env.evaluation:
        assigned=ids.remainder(100)<round(100*cfg['unified_backward_env_fraction'])
        elapsed=max(0.,getattr(env,'iteration',0)-cfg.get('unified_backward_curriculum_start',0))
        progress=min(1.,elapsed/cfg.get('unified_backward_curriculum_iterations',250))
        limit=cfg.get('unified_backward_initial_speed',.6)+(cfg['max_speed']-cfg.get('unified_backward_initial_speed',.6))*progress
        speed=.4+torch.rand(n,device=device)*max(0.,limit-.4)
        speed=torch.where(torch.rand(n,device=device)<.6,torch.full_like(speed,limit),speed)
        steering=min(1.,max(0.,(elapsed-100)/150))
        spread=cfg.get('unified_backward_spread_radians',torch.pi/4)*steering
        angle=torch.pi+(torch.rand(n,device=device)*2-1)*spread
        choice=torch.rand(n,device=device);pure=cfg.get('unified_backward_pure_fraction',.5)
        boundary=cfg.get('unified_backward_boundary_fraction',0.)
        angle=torch.where(choice<pure,torch.full_like(angle,torch.pi),angle)
        sign=torch.where(torch.rand(n,device=device)<.5,-torch.ones_like(angle),torch.ones_like(angle))
        angle=torch.where((choice>=pure)&(choice<pure+boundary),torch.pi+sign*spread,angle)
        dedicated=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),(torch.rand(n,device=device)*2-1)*cfg['max_yaw']*steering],-1)
        dedicated[torch.rand(n,device=device)<.5,2]=0.
        if not on_reset:
            stopping=(env.commands[ids,:2].norm(dim=-1)>.4)&(torch.rand(n,device=device)<cfg.get('unified_transition_stop_fraction',.35))
            dedicated[stopping]=0.
        cmd[assigned]=dedicated[assigned]
    if cfg.get('unified_right_env_fraction',0.) and not env.evaluation:
        first=round(100*cfg.get('unified_backward_env_fraction',0.))
        last=first+round(100*cfg['unified_right_env_fraction'])
        if last>100:raise ValueError('Reserved directional environment fractions exceed one')
        assigned=(ids.remainder(100)>=first)&(ids.remainder(100)<last)
        elapsed=max(0.,getattr(env,'iteration',0)-cfg.get('unified_backward_curriculum_start',0))
        progress=min(1.,elapsed/cfg.get('unified_right_curriculum_iterations',100))
        limit=cfg.get('unified_right_initial_speed',1.)+(cfg['max_speed']-cfg.get('unified_right_initial_speed',1.))*progress
        speed=.4+torch.rand(n,device=device)*max(0.,limit-.4)
        speed=torch.where(torch.rand(n,device=device)<.6,torch.full_like(speed,limit),speed)
        steering=min(1.,max(0.,(elapsed-100)/150))
        angle=-torch.pi/2+(torch.rand(n,device=device)*2-1)*(torch.pi/4)*steering
        choice=torch.rand(n,device=device);pure=cfg.get('unified_right_pure_fraction',.5)
        boundary=cfg.get('unified_right_boundary_fraction',0.)
        angle=torch.where(choice<pure,torch.full_like(angle,-torch.pi/2),angle)
        sign=torch.where(torch.rand(n,device=device)<.5,-torch.ones_like(angle),torch.ones_like(angle))
        angle=torch.where((choice>=pure)&(choice<pure+boundary),-torch.pi/2+sign*(torch.pi/4)*steering,angle)
        dedicated=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),(torch.rand(n,device=device)*2-1)*cfg['max_yaw']*steering],-1)
        dedicated[torch.rand(n,device=device)<.5,2]=0.
        if not on_reset:
            stopping=(env.commands[ids,:2].norm(dim=-1)>.4)&(torch.rand(n,device=device)<cfg.get('unified_transition_stop_fraction',.35))
            dedicated[stopping]=0.
        cmd[assigned]=dedicated[assigned]
        if cfg.get('unified_lateral_both_directions'):
            left=assigned&(torch.rand(n,device=device)<.5)
            cmd[left,:2]*=-1
    pivot_hold=None
    if cfg.get('unified_pivot_sampling',cfg.get('unified_pivot_training',False)) and not env.evaluation:
        # Apply after directional pools and stop sampling: these commands must
        # never be overwritten by the running curriculum or zero-yaw branch.
        first=round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))
        pure_count=round(100*cfg.get('unified_pivot_env_fraction',.10))
        reverse_count=round(100*cfg.get('unified_reversal_env_fraction',.10))
        if min(first,pure_count,reverse_count)<0 or first+pure_count+reverse_count>100:
            raise ValueError('Directional, pivot and reversal environment pools must fit within 100 percent')
        bucket=ids.remainder(100)
        pure=(bucket>=first)&(bucket<first+pure_count)
        reverse=(bucket>=first+pure_count)&(bucket<first+pure_count+reverse_count)
        reserved=pure|reverse
        sign=torch.where(bucket.remainder(2)==0,1.,-1.)
        previous=env.commands[ids,2]
        if not on_reset:
            sign=torch.where(reverse&(previous.abs()>1e-4),-previous.sign(),sign)
        maximum_yaw=float(cfg['max_yaw'])
        minimum_yaw=float(cfg.get('unified_pivot_min_yaw',.15))
        boundary=float(cfg.get('unified_pivot_boundary_fraction',.5))
        if not 0<minimum_yaw<=maximum_yaw or not 0<=boundary<=1:
            raise ValueError('Pivot yaw range or boundary fraction is invalid')
        magnitude=minimum_yaw+torch.rand(n,device=device)*(maximum_yaw-minimum_yaw)
        magnitude=torch.where(torch.rand(n,device=device)<boundary,maximum_yaw,magnitude)
        # A reversal changes direction at the same speed, including boundaries.
        if not on_reset:magnitude=torch.where(reverse&(previous.abs()>1e-4),previous.abs().clamp(minimum_yaw,maximum_yaw),magnitude)
        cmd[reserved,:2]=0.;cmd[reserved,2]=(sign*magnitude)[reserved]
        pivot_hold=(reserved,float(cfg.get('unified_pivot_hold_seconds',6.))+torch.rand(n,device=device)*float(cfg.get('unified_pivot_hold_jitter_seconds',2.)))
    transition_hold=None
    if cfg.get('unified_transition_env_fraction',0.) and not env.evaluation:
        first=(round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))+
               round(100*cfg.get('unified_pivot_env_fraction',0.))+round(100*cfg.get('unified_reversal_env_fraction',0.)))
        count=round(100*cfg['unified_transition_env_fraction'])
        if count<0 or first+count>100:raise ValueError('Transition pool overlaps reserved environments')
        selected=(ids.remainder(100)>=first)&(ids.remainder(100)<first+count)
        if not hasattr(env,'transition_stage'):
            env.transition_stage=torch.zeros(env.num_envs,device=device,dtype=torch.long)
        selected_ids=ids[selected]
        if on_reset:
            env.transition_stage[selected_ids]=torch.randint(3,(len(selected_ids),),device=device)
        else:env.transition_stage[selected_ids]=(env.transition_stage[selected_ids]+1)%3
        bank=torch.tensor([[1.1,0.,.6],[1.1,0.,-.6],[-.9,0.,.6]],device=device)
        cmd[selected]=bank[env.transition_stage[selected_ids]]
        transition_hold=selected
    pivot_stop_enabled = not env.evaluation and cfg.get('unified_pivot_stop_env_fraction', 0.)
    previous_yaw = env.commands[ids, 2] if pivot_stop_enabled else None
    if env.evaluation:cmd.zero_()
    env.commands[ids]=cmd;env.command_age[ids]=0.
    if not hasattr(env,'unified_command_hold'):env.unified_command_hold=torch.zeros(env.num_envs,device=device)
    env.unified_command_hold[ids]=torch.where(cmd.abs().sum(-1)<.001,torch.full((n,),cfg.get('unified_idle_hold_seconds',40.),device=device),
                                            cfg.get('unified_moving_hold_min_seconds',6.)+torch.rand(n,device=device)*6.)
    if pivot_hold is not None:
        reserved,hold=pivot_hold
        env.unified_command_hold[ids[reserved]]=hold[reserved]
    if transition_hold is not None:
        env.unified_command_hold[ids[transition_hold]]=float(cfg.get('unified_transition_hold_seconds',5.))+torch.rand(int(transition_hold.sum()),device=device)*float(cfg.get('unified_transition_hold_jitter_seconds',1.))
    if not env.evaluation:
        from low_speed_commands import apply_low_speed_pools, apply_pivot_stop_pool, apply_standing_resets
        apply_low_speed_pools(env,ids,on_reset)
        if pivot_stop_enabled:
            apply_pivot_stop_pool(env, ids, previous_yaw, on_reset)
        if on_reset and cfg.get('unified_stand_reset_env_fraction', 0.):
            apply_standing_resets(env, ids)
    if on_reset:
        from command_plan import limit_yaw_command
        env.continuous_ref_commands[ids]=limit_yaw_command(env.commands[ids],cfg)
