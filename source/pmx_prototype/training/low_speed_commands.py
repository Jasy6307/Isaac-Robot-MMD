"""Explicit sustained low-speed and walk/stop pools for the unified sampler."""
import torch


def apply_low_speed_pools(env,ids,on_reset):
    cfg=env.cfg
    steady=round(100*cfg.get('unified_low_speed_env_fraction',0.))
    stopping=round(100*cfg.get('unified_low_stop_env_fraction',0.))
    if steady==0 and stopping==0:return  # No RNG or behavior change for old recipes.
    first=round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))
    if cfg.get('unified_pivot_sampling',cfg.get('unified_pivot_training',False)):
        first+=round(100*cfg.get('unified_pivot_env_fraction',.1))+round(100*cfg.get('unified_reversal_env_fraction',.1))
    first+=round(100*cfg.get('unified_transition_env_fraction',0.))
    if min(first,steady,stopping)<0 or first+steady+stopping>100:
        raise ValueError('Unified command pools must fit within100 percent')
    bucket=ids.remainder(100)
    steady_mask=(bucket>=first)&(bucket<first+steady)
    stop_mask=(bucket>=first+steady)&(bucket<first+steady+stopping)
    selected=steady_mask|stop_mask
    n=int(selected.sum())
    if not n:return
    lower=float(cfg.get('unified_low_speed_min',.15));upper=float(cfg.get('unified_low_speed_max',.4))
    if not 0<lower<upper<=cfg['max_speed']:raise ValueError('Invalid low-speed range')
    speed=lower+torch.rand(n,device=env.device)*(upper-lower)
    anchors=torch.tensor([.15,.2,.25,.3,.35,.4],device=env.device)
    anchors=anchors[(anchors>=lower)&(anchors<=upper)]
    if len(anchors):
        speed=torch.where(torch.rand(n,device=env.device)<.5,anchors[torch.randint(len(anchors),(n,),device=env.device)],speed)
    angle=torch.rand(n,device=env.device)*2*torch.pi
    direction=torch.rand(n,device=env.device)
    compass=torch.randint(8,(n,),device=env.device)*torch.pi/4
    angle=torch.where(direction<.6,torch.zeros_like(angle),torch.where(direction<.85,compass,angle))
    yaw=(torch.rand(n,device=env.device)*2-1)*cfg['max_yaw']
    yaw=torch.where(torch.rand(n,device=env.device)<.8,torch.zeros_like(yaw),yaw)
    commands=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),yaw],-1)
    if not hasattr(env,'low_stop_stage'):
        env.low_stop_stage=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    stop_ids=ids[stop_mask]
    if on_reset:env.low_stop_stage[stop_ids]=torch.rand(len(stop_ids),device=env.device)<.5
    else:env.low_stop_stage[stop_ids]=~env.low_stop_stage[stop_ids]
    stationary=env.low_stop_stage[ids[selected]]&stop_mask[selected]
    commands[stationary]=0.
    env.commands[ids[selected]]=commands
    walk_hold=float(cfg.get('unified_low_speed_hold_seconds',12.))
    walk_jitter=float(cfg.get('unified_low_speed_hold_jitter_seconds',4.))
    stop_hold=float(cfg.get('unified_low_stop_hold_seconds',6.))
    stop_jitter=float(cfg.get('unified_low_stop_hold_jitter_seconds',2.))
    if min(walk_hold,stop_hold)<=0 or min(walk_jitter,stop_jitter)<0:raise ValueError('Invalid low-speed hold times')
    duration=walk_hold+torch.rand(n,device=env.device)*walk_jitter
    duration=torch.where(stationary,stop_hold+torch.rand(n,device=env.device)*stop_jitter,duration)
    env.unified_command_hold[ids[selected]]=duration

