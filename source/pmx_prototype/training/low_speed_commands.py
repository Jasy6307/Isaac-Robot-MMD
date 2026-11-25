"""Dedicated low-speed and stop command pools for the unified sampler."""
import math

import torch


def _low_speed_start(cfg):
    first=round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))
    if cfg.get('unified_pivot_sampling',cfg.get('unified_pivot_training',False)):
        first+=round(100*cfg.get('unified_pivot_env_fraction',.1))+round(100*cfg.get('unified_reversal_env_fraction',.1))
    return first+round(100*cfg.get('unified_transition_env_fraction',0.))


def apply_low_speed_pools(env,ids,on_reset):
    cfg=env.cfg
    steady=round(100*cfg.get('unified_low_speed_env_fraction',0.))
    stopping=round(100*cfg.get('unified_low_stop_env_fraction',0.))
    backward_fraction = float(cfg.get('unified_backward_start_env_fraction', 0.))
    if not 0. <= backward_fraction <= 1.:
        raise ValueError('Invalid backward-start environment fraction')
    backward_count = round(100 * backward_fraction)
    if backward_count > stopping:
        raise ValueError('Backward-start lanes must fit inside the low-speed stop pool')
    if steady==0 and stopping==0:return  # No RNG or behavior change for old recipes.
    first=_low_speed_start(cfg)
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
    speed_draw=torch.rand(n,device=env.device)
    speed=lower+speed_draw*(upper-lower)
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
    if backward_count:
        backward_mask = stop_mask & (bucket < first + steady + backward_count)
        backward = backward_mask[selected]
        if on_reset:
            env.low_stop_stage[ids[backward_mask]] = True
        # Reuse the draw and anchor half the starts at the acceptance speed.
        draw = speed_draw[backward]
        backward_speed = torch.where(draw < .5, .40, .30 + .30 * (draw - .5))
        commands[backward] = 0.
        commands[backward, 0] = -backward_speed.clamp_max(cfg['max_speed'])
    stationary=env.low_stop_stage[ids[selected]]&stop_mask[selected]
    commands[stationary]=0.
    env.commands[ids[selected]]=commands
    walk_hold=float(cfg.get('unified_low_speed_hold_seconds',12.))
    walk_jitter=float(cfg.get('unified_low_speed_hold_jitter_seconds',4.))
    stop_hold=float(cfg.get('unified_low_stop_hold_seconds',6.))
    stop_jitter=float(cfg.get('unified_low_stop_hold_jitter_seconds',2.))
    if min(walk_hold,stop_hold)<=0 or min(walk_jitter,stop_jitter)<0:raise ValueError('Invalid low-speed hold times')
    walk_draw=torch.rand(n,device=env.device)
    stop_draw=torch.rand(n,device=env.device)
    duration=torch.where(stationary,stop_hold+stop_draw*stop_jitter,walk_hold+walk_draw*walk_jitter)
    if backward_count:
        duration[backward] = torch.where(stationary[backward], 3. + stop_draw[backward],
                                         6. + 2. * walk_draw[backward])
    env.unified_command_hold[ids[selected]]=duration


def _pivot_stop_bounds(cfg):
    fraction = float(cfg.get('unified_pivot_stop_env_fraction', 0.))
    if not 0. <= fraction <= 1.:
        raise ValueError('Invalid pivot-stop environment fraction')
    count = round(100 * fraction)
    low_speed_count = round(100 * cfg.get('unified_low_speed_env_fraction', 0.))
    low_stop_count = round(100 * cfg.get('unified_low_stop_env_fraction', 0.))
    first = _low_speed_start(cfg) + low_speed_count + low_stop_count
    if cfg.get('unified_transition_env_fraction', 0.):
        # The running transition pool uses the configured offsets even when
        # standalone pivot sampling was explicitly turned off.
        transition_keys = ('unified_backward_env_fraction', 'unified_right_env_fraction',
                           'unified_pivot_env_fraction', 'unified_reversal_env_fraction',
                           'unified_transition_env_fraction')
        transition_end = sum(round(100 * cfg.get(key, 0.)) for key in transition_keys)
        first = max(first, transition_end)
    if first < 0 or (fraction and (count < 2 or count % 2)) or first + count > 100:
        raise ValueError('Pivot-stop lanes require an even number of unused percentage buckets')
    return first, first + count


def pivot_stop_directions(cfg, ids):
    """Signed turn-stop lane assignment, including the stationary portion."""
    first, end = _pivot_stop_bounds(cfg)
    bucket = ids.remainder(100)
    sign = torch.where((bucket - first).remainder(2) == 0, 1., -1.)
    return torch.where((bucket >= first) & (bucket < end), sign, 0.)


def apply_pivot_stop_pool(env, ids, previous_yaw, on_reset):
    """Alternate between boundary-speed pivots and a hold at zero command."""
    cfg = env.cfg
    direction = pivot_stop_directions(cfg, ids)
    selected = direction != 0.
    pool_ids = ids[selected]
    if not len(pool_ids):
        return

    yaw = float(cfg['max_yaw'])
    turn_hold = float(cfg.get('unified_pivot_hold_seconds', 6.))
    turn_jitter = float(cfg.get('unified_pivot_hold_jitter_seconds', 2.))
    stop_hold = float(cfg.get('unified_pivot_stop_hold_seconds', 8.))
    stop_jitter = float(cfg.get('unified_pivot_stop_hold_jitter_seconds', 2.))
    if (not all(map(math.isfinite, (yaw, turn_hold, turn_jitter, stop_hold, stop_jitter)))
            or min(yaw, turn_hold, stop_hold) <= 0. or min(turn_jitter, stop_jitter) < 0.):
        raise ValueError('Invalid pivot-stop yaw or hold durations')

    at_stop = previous_yaw[selected].abs() == yaw
    if on_reset:
        at_stop.zero_()
    env.commands[pool_ids, :2] = 0.
    env.commands[pool_ids, 2] = torch.where(at_stop, 0., yaw * direction[selected])
    hold = torch.where(at_stop, stop_hold, turn_hold)
    jitter = torch.where(at_stop, stop_jitter, turn_jitter)
    env.unified_command_hold[pool_ids] = hold + jitter * torch.rand_like(hold)


def apply_standing_resets(env, ids):
    """Begin selected general lanes at idle, then resume general sampling."""
    cfg = env.cfg
    _, first = _pivot_stop_bounds(cfg)
    fraction = float(cfg['unified_stand_reset_env_fraction'])
    if not 0. < fraction <= 1.:
        raise ValueError('Invalid initial standing environment fraction')
    count = round(100 * fraction)
    if count < 1 or first + count > 100:
        raise ValueError('Initial standing lanes must fit within the general pool')
    bucket = ids.remainder(100)
    standing = ids[(bucket >= first) & (bucket < first + count)]
    env.commands[standing] = 0.
    env.unified_command_hold[standing] = float(cfg.get('unified_idle_hold_seconds', 40.))

