"""Command planning shared by locomotion training and both evaluators."""
import numpy as np
import torch


YAW_LIMIT_SPEED_START = .55
YAW_LIMIT_SPEED_END = .90
HIGH_SPEED_YAW_LIMIT = .40


def yaw_speed_limit(speed, config):
    """Return the symmetric yaw limit at the planned planar speed."""
    maximum = float(config.get('max_yaw', .6))
    if not config.get('command_yaw_limit_by_speed'):
        return speed * 0 + maximum
    t = ((speed - YAW_LIMIT_SPEED_START) / (YAW_LIMIT_SPEED_END - YAW_LIMIT_SPEED_START)).clip(0., 1.)
    return maximum + (min(maximum, HIGH_SPEED_YAW_LIMIT) - maximum) * (t.square() * (3 - 2 * t))


def limit_yaw_command(commands, config):
    if not config.get('command_yaw_limit_by_speed'):
        return commands
    limit = yaw_speed_limit(commands[..., :2].norm(dim=-1), config)
    yaw = commands[..., 2].clamp(-limit, limit)
    return torch.cat((commands[..., :2], yaw[..., None]), dim=-1)


def _limit_yaw_increment(commands, reference, delta, dt, smoothing, yaw_acceleration, config):
    if config.get('command_yaw_limit_by_speed'):
        limit = yaw_speed_limit((reference[:, :2] + delta[:, :2]).norm(dim=-1), config)
        requested = commands[:, 2].clamp(-limit, limit)
        current = reference[:, 2]
        change = ((requested - current) * smoothing).clamp(-yaw_acceleration * dt, yaw_acceleration * dt)
        delta[:, 2] = (current + change).clamp(-limit, limit) - current
    return delta


def command_increment(commands,reference,dt,config):
    tau=float(config.get('command_smoothing_seconds',.35))
    linear=float(config.get('command_linear_acceleration_limit',.6))
    yaw=float(config.get('command_yaw_acceleration_limit',1.2))
    if min(tau,linear,yaw)<=0:raise ValueError('Command planning parameters must be positive')
    smoothing=1-np.exp(-dt/tau)
    delta=(commands-reference)*smoothing
    length=delta[:,:2].norm(dim=-1).clamp_min(1e-6)
    delta[:,:2]*=(linear*dt/length).clamp(max=1)[:,None]
    delta[:,2]=delta[:,2].clamp(-yaw*dt,yaw*dt)
    mode=config.get('command_plan_mode','legacy')
    if mode=='legacy':return _limit_yaw_increment(commands,reference,delta,dt,smoothing,yaw,config)
    if mode!='split':raise ValueError('Unknown command plan mode: '+str(mode))
    accelerate=float(config.get('command_speedup_limit',1.2))
    turn=float(config.get('command_direction_change_limit',1.2))
    brake=float(config.get('command_braking_limit',linear))
    if min(accelerate,turn,brake)<=0:raise ValueError('Split command limits must be positive')
    # Project the desired command increment onto the current movement axis.
    # A reversal first loses forward speed under the braking limit, then
    # accelerates along the new axis. Tangential direction changes have their
    # own limit. No robot state, joint targets or policy outputs are modified.
    current=reference[:,:2]
    target=commands[:,:2]
    speed=current.norm(dim=-1,keepdim=True)
    target_speed=target.norm(dim=-1,keepdim=True)
    axis=current/speed.clamp_min(1e-6)
    axis=axis.where(speed>1e-6,target/target_speed.clamp_min(1e-6))
    desired=(target-current)*smoothing
    radial=(desired*axis).sum(-1,keepdim=True)
    tangent=desired-axis*radial
    tangent*=((turn*dt)/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp(max=1)
    planned=axis*radial.clamp(-brake*dt,accelerate*dt)+tangent
    planned*=((max(brake,accelerate,turn)*dt)/planned.norm(dim=-1,keepdim=True).clamp_min(1e-6)).clamp(max=1)
    # Preserve the proven stop and straight-line braking path exactly when
    # the braking limit retains the legacy value.
    alignment=(axis*target/target_speed.clamp_min(1e-6)).sum(-1,keepdim=True)
    straight_brake=(target_speed<1e-6)|((target_speed<speed)&(alignment>1-1e-6))
    if brake==linear:planned=planned.where(~straight_brake,delta[:,:2])
    delta[:,:2]=planned
    return _limit_yaw_increment(commands,reference,delta,dt,smoothing,yaw,config)
