"""Command increment with legacy-compatible defaults for controlled audits."""
import numpy as np


def command_increment(commands,reference,dt,config):
    tau=float(config.get('command_smoothing_seconds',.35))
    linear=float(config.get('command_linear_acceleration_limit',.6))
    yaw=float(config.get('command_yaw_acceleration_limit',1.2))
    if min(tau,linear,yaw)<=0:raise ValueError('Command planning parameters must be positive')
    delta=(commands-reference)*(1-np.exp(-dt/tau))
    length=delta[:,:2].norm(dim=-1).clamp_min(1e-6)
    delta[:,:2]*=(linear*dt/length).clamp(max=1)[:,None]
    delta[:,2]=delta[:,2].clamp(-yaw*dt,yaw*dt)
    mode=config.get('command_plan_mode','legacy')
    if mode=='legacy':return delta
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
    desired=(target-current)*(1-np.exp(-dt/tau))
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
    return delta
