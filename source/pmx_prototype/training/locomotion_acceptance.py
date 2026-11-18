"""Versioned response/steady-state scoring, independent of simulator startup."""
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from command_plan import command_increment

PROTOCOL='response_steady_v2'


def specification():
    return dict(protocol=PROTOCOL,min_steady_seconds=2.,response_min_seconds=2.,response_cap_seconds=5.,
        response_grace_after_planner='max(1.5 seconds, one target gait cycle +0.5 second confirmation)',
        fallback_gait_period_seconds=1.5,planner_linear_tolerance=.02,planner_yaw_tolerance=.02,
        linear_filter_seconds=1.,yaw_filter_seconds=2.,response_hold_seconds=.5,
        variable_direction_lag_limit_seconds=1.,
        steady_linear_limit='max(0.06,0.22*requested_speed)',steady_idle_speed_limit=.08,
        steady_yaw_limit_stationary=.1,steady_yaw_limit_moving=.2,
        note='Centered filters estimate response time offline. Short screens may be inconclusive; they never grant full acceptance.')


@lru_cache(maxsize=8)
def _clock_metadata(path,mtime):
    with np.load(path) as data:
        return SimpleNamespace(commands=torch.tensor(data['commands'],dtype=torch.float32),
            gait_cycles=torch.tensor(data['gait_cycles'],dtype=torch.float32),periods=torch.tensor(data['periods'],dtype=torch.float32))


def gait_period(target,config):
    if not config.get('full_motion_path'):return 1.5
    from unified_walk_run_reference import components
    path=Path(config['full_motion_path'])
    motion=_clock_metadata(str(path),path.stat().st_mtime_ns)
    frequency=float(components(motion,torch.as_tensor(np.asarray(target,dtype=np.float32)).reshape(1,3),config)[2][0])
    return 1/max(frequency,.1)


def planning_time(initial,target,config,dt):
    current=torch.tensor(initial,dtype=torch.float32).reshape(1,3)
    requested=torch.tensor(target,dtype=torch.float32).reshape(1,3)
    for step in range(round(10/dt)+1):
        difference=current-requested
        if float(difference[0,:2].norm())<=.02 and abs(float(difference[0,2]))<=.02:return step*dt
        current+=command_increment(requested,current,dt,config)
    return None


def _response(values,target,limit,dt,window_seconds,stationary=False):
    values=np.asarray(values);width=max(1,round(window_seconds/dt))
    if len(values)<width:return None
    if values.ndim==1:
        averaged=np.convolve(values,np.ones(width)/width,mode='valid')
        flags=abs(averaged-target)<limit
    else:
        averaged=np.stack([np.convolve(values[:,k],np.ones(width)/width,mode='valid') for k in range(values.shape[1])],-1)
        flags=np.linalg.norm(averaged-target,axis=-1)<limit
        if stationary:
            speed=np.convolve(np.linalg.norm(values,axis=-1),np.ones(width)/width,mode='valid')
            flags&=speed<.08
    hold=max(1,math.ceil(.5/dt)+1)
    if len(flags)<hold:return None
    indices=np.flatnonzero(np.convolve(flags.astype(int),np.ones(hold,dtype=int),mode='valid')==hold)
    if not len(indices):return None
    # End of the confirmed hold, timestamped at the centered averaging window.
    return float((indices[0]+hold-1+(width-1)/2)*dt)


def _tracking_error(error,variable,dt):
    if not variable:return float(np.linalg.norm(np.mean(error,axis=0)))
    width=min(len(error),max(1,round(1/dt)))
    averaged=np.stack([np.convolve(error[:,k],np.ones(width)/width,mode='valid') for k in range(2)],-1)
    return float(np.linalg.norm(averaged,axis=-1).mean())


def _direction_lag(raw,actual,dt):
    width=max(1,round(1/dt))
    if len(raw)<width+round(1/dt):return None
    averaged=np.stack([np.convolve(actual[:,k],np.ones(width)/width,mode='valid') for k in range(2)],-1)
    centers=np.arange(len(averaged))+(width-1)/2
    target=np.stack([np.interp(centers,np.arange(len(raw)),raw[:,k]) for k in range(2)],-1)
    angle=np.unwrap(np.arctan2(raw[:,1],raw[:,0]));rate=float(np.median(np.diff(angle))/dt)
    if abs(rate)<1e-4:return None
    valid=(centers*dt>=2)&(np.linalg.norm(averaged,axis=-1)>.1)
    if valid.sum()<round(1/dt):return None
    difference=np.arctan2(np.sin(np.arctan2(target[:,1],target[:,0])-np.arctan2(averaged[:,1],averaged[:,0])),
        np.cos(np.arctan2(target[:,1],target[:,0])-np.arctan2(averaged[:,1],averaged[:,0])))
    return float(np.median(difference[valid]/rate))


def quality_passed(item,config):
    """Keep the original physical/style gates; tracking is scored separately."""
    metrics=item['metrics']
    passed=item['falls']==0 and metrics['Imitation/keypoint_rmse_m']<.075
    passed&=metrics.get('Imitation/near_hip_clearance_deficit_m',1)<.015
    if config.get('unified_command_frame_metrics'):
        passed&=metrics['Imitation/pelvis_orientation_rmse_rad']<.25
    if item['label'] in ['stand','stand_end']:
        passed&=item['displacement_m']<.25 and item['pelvis_yaw_drift_rad']<.35
    if item['label'] in ['run_forward','run_backward','run_left','run_right']:
        flight=item.get('force_flight_fraction',item['flight_fraction']) if config.get('unified_eval_force_flight') else item['flight_fraction']
        passed&=flight>.01
    return bool(passed)


def score_segment(item,actual,planned,requested,geometric,initial_plan,config,dt):
    actual=np.asarray(actual);planned=np.asarray(planned);requested=np.asarray(requested);geometric=np.asarray(geometric)
    if not(len(actual)==len(planned)==len(requested)==len(geometric)) or not np.isfinite(actual).all():
        raise ValueError('Invalid tracking trace')
    duration=len(actual)*dt
    variable=item['label'].startswith('ramp_') or item['label']=='direction_sweep'
    stationary=item['label'] in ['stand','stand_end','stop']
    hard=quality_passed(item,config)
    result=dict(protocol=PROTOCOL,quality_passed=hard,classification='continuous_tracking' if variable else 'step_response')
    if variable:
        start=round(2/dt)
        error=_tracking_error(actual[start:,:2]-planned[start:,:2],True,dt)
        geometry=_tracking_error(geometric[start:]-planned[start:,:2],True,dt)
        yaw=float(abs((actual[start:,2]-requested[start:,2]).mean()))
        lag=_direction_lag(requested[:,:2],actual[:,:2],dt) if item['label']=='direction_sweep' else None
        lag_pass=item['label']!='direction_sweep' or (lag is not None and abs(lag)<=1.)
        passed=hard and error<.2 and geometry<.2 and yaw<.2 and lag_pass
        result.update(status='passed' if passed else 'failed',passed=bool(passed),
            tracking_against='planned_command',velocity_error=error,geometric_velocity_error=geometry,
            yaw_error=yaw,raw_command_error=_tracking_error(actual[start:,:2]-requested[start:,:2],True,dt),
            apparent_direction_lag_seconds=lag,direction_lag_limit_seconds=1.,
            lag_note='Direction lag is measured only for the excited circular command; a monotone ramp does not identify lag independently of speed bias.')
        return result
    target=requested[-1];limit=max(.06,float(np.linalg.norm(target[:2]))*.22);yaw_limit=.1 if stationary else .2
    plan_seconds=planning_time(initial_plan,target,config,dt)
    cycle=gait_period(target,config);grace=max(1.5,cycle+.5)
    deadline=min(5.,max(2.,(plan_seconds if plan_seconds is not None else 10.)+grace))
    linear_response=_response(actual[:,:2],target[:2],limit,dt,1.,stationary)
    yaw_response=_response(actual[:,2],target[2],yaw_limit,dt,2.)
    responses=[linear_response,yaw_response]
    required=plan_seconds!=0.
    if not required:
        response_status='not_applicable'
    elif plan_seconds is None or plan_seconds>deadline or any(t is not None and t>deadline for t in responses):
        response_status='failed'
    elif all(t is not None for t in responses):response_status='passed'
    elif duration-1>=deadline:response_status='failed'
    else:response_status='insufficient_duration'
    start=math.ceil(deadline/dt-1e-8);enough=len(actual)-start>=round(2/dt)
    steady=None
    if enough:
        bias=_tracking_error(actual[start:,:2]-target[:2],False,dt)
        geom=_tracking_error(geometric[start:]-target[:2],False,dt)
        yaw=float(abs((actual[start:,2]-target[2]).mean()))
        speed=float(np.linalg.norm(actual[start:,:2],axis=-1).mean())
        passed=bias<limit and geom<limit and yaw<yaw_limit and (not stationary or speed<.08)
        steady=dict(start_seconds=start*dt,duration_seconds=(len(actual)-start)*dt,velocity_error=bias,
            geometric_velocity_error=geom,yaw_error=yaw,mean_speed=speed,linear_limit=limit,yaw_limit=yaw_limit,passed=bool(passed))
    failed=not hard or response_status=='failed' or (steady is not None and not steady['passed'])
    status='failed' if failed else ('passed' if response_status in ['passed','not_applicable'] and enough else 'insufficient_duration')
    result.update(status=status,passed=status=='passed',planner_settle_seconds=plan_seconds,response_deadline_seconds=deadline,
        target_gait_period_seconds=cycle,response_grace_seconds=grace,
        linear_response_seconds=linear_response,yaw_response_seconds=yaw_response,response_status=response_status,
        response_required=required,
        steady=steady,steady_minimum_seconds=2.)
    return result


def summarize(items):
    states=[s['response_steady']['status'] for s in items]
    return dict(passed=all(s=='passed' for s in states),promising=all(s!='failed' for s in states),
        passed_segments=states.count('passed'),failed_segments=states.count('failed'),inconclusive_segments=states.count('insufficient_duration'))
