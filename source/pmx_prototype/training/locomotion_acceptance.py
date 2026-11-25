"""Versioned response/steady-state scoring, independent of simulator startup."""
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from command_plan import command_increment,limit_yaw_command
from idle_head_alignment import MEAN_BIAS_LIMIT_DEG,P95_LIMIT_DEG,RECENTER_HOLD_SECONDS,RECENTER_LIMIT_DEG

PROTOCOL='response_steady_v2'
CURRENT_PROTOCOL='response_steady_v3'
RESPONSE_STEADY_PROTOCOLS=(PROTOCOL,CURRENT_PROTOCOL)
DEFAULT_BACKWARD_DIAGONAL_RUN_SPEED = .85
FULL_RUN_SPEED = 1.2
YAW_RESPONSE_DEADLINE_SECONDS=2.1
STOP_YAW_RESPONSE_DEADLINE_SECONDS=3.0
WALK_BACKWARD_RESPONSE_TOLERANCE_FRAMES=4
# stand_end begins with the stop after the final right pivot.
STOP_SEGMENTS=('stand_end','stop','stop_after_walk','stop_after_pivot_left')
STATIONARY_SEGMENTS=('stand',*STOP_SEGMENTS)

LIGHT_ACCEPTANCE = dict(
    version='light_v2', tracking_tolerance_multiplier=1.20,
    extra_response_seconds=.25, response_hold_seconds=.35,
    direction_lag_limit_seconds=1.25, requires_zero_falls=True,
)


def full_backward_diagonal_run_speed(protocol, configured=None):
    if configured is None:
        return DEFAULT_BACKWARD_DIAGONAL_RUN_SPEED if protocol == CURRENT_PROTOCOL else FULL_RUN_SPEED
    speed = float(configured)
    if not math.isfinite(speed) or not 0 < speed <= FULL_RUN_SPEED:
        raise ValueError('unified_eval_backward_diagonal_run_speed must be in (0, 1.2] m/s')
    return speed


def inspector_planar_speed_limit(command, config):
    """Apply the current unified acceptance targets as manual direction caps."""
    maximum = min(float(config.get('max_speed', .4)), FULL_RUN_SPEED)
    forward, lateral = command[:2]
    axis_tolerance = 1e-4
    if forward < -axis_tolerance or (forward > axis_tolerance and abs(lateral) > axis_tolerance):
        restricted = full_backward_diagonal_run_speed(
            CURRENT_PROTOCOL, config.get('unified_eval_backward_diagonal_run_speed'),
        )
        return min(maximum, restricted)
    return maximum


def specification(idle_head=False,protocol=PROTOCOL):
    spec=dict(protocol=protocol,min_steady_seconds=2.,response_min_seconds=2.,response_cap_seconds=5.,
        response_grace_after_planner='max(1.5 seconds, one target gait cycle +0.5 second confirmation)',
        yaw_response_deadline_seconds=YAW_RESPONSE_DEADLINE_SECONDS,
        fallback_gait_period_seconds=1.5,planner_linear_tolerance=.02,planner_yaw_tolerance=.02,
        linear_filter_seconds=1.,yaw_filter_seconds=2.,response_hold_seconds=.5,
        variable_direction_lag_limit_seconds=1.,
        steady_linear_limit='max(0.06,0.22*requested_speed)',steady_idle_speed_limit=.08,
        steady_yaw_limit_stationary=.1,steady_yaw_limit_moving=.2,
        running_flight_required=protocol!=CURRENT_PROTOCOL,
        note='Centered filters estimate response time offline. Full acceptance requires sufficient steady-state data for every scored segment.')
    if protocol==CURRENT_PROTOCOL:
        spec.update(stop_yaw_response_deadline_seconds=STOP_YAW_RESPONSE_DEADLINE_SECONDS,
            stop_yaw_response_segments=STOP_SEGMENTS,
            walk_backward_linear_response_tolerance_frames=WALK_BACKWARD_RESPONSE_TOLERANCE_FRAMES)
    if idle_head:
        head_spec=dict(target='Horizontal forward in the controller heading frame',
            window='Uses the stationary segment steady window, after the body response deadline',
            required_segments=())
        if protocol!=CURRENT_PROTOCOL:
            head_spec.update(mean_bias_limit_deg=MEAN_BIAS_LIMIT_DEG,p95_abs_limit_deg=P95_LIMIT_DEG,
                recenter_report_limit_deg=RECENTER_LIMIT_DEG,recenter_report_hold_seconds=RECENTER_HOLD_SECONDS,
                required_segments=STATIONARY_SEGMENTS)
        spec['idle_head']=head_spec
    return spec


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
    settled=limit_yaw_command(requested,config)
    for step in range(round(10/dt)+1):
        difference=current-settled
        if float(difference[0,:2].norm())<=.02 and abs(float(difference[0,2]))<=.02:return step*dt
        current+=command_increment(requested,current,dt,config)
    return None


def _response(values,target,limit,dt,window_seconds,stationary=False,*,hold_seconds=.5):
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
    hold=max(1,math.ceil(hold_seconds/dt)+1)
    if len(flags)<hold:return None
    indices=np.flatnonzero(np.convolve(flags.astype(int),np.ones(hold,dtype=int),mode='valid')==hold)
    if not len(indices):return None
    # End of the confirmed hold, timestamped at the centered averaging window.
    return float((indices[0]+hold-1+(width-1)/2)*dt)


def _response_status(plan_seconds, deadline, duration, responses):
    if plan_seconds == 0.:
        return 'not_applicable'
    response_late = any(seconds is not None and seconds > budget for seconds, budget in responses)
    confirmation_missing = any(seconds is None and duration - 1 >= budget for seconds, budget in responses)
    if plan_seconds is None or plan_seconds > deadline or response_late or confirmation_missing:
        return 'failed'
    if all(seconds is not None for seconds, _ in responses):
        return 'passed'
    return 'insufficient_duration'


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
    if (config.get('unified_acceptance_protocol', PROTOCOL) != CURRENT_PROTOCOL and
            item['label'] in ('run_forward', 'run_backward', 'run_left', 'run_right')):
        flight=item.get('force_flight_fraction',item['flight_fraction']) if config.get('unified_eval_force_flight') else item['flight_fraction']
        passed&=flight>.01
    return bool(passed)


def _idle_head_metrics(errors,frames,start,body_response,dt,*,required):
    errors=np.asarray(errors)
    if errors.ndim not in (2,3) or errors.shape[0]!=frames or errors.shape[-1]!=2 or not errors.size:
        if required:raise ValueError('Idle head scoring requires yaw and pitch for each step')
        return dict(status='unavailable',reason='missing_or_incomplete_head_pose')
    errors=np.rad2deg(errors.reshape(len(errors),-1,2))
    if not np.isfinite(errors).all():
        return dict(status='failed' if required else 'unavailable',reason='non_finite_head_pose')
    result=dict(status='insufficient_duration')
    if required:
        hold=math.ceil(RECENTER_HOLD_SECONDS/dt)
        onset=math.ceil(body_response/dt)
        within=(np.abs(errors[onset:])<=RECENTER_LIMIT_DEG).all(axis=(1,2))
        recentered=None
        if len(within)>=hold:
            starts=np.flatnonzero(np.convolve(within.astype(int),np.ones(hold,dtype=int),mode='valid')==hold)
            if len(starts):recentered=float((onset+starts[0]+hold)*dt)
        result['recenter_confirmed_seconds']=recentered
    if len(errors)-start>=round(2/dt):
        window=errors[start:]
        bias=window.mean(axis=0)
        # Use the largest absolute bias per axis so opposite environments cannot cancel.
        yaw,pitch=bias[np.argmax(np.abs(bias),axis=0),np.arange(2)]
        yaw_p95,pitch_p95=np.quantile(np.abs(window),.95,axis=0).max(axis=0)
        result.update(status='recorded',yaw_bias_deg=float(yaw),pitch_bias_deg=float(pitch),
            yaw_abs_p95_deg=float(yaw_p95),pitch_abs_p95_deg=float(pitch_p95),start_seconds=start*dt,
            duration_seconds=(len(errors)-start)*dt)
        if required:
            passed=max(abs(yaw),abs(pitch))<=MEAN_BIAS_LIMIT_DEG and max(yaw_p95,pitch_p95)<=P95_LIMIT_DEG
            result['status']='passed' if passed else 'failed'
    return result


def step_body_status(quality, response_status, steady):
    if not quality or response_status == 'failed' or (steady is not None and not steady['passed']):
        return 'failed'
    if response_status in ('passed', 'not_applicable') and steady is not None:
        return 'passed'
    return 'insufficient_duration'


def score_segment(item,actual,planned,requested,geometric,initial_plan,config,dt,head_errors=None):
    actual=np.asarray(actual);planned=np.asarray(planned);requested=np.asarray(requested);geometric=np.asarray(geometric)
    if not(len(actual)==len(planned)==len(requested)==len(geometric)) or not np.isfinite(actual).all():
        raise ValueError('Invalid tracking trace')
    duration=len(actual)*dt
    variable=item['label'].startswith('ramp_') or item['label']=='direction_sweep'
    stationary=item['label'] in STATIONARY_SEGMENTS
    protocol=config.get('unified_acceptance_protocol',PROTOCOL)
    hard=quality_passed(item,config)
    result=dict(protocol=protocol,quality_passed=hard,classification='continuous_tracking' if variable else 'step_response')
    if variable:
        start=round(2/dt)
        error=_tracking_error(actual[start:,:2]-planned[start:,:2],True,dt)
        geometry=_tracking_error(geometric[start:]-planned[start:,:2],True,dt)
        yaw_target=planned if config.get('command_yaw_limit_by_speed') else requested
        yaw=float(abs((actual[start:,2]-yaw_target[start:,2]).mean()))
        lag=_direction_lag(requested[:,:2],actual[:,:2],dt) if item['label']=='direction_sweep' else None
        lag_pass=item['label']!='direction_sweep' or (lag is not None and abs(lag)<=1.)
        passed=hard and error<.2 and geometry<.2 and yaw<.2 and lag_pass
        status='passed' if passed else 'failed'
        result.update(status=status,body_status=status,passed=bool(passed),
            tracking_against='planned_command',velocity_error=error,geometric_velocity_error=geometry,
            yaw_error=yaw,raw_command_error=_tracking_error(actual[start:,:2]-requested[start:,:2],True,dt),
            apparent_direction_lag_seconds=lag,direction_lag_limit_seconds=1.,
            lag_note='Direction lag is measured only for the excited circular command; a monotone ramp does not identify lag independently of speed bias.')
        return result
    requested_target=requested[-1]
    target=requested_target
    if config.get('command_yaw_limit_by_speed'):
        target=limit_yaw_command(torch.tensor(requested_target),config).numpy()
    limit=max(.06,float(np.linalg.norm(target[:2]))*.22);yaw_limit=.1 if stationary else .2
    plan_seconds=planning_time(initial_plan,requested_target,config,dt)
    cycle=gait_period(target,config);grace=max(1.5,cycle+.5)
    deadline=min(5.,max(2.,(plan_seconds if plan_seconds is not None else 10.)+grace))
    linear_response=_response(actual[:,:2],target[:2],limit,dt,1.,stationary)
    yaw_response=_response(actual[:,2],target[2],yaw_limit,dt,2.)
    yaw_deadline=YAW_RESPONSE_DEADLINE_SECONDS
    if protocol==CURRENT_PROTOCOL and item['label'] in STOP_SEGMENTS:
        yaw_deadline=STOP_YAW_RESPONSE_DEADLINE_SECONDS
    linear_deadline=deadline
    if protocol==CURRENT_PROTOCOL and item['label']=='walk_backward':
        tolerance=WALK_BACKWARD_RESPONSE_TOLERANCE_FRAMES*dt
        linear_deadline+=tolerance
        result.update(linear_response_deadline_seconds=linear_deadline,linear_response_tolerance_seconds=tolerance)
    responses=((linear_response,linear_deadline),(yaw_response,yaw_deadline))
    required=plan_seconds!=0.
    response_status=_response_status(plan_seconds,deadline,duration,responses)
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
    body_status=step_body_status(hard,response_status,steady)
    status=body_status
    if stationary and config.get('unified_idle_head_acceptance'):
        head_required=protocol!=CURRENT_PROTOCOL
        head=_idle_head_metrics(head_errors,len(actual),start,max(linear_response or 0.,yaw_response or 0.),dt,
            required=head_required)
        head['required']=head_required
        result['idle_head']=head
        if head_required and (head['status']=='failed' or status=='passed'):
            status=head['status']
    result.update(status=status,body_status=body_status,passed=status=='passed',planner_settle_seconds=plan_seconds,response_deadline_seconds=deadline,
        yaw_response_deadline_seconds=yaw_deadline,
        target_gait_period_seconds=cycle,response_grace_seconds=grace,
        linear_response_seconds=linear_response,yaw_response_seconds=yaw_response,response_status=response_status,
        response_required=required,
        steady=steady,steady_minimum_seconds=2.)
    return result


def score_light_segment(item, actual, dt):
    """Loosen tracking and response tolerances on the scored motion."""
    standard = item['response_steady']
    scale = LIGHT_ACCEPTANCE['tracking_tolerance_multiplier']
    quality = standard['quality_passed']
    if standard['classification'] == 'continuous_tracking':
        limit = .2 * scale
        lag_limit = LIGHT_ACCEPTANCE['direction_lag_limit_seconds']
        lag = standard['apparent_direction_lag_seconds']
        lag_passed = item['label'] != 'direction_sweep' or (lag is not None and abs(lag) <= lag_limit)
        passed = quality and lag_passed and all(standard[key] < limit for key in (
            'velocity_error', 'geometric_velocity_error', 'yaw_error'))
        return dict(status='passed' if passed else 'failed', passed=passed,
                    tracking_error_limit=limit, direction_lag_limit_seconds=lag_limit)

    target = np.asarray(item['command'])
    stationary = item['label'] in STATIONARY_SEGMENTS
    linear_limit = max(.06, float(np.linalg.norm(target[:2])) * .22) * scale
    yaw_limit = (.1 if stationary else .2) * scale
    extra = LIGHT_ACCEPTANCE['extra_response_seconds']
    hold = LIGHT_ACCEPTANCE['response_hold_seconds']
    deadline = standard['response_deadline_seconds']
    linear_deadline = standard.get('linear_response_deadline_seconds', deadline) + extra
    yaw_deadline = standard['yaw_response_deadline_seconds'] + extra
    linear = _response(actual[:, :2], target[:2], linear_limit, dt, 1., stationary, hold_seconds=hold)
    yaw = _response(actual[:, 2], target[2], yaw_limit, dt, 2., hold_seconds=hold)
    response = _response_status(standard['planner_settle_seconds'], deadline + extra, len(actual) * dt,
                                ((linear, linear_deadline), (yaw, yaw_deadline)))
    steady = standard['steady']
    if steady is not None:
        passed = (steady['velocity_error'] < linear_limit and steady['geometric_velocity_error'] < linear_limit
                  and steady['yaw_error'] < yaw_limit and (not stationary or steady['mean_speed'] < .08))
        steady = dict(steady, linear_limit=linear_limit, yaw_limit=yaw_limit, passed=passed)
    status = step_body_status(quality, response, steady)
    return dict(status=status, passed=status == 'passed', response_status=response, steady=steady,
                linear_response_seconds=linear, yaw_response_seconds=yaw,
                linear_response_deadline_seconds=linear_deadline, yaw_response_deadline_seconds=yaw_deadline)


def add_light_acceptance(report, velocities, dt):
    """Add the light score to the current protocol using the same measured velocities."""
    if report['acceptance_protocol'] != CURRENT_PROTOCOL:
        return
    actual = np.asarray(velocities).mean(axis=1)
    items = report['segments']
    steps = [round(item['seconds'] / dt) for item in items]
    if sum(steps) != len(actual):
        raise ValueError('Light acceptance segments do not match the measured rollout')
    start = 0
    for item, count in zip(items, steps, strict=True):
        end = start + count
        item['light_acceptance'] = score_light_segment(item, actual[start:end], dt)
        start = end
    scored = [item for item in items if item.get('counts_toward_gate', True)]
    summary = summarize(scored, field='light_acceptance')
    report.update(light_acceptance_specification=LIGHT_ACCEPTANCE.copy(), light_acceptance_summary=summary,
                  passed_light_gate=bool(report['complete_scope'] and report['total_falls'] == 0 and summary['passed']))


def summarize(items, *, field='response_steady'):
    states=[s[field]['status'] for s in items]
    return dict(passed=all(s=='passed' for s in states),
        passed_segments=states.count('passed'),failed_segments=states.count('failed'),inconclusive_segments=states.count('insufficient_duration'))
