"""Isolated Isaac Lab inference worker for the secondary-physics viewer."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source, config_path

import argparse
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--ipc',type=Path,required=True)
parser.add_argument('--config-override',type=config_path)
parser.add_argument('--body-device',choices=['cpu','cuda:0'],default='cuda:0')
parser.add_argument('--address')
parser.add_argument('--authkey')
parser.add_argument('--legacy-compute',action='store_true')
parser.add_argument('--physics-threads',type=int)
parser.add_argument('--manual-reset-only',action='store_true')
parser.add_argument('--joint-telemetry',action='store_true',help='Read-only articulation monitoring for the inspector')
args=parser.parse_args()
from anon_startup_progress import StartupJournal
startup=StartupJournal(args.ipc.parent,'policy')
startup.mark('engine')
extra=[] if args.physics_threads is None else [f'--/persistent/physics/numThreads={args.physics_threads}']
from isaaclab.app import AppLauncher
app=AppLauncher({'headless':True,'device':args.body_device,'disable_viewport_updates':True,
                 'limit_cpu_threads':2,'kit_args':' '.join(extra)}).app
startup.mark('imports')
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

env=None
connection=None
result=0
try:
    torch.set_num_threads(1)
    from anon_policy_catalog import identify,label_for
    from anon_config_compat import clean_legacy_config
    from command_plan import yaw_speed_limit
    from locomotion_acceptance import inspector_planar_speed_limit
    from anon_environment import create_environment

    path=args.checkpoint
    active=identify(path)
    checkpoint=str(path.resolve())
    startup.mark('config',path.name)
    config=json.loads((path.parent/'env_config.json').read_text(encoding='utf-8'))
    if args.config_override:
        config.update(json.loads(args.config_override.read_text(encoding='utf-8')))
    config=clean_legacy_config(config)
    config['simulation_device']=args.body_device
    config['runtime_inference_only']=not args.legacy_compute
    config['runtime_manual_reset_only']=args.manual_reset_only
    if config.get('amp_enabled'):
        config['amp_reward_diagnostics']=False
    agent=json.loads((path.parent/'agent_config.json').read_text(encoding='utf-8'))
    if config.get('full_motion_path'):
        with np.load(config['full_motion_path']) as motion_data:
            if 'commands' in motion_data:
                config['imitation_evaluation_initial_clip']=int(np.linalg.norm(motion_data['commands'],axis=-1).argmin())
    startup.mark('environment',Path(config['asset']).name)
    env=create_environment(1,42,app,evaluation=True,config=config)
    env.max_episode_length=round(3600/env.dt)
    startup.mark('network',path.name)
    runner=OnPolicyRunner(env,agent,log_dir=None,device=args.body_device)
    if config.get('guarded_recovery_actor',False):
        runner.load(str(path),load_cfg={'actor':True},map_location=args.body_device)
    else:
        runner.load(str(path),map_location=args.body_device)
    policy=runner.get_inference_policy()
    catalog={active:dict(label=label_for(active,config,path),checkpoint=checkpoint,available=True)}
    startup.mark('reset')
    config=env.cfg
    use_inspector_limits = args.manual_reset_only and config.get('unified_walk_run', False)
    env.reset(env.ids)
    if not config.get('full_motion_path'):env.commands.zero_()
    env.refresh()
    args.ipc.mkdir(parents=True,exist_ok=True)
    if args.address:
        from anon_policy_transport import connect_worker
        connection=connect_worker(args.address,args.authkey)
    joint_reader=None
    if args.joint_telemetry:
        from anon_joint_telemetry import JointTelemetry
        joint_reader=JointTelemetry(env)
    import omni.usd
    stage_id=int(omni.usd.get_context().get_stage_id())
    def publish(seq,done=False,reset=False,reason='',turn_state=None,fallen=False,
                requested=None,planar_limit=None):
        state=dict(seq=seq,poses=env.poses[0].cpu().numpy(),
                   velocity=env.logical_vel[0].cpu().numpy(),command=env.commands[0].cpu().numpy(),done=done,reset=reset,link_names=env.links,
                   policy_id=active,reset_reason=reason,worker_pid=os.getpid(),
                   physics_stage_id=stage_id,checkpoint=checkpoint)
        state['fallen']=fallen
        state['restart']=reset or (done and not fallen)
        if joint_reader is not None:
            try:
                state.update(joint_reader.read(env,reset=seq==-1 or state['restart']))
            except Exception as telemetry_error:
                state.update(joint_state_valid=False,joint_state_error=str(telemetry_error))
        if config.get('command_yaw_limit_by_speed'):
            if turn_state is None:
                initial=env.continuous_ref_commands[0]
                limit=yaw_speed_limit(initial[:2].norm(),config)
                turn_state=torch.cat((env.commands[0],initial[2:3],limit.reshape(1)))
            target_x,target_y,request_yaw,applied_yaw,limit=turn_state.cpu().numpy()
            state['requested_command']=np.array([target_x,target_y,request_yaw])
            state['command']=np.array([target_x,target_y,applied_yaw])
            state['yaw_limit']=float(limit)
        if use_inspector_limits:
            if requested is None:
                requested = state['command']
                planar_limit = inspector_planar_speed_limit(requested, config)
            state['requested_command'] = np.asarray(requested, dtype=np.float32)
            state['planar_speed_limit'] = planar_limit
        if seq==-1:state['config_json']=json.dumps(config)
        if seq==-1:state['catalog_json']=json.dumps(catalog)
        if connection is not None:
            connection.send(state)
            return
        temporary=args.ipc/f'state_{seq}.tmp.npz'
        np.savez(temporary,**state)
        os.replace(temporary,args.ipc/f'state_{seq}.npz')
    publish(-1)
    startup.mark('ready')
    last=-1; last_request=time.monotonic(); previously_fallen=False
    with torch.inference_mode():
        while time.monotonic()-last_request<120:
            if connection is not None:
                if not connection.poll(.1):continue
                data=connection.recv()
                if data.get('stop'):break
            else:
                if (args.ipc/'stop.request').exists(): break
                request=args.ipc/f'request_{last+1}.json'
                if not request.exists(): time.sleep(.001); continue
                try:
                    data=json.loads(request.read_text())
                    request.unlink()
                except (PermissionError,FileNotFoundError):
                    time.sleep(.002); continue
            seq=int(data['seq'])
            if seq<=last: time.sleep(.001); continue
            last_request=time.monotonic()
            acceptance_inputs = use_inspector_limits and bool(data.get('acceptance_inputs'))
            manual_reset_only = args.manual_reset_only and (not acceptance_inputs or bool(data.get('manual_reset_only')))
            if args.manual_reset_only:
                config['runtime_manual_reset_only'] = manual_reset_only
            reset=bool(data.get('reset',False))
            reason=('acceptance_group' if acceptance_inputs else 'user') if reset else ''
            if reset:env.reset(env.ids)
            if data.get('push_character'):
                from anon_character_push import apply_requested_push
                print('CHARACTER_PUSH',json.dumps(apply_requested_push(env,data['push_character'])),flush=True)
            requested = data['command']
            planar_limit = (inspector_planar_speed_limit(requested, config) if use_inspector_limits and not acceptance_inputs
                            else config.get('max_speed', .4))
            command=torch.tensor(requested,device=env.device,dtype=torch.float32)
            speed=torch.linalg.vector_norm(command[:2]); command[:2]*=torch.minimum(torch.ones_like(speed),planar_limit/speed.clamp_min(1e-6))
            command[2]=command[2].clamp(-config.get('max_yaw',.6),config.get('max_yaw',.6))
            if hasattr(env,'set_command'):env.set_command(command.cpu().tolist())
            else:env.commands[:]=command
            observation=env.get_observations()
            turn_state=None
            if config.get('command_yaw_limit_by_speed'):
                planned=env.next_planned_command()[0]
                limit=yaw_speed_limit(planned[:2].norm(),config)
                turn_state=torch.cat((env.commands[0],planned[2:3],limit.reshape(1)))
            _,_,done,_=env.step(policy(observation))
            env.refresh()
            ended=bool(done[0])
            fallen=manual_reset_only and ended
            # A character lying on the floor counts as one fall until it recovers or is reset.
            fall_event=fallen and (reset or not previously_fallen) if manual_reset_only else ended
            previously_fallen=fallen
            publish(seq,fall_event,reset,reason,turn_state,fallen=fallen,
                    requested=requested,planar_limit=planar_limit)
            last=seq
except Exception:
    import traceback
    traceback.print_exc(); result=1
    startup.mark('failed','See policy_worker.log and ipc error.txt')
    args.ipc.mkdir(parents=True,exist_ok=True)
    (args.ipc/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
finally:
    if connection is not None:connection.close()
    if env is not None:env.close()
    app.close(exit_code=result)
