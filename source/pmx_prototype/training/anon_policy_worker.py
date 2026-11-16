"""Isolated GPU PhysX inference worker for the CPU secondary-physics viewer."""

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
parser.add_argument('--preload-presets',action='store_true')
args=parser.parse_args()
from isaacsim import SimulationApp
extra=[] if args.physics_threads is None else [f'--/persistent/physics/numThreads={args.physics_threads}']
app=SimulationApp({'headless':True,'fast_shutdown':True,'disable_viewport_updates':True,'limit_cpu_threads':2,'extra_args':extra})
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from anon_locomotion_env import AnonLocomotionEnv

env=None
owner=None
connection=None
result=0
try:
    torch.set_num_threads(1)
    from anon_policy_catalog import PRESETS,LABELS,identify,label_for
    from anon_config_compat import clean_legacy_config
    profiles={};catalog={};initial_actions={};active=identify(args.checkpoint)
    paths={active:args.checkpoint}
    if args.preload_presets:paths.update({k:v for k,v in PRESETS.items() if k!=active})
    for key,path in paths.items():
        try:
            config=json.loads((path.parent/'env_config.json').read_text(encoding='utf-8'))
            if key==active and args.config_override:config.update(json.loads(args.config_override.read_text(encoding='utf-8')))
            config=clean_legacy_config(config)
            config['simulation_device']=args.body_device
            config['runtime_inference_only']=not args.legacy_compute
            if config.get('amp_enabled'):config['amp_reward_diagnostics']=False
            agent=json.loads((path.parent/'agent_config.json').read_text(encoding='utf-8'))
            if owner is not None and (not config.get('full_motion_path') or not owner.cfg.get('full_motion_path')):
                raise ValueError('Hot switching requires full-motion imitation controllers')
            environment=AnonLocomotionEnv
            if config.get('full_motion_path'):
                from anon_imitation_env import AnonImitationEnv
                with np.load(config['full_motion_path']) as motion_data:
                    if 'commands' in motion_data:
                        config['imitation_evaluation_initial_clip']=int(np.linalg.norm(motion_data['commands'],axis=-1).argmin())
                environment=AnonImitationEnv
            candidate=environment(1,42,app,evaluation=True,config=config,shared_physics=owner)
            if owner is None:owner=candidate
            candidate.max_episode_length=round(3600/candidate.dt)
            runner=OnPolicyRunner(candidate,agent,log_dir=None,device=args.body_device)
            runner.load(str(path),map_location=args.body_device)
            infer=runner.get_inference_policy()
            if args.preload_presets:
                with torch.inference_mode():initial_actions[key]=infer(candidate.get_observations()).clone()
            profiles[key]=(candidate,runner,infer,str(path.resolve()))
            catalog[key]=dict(label=label_for(key,config,path),checkpoint=str(path.resolve()),available=True)
            print('POLICY_PRELOADED',key,path,flush=True)
        except Exception as error:
            if key==active:raise
            catalog[key]=dict(label=LABELS[key],checkpoint=str(path.resolve()),available=False,error=str(error))
            print('POLICY_UNAVAILABLE',key,repr(error),flush=True)
    # Verify that loading other networks/motion banks did not mutate a model's
    # reference state or observation normalization. All profiles share only PhysX.
    if args.preload_presets:
        for key,(candidate,_,infer,_) in profiles.items():
            candidate.reset(candidate.ids);candidate.refresh()
            with torch.inference_mode():difference=(infer(candidate.get_observations())-initial_actions[key]).abs().max().item()
            if difference>1e-6:raise RuntimeError(f'Controller isolation check failed for {key}: {difference}')
            catalog[key]['isolation_max_action_delta']=difference
    env,runner,policy,checkpoint=profiles[active]
    config=env.cfg
    env.reset(env.ids)
    if not config.get('full_motion_path'):env.commands.zero_()
    env.refresh()
    args.ipc.mkdir(parents=True,exist_ok=True)
    if args.address:
        from anon_policy_transport import connect_worker
        connection=connect_worker(args.address,args.authkey)
    generation=0;pending='';settled=0;switch_age=0
    import omni.usd
    stage_id=int(omni.usd.get_context().get_stage_id())
    def publish(seq,done=False,reset=False,error='',reason='',complete=''):
        state=dict(seq=seq,poses=env.poses[0].cpu().numpy(),
                   velocity=env.logical_vel[0].cpu().numpy(),command=env.commands[0].cpu().numpy(),done=done,reset=reset,link_names=env.links,
                   policy_id=active,generation=generation,switch_pending=pending,switch_error=error,reset_reason=reason,
                   worker_pid=os.getpid(),physics_stage_id=stage_id,checkpoint=checkpoint,switch_complete=complete)
        if seq==-1 or reason=='policy_switch':state['config_json']=json.dumps(config)
        if seq==-1:state['catalog_json']=json.dumps(catalog)
        if connection is not None:
            connection.send(state)
            return
        temporary=args.ipc/f'state_{seq}.tmp.npz'
        np.savez(temporary,**state)
        os.replace(temporary,args.ipc/f'state_{seq}.npz')
    publish(-1)
    last=-1; last_request=time.monotonic()
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
            reset=bool(data.get('reset',False))
            error='';complete='';reason='user' if reset else ''
            if data.get('switch_to'):
                target=data['switch_to']
                if target not in profiles:error=f'Model unavailable: {target}'
                elif target!=active:pending=target;settled=0;switch_age=0
                else:complete=active
            if reset:env.reset(env.ids)
            command=torch.tensor([0.,0.,0.] if pending else data['command'],device=env.device,dtype=torch.float32)
            speed=torch.linalg.vector_norm(command[:2]); command[:2]*=torch.minimum(torch.ones_like(speed),config.get('max_speed',.4)/speed.clamp_min(1e-6))
            command[2]=command[2].clamp(-config.get('max_yaw',.6),config.get('max_yaw',.6))
            if hasattr(env,'set_command'):env.set_command(command.cpu().tolist())
            else:env.commands[:]=command
            _,_,done,_=env.step(policy(env.get_observations()))
            env.refresh()
            if pending:
                switch_age+=1
                quiet=env.logical_vel[0,:2].norm()<.08 and env.logical_vel[0,2].abs()<.15 and env.gravity[0,2]<-.8
                settled=settled+1 if quiet else 0
                if bool(done[0]) or switch_age>120:
                    error='Switch cancelled: unable to stop safely. Press Space/R and retry.';pending=''
                elif settled>=6:
                    root_xy=env.root[:,:2].clone()
                    heading=env.reference_heading.clone()
                    physics_time=env.physics_time
                    active=pending;pending='';generation+=1
                    env,runner,policy,checkpoint=profiles[active];config=env.cfg
                    env.physics_time=physics_time
                    env.reset(env.ids,heading=heading,root_xy=root_xy)
                    env.refresh();reset=True;reason='policy_switch'
            publish(seq,bool(done[0]),reset,error,reason,complete)
            last=seq
except Exception:
    import traceback
    traceback.print_exc(); result=1
    args.ipc.mkdir(parents=True,exist_ok=True)
    (args.ipc/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
finally:
    if connection is not None:connection.close()
    if owner: owner.close()
    app.close(exit_code=result)
