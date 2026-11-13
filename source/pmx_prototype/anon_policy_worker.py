"""Isolated GPU PhysX inference worker for the CPU secondary-physics viewer."""
import argparse
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--ipc',type=Path,required=True)
parser.add_argument('--config-override',type=Path)
parser.add_argument('--body-device',choices=['cpu','cuda:0'],default='cuda:0')
parser.add_argument('--address')
parser.add_argument('--authkey')
parser.add_argument('--legacy-compute',action='store_true')
parser.add_argument('--physics-threads',type=int)
args=parser.parse_args()
from isaacsim import SimulationApp
extra=[] if args.physics_threads is None else [f'--/persistent/physics/numThreads={args.physics_threads}']
app=SimulationApp({'headless':True,'fast_shutdown':True,'extra_args':extra})
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from anon_locomotion_env import AnonLocomotionEnv

env=None
connection=None
result=0
try:
    torch.set_num_threads(4)
    config=json.loads((args.checkpoint.parent/'env_config.json').read_text(encoding='utf-8'))
    if args.config_override:config.update(json.loads(args.config_override.read_text(encoding='utf-8')))
    config['simulation_device']=args.body_device
    config['runtime_inference_only']=not args.legacy_compute
    if config.get('amp_enabled'):config['amp_reward_diagnostics']=False
    agent=json.loads((args.checkpoint.parent/'agent_config.json').read_text(encoding='utf-8'))
    environment=AnonLocomotionEnv
    if config.get('full_motion_path'):
        from anon_imitation_env import AnonImitationEnv
        with np.load(config['full_motion_path']) as motion_data:
            if 'commands' in motion_data:
                config['imitation_evaluation_initial_clip']=int(np.linalg.norm(motion_data['commands'],axis=-1).argmin())
        environment=AnonImitationEnv
    env=environment(1,42,app,evaluation=True,config=config)
    env.max_episode_length=round(3600/env.dt)
    runner=OnPolicyRunner(env,agent,log_dir=None,device=args.body_device)
    runner.load(str(args.checkpoint),map_location=args.body_device)
    policy=runner.get_inference_policy()
    env.reset(env.ids)
    if not config.get('full_motion_path'):env.commands.zero_()
    env.refresh()
    args.ipc.mkdir(parents=True,exist_ok=True)
    if args.address:
        from anon_policy_transport import connect_worker
        connection=connect_worker(args.address,args.authkey)
    def publish(seq,done=False,reset=False):
        state=dict(seq=seq,poses=env.poses[0].cpu().numpy(),
                   velocity=env.logical_vel[0].cpu().numpy(),command=env.commands[0].cpu().numpy(),done=done,reset=reset,link_names=env.links)
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
            if reset:env.reset(env.ids)
            command=torch.tensor(data['command'],device=env.device,dtype=torch.float32)
            speed=torch.linalg.vector_norm(command[:2]); command[:2]*=torch.minimum(torch.ones_like(speed),config.get('max_speed',.4)/speed.clamp_min(1e-6))
            command[2]=command[2].clamp(-config.get('max_yaw',.6),config.get('max_yaw',.6))
            if hasattr(env,'set_command'):env.set_command(command.cpu().tolist())
            else:env.commands[:]=command
            _,_,done,_=env.step(policy(env.get_observations()))
            env.refresh(); publish(seq,bool(done[0]),reset)
            last=seq
except Exception:
    import traceback
    traceback.print_exc(); result=1
    args.ipc.mkdir(parents=True,exist_ok=True)
    (args.ipc/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
finally:
    if connection is not None:connection.close()
    if env: env.close()
    app.close(exit_code=result)
