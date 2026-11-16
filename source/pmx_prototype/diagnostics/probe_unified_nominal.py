"""Measure unassisted nominal-PD survival from one running reference state.

This diagnostic is not acceptance and never drives a published policy.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import argparse,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--command',type=float,nargs=3,default=[-1.2,0.,0.]);p.add_argument('--seconds',type=float,default=12.)
p.add_argument('--zero-actions',action='store_true')
p.add_argument('--check-metric-parity',action='store_true')
p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');args=p.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True,'disable_viewport_updates':True,'limit_cpu_threads':2})
import torch
from anon_imitation_env import AnonImitationEnv
from rsl_rl.runners import OnPolicyRunner
try:
    torch.set_num_threads(1)
    cfg=json.loads((args.checkpoint.parent/'env_config.json').read_text())
    cfg.update(simulation_device=args.device,amp_reward_diagnostics=False,runtime_inference_only=True)
    env=AnonImitationEnv(1,42,app,evaluation=True,config=cfg)
    env.max_episode_length=round(600/env.dt)
    runner=OnPolicyRunner(env,json.loads((args.checkpoint.parent/'agent_config.json').read_text()),device=args.device)
    runner.load(str(args.checkpoint),map_location=args.device);infer=runner.get_inference_policy()
    samples=[];max_observation_delta=0.;max_action_delta=0.;telemetry_changed_frames=0
    with torch.inference_mode():
        for phase in [0.,1.,2.,3.]:
            env.reset(env.ids,command=args.command,phase=phase);env.refresh();fell=False
            for step in range(round(args.seconds/env.dt)):
                env.set_command(args.command)
                if args.check_metric_parity:
                    env.cfg['unified_command_frame_metrics']=False;before=env.get_observations();old=infer(before);old_velocity=env.logical_vel.clone()
                    env.cfg['unified_command_frame_metrics']=True;after=env.get_observations();a=infer(after)
                    max_observation_delta=max(max_observation_delta,float((before['policy']-after['policy']).abs().max()))
                    max_action_delta=max(max_action_delta,float((old-a).abs().max()))
                    telemetry_changed_frames+=int((env.logical_vel-old_velocity).abs().max()>1e-6)
                    if max(max_observation_delta,max_action_delta)>1e-6:raise RuntimeError('Metric-frame change altered actor observations/actions')
                else:a=torch.zeros_like(env.actions) if args.zero_actions else infer(env.get_observations())
                _,_,done,_=env.step(a)
                if bool(done.any()):fell=True;break
            samples.append(dict(phase=phase,survival_seconds=(step+1)*env.dt,fell=fell))
            print('NOMINAL_PROBE',samples[-1],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),device=args.device,zero_actions=args.zero_actions,command=args.command,samples=samples,
                                         metric_parity_checked=args.check_metric_parity,max_observation_delta=max_observation_delta,max_action_delta=max_action_delta,telemetry_changed_frames=telemetry_changed_frames,
                                         diagnostic_only=True,full_acceptance=False),indent=2))
    env.close()
finally:app.close()
