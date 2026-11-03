"""Read-only checkpoint rollout from zero with 120 Hz turn telemetry."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat
parser=argparse.ArgumentParser()
parser.add_argument("--bundle",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
parser.add_argument("--num_envs",type=int,default=16)
parser.add_argument("--start",type=float,default=10.)
parser.add_argument("--end",type=float,default=16.)
parser.add_argument("--no_obs_noise",action="store_true",help="Diagnostic ablation only; leave physical/reset settings unchanged.")
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
if not 0<=args.start<args.end or len(args.seeds)>args.num_envs:parser.error("Invalid time range or insufficient environments")
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import torch
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.mdp.validation_random import ValidationRandom,PROTOCOL
from source.my_task.motion_reference import get_or_create_motion_buffer,motion_steps
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.deployment import load_bundle,atomic_write_json
from source.train_workflow.utils.jump_diagnostics import JumpDiagnostics


def main():
    import tempfile
    bundle=load_bundle(args.bundle);cfg=bundle["env_cfg"];agent=bundle["agent_cfg"]
    cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;agent.device=args.device;cfg.seed=args.seeds[0]
    if args.no_obs_noise:cfg.observations.policy.enable_corruption=False
    cfg.events.reset_robot_joints.params.update(random_start=False,motion_start_seconds=0.,motion_start_choices=None,
                                              segment_seconds=cfg.actions.joint_pos.motion_window_seconds)
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    wrapped=RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.clip_actions)
    env=wrapped.unwrapped;env._auto_jump_strength=1.
    diagnostics=[]
    try:
        runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
        args.output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            checkpoint=Path(temporary)/"embedded.pt";checkpoint.write_bytes(bundle["checkpoint"])
            runner.load(str(checkpoint))
        policy=runner.get_inference_policy(device=env.device)
        action=cfg.actions.joint_pos
        buf=get_or_create_motion_buffer(env,action.motion_h5_path,action.motion_window_seconds)
        active=[True]*len(args.seeds)
        with ValidationRandom(env,args.seeds):
            env.reset();obs=wrapped.get_observations()
            for i,seed in enumerate(args.seeds):
                diag=JumpDiagnostics(env,args.output/f"seed_{seed}",env_index=i,start=args.start,end=args.end,extended=True)
                diag.begin(0,buf,dict(bundle=str(args.bundle.resolve()),seed=seed,validation_protocol=PROTOCOL,
                    simulation_device=str(env.device),num_envs=env.num_envs,observation_corruption=cfg.observations.policy.enable_corruption,
                    action_smoothing=1.,rollout_start_s=0.,requested_record_start_s=args.start,requested_rollout_end_s=args.end))
                diagnostics.append(diag)
            with torch.inference_mode():
                for step in range(round(args.end/env.step_dt)):
                    steps=motion_steps(env)
                    for diag,is_active in zip(diagnostics,active):
                        if is_active:diag.begin_control(steps)
                    try:obs,_,done,_=wrapped.step(policy(obs))
                    finally:
                        for diag in diagnostics:diag.end_control()
                    for i,diag in enumerate(diagnostics):
                        if active[i] and bool(done[i]):
                            active[i]=False;diag.flush("terminated_before_diagnostic_end")
                    if step%150==0:print(f"[TURN] {(step+1)*env.step_dt:.2f}/{args.end:.2f}s",flush=True)
            for diag,is_active in zip(diagnostics,active):
                if is_active:diag.flush("diagnostic_window_complete; remaining motion not evaluated")
        atomic_write_json(args.output/"run.json",dict(bundle=str(args.bundle.resolve()),seeds=args.seeds,
            started_at_zero=True,recording_hz=1/env.physics_dt,control_hz=1/env.step_dt,
            reached_window_end=active,validation_protocol=PROTOCOL,training_updates=0))
    finally:
        for diag in reversed(diagnostics):diag.close()
        wrapped.close()


try:main()
finally:app.close()
