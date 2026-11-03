"""Matched low-LR continuation arm; source deployment is never overwritten."""
from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat
parser=argparse.ArgumentParser()
parser.add_argument("--bundle",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--iterations",type=int,default=100)
parser.add_argument("--num_envs",type=int,default=8192)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--smoothing",action="store_true")
parser.add_argument("--rate_weight",type=float,default=1.)
parser.add_argument("--acceleration_weight",type=float,default=.5)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
if args.iterations<1 or args.num_envs<3:parser.error("Positive iteration count and >=3 environments required")
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import tempfile
import gymnasium as gym
import torch
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.motion_reference import get_or_create_motion_buffer
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.my_task.mdp.turn_smoothing import configure
from source.train_workflow.utils.auto_jump_training import AutoJumpController,install_checkpoint_metadata
from source.train_workflow.utils.deployment import load_bundle,save_bundle,atomic_write_json
from source.train_workflow.utils.quality_selection import continuous_coverage,QualitySelection
from source.train_workflow.utils.validation_recording import save_validation_scene


def main():
    args.output.mkdir(parents=True,exist_ok=False)
    bundle=load_bundle(args.bundle);cfg=bundle["env_cfg"];agent=bundle["agent_cfg"]
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    agent.seed=args.seed;agent.device=args.device;agent.save_interval=25
    agent.actor.freeze_base=False;agent.actor.freeze_normalization=True
    agent.algorithm.learning_rate=1.e-5;agent.algorithm.entropy_coef=.001
    if args.smoothing:configure(cfg,args.rate_weight,args.acceleration_weight)
    elif getattr(cfg,"turn_smoothing_enabled",False):raise ValueError("Control source already has turn rewards")
    cfg.auto_jump_report.pop("last_validation",None)
    cfg.auto_jump_report["turn_continuation"]=dict(smoothing=args.smoothing,source=str(args.bundle.resolve()))
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    wrapped=RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.clip_actions)
    try:
        env=wrapped.unwrapped
        controller=AutoJumpController(env,cfg.auto_jump_report,args.output)
        env._auto_jump_controller=controller
        runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(args.output),device=args.device)
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            path=Path(temporary)/"source.pt";path.write_bytes(bundle["checkpoint"])
            infos=runner.load(str(path),load_cfg=dict(actor=True,critic=True,optimizer=False,iteration=True))
        state=(infos or {}).get("auto_curriculum")
        if not state:raise ValueError("Source must have Auto-Jump curriculum metadata")
        saved_profile=(state.get("quality_selection") or {}).get("profile")
        if saved_profile is not None:
            controller.quality_profile=copy.deepcopy(saved_profile)
            atomic_write_json(args.output/"quality_profile.json",saved_profile)
        controller.load_state_dict(state);controller.configure(1.)
        policy=runner.alg.get_policy();policy.freeze_normalization=True
        for p in policy.jump_mlp.parameters():p.requires_grad_(False)
        with torch.no_grad():policy.distribution.std_param.fill_(.25)
        for p in policy.distribution.parameters():p.requires_grad_(False)
        frozen_jump={k:v.detach().cpu().clone() for k,v in policy.jump_mlp.state_dict().items()}
        frozen_norm={k:v.detach().cpu().clone() for k,v in policy.state_dict().items() if "normalizer" in k}
        runner.alg.configure_learning_rate("full",repair=True)
        start=int(runner.current_learning_iteration)+1;runner.current_learning_iteration=start
        install_checkpoint_metadata(runner,controller,start=start,total=args.iterations,initial_progress=1.)
        experiment=dict(source=str(args.bundle.resolve()),smoothing=args.smoothing,num_envs=args.num_envs,
            additional_iterations=args.iterations,learning_rate=1.e-5,exploration_std=.25,
            jump_branch_frozen=True,actor_normalizers_frozen=True,optimizer_reinitialized=True,
            rollout_steps=agent.num_steps_per_env,settings=getattr(cfg,"turn_smoothing_settings",None))
        atomic_write_json(args.output/"experiment.json",experiment)
        original=runner.logger.log;began=time.monotonic()
        def log(**kwargs):
            original(**kwargs)
            atomic_write_json(args.output/"progress.json",dict(phase="learning",completed=kwargs["it"]-start+1,
                total=args.iterations,elapsed_s=time.monotonic()-began,iteration_s=kwargs["collect_time"]+kwargs["learn_time"]))
        runner.logger.log=log
        env.reset();runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        assert all(torch.equal(frozen_jump[k],v.detach().cpu()) for k,v in policy.jump_mlp.state_dict().items())
        assert all(torch.equal(v,policy.state_dict()[k].detach().cpu()) for k,v in frozen_norm.items())
        checkpoint=args.output/f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(checkpoint),infos={"auto_curriculum":controller.state_dict(),"turn_experiment":experiment})
        # Export candidate with full reset config; existing source best is immutable.
        export_cfg=copy.deepcopy(cfg)
        export_cfg.events.reset_robot_joints.params.update(random_start=False,motion_start_seconds=0.,motion_start_choices=None,
            segment_seconds=cfg.actions.joint_pos.motion_window_seconds)
        save_bundle(args.output/"candidate.pt",checkpoint_path=checkpoint,reference=bundle["reference"],env_cfg=export_cfg,
                    agent_cfg=agent,task=bundle["task"],dance=bundle["dance"])
        controller.record_validation=True;controller.render_validation=False
        controller.validation_scene=save_validation_scene(args.output/"validation_media/scene.pt",env_cfg=export_cfg,
                    reference=bundle["reference"],task=bundle["task"],report=controller.report)
        atomic_write_json(args.output/"progress.json",dict(phase="validation",completed=args.iterations,total=args.iterations))
        records=controller.validate_seeds(runner,wrapped,[args.seed+i for i in range(3)])
        coverage=continuous_coverage(records[0]);passed=QualitySelection.passed([coverage,*records])
        atomic_write_json(args.output/"validation.json",dict(passed=passed,coverage=coverage,records=records))
        atomic_write_json(args.output/"progress.json",dict(phase="completed",completed=args.iterations,total=args.iterations,
            full_validation_passed=passed,elapsed_s=time.monotonic()-began))
        print(f"[TURN EXPERIMENT COMPLETE] smoothing={args.smoothing} full validation passed={passed}",flush=True)
    finally:wrapped.close()


try:main()
finally:app.close()
