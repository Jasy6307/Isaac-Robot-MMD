"""Retrospective full-motion checks using a fixed packaged reference and config."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat

parser=argparse.ArgumentParser()
parser.add_argument("--bundle",required=True,type=Path)
parser.add_argument("--iterations",required=True,nargs="+",type=int)
parser.add_argument("--output",required=True,type=Path)
parser.add_argument("--num_envs",type=int,default=16)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--confirm_seeds",nargs="+",type=int,default=[43,44])
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import source.my_task  # register tasks/configs before loading trusted bundle
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.deployment import load_bundle


def main():
    bundle=load_bundle(args.bundle)
    cfg=bundle["env_cfg"]
    cfg.scene.num_envs=args.num_envs
    cfg.seed=args.seed
    cfg.sim.device=args.device
    agent=bundle["agent_cfg"]
    agent.device=args.device
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    args.output.mkdir(parents=True,exist_ok=True)
    env=gym.make(bundle["task"],cfg=cfg)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    controller=AutoJumpController(wrapped.unwrapped,cfg.auto_jump_report,args.output)
    controller.configure(1.)
    wrapped.unwrapped._auto_jump_controller=controller
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
    run=args.bundle.parent.parent
    records=[]
    def persist():
        (args.output/"results.json").write_text(json.dumps(dict(
            reference_bundle=str(args.bundle.resolve()),num_envs=args.num_envs,
            protocol="screen all at one seed; confirm earliest passing screen and final model at additional seeds",
            records=records),ensure_ascii=False,indent=2),encoding="utf-8")
    def check(iteration,seed,kind):
        runner.load(str(run/f"model_{iteration}.pt"))
        started=time.monotonic()
        result=controller.validate(runner,wrapped,full=True,seed=seed)
        result.update(checkpoint_iteration=iteration,elapsed_wall_s=time.monotonic()-started,check_kind=kind)
        records.append(result);persist()
        print(f"[CHECKPOINT CHECK] iter={iteration} seed={seed} completed={result['completed']}/{result['tested']} "
              f"quality={result['quality_passed']} seconds={result['elapsed_wall_s']:.1f}",flush=True)
        return result
    try:
        screens=[check(i,args.seed,"screen") for i in sorted(set(args.iterations))]
        candidates=[r["checkpoint_iteration"] for r in screens if r["quality_passed"]]
        final=max(args.iterations)
        # Ascending candidates: if confirmation fails, try the next screened pass.
        earliest=None
        for iteration in candidates:
            confirmations=[check(iteration,seed,"confirmation") for seed in args.confirm_seeds]
            if all(r["quality_passed"] for r in confirmations):
                earliest=iteration;break
        if earliest != final:
            for seed in args.confirm_seeds:check(final,seed,"final_baseline")
        (args.output/"summary.json").write_text(json.dumps(dict(
            earliest_tested_passing_iteration=earliest,screened_iterations=sorted(set(args.iterations)),
            seeds=[args.seed,*args.confirm_seeds],
            limitation="Earliest among tested checkpoints only. Quality thresholds do not certify jump naturalness. "
                       "Results from this training seed do not establish a universal training budget."),indent=2),encoding="utf-8")
    finally:
        wrapped.close()


try:main()
finally:app.close()
