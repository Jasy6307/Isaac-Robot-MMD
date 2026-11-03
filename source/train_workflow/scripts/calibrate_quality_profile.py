"""Calibrate frozen gates from explicitly accepted checkpoints, without learning."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat
parser=argparse.ArgumentParser()
parser.add_argument("--bundle",type=Path,required=True)
parser.add_argument("--iterations",type=int,nargs="+",required=True,help="Completed iteration counts")
parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--num_envs",type=int,default=16)
parser.add_argument("--verify_profile",type=Path,help="Verify an existing frozen profile instead of calibrating it")
parser.add_argument("--coverage",action="store_true",help="With --verify_profile, test all local reset starts")
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.train_workflow.utils.deployment import load_bundle
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.quality_selection import generic_profile,calibrate_profile,assess_trace,reference_key,resolve_profile
from source.my_task.motion_reference.motion_buffer import register_packaged_reference

def main():
    args.output.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(args.bundle);cfg=bundle["env_cfg"];agent=bundle["agent_cfg"]
    cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;agent.device=args.device
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    wrapped=RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.clip_actions)
    controller=AutoJumpController(wrapped.unwrapped,cfg.auto_jump_report,args.output)
    controller.quality_profile=(resolve_profile(controller.report,args.verify_profile) if args.verify_profile else generic_profile(controller.report))
    controller.configure(1.)
    wrapped.unwrapped._auto_jump_controller=controller
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
    records=[];provenance=[];traces=[]
    try:
        for iteration in args.iterations:
            checkpoint=args.bundle.parent.parent/f"model_{iteration-1}.pt"
            runner.load(str(checkpoint))
            provenance.append(dict(completed_iterations=iteration,checkpoint=str(checkpoint.resolve()),seeds=args.seeds))
            for seed in args.seeds:
                directory=args.output/f"iter_{iteration}_seed_{seed}"
                record=controller.validate(runner,wrapped,full=not args.coverage,seed=seed,trace_dir=directory)
                record["completed_iterations"]=iteration
                records.append(record);traces.append(directory/"trace_0000.npz")
                (args.output/"baseline_results.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        if args.verify_profile:
            passed=all(r["quality_passed"] and r["acceptance_passed"] for r in records)
            print(f"[{'RESET DIAGNOSTIC' if args.coverage else 'PROFILE VERIFIED'}] {passed} | {args.verify_profile}",flush=True)
            if not passed and not args.coverage:raise RuntimeError("Frozen profile verification failed; inspect baseline_results.json")
            return
        if args.coverage:raise ValueError("Calibration uses accepted full rollouts; --coverage requires --verify_profile")
        profile=calibrate_profile(controller.report,records,provenance)
        # A generated profile must actually accept every baseline, including
        # boundaries and event checks. Never silently relax a missing jump.
        verification=[]
        for record,path in zip(records,traces):
            acceptance=assess_trace(np.load(path)["trace"],controller.report,profile,full=True)
            verification.append(dict(iteration=record["completed_iterations"],seed=record["seed"],**acceptance))
        (args.output/"calibration_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
        if not all(v["passed"] for v in verification):raise RuntimeError("Calibrated profile rejected an accepted baseline; inspect verification")
        path=args.output/f"{reference_key(controller.report)}.json"
        path.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"[CALIBRATION SAVED] {path.resolve()}",flush=True)
    finally:wrapped.close()

try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()
