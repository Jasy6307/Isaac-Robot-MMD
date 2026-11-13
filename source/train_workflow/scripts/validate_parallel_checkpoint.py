"""Revalidate a trusted checkpoint: independent seeds, shared physics, one MP4."""
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
source_group=parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--bundle",type=Path)
source_group.add_argument("--scene",type=Path,help="Training validation_media/scene.pt snapshot; no completed deployment required")
parser.add_argument("--agent_config",type=Path,help="Saved params/agent.yaml; defaults to checkpoint run")
parser.add_argument("--quality_profile",type=Path,help="Frozen training quality_profile.json; defaults to checkpoint run")
parser.add_argument("--audio",type=Path,help="Explicit music for old references without provenance metadata")
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--num_envs",type=int,default=16)
parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
parser.add_argument("--no_video",action="store_true")
parser.add_argument("--auto_motion_feedback",action="store_true",help="Exercise reference-aware diagnostics on this fixed checkpoint, without training")
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
from source.paths import resolve_dance_asset
if args.audio:
    args.audio = resolve_dance_asset(args.audio)
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.deployment import load_bundle,atomic_write_json
from source.train_workflow.utils.quality_selection import resolve_profile
from source.train_workflow.utils.validation_recording import save_validation_scene


def main():
    if args.num_envs < len(args.seeds):
        raise ValueError("num_envs must cover every validation seed")
    if args.scene:
        bundle=torch.load(args.scene,map_location="cpu",weights_only=False)
        if bundle.get("format")!="g1-validation-scene":
            raise ValueError("Expected the training scene.pt snapshot")
        config_path=args.agent_config or args.checkpoint.parent/"params/agent.yaml"
        agent=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        bundle=load_bundle(args.bundle)
        saved_agent=bundle["agent_cfg"]
        agent=saved_agent.to_dict() if hasattr(saved_agent,"to_dict") else dict(saved_agent)
    cfg=bundle["env_cfg"]
    cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;agent["device"]=args.device
    cfg.seed=args.seeds[0]
    if args.auto_motion_feedback:cfg.auto_motion_feedback=True
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    wrapped=RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.get("clip_actions"))
    try:
        controller=AutoJumpController(wrapped.unwrapped,cfg.auto_jump_report,args.output)
        controller.configure(1.);wrapped.unwrapped._auto_jump_controller=controller
        profile_path=args.quality_profile or args.checkpoint.parent/"quality_profile.json"
        controller.quality_profile=resolve_profile(controller.report,profile_path if profile_path.exists() else None)
        controller.record_validation=True;controller.render_validation=not args.no_video
        controller.validation_source="checkpoint_revalidation"
        controller.validation_scene=save_validation_scene(args.output/"validation_media/scene.pt",
            env_cfg=cfg,reference=bundle["reference"],task=bundle["task"],report=controller.report,
            audio=str(args.audio.resolve()) if args.audio else bundle.get("audio"))
        runner=OnPolicyRunner(wrapped,agent,log_dir=None,device=args.device)
        runner.load(str(args.checkpoint),load_cfg={"actor":True,"critic":True,"optimizer":False,"iteration":True})
        started=time.monotonic()
        records=controller.validate_seeds(runner,wrapped,args.seeds,trace_dir=args.output/"traces")
        atomic_write_json(args.output/"results.json",dict(checkpoint=str(args.checkpoint.resolve()),
            source_snapshot=str((args.scene or args.bundle).resolve()),training=False,
            num_envs=args.num_envs,elapsed_wall_s=time.monotonic()-started,records=records))
    finally:wrapped.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()
