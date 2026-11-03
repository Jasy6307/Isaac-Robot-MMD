"""Record policy trajectories, then export four synchronized comparison views."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat

parser=argparse.ArgumentParser()
parser.add_argument("--mode",choices=("collect","render"),required=True)
parser.add_argument("--bundle",type=Path,required=True)
parser.add_argument("--iterations",type=int,nargs=4,required=True,help="Completed iteration counts, not zero-based filenames")
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--panel_width",type=int,default=960)
parser.add_argument("--panel_height",type=int,default=540)
parser.add_argument("--max_frames",type=int,default=0,help="Short rendering preview only; zero renders full motion")
parser.add_argument("--start_frame",type=int,default=0,help="Preview frame offset; full exports start at zero")
parser.add_argument("--preview_stride",type=int,default=1,help="Frame stride for short visual checks")
parser.add_argument("--ffmpeg",type=Path)
parser.add_argument("--audio",type=Path)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
args.enable_cameras=args.mode=="render"
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.deployment import load_bundle
from source.train_workflow.utils.motion.audio import resolve_motion_audio, ffmpeg_audio_args


def collect(bundle,cfg,agent):
    cfg.scene.num_envs=1
    wrapped=RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.clip_actions)
    env=wrapped.unwrapped
    robot=env.scene["robot"]
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
    total=int(bundle["reference"]["tables"]["_phase"].shape[0])
    fps=1./env.step_dt
    capture=dict(armed=False,latest=None)
    original=env.scene.update
    def update(*a,**kw):
        result=original(*a,**kw)
        if capture["armed"]:
            capture["latest"]=torch.cat((robot.data.root_state_w.torch[0,:7],robot.data.joint_pos.torch[0])).detach().clone()
        return result
    env.scene.update=update
    try:
        for count in args.iterations:
            checkpoint=args.bundle.parent.parent/f"model_{count-1}.pt"
            runner.load(str(checkpoint))
            policy=runner.get_inference_policy(device=args.device)
            with torch.no_grad():
                wrapped.seed(args.seed)
                obs,_=wrapped.reset()
                poses=[torch.cat((robot.data.root_state_w.torch[0,:7],robot.data.joint_pos.torch[0])).cpu().numpy().copy()]
            completed=False
            for step in range(total):
                with torch.inference_mode():
                    action=policy(obs)
                    capture["armed"]=True
                    try:obs,_,done,_=wrapped.step(action)
                    finally:capture["armed"]=False
                    poses.append(capture["latest"].cpu().numpy().copy())
                if (step+1)%300==0:print(f"[POSES] {count} iter: {(step+1)/fps:.1f}/{total/fps:.1f}s",flush=True)
                if bool(done[0]):
                    completed=step+1>=total and not bool(env.reset_terminated[0])
                    break
                completed=step+1==total
            metadata=dict(completed_iterations=count,checkpoint=str(checkpoint.resolve()),seed=args.seed,
                num_envs=1,control_hz=fps,physics_hz=1./env.physics_dt,decimation=cfg.decimation,
                expected_steps=total,recorded_steps=len(poses)-1,completed=completed,
                joint_names=list(robot.joint_names),pose_timing="initial pose plus post-control-step states, sampled before automatic reset",
                action_smoothing=1.,observation_corruption=cfg.observations.policy.enable_corruption)
            np.savez_compressed(args.output/f"poses_{count}.npz",poses=np.stack(poses),metadata=json.dumps(metadata))
            print(f"[POSES SAVED] {count} iter: {len(poses)} poses, completed={completed}",flush=True)
    finally:
        env.scene.update=original
        wrapped.close()


def render(bundle,cfg):
    from PIL import Image,ImageDraw,ImageFont
    from isaaclab.sensors import CameraCfg
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    # A one-microsecond render-only physics tick publishes teleported poses to
    # RTX. Every frame starts from a recorded pose; drift is measured below.
    cfg.sim.use_fabric=True
    cfg.scene.num_envs=4
    cfg.scene.env_spacing=10.
    cfg.scene.replicate_physics=False
    cfg.scene.clone_in_fabric=False
    # Rendering uses recorded poses, so a local matte floor suffices. Avoid
    # remote ground USD lookups and keep the same visible contact-plane height.
    ground_z=float(getattr(cfg.scene.terrain,"ground_z_offset",0.))
    cfg.scene.terrain=None
    cfg.scene.render_floor=AssetBaseCfg(prim_path="/World/ground/terrain/CollisionPlane",
        spawn=sim_utils.CuboidCfg(size=(100.,100.,.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.38,.43),roughness=.95,metallic=0.)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,ground_z-.01)))
    cfg.scene.record_camera=CameraCfg(prim_path="{ENV_REGEX_NS}/RecordCamera",update_period=0.,
        height=args.panel_height,width=args.panel_width,data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=26.,horizontal_aperture=20.955,clipping_range=(.1,8.)))
    env=gym.make(bundle["task"],cfg=cfg)
    robot=env.unwrapped.scene["robot"]
    sim=env.unwrapped.sim
    sim.cfg.dt=1.e-6
    camera=env.unwrapped.scene["record_camera"]
    origins=sim_compat.as_torch(sim.get_clone_plan().positions)
    eyes=origins+torch.tensor([3.,5.2,2.],device=args.device)
    targets=origins+torch.tensor([0.,.5,.7],device=args.device)
    camera.set_world_poses_from_view(eyes=eyes,targets=targets)
    archives=[np.load(args.output/f"poses_{n}.npz") for n in args.iterations]
    metas=[json.loads(str(z["metadata"])) for z in archives]
    for iteration,meta in zip(args.iterations,metas):
        if meta["completed_iterations"]!=iteration or meta["joint_names"]!=list(robot.joint_names):
            raise ValueError("Recorded checkpoint or joint ordering does not match the requested comparison")
    poses=[torch.as_tensor(z["poses"],device=args.device) for z in archives]
    fps=metas[0]["control_hz"]
    count=metas[0]["expected_steps"]
    if args.max_frames:count=min(count,args.max_frames)
    if any(m["control_hz"]!=fps or m["expected_steps"]!=metas[0]["expected_steps"] for m in metas):
        raise ValueError("Pose archives do not share the same motion clock")
    width,height=args.panel_width,args.panel_height
    scale=width/960.
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",round(32*scale))
    small=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",round(22*scale))
    output=args.output/("preview.mp4" if args.max_frames else "comparison_4400_6000_7000_8000.mp4")
    ffmpeg=str(args.ffmpeg) if args.ffmpeg else __import__("imageio_ffmpeg").get_ffmpeg_exe()
    cmd=[ffmpeg,"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width*2}x{height*2}","-r",str(fps),"-i","pipe:0"]
    audio = args.audio or resolve_motion_audio(cfg.actions.joint_pos.motion_h5_path)
    if audio:
        print(f"[VIDEO AUDIO] {audio}", flush=True)
        cmd += ffmpeg_audio_args(audio, start_seconds=args.start_frame/fps, playback_rate=args.preview_stride)
    cmd += ["-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-t",str(count/fps),"-movflags","+faststart",str(output)]
    encoder_log=(args.output/("preview_encoder.log" if args.max_frames else "encoder.log")).open("wb")
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=encoder_log,stderr=subprocess.STDOUT)
    first_image=None
    image_checks=[]
    debug=[]
    max_joint_drift=0.
    max_root_drift=0.
    try:
        for video_frame in range(count):
            frame=video_frame*args.preview_stride+args.start_frame
            state=torch.stack([p[min(frame,len(p)-1)] for p in poses])
            root=state[:,:7].clone();root[:,:3]+=origins
            sim_compat.write_root_pose(robot,root)
            sim_compat.write_root_velocity(robot,torch.zeros((4,6),device=args.device))
            sim_compat.write_joint_state(robot,state[:,7:],torch.zeros_like(state[:,7:]))
            _=robot.data.body_link_pose_w.torch  # explicitly refresh FK after pose writes
            sim.step(render=False)
            robot.update(sim.cfg.dt)
            joint_drift=float((robot.data.joint_pos.torch-state[:,7:]).abs().max())
            root_drift=float((robot.data.root_state_w.torch[:,:3]-root[:,:3]).abs().max())
            max_joint_drift=max(max_joint_drift,joint_drift)
            max_root_drift=max(max_root_drift,root_drift)
            if joint_drift>1.e-3 or root_drift>1.e-4:
                raise RuntimeError(f"Render synchronization changed the recorded pose: {joint_drift=}, {root_drift=}")
            if args.max_frames:
                body=robot.data.body_link_pose_w.torch.detach().cpu().numpy()
                debug.append(dict(frame=frame,is_rendering=sim.is_rendering,is_playing=sim.is_playing(),
                    fabric=getattr(sim.physics_manager,"_fabric",None) is not None,
                    q_error=float((robot.data.joint_pos.torch-state[:,7:]).abs().max()),
                    body_positions=body[0,:,:3].tolist(),requested_root=root[0].detach().cpu().tolist()))
                (args.output/"render_debug.json").write_text(json.dumps(debug),encoding="utf-8")
            # RTX deduplicates by physics step AND render generation. A pose
            # replay must advance the render generation without stepping physics.
            sim.render()
            ensure_isaac_rtx_render_update(force=True)
            sim.render_context.reset_scene_state_cadence()
            camera.update(1./fps,force_recompute=True)
            rgb=sim_compat.as_torch(camera.data.output["rgb"])[...,:3].cpu().numpy()
            if rgb.dtype!=np.uint8:rgb=(rgb*255.).clip(0,255).astype(np.uint8)
            if first_image is None:first_image=rgb[0].copy()
            if video_frame>0 and (frame in (510,2220) or args.max_frames):
                changed=float(np.abs(rgb[0].astype(np.float32)-first_image.astype(np.float32)).mean())
                image_checks.append(dict(frame=frame,mean_image_change=changed))
                if changed<.5 and float((state[0]-poses[0][args.start_frame]).abs().max())>.5:
                    raise RuntimeError("Recorded pose changed but rendered image did not refresh")
            tile=np.concatenate((np.concatenate((rgb[0],rgb[1]),axis=1),np.concatenate((rgb[2],rgb[3]),axis=1)),axis=0)
            canvas=Image.fromarray(tile);draw=ImageDraw.Draw(canvas)
            for i,iteration in enumerate(args.iterations):
                x,y=(i%2)*width,(i//2)*height
                draw.rectangle((x,y,x+width,y+round(55*scale)),fill=(17,23,33))
                draw.text((x+round(18*scale),y+round(9*scale)),f"{iteration} iterations",font=font,fill=(255,255,255))
                draw.text((x+width-round(230*scale),y+round(12*scale)),f"{frame/fps:05.2f} s",font=font,fill=(123,218,255))
                draw.text((x+round(15*scale),y+height-round(30*scale)),"seed 42 | raw policy | 30 fps",font=small,fill=(235,235,235),stroke_width=1,stroke_fill=(0,0,0))
                if not metas[i]["completed"] and frame>metas[i]["recorded_steps"]:
                    draw.text((x+15,y+65),"ROLLOUT TERMINATED - last pose held",font=small,fill=(255,80,70))
            if args.max_frames or frame in (0,round(17*fps),round(21*fps),round(48*fps),round(74*fps),count-1):
                canvas.save(args.output/f"frame_{frame:04d}.jpg",quality=95)
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if frame%150==0:print(f"[RENDER] {frame}/{count} frames ({frame/fps:.1f}s)",flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError("Video encoder failed; inspect encoder.log")
        (args.output/"video_metadata.json").write_text(json.dumps(dict(video=str(output.resolve()),fps=fps,frames=count,
            duration_s=count/fps,width=width*2,height=height*2,source_poses=metas,camera_eye_relative=[3.,5.2,2.],
            camera_lookat_relative=[0.,.5,.7],image_refresh_checks=image_checks,
            render_sync_dt_s=sim.cfg.dt,max_render_joint_drift_rad=max_joint_drift,max_render_root_drift_m=max_root_drift,
            method="Render saved real policy trajectories; each pose has a checked one-microsecond synchronization tick"),indent=2),encoding="utf-8")
        print(f"[VIDEO SAVED] {output.resolve()}",flush=True)
    finally:
        if encoder.poll() is None:encoder.kill()
        encoder_log.close();env.close()


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(args.bundle)
    cfg=bundle["env_cfg"];cfg.seed=args.seed;cfg.sim.device=args.device
    agent=bundle["agent_cfg"];agent.device=args.device
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    if args.mode=="collect":collect(bundle,cfg,agent)
    else:render(bundle,cfg)


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()



