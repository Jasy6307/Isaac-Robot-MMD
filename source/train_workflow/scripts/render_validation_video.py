"""Render saved acceptance poses; never rerun the policy or change training physics."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# A concurrent diagnostic simulator may already consume the remaining VRAM.
# Check before importing/starting Kit; a skipped video must not disrupt training.
try:
    memory=subprocess.run(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                          capture_output=True,text=True,timeout=10,check=True)
    free_mib=int(memory.stdout.splitlines()[0])
except (OSError,ValueError,IndexError,subprocess.SubprocessError):
    free_mib=None
if free_mib is not None and free_mib<4864:
    raise RuntimeError(f"Insufficient free render VRAM ({free_mib} MiB; need >=4864 MiB); poses are saved, retry rendering later")

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from isaaclab.app import AppLauncher
from source import sim_compat
from source.train_workflow.utils.validation_reference import (
    add_reference_video_arguments, reference_video_from_args, has_reference_override,
    resolve_reference_video, compose_reference_video,
)
parser=argparse.ArgumentParser()
parser.add_argument("--poses",type=Path,nargs="+",required=True)
parser.add_argument("--scene",type=Path)
parser.add_argument("--bundle",type=Path,help="Existing trusted bundle, for historical render tests")
parser.add_argument("--output",type=Path)
parser.add_argument("--width",type=int,default=1280)
parser.add_argument("--height",type=int,default=720)
parser.add_argument("--ffmpeg",type=Path)
parser.add_argument("--audio",type=Path)
parser.add_argument("--max_frames",type=int,default=0)
parser.add_argument("--preview_stride",type=int,default=1)
parser.add_argument("--checkpoint_comparison",action="store_true",help="Explicitly compare different checkpoints at the same reference time")
parser.add_argument("--labels",nargs="+",help="Left-to-right checkpoint labels (checkpoint comparison only)")
parser.add_argument("--start_seconds",type=float,default=0.)
parser.add_argument("--end_seconds",type=float)
parser.add_argument("--playback_rate",type=float,default=1.,help="0.5 encodes recorded poses at half speed; source seconds remain accurate")
parser.add_argument("--camera_layout",choices=("close","wide"),default="close",help="Close full-motion framing (default), or the legacy wide view for comparison")
parser.add_argument("--robot_spacing",type=float,help="Replay robot center spacing in metres; close default 1.25, wide 2.8")
parser.add_argument("--fixed_camera",action="store_true",help="Disable close-layout reference-XY camera tracking")
add_reference_video_arguments(parser)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args=parser.parse_args()
from source.paths import resolve_dance_asset
if args.audio:
    args.audio = resolve_dance_asset(args.audio)
if args.start_seconds<0 or (args.end_seconds is not None and args.end_seconds<=args.start_seconds) or args.playback_rate<=0:
    parser.error("Invalid time range or playback rate")
if args.labels and (not args.checkpoint_comparison or len(args.labels)!=len(args.poses)):
    parser.error("--labels needs checkpoint comparison and one label per recording")
if args.robot_spacing is not None and (not math.isfinite(args.robot_spacing) or args.robot_spacing<=0):
    parser.error("--robot_spacing must be finite and positive")
args.enable_cameras=True
sim_compat.prepare_launcher_args(args,default_visualizer="none")
app=AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import source.my_task
from PIL import Image,ImageDraw,ImageFont
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.deployment import atomic_write_json
from source.train_workflow.utils.validation_recording import load_recordings,find_ffmpeg
from source.train_workflow.utils.validation_camera import replica_offsets,fit_camera,motion_bounds,reference_camera_track
from source.train_workflow.utils.motion.audio import resolve_motion_audio, ffmpeg_audio_args


def main():
    recordings=load_recordings(args.poses,checkpoint_comparison=args.checkpoint_comparison)
    poses,meta=recordings[0];replicas=len(recordings)
    scene_path=args.bundle or args.scene or Path(meta["scene"])
    scene=torch.load(scene_path,map_location="cpu",weights_only=False)
    if scene.get("format") not in ("g1-validation-scene","g1-dance-deployment"):
        raise ValueError("Not a supported render scene or deployment bundle")
    cfg=scene["env_cfg"]
    # New scenes freeze the selected reference. Legacy scenes resolve the dance
    # list here, so already-running jobs can use a registered PMX video too.
    reference_video=None
    reference_error=None
    try:
        if has_reference_override(args):
            reference_video=reference_video_from_args(args,cfg.actions.joint_pos.motion_h5_path)
        elif "reference_video" in scene:
            reference_video=scene["reference_video"]
        else:
            reference_video=resolve_reference_video(cfg.actions.joint_pos.motion_h5_path)
    except (OSError,ValueError,TypeError) as error:
        reference_error=str(error)
        print(f"[PMX对照视频不可用，继续生成原验收录像] {error}",flush=True)
    motion_start_s=float(meta.get("result",{}).get("start_s",0.))
    audio = args.audio
    if audio is None:
        saved_audio = scene.get("audio")
        audio = (Path(saved_audio) if saved_audio and Path(saved_audio).is_file() else
                 resolve_motion_audio(cfg.actions.joint_pos.motion_h5_path))
    if audio:
        print(f"[VIDEO AUDIO] {audio}", flush=True)
    if meta.get("source_sha256")!=cfg.auto_jump_report.get("source_sha256"):
        raise ValueError("Pose and render reference differ")
    cfg.sim.device=args.device;cfg.sim.use_fabric=True
    spacing=args.robot_spacing if args.robot_spacing is not None else (1.25 if args.camera_layout=="close" else 2.8)
    replay_offsets=replica_offsets(replicas,spacing)
    font_size=max(18,round(args.width/48)) if args.camera_layout=="close" else max(16,round(args.width/40))
    small_size=max(14,round(args.width/64)) if args.camera_layout=="close" else max(13,round(args.width/60))
    label_y=round(4+font_size*1.25) if args.camera_layout=="close" else 50
    ended_y=round(label_y+small_size*1.3) if args.camera_layout=="close" else 75
    has_short_recording=len({len(p) for p,m in recordings})>1
    header_pixels=(round(label_y+small_size*1.5+6+(small_size*1.3 if has_short_recording else 0))
                   if args.camera_layout=="close" else round(args.height*.15))
    footer_pixels=max(34,round(small_size*1.5))
    ground_z=float(getattr(cfg.scene.terrain,"ground_z_offset",0.))
    camera_track=None
    if args.camera_layout=="close":
        if not args.fixed_camera:camera_track=reference_camera_track(scene["reference"],max(len(p) for p,m in recordings))
        bounds=motion_bounds(recordings,cfg.scene.robot.spawn.usd_path,replay_offsets,ground_z,camera_track)
        framing=fit_camera(bounds,width=args.width,height=args.height,header_pixels=header_pixels,footer_pixels=footer_pixels)
        framing["follow"]= "reference XY only, smoothed; no actual-state or jump-height tracking" if camera_track is not None else "fixed world camera"
        framing["fixed_scale_for_entire_recording"]=True
        framing["fixed_for_entire_recording"]=camera_track is None
    else:
        framing=dict(eye=[3.,5.2,2.] if replicas==1 else [0.,max(8.,replicas*3.5),3.],target=[0.,.5,.7],
                     fixed_for_entire_recording=True)
    # Only the replay scene is changed. Close replicas must not interact during
    # the microscopic RTX synchronization tick; acceptance physics is untouched.
    collision=cfg.scene.robot.spawn.collision_props
    cfg.scene.robot.spawn.collision_props=(collision.replace(collision_enabled=False) if collision is not None
                                         else sim_utils.CollisionPropertiesCfg(collision_enabled=False))
    cfg.scene.num_envs=1;cfg.scene.replicate_physics=False;cfg.scene.clone_in_fabric=False
    # One env, several independently posed articulations, and exactly one camera.
    # CameraCfg requires one camera per env; creating three envs would create
    # three render products rather than the requested single shared view.
    for i in range(1,replicas):
        extra=copy.deepcopy(cfg.scene.robot)
        extra.prim_path=f"{{ENV_REGEX_NS}}/ValidationRobot_{i}"
        x,y,z=extra.init_state.pos
        extra.init_state.pos=(x+i*spacing,y,z)
        setattr(cfg.scene,f"validation_robot_{i}",extra)
    cfg.scene.terrain=None
    cfg.scene.render_floor=AssetBaseCfg(prim_path="/World/ground/terrain/CollisionPlane",
        spawn=sim_utils.CuboidCfg(size=(100.,100.,.02),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.38,.43),roughness=.95,metallic=0.)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,ground_z-.01)))
    cfg.scene.validation_camera=CameraCfg(prim_path="/World/ValidationCamera",update_period=0.,
        height=args.height,width=args.width,data_types=["rgb"],
        renderer_cfg=IsaacRtxRendererCfg(enable_scene_partitioning=False),
        spawn=sim_utils.PinholeCameraCfg(focal_length=26.,horizontal_aperture=20.955,clipping_range=(.1,30.)))
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,scene["reference"])
    env=gym.make(scene["task"],cfg=cfg)
    try:
        sim=env.unwrapped.sim;robot=env.unwrapped.scene["robot"];camera=env.unwrapped.scene["validation_camera"]
        robots=[robot]+[env.unwrapped.scene[f"validation_robot_{i}"] for i in range(1,replicas)]
        if meta["joint_names"]!=list(robot.joint_names):raise ValueError("Recorded joint ordering differs")
        if poses.ndim!=2 or poses.shape[1]!=7+len(robot.joint_names) or not np.isfinite(poses).all():
            raise ValueError("Invalid recorded pose data")
        sim.cfg.dt=1.e-6
        origins=torch.as_tensor(replay_offsets,device=args.device)
        camera_eye=origins.new_tensor(framing["eye"]).unsqueeze(0)
        camera_target=origins.new_tensor(framing["target"]).unsqueeze(0)
        camera_shifts=None if camera_track is None else origins.new_tensor(camera_track)
        camera.set_world_poses_from_view(eyes=camera_eye,targets=camera_target)
        states=[torch.as_tensor(p,device=args.device) for p,m in recordings]
        fps=float(meta["control_hz"])
        video_fps=fps*args.playback_rate
        first_frame=round(args.start_seconds*fps)
        last_frame=max(len(p) for p,m in recordings)
        if args.end_seconds is not None:last_frame=min(last_frame,round(args.end_seconds*fps)+1)
        frames=list(range(first_frame,last_frame,args.preview_stride))
        if not frames:raise ValueError("No recorded frames in requested time range")
        if args.max_frames:frames=frames[:args.max_frames]
        output=args.output or args.poses[0].with_name("validation.mp4" if replicas==1 else "validation_seeds.mp4")
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_name(output.stem+".partial.mp4")
        encoder_exe=str(args.ffmpeg) if args.ffmpeg else find_ffmpeg()
        if not encoder_exe:raise RuntimeError("Install imageio-ffmpeg in the recording Python environment")
        cmd=[encoder_exe,"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{args.width}x{args.height}",
             "-r",str(video_fps),"-i","pipe:0"]
        if audio:cmd += ffmpeg_audio_args(audio, start_seconds=motion_start_s+first_frame/fps, playback_rate=args.playback_rate*args.preview_stride)
        cmd += ["-c:v","libx264","-preset","fast","-crf","19","-pix_fmt","yuv420p",
                "-t",str(len(frames)/video_fps),"-movflags","+faststart",str(temporary)]
        verdicts=[]
        for p,m in recordings:
            result=m.get("result",{})
            verified=(result.get("completed") is True and result.get("quality",{}).get("passed") is True
                      and result.get("acceptance",{}).get("passed") is True)
            known=result.get("acceptance",{}).get("passed") in (True,False)
            verdicts.append("PASSED" if verified else "NOT PASSED" if known else "NOT ASSESSED")
        font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",font_size)
        small=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",small_size)
        checks=[];first_rgb=None;max_joint_drift=max_root_drift=0.
        with output.with_suffix(".encoder.log").open("wb") as encoder_log:
            encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=encoder_log,stderr=subprocess.STDOUT)
            try:
                for n,frame in enumerate(frames):
                    if camera_shifts is not None:
                        shift=camera_shifts[frame]
                        camera.set_world_poses_from_view(eyes=camera_eye+shift,targets=camera_target+shift)
                    state=torch.stack([s[min(frame,len(s)-1)] for s in states]);root=state[:,:7].clone();root[:,:3]+=origins
                    for i,replica in enumerate(robots):
                        sim_compat.write_root_pose(replica,root[i:i+1])
                        sim_compat.write_root_velocity(replica,torch.zeros((1,6),device=args.device))
                        sim_compat.write_joint_state(replica,state[i:i+1,7:],torch.zeros_like(state[i:i+1,7:]))
                        _=replica.data.body_link_pose_w.torch
                    sim.step(render=False)
                    for replica in robots:replica.update(sim.cfg.dt)
                    jd=max(float((r.data.joint_pos.torch-state[i:i+1,7:]).abs().max()) for i,r in enumerate(robots))
                    rd=max(float((r.data.root_state_w.torch[:,:3]-root[i:i+1,:3]).abs().max()) for i,r in enumerate(robots))
                    max_joint_drift=max(max_joint_drift,jd);max_root_drift=max(max_root_drift,rd)
                    if jd>1.e-3 or rd>1.e-4:raise RuntimeError(f"Render pose drift exceeded tolerance: {jd}, {rd}")
                    sim.render();ensure_isaac_rtx_render_update(force=True)
                    sim.render_context.reset_scene_state_cadence();camera.update(1./fps,force_recompute=True)
                    rgb=sim_compat.as_torch(camera.data.output["rgb"])[0,...,:3].cpu().numpy()
                    if rgb.dtype!=np.uint8:rgb=(rgb*255.).clip(0,255).astype(np.uint8)
                    if first_rgb is None:first_rgb=rgb.copy()
                    if n and (args.max_frames or frame in (510,2220) or n in (len(frames)//2,len(frames)-1)):
                        change=float(np.abs(rgb.astype(np.float32)-first_rgb.astype(np.float32)).mean())
                        checks.append(dict(frame=frame,mean_image_change=change))
                        pose_change=max(float((s[min(frame,len(s)-1)]-s[min(first_frame,len(s)-1)]).abs().max()) for s in states)
                        if change<.5 and pose_change>.5:
                            raise RuntimeError("Robot pose changed but image did not refresh")
                    canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
                    draw.rectangle((0,0,args.width,header_pixels),fill=(17,23,33))
                    heading="Checkpoint comparison" if args.checkpoint_comparison else f"iter {meta['completed_iterations']}"
                    draw.text((12,6),f"{heading} | {motion_start_s+frame/fps:05.2f} s | {args.playback_rate:g}x recorded poses",font=font,fill="white")
                    for i,((p,m),verdict) in enumerate(zip(recordings,verdicts)):
                        ended=f" | ENDED @ {(len(p)-1)/fps:.2f}s" if frame>=len(p) else ""
                        label=args.labels[i] if args.labels else f"seed {m['seed']}"
                        draw.text((12+i*args.width//replicas,label_y),f"{label} | {verdict}",font=small,fill=(123,218,255))
                        if ended:draw.text((12+i*args.width//replicas,ended_y),ended.strip(" |"),font=small,fill=(255,155,95))
                    draw.text((12,args.height-32),f"{meta['source']} | {replicas} replicas, one camera",font=small,fill=(123,218,255),stroke_width=1,stroke_fill="black")
                    if args.max_frames or n in (0,len(frames)//2,len(frames)-1) or frame in (0,510,2220):canvas.save(output.with_name(f"{output.stem}_frame_{frame:04d}.jpg"),quality=95)
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                    if n%300==0:print(f"[VIDEO] {n}/{len(frames)} | source time {frame/fps:.2f}s",flush=True)
                encoder.stdin.close()
                if encoder.wait()!=0:raise RuntimeError("Video encoder failed; inspect encoder log")
                os.replace(temporary,output)
            finally:
                if encoder.poll() is None:encoder.kill();encoder.wait()
        atomic_write_json(output.with_suffix(".json"),dict(video=str(output.resolve()),poses=[str(p.resolve()) for p in args.poses],
            metadata=[m for p,m in recordings],replicas=replicas,cameras=1,render_device=args.device,frames=len(frames),fps=video_fps,
            video_duration_s=len(frames)/video_fps,source_control_hz=fps,source_start_s=motion_start_s+first_frame/fps,
            source_frame_stride=args.preview_stride,audio=str(audio) if audio else None,
            playback_rate=args.playback_rate,checkpoint_comparison=args.checkpoint_comparison,labels=args.labels,
            camera_layout=args.camera_layout,robot_spacing_m=spacing,camera=framing,replay_collision_enabled=False,
            image_refresh_checks=checks,max_joint_drift_rad=max_joint_drift,max_root_drift_m=max_root_drift,
            render_sync_dt_s=sim.cfg.dt,method="Recorded physical poses replayed; no policy inference"))
        print(f"[VIDEO SAVED] {output.resolve()}",flush=True)
    finally:env.close()
    # Compose after closing the render environment; acceptance and its original
    # MP4 remain intact even if the reference movie cannot be decoded.
    comparison={"error":reference_error} if reference_error else None
    if reference_video:
        try:
            comparison=compose_reference_video(output,reference_video,ffmpeg=encoder_exe)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
            comparison=dict(error=str(error),reference=reference_video)
            print(f"[PMX对照录像失败，原验收录像已保留] {error}",flush=True)
    if comparison:
        sidecar=output.with_suffix(".json")
        info=json.loads(sidecar.read_text(encoding="utf-8"))
        info["reference_comparison"]=comparison
        atomic_write_json(sidecar,info)


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()
