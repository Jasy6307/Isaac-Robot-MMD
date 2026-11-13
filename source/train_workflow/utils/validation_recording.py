"""Capture the actual acceptance trajectory without rendering or stepping physics."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import subprocess
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def save_validation_scene(path, *, env_cfg, reference, task, report, audio=None, reference_video=None):
    """Render-only config/reference snapshot, containing no policy or optimizer."""
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(env_cfg)
    cfg.scene.num_envs=1
    from source.train_workflow.utils.motion.audio import resolve_motion_audio
    audio = audio or resolve_motion_audio(cfg.actions.joint_pos.motion_h5_path)
    if audio and not Path(audio).is_file():
        raise FileNotFoundError(f"Validation audio not found: {audio}")
    payload=dict(format="g1-validation-scene",version=1,env_cfg=cfg,reference=reference,task=task,audio=audio,
                 reference_video=copy.deepcopy(reference_video),
                 source_sha256=report["source_sha256"],control_hz=report["control_hz"])
    fd,temporary=tempfile.mkstemp(dir=target.parent,suffix=".tmp");os.close(fd)
    try:
        torch.save(payload,temporary);os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    return target.resolve()


class ValidationPoseCapture:
    def __init__(self, *, origin, joint_names, control_hz):
        self.origin=origin.detach().clone()
        self.joint_names=list(joint_names)
        self.control_hz=float(control_hz)
        self.frames=[]

    def append(self, root_pose, joint_positions):
        root=root_pose.detach().clone()
        root[:3]-=self.origin
        self.frames.append(torch.cat((root,joint_positions.detach())).clone())

    def save(self, path, *, metadata, result):
        if not self.frames:raise ValueError("Cannot save an empty validation recording")
        poses=torch.stack(self.frames).cpu().numpy()
        if poses.shape[1]!=7+len(self.joint_names) or not np.isfinite(poses).all():
            raise ValueError("Invalid recorded poses")
        info=dict(metadata,format="g1-validation-poses",version=1,
            joint_names=self.joint_names,control_hz=self.control_hz,frames=len(poses),
            simulation_duration_s=(len(poses)-1)/self.control_hz,
            pose_order="root XYZ + XYZW quaternion + physical joint positions",
            timing="initial state plus actual post-control-step states, before automatic reset",
            action_smoothing=1.,result=result)
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(dir=target.parent,suffix=".tmp")
        try:
            with os.fdopen(fd,"wb") as stream:
                np.savez_compressed(stream,poses=poses,metadata=json.dumps(info,ensure_ascii=False))
            os.replace(temporary,target)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        return str(target.resolve())


def find_ffmpeg():
    found=shutil.which("ffmpeg")
    if found:return found
    try:return __import__("imageio_ffmpeg").get_ffmpeg_exe()
    except (ImportError,RuntimeError,FileNotFoundError):pass
    # A binary supplied by another local Python installation is usable too.
    appdata=os.environ.get("APPDATA")
    if appdata:
        candidates=sorted((Path(appdata)/"Python").glob("Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe"))
        if candidates:return str(candidates[-1])
    return None


def load_recordings(paths,*,checkpoint_comparison=False):
    """Validate common clock/reference before synchronous multi-robot replay."""
    recordings=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            poses=archive["poses"].copy();meta=json.loads(str(archive["metadata"]))
        if meta.get("format")!="g1-validation-poses" or meta.get("version")!=1:
            raise ValueError("Not a supported validation recording")
        if poses.ndim!=2 or not len(poses) or poses.shape[1]!=7+len(meta["joint_names"]) or not np.isfinite(poses).all():
            raise ValueError("Invalid recorded pose data")
        if not np.isfinite(meta["control_hz"]) or meta["control_hz"]<=0:
            raise ValueError("Invalid recording clock")
        if recordings:
            first=recordings[0][1]
            keys=["joint_names","control_hz","source_sha256","validation_protocol"]
            if not checkpoint_comparison:keys.append("completed_iterations")
            for key in keys:
                if meta.get(key)!=first.get(key):raise ValueError(f"Incompatible recordings: {key}")
            if not checkpoint_comparison and meta["seed"] in [m["seed"] for _,m in recordings]:raise ValueError("Duplicate recording seed")
        recordings.append((poses,meta))
    if not recordings:raise ValueError("No validation recordings")
    return recordings


def render_recording(path, *, emit=print):
    return render_recordings([path],emit=emit)


def render_recordings(paths, *, emit=print):
    """Serial isolated renderer. Failures never change acceptance or best ranking."""
    paths=[Path(path).resolve() for path in paths]
    if not paths:raise ValueError("No validation recordings")
    poses=paths[0]
    output=poses.with_name("validation.mp4") if len(paths)==1 else poses.parent.parent/"validation_seeds.mp4"
    encoder=find_ffmpeg()
    if encoder is None:return dict(video=None,error="ffmpeg unavailable; install imageio-ffmpeg",poses=str(poses))
    root=Path(__file__).resolve().parents[3]
    command=[sys.executable,str(root/"source/train_workflow/scripts/render_validation_video.py"),
             "--poses",*[str(path) for path in paths],"--output",str(output),"--device","cpu","--headless","--ffmpeg",encoder]
    log=output.with_suffix(".render.log")
    started=time.time_ns()
    emit(f"[验收录像] 生成MP4：{output}；录像子进程只回放已记录姿态，不运行策略")
    try:
        with log.open("wb") as stream:
            process=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            try:
                while True:
                    try:code=process.wait(timeout=30.);break
                    except subprocess.TimeoutExpired:emit(f"[验收录像] 仍在生成，详情：{log}")
            finally:
                if process.poll() is None:process.terminate();process.wait()
        sidecar=output.with_suffix(".json")
        if code!=0 or not output.exists() or output.stat().st_mtime_ns<started or not sidecar.exists() or sidecar.stat().st_mtime_ns<started:
            return dict(video=None,error=f"Renderer failed (exit {code}); inspect {log}",poses=str(poses))
        emit(f"[验收录像已保存] {output}")
        result=dict(video=str(output),render_log=str(log),poses=[str(path) for path in paths])
        try:
            comparison=json.loads(sidecar.read_text(encoding="utf-8")).get("reference_comparison")
            if comparison:
                result["reference_comparison"]=comparison
                if comparison.get("video"):
                    result["comparison_video"]=comparison["video"]
                    emit(f"[PMX对照录像已保存] {comparison['video']}")
                elif comparison.get("error"):
                    emit(f"[PMX对照录像失败，原验收录像已保留] {comparison['error']}")
        except (OSError, ValueError, TypeError) as error:
            emit(f"[PMX对照录像信息读取失败] {error}")
        return result
    except OSError as error:
        return dict(video=None,error=str(error),poses=str(poses))
