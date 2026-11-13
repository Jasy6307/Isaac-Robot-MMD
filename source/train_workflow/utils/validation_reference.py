"""Resolve PMX reference movies and compose them with recorded validation video.

No simulator or policy imports: composition cannot affect acceptance results.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from source.paths import resolve_dance_asset

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "source/train_workflow/validation_videos.json"


def add_reference_video_arguments(parser):
    parser.add_argument("--validation_reference_video", type=str,
                        help="PMX MP4 path; overrides the per-dance validation_videos.json default.")
    parser.add_argument("--validation_reference_start_seconds", type=float,
                        help="MP4 time corresponding to motion frame 0 (seconds; default from dance list or 0).")
    parser.add_argument("--validation_reference_config", type=str,
                        help="Alternate per-dance video list; paths are relative to the repository root.")
    parser.add_argument("--no_validation_reference_video", action="store_true",
                        help="Disable PMX comparison for this run; keep ordinary validation video.")


def _path(value):
    path = Path(value).expanduser()
    return resolve_dance_asset(path if path.is_absolute() else ROOT / path)


def _key(value):
    stem = Path(str(value).replace("\\", "/")).stem
    return "_".join(stem.removesuffix("_z_editted").casefold().split())


def resolve_reference_video(motion_path=None, *, video=None, start_seconds=None,
                            config_path=None, disabled=False):
    if disabled:
        if video is not None or start_seconds is not None:
            raise ValueError("Cannot combine --no_validation_reference_video with a reference override")
        return None
    entry = {}
    matched = None
    config = _path(config_path) if config_path else DEFAULT_CONFIG
    if video is None:
        candidates = []
        if motion_path:
            motion = _path(motion_path)
            candidates.append(_key(motion))
            sidecar = motion.with_suffix(".motion.json")
            if sidecar.is_file():
                provenance = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                for field in ("source_dance", "vmd", "audio"):
                    if provenance.get(field):
                        candidates.append(_key(provenance[field]))
            # Audio provenance also covers corrected dances without a motion sidecar.
            from source.train_workflow.utils.motion.audio import resolve_motion_audio
            audio = resolve_motion_audio(str(motion))
            if audio:
                candidates.append(_key(audio))
        if config.is_file():
            data = json.loads(config.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("dances"), dict):
                raise ValueError(f"Invalid validation video list: {config}")
            entries = {_key(name): item for name, item in data["dances"].items()}
            for candidate in candidates:
                if candidate in entries:
                    entry = entries[candidate]
                    if not isinstance(entry, dict):
                        raise ValueError(f"Invalid validation video entry: {candidate}")
                    matched = candidate
                    break
        elif config_path:
            raise FileNotFoundError(f"Validation video list not found: {config}")
        video = entry.get("video")
    offset = float(start_seconds if start_seconds is not None else entry.get("start_seconds", 0.))
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Reference video start seconds must be finite and nonnegative")
    if not video:
        if start_seconds is not None:
            raise ValueError("Reference start seconds requires an explicit or configured MP4")
        return None
    path = _path(video)
    if not path.is_file():
        raise FileNotFoundError(f"Validation reference MP4 not found: {path}")
    return dict(video=str(path), start_seconds=offset, dance=matched,
                config=str(config) if matched is not None else None,
                actor=entry.get("actor"))


def reference_video_from_args(args, motion_path=None):
    return resolve_reference_video(motion_path,
        video=getattr(args, "validation_reference_video", None),
        start_seconds=getattr(args, "validation_reference_start_seconds", None),
        config_path=getattr(args, "validation_reference_config", None),
        disabled=getattr(args, "no_validation_reference_video", False))


def has_reference_override(args):
    return any(getattr(args, name, None) is not None for name in (
        "validation_reference_video", "validation_reference_start_seconds", "validation_reference_config")) or bool(
        getattr(args, "no_validation_reference_video", False))


def _probe_video(encoder, path):
    result = subprocess.run([str(encoder), "-hide_banner", "-i", str(path),
        "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    message = result.stderr.decode("utf-8", errors="replace")
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", message)
    size = re.search(r"Video: [^\n]*?\b(\d{2,6})x(\d{2,6})\b", message)
    if result.returncode or duration is None or size is None:
        raise ValueError(f"Cannot read video stream/duration: {path}\n{message[-1500:]}")
    hours, minutes, seconds = map(float, duration.groups())
    return dict(duration_s=hours*3600+minutes*60+seconds,
                width=int(size[1]), height=int(size[2]))


def compose_reference_video(validation, reference, *, ffmpeg, output=None, emit=print, output_height=None):
    """PMX left, untouched validation view right; validation audio/clock is master."""
    validation = Path(validation).resolve()
    movie = _path(reference["video"])
    if not movie.is_file():
        raise FileNotFoundError(movie)
    meta = json.loads(validation.with_suffix(".json").read_text(encoding="utf-8-sig"))
    fps = float(meta["fps"])
    duration = float(meta["video_duration_s"])
    source_start = float(meta.get("source_start_s", 0.))
    rate = float(meta.get("playback_rate", 1.))*int(meta.get("source_frame_stride", 1))
    offset = float(reference.get("start_seconds", 0.))
    if (not all(math.isfinite(v) for v in (fps, duration, source_start, rate, offset)) or
            min(fps, duration, rate) <= 0 or min(source_start, offset) < 0):
        raise ValueError("Invalid reference/validation video clock")
    val_info = _probe_video(ffmpeg, validation)
    ref_info = _probe_video(ffmpeg, movie)
    if offset >= ref_info["duration_s"]:
        raise ValueError("Reference start offset is past the end of the MP4")
    # Pad before selecting a tail/preview window. If its start is past the PMX
    # end, this selects the true final frame instead of replaying the last .1s.
    seek = offset
    trim_start = source_start/rate
    # One extra endpoint frame lets FFmpeg's framesync flush the final real
    # validation sample; output -t still uses the exact original duration.
    padded_duration = duration+1/fps
    if output_height is not None and (output_height < 2 or output_height % 2):
        raise ValueError("Comparison height must be a positive even integer")
    height = output_height or val_info["height"]//2*2
    validation_width = max(2, round(val_info["width"]*height/val_info["height"]/2)*2)
    width = max(2, round(height*8/9)//2*2)
    output = Path(output).resolve() if output else validation.with_name(validation.stem+"_comparison.mp4")
    if output in (validation, movie):
        raise ValueError("Comparison output must differ from both input videos")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+".partial.mp4")
    reference_title = ("Mutsumi PhysX reference (holds at end)" if reference.get("actor") == "若叶睦"
                       else "PMX reference (holds at end)")
    filters = (
        f"[0:v]setpts=(PTS-STARTPTS)/{rate:.12g},fps={fps:.12g},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={trim_start+padded_duration:.12g},"
        f"trim=start={trim_start:.12g}:duration={padded_duration:.12g},setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=42:color=0x111721:t=fill,"
        f"drawtext=text='{reference_title}':x=12:y=10:fontsize=22:fontcolor=white[ref];"
        f"[1:v]setpts=PTS-STARTPTS,fps={fps:.12g},setsar=1,"
        f"scale={validation_width}:{height},"
        f"tpad=stop_mode=clone:stop_duration={1/fps:.12g}[val];"
        f"[ref][val]hstack=inputs=2,fps={fps:.12g}:eof_action=pass[out]"
    )
    command = [str(ffmpeg), "-nostdin", "-y", "-ss", str(seek), "-i", str(movie),
        "-i", str(validation), "-filter_complex_threads", "2", "-filter_complex", filters,
        "-map", "[out]", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-threads", "4", "-pix_fmt", "yuv420p", "-r", str(fps), "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration), "-movflags", "+faststart", str(temporary)]
    log = output.with_suffix(".encoder.log")
    emit(f"[PMX对照录像] 合成参考视频与验收录像：{output}")
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while True:
                    try:
                        code = process.wait(timeout=30.)
                        break
                    except subprocess.TimeoutExpired:
                        emit(f"[PMX对照录像] 仍在合成，详情：{log}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        if code != 0 or not temporary.is_file():
            raise RuntimeError(f"PMX video composition failed (exit {code}); inspect {log}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(video=str(output), validation_video=str(validation), reference=reference,
        reference_seek_s=seek, source_start_s=source_start, source_playback_rate=rate,
        fps=fps, video_duration_s=duration, width=width+validation_width, height=height,
        layout="PMX left; original multi-seed validation right (aspect ratios preserved)",
        reference_end="hold last frame", audio_source="validation video only",
        method="Video composition only; no policy inference or acceptance changes", encoder_log=str(log))
    sidecar = output.with_suffix(".json")
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)
    emit(f"[PMX对照录像已保存] {output}")
    return result
