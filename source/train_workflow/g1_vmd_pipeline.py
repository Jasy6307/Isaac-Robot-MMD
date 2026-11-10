"""VMD -> freshly captured H5 -> optional collision repair -> H5 video with music.

Run in the same Isaac Python environment as g1_vmd_0_replay.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def find_ffmpeg(explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, FileNotFoundError):
        pass
    # The binary is standalone: it can be shared with another local Python
    # installation without importing that installation's Python packages.
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates = sorted((Path(appdata) / "Python").glob(
            "Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe"))
        if candidates:
            return str(candidates[-1])
    raise RuntimeError("FFmpeg is required; install imageio-ffmpeg or pass --ffmpeg PATH")


def run_step(label, command):
    print(f"\n[PIPELINE] {label}", flush=True)
    subprocess.run([str(v) for v in command], cwd=ROOT, check=True)


def select_output_paths(directory, video_dir, base_name, corrected):
    """Version the entire output set together, preserving existing artifacts."""
    version = 1
    while True:
        name = base_name if version == 1 else f"{base_name}_v{version}"
        raw_stem = f"{name}_raw" if corrected else name
        paths = dict(csv=directory / f"{raw_stem}.csv",
            raw=directory / f"{raw_stem}.h5", h5=directory / f"{name}.h5",
            avi=video_dir / f"{name}.avi", video=video_dir / f"{name}.mp4",
            manifest=video_dir / f"{name}.pipeline.json")
        occupied = [*paths.values(), paths["h5"].with_suffix(".collision.json"),
            paths["h5"].with_suffix(".candidate.npz"), paths["h5"].with_suffix(".motion.json"),
            paths["raw"].with_suffix(".motion.json")]
        if not any(path.exists() for path in occupied):
            return paths
        version += 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmd", required=True, type=Path)
    parser.add_argument("--audio", type=Path, help="Music file; defaults to same-stem WAV/OGG/MP3/FLAC/M4A")
    parser.add_argument("--collision_correction", action="store_true", help="Repair self collisions and spikes; publish only if validation passes")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "media/dance", help="CSV/H5 directory (default media/dance)")
    parser.add_argument("--name", help="Output stem; defaults to VMD_collision_corrected or VMD_rebuilt; repeats use _v2, _v3")
    parser.add_argument("--video_dir", type=Path, default=ROOT / "media/recordings")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--audio_volume", type=float, default=0.25,
                        help="Muxed music gain (0.25 = 25%%; default 0.25)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    vmd = args.vmd.resolve()
    if not vmd.is_file() or vmd.suffix.lower() != ".vmd":
        parser.error("--vmd must be an existing VMD file")
    audio = args.audio.resolve() if args.audio else next(
        (vmd.with_suffix(ext) for ext in (".wav", ".ogg", ".mp3", ".flac", ".m4a") if vmd.with_suffix(ext).is_file()), None)
    if audio is None or not audio.is_file():
        parser.error("Music not found; provide --audio PATH")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("Video width and height must be positive even integers")
    if not math.isfinite(args.audio_volume) or args.audio_volume < 0.0 or args.audio_volume > 2.0:
        parser.error("--audio_volume must be finite and in [0, 2]")
    name = args.name or f"{vmd.stem}_{'collision_corrected' if args.collision_correction else 'rebuilt'}"
    if not name or name in (".", "..") or any(c in name for c in '<>:"/\\|?*') or name.endswith((" ", ".")):
        parser.error("--name must be a valid filename stem")
    ffmpeg = find_ffmpeg(args.ffmpeg)
    import h5py
    import cv2
    # Preflight before launching either simulator.
    run_step("Check FFmpeg", [ffmpeg, "-version"])
    directory, video_dir = args.output_dir.resolve(), args.video_dir.resolve()
    paths = select_output_paths(directory, video_dir, name, args.collision_correction)
    csv, raw, h5 = paths["csv"], paths["raw"], paths["h5"]
    avi, video, manifest = paths["avi"], paths["video"], paths["manifest"]
    directory.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    replay = ROOT / "source/train_workflow/g1_vmd_0_replay.py"
    scripts = ROOT / "source/train_workflow/scripts"
    run_step("1/5 VMD -> new CSV", [sys.executable, scripts / "vmd_2_csv.py", vmd, "--output", csv])
    run_step("2/5 Capture all CSV frames -> raw H5", [sys.executable, replay,
        "--motion_csv", csv, "--num_envs", "1", "--pd_profile", "deploy",
        "--auto_record_h5", "--exit_after_h5_record", "--width", args.width, "--height", args.height])
    if not raw.is_file():
        raise RuntimeError(f"H5 capture did not produce {raw}")
    if args.collision_correction:
        run_step("3/5 Collision/spike repair and validation", [sys.executable,
            scripts / "record_h5_without_collision.py", "--input", raw, "--output", h5])
    else:
        print("[PIPELINE] 3/5 Collision correction disabled", flush=True)
    with h5py.File(h5) as stream:
        frames = len(stream["joint_pos_delta"])
        fps = float(stream.attrs["fps"])
        if args.collision_correction and not stream.attrs.get("without_collision_validated", False):
            raise RuntimeError("Collision-corrected H5 was not validated")
    if frames < 1 or abs(fps - 30.) > 1e-6:
        raise ValueError("Pipeline replay expects nonempty 30 FPS motion")
    for motion in {raw, h5}:
        with motion.with_suffix(".motion.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(vmd=str(vmd), source_dance=vmd.stem, audio=str(audio),
                h5=str(motion)), stream, ensure_ascii=False, indent=2)
    run_step("4/5 Record final H5 frame by frame", [sys.executable, replay,
        "--motion_h5", h5, "--auto_play", "--record_avi_path", avi,
        "--exit_after_playback", "--num_envs", "1", "--pd_profile", "deploy",
        "--width", args.width, "--height", args.height])
    capture = cv2.VideoCapture(str(avi))
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != frames or abs(capture.get(cv2.CAP_PROP_FPS) - fps) > .01:
            raise RuntimeError("Recorded AVI frame count/FPS does not match final H5")
    finally:
        capture.release()
    run_step("5/5 Encode MP4 and mux music", [ffmpeg, "-nostdin", "-n", "-i", avi,
        "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-af", f"volume={args.audio_volume:.6g},apad", "-shortest",
        "-movflags", "+faststart", video])
    if not video.is_file() or not video.stat().st_size:
        raise RuntimeError("MP4 output is missing or empty")
    result = dict(vmd=str(vmd), audio=str(audio), csv=str(csv), raw_h5=str(raw),
        h5=str(h5), collision_corrected=args.collision_correction,
        avi=str(avi), video=str(video), frames=frames, fps=fps,
        duration_seconds=frames/fps, audio_volume=float(args.audio_volume))
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[PIPELINE DONE]\nH5: {h5}\nVideo: {video}\nManifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
