"""VMD -> in-memory targets -> H5 -> optional collision repair -> video.

Default: general upper/lower-body retargeting, FK source legs, one H5 write.
Use --retarget_mode legacy to retain the original capture chain for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from source.paths import DANCE_DIR, resolve_dance_asset


def validate_final_joint_limits(path):
    import h5py
    from source.train_workflow.utils.motion.joint_constraints import validate_joint_positions
    with h5py.File(path) as stream:
        names=list(stream['joint_names'].asstr()[:])
        absolute=stream['joint_pos_delta'][:]+stream.attrs['record_baseline_joint_pos']
        fps=float(stream.attrs['fps'])
    return validate_joint_positions(absolute,names,fps,ROOT/'assets/g1_29dof_o6_hand_V3.usd')


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
            paths["h5"].with_name(paths["h5"].stem + "_upper.h5"),
            paths["h5"].with_suffix(".upper_body.json"), paths["raw"].with_suffix(".upper_body.json"),
            paths["h5"].with_suffix(".lower_body.json"), paths["raw"].with_suffix(".lower_body.json"),
            paths["h5"].with_suffix(".joint_limits.json"), paths["raw"].with_suffix(".joint_limits.json"),
            paths["h5"].with_suffix(".candidate.npz"), paths["h5"].with_suffix(".motion.json"),
            paths["raw"].with_suffix(".motion.json")]
        if not any(path.exists() for path in occupied):
            return paths
        version += 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmd", required=True, type=Path)
    parser.add_argument('--retarget_mode',choices=('new','legacy'),default='new')
    parser.add_argument('--source_leg_mode',choices=('fk','ik'),default='fk',
                        help='Default joint-driven FK; explicitly select ik for foot-IK-driven VMDs')
    parser.add_argument("--audio", type=Path, help="Music file; defaults to same-stem WAV/OGG/MP3/FLAC/M4A")
    parser.add_argument("--collision_correction", action="store_true", help="Repair self collisions and spikes; publish only if validation passes")
    parser.add_argument('--collision_mode',choices=('local','gesture'),default='local',help='Default: bounded local avoidance; gesture enables expensive full-motion refinement')
    parser.add_argument('--collision_max_seconds',type=float,default=60.,help='Local posture-restoration budget; never skips collision/mesh validation')
    parser.add_argument("--upper_body_source_animation", type=Path,
                        help="Opt in to general bounded upper-body IK using a PMX UsdSkel baked from this same VMD")
    parser.add_argument("--lower_body_source_animation", type=Path,
                        help="Opt in to contact-aware lower-body IK before the first H5 write")
    parser.add_argument("--output_dir", type=Path, help="CSV/H5 directory (default: source dance folder)")
    parser.add_argument("--name", help="Output stem; defaults to VMD_collision_corrected or VMD_rebuilt; repeats use _v2, _v3")
    parser.add_argument("--video_dir", type=Path, default=ROOT / "media/recordings")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--audio_volume", type=float, default=0.25,
                        help="Muxed music gain (0.25 = 25%%; default 0.25)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def dance_output_directory(vmd, output_dir=None):
    root = Path(DANCE_DIR)
    if output_dir is not None and Path(output_dir).resolve() != root:
        return Path(output_dir).resolve()
    return vmd.parent if vmd.is_relative_to(root) and vmd.parent != root else root / vmd.stem


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.collision_max_seconds) or args.collision_max_seconds<=0:
        parser.error('--collision_max_seconds must be finite and positive')
    vmd = resolve_dance_asset(args.vmd)
    if not vmd.is_file() or vmd.suffix.lower() != ".vmd":
        parser.error("--vmd must be an existing VMD file")
    upper_source = args.upper_body_source_animation.resolve() if args.upper_body_source_animation else None
    lower_source = args.lower_body_source_animation.resolve() if args.lower_body_source_animation else None
    if args.retarget_mode == 'legacy' and (upper_source or lower_source):
        parser.error('Legacy mode cannot use new limb source animations')
    for source in (upper_source, lower_source):
        if source is None:
            continue
        if not source.is_file():
            parser.error("Source animation must exist")
        source_metadata = source.with_suffix(".json")
        if source_metadata.is_file():
            source_vmd = json.loads(source_metadata.read_text(encoding="utf-8")).get("vmd")
            if source_vmd and resolve_dance_asset(source_vmd) != vmd:
                parser.error("PMX animation was baked from a different VMD")
    audio = resolve_dance_asset(args.audio) if args.audio else next(
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
    directory = dance_output_directory(vmd, args.output_dir)
    video_dir = args.video_dir.resolve()
    paths = select_output_paths(directory, video_dir, name, args.collision_correction)
    csv, raw, h5 = paths["csv"], paths["raw"], paths["h5"]
    avi, video, manifest = paths["avi"], paths["video"], paths["manifest"]
    directory.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    replay = ROOT / "source/train_workflow/g1_vmd_0_replay.py"
    scripts = ROOT / "source/train_workflow/scripts"
    run_step("1/5 VMD -> new CSV", [sys.executable, scripts / "vmd_2_csv.py", vmd, "--output", csv])
    capture_command = [sys.executable, replay,
        "--motion_csv", csv, "--num_envs", "1", "--pd_profile", "deploy",
        "--auto_record_h5", "--exit_after_h5_record", "--width", args.width, "--height", args.height]
    capture_command += ['--retarget_mode', args.retarget_mode, '--source_vmd', vmd,
                        '--source_leg_mode', args.source_leg_mode]
    if upper_source:
        capture_command += ["--upper_body_source_animation", upper_source]
    if lower_source:
        capture_command += ["--lower_body_source_animation", lower_source]
    run_step("2/5 Capture targets; solve limbs in memory; write H5" if args.retarget_mode == 'new' else
             "2/5 Legacy CSV capture -> H5", capture_command)
    if not raw.is_file():
        raise RuntimeError(f"H5 capture did not produce {raw}")
    if args.collision_correction:
        run_step("3/5 Collision/spike repair and validation", [sys.executable,
            scripts / "record_h5_without_collision.py", "--input", raw, "--output", h5,
            '--collision_mode',args.collision_mode,'--max_seconds',args.collision_max_seconds])
    else:
        print("[PIPELINE] 3/5 Collision correction disabled", flush=True)
    with h5py.File(h5) as stream:
        frames = len(stream["joint_pos_delta"])
        fps = float(stream.attrs["fps"])
        if args.collision_correction and not stream.attrs.get("without_collision_validated", False):
            raise RuntimeError("Collision-corrected H5 was not validated")
        arm_motion_validated=bool(stream.attrs.get('arm_motion_validated',False))
        local_continuity=bool(stream.attrs.get('collision_continuity_validated',False))
        local_mode=stream.attrs.get('collision_repair_mode')=='local'
        if args.collision_correction and not (local_continuity if local_mode else arm_motion_validated):
            raise RuntimeError('Collision-corrected H5 local continuity was not validated' if local_mode else
                               'Collision-corrected H5 arm motion was not validated')
        hand_geometry_mode=stream.attrs.get('collision_hand_geometry_mode')
        if args.collision_correction and hand_geometry_mode!='neutral-hand-box-and-articulated-finger-capsules-v1':
            raise RuntimeError('Collision-corrected H5 articulated hand geometry was not validated')
    if frames < 1 or abs(fps - 30.) > 1e-6:
        raise ValueError("Pipeline replay expects nonempty 30 FPS motion")
    joint_validation=validate_final_joint_limits(h5) if args.retarget_mode=='new' else None
    if joint_validation and not joint_validation['passed']:
        raise RuntimeError(f'Final H5 joint constraints failed: {joint_validation}')
    if joint_validation:
        print(f"[PIPELINE] All {joint_validation['joint_count']} joints passed angle/rate/mimic validation",flush=True)
    mesh_audit_path=None
    if args.collision_correction:
        mesh_directory=h5.parent/(h5.stem+'.hand-mesh-audit')
        run_step('3b/5 Full-motion visible hand mesh validation',[sys.executable,
            scripts/'audit_full_meshes.py','--pair-scope','hands','--h5',h5,'--output',mesh_directory,'--fail-on-intersection'])
        mesh_audit_path=mesh_directory/'mesh_scan.json'
        mesh_result=json.loads(mesh_audit_path.read_text(encoding='utf-8'))
        if (not mesh_result.get('passed') or mesh_result.get('h5_sha256')!=hashlib.sha256(h5.read_bytes()).hexdigest()
            or mesh_result.get('sample_count')!=4*(frames-1)+1 or mesh_result.get('sample_step_frames')!=.25):
            raise RuntimeError('Final H5 full-motion hand mesh validation failed')
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
    result = dict(vmd=str(vmd), audio=str(audio), csv=str(csv),
        raw_h5=str(raw) if args.collision_correction or args.retarget_mode == 'legacy' else None,
        retarget_chain="direct-limbs" if args.retarget_mode == 'new' else "legacy",
        retarget_mode=args.retarget_mode,source_leg_mode=args.source_leg_mode,
        h5=str(h5), collision_corrected=args.collision_correction,
        joint_constraints_validation=joint_validation,
        arm_motion_validated=arm_motion_validated if args.collision_correction else None,
        collision_mode=args.collision_mode if args.collision_correction else None,
        collision_continuity_validated=local_continuity if args.collision_correction else None,
        collision_hand_geometry_mode=hand_geometry_mode if args.collision_correction else None,
        hand_mesh_validation_report=str(mesh_audit_path) if mesh_audit_path else None,
        upper_body_source_animation=str(upper_source) if upper_source else None,
        lower_body_source_animation=str(lower_source) if lower_source else None,
        avi=str(avi), video=str(video), frames=frames, fps=fps,
        duration_seconds=frames/fps, audio_volume=float(args.audio_volume))
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[PIPELINE DONE]\nH5: {h5}\nVideo: {video}\nManifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
