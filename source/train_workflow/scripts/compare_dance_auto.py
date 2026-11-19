"""Prepare a validated G1 training reference H5 from a VMD clip and optionally record a review video."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
PMX_SOURCE = ROOT/'source/pmx_prototype'
ROBOT = ROOT/'assets/g1_29dof_o6_hand_V3.usd'
ROBOT_GEOMETRY = ROOT/'assets/g1_29dof_o6_hand_V2.usdz'
ANON = ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'
FPS = 30
LOOPBACK_PROBE = 'import socket; socket.setdefaulttimeout(3); a,b=socket.socketpair(); a.close(); b.close()'

sys.path.insert(0, str(ROOT))
from source.paths import resolve_dance_asset


class Run:
    def __init__(self, directory, accept_eula):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.logs = directory/'logs'
        self.logs.mkdir()
        self.reports = directory/'reports'
        self.reports.mkdir()
        self.env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
        if accept_eula:
            self.env['OMNI_KIT_ACCEPT_EULA'] = 'YES'
        self.stage = 'preflight'
        self.started = self.stage_started = time.monotonic()
        self.phases = {}
        self.data = {}

    def status(self, state, **details):
        path = self.directory/'result.json'
        temporary = path.with_suffix('.tmp')
        record = dict(state=state, stage=self.stage, seconds_elapsed=round(time.monotonic()-self.started, 3),
                      completed_stage_seconds=self.phases, **self.data, **details)
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def phase(self, stage):
        now = time.monotonic()
        self.phases[self.stage] = round(now-self.stage_started, 3)
        self.stage, self.stage_started = stage, now
        self.status('running')
        print(f'[dance] {stage}  {self.directory}', flush=True)

    def progress(self, name):
        """Update a lightweight window receipt independently of solver stdout."""
        def write(value):
            path = self.directory/'progress.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(stage=self.stage, event=name, window=value,
                                                elapsed_seconds=round(time.monotonic()-self.started, 3)), indent=2),
                                 encoding='utf-8')
            temporary.replace(path)
        return write

    def report(self, name, data):
        path = self.reports/(name+'.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        return path

    def command(self, name, *args, timeout):
        log_path = self.logs/(name+'.log')
        command = [str(arg) for arg in args]
        print(f'[dance] {name}  {log_path}', flush=True)
        with log_path.open('xb') as log:
            try:
                subprocess.run(command, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, check=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f'{name} exceeded {timeout:g}s; see {log_path}') from error
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'{name} exited with code {error.returncode}; see {log_path}') from error
        return log_path

    def python(self, name, *args, timeout):
        return self.command(name, sys.executable, '-u', *args, timeout=timeout)


def select_audio(vmd, explicit):
    if explicit:
        path = resolve_dance_asset(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for extension in ('.wav', '.flac', '.mp3', '.m4a', '.ogg'):
        candidate = vmd.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def import_dance(run, vmd, seconds, leg_mode):
    # The established PMX entry points use imports relative to this directory.
    sys.path.insert(0, str(PMX_SOURCE))
    try:
        from load_vmd import read_motion
    finally:
        sys.path.remove(str(PMX_SOURCE))
    tracks, switches, _ = read_motion(vmd)
    source_count = max(max(frames) for frames in tracks.values())+1
    count = round(min(seconds, source_count/FPS)*FPS)
    if count < 9:
        raise ValueError('Source clip must contain at least nine frames at 30 fps')
    foot_keys = {key: len(tracks.get(key, {})) for key in ('左足IK', '右足IK')}
    requested_mode = leg_mode
    if leg_mode == 'auto':
        leg_mode = 'ik' if max(foot_keys.values()) > 1 else 'fk'
    source = run.directory/'input'/'source.usdc'
    asset = source.parent/'character'
    run.data.update(frames=count, fps=FPS, source_frames=source_count, duration_seconds=count/FPS,
                    source_leg_mode=leg_mode)
    run.report('source_clock', dict(frames=count, full_source_frames=source_count, selected_leg_mode=leg_mode,
                                   requested_mode=requested_mode, foot_ik_key_counts=foot_keys,
                                   source_ik_switch_channels=sorted(switches)))
    run.python('convert_anon', PMX_SOURCE/'convert_pmx.py', ANON, '--output', asset, timeout=120)
    run.python('bake_vmd', PMX_SOURCE/'load_vmd.py', '--vmd', vmd, '--pmx', ANON,
               '--asset', asset/'character.usdc', '--output', source, '--leg-mode', leg_mode,
               '--max-frames', count, timeout=180+count)
    report = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if report['frames'] != count or report['leg_mode'] != leg_mode:
        raise RuntimeError('PMX import did not match requested frame count or source mode')
    csv = source.parent/'input.csv'
    run.python('vmd_csv', ROOT/'source/train_workflow/scripts/vmd_2_csv.py', vmd, '--output', csv, timeout=180)
    return source, csv, count, leg_mode


def retarget(run, csv, source, count, leg_mode):
    import numpy as np
    from source.train_workflow.utils.format.csv_loader import load_csv_motion, get_frame_indices, get_bone_frame_lists
    from source.train_workflow.utils.grounding import pipeline
    from source.train_workflow.utils.grounding.contact import project_contact, robot_sole_vertices
    from source.train_workflow.utils.grounding.ik import source_contact_report
    from source.train_workflow.utils.grounding.motion import read_motion, write_motion
    from source.train_workflow.utils.grounding.smoothing import GroundProblem
    from source.train_workflow.utils.motion.joint_constraints import constrain_joint_positions, validate_joint_positions
    from source.train_workflow.utils.motion.self_collision import playback_baseline
    from source.train_workflow.utils.playback.recorder import PlaybackH5Recorder
    from source.train_workflow.utils.playback.targets import (MotionRootTrackState, PlaybackUiDebugState,
                                                            compute_targets_for_motion_frame)
    from source.train_workflow.utils.retarget.upper_body_export import solve_upper_body_arrays
    from source.train_workflow.utils.retarget.upper_body_ik import read_joints

    names = [j.name for j in read_joints(ROBOT) if j.axis is not None]
    baseline = playback_baseline(names, ROOT/'source/my_task/g1_replay_env_cfg.py')
    frames = load_csv_motion(str(csv))
    indices = get_frame_indices(frames)
    bones = set().union(*(frame.keys() for frame in frames.values()))
    bone_frames = get_bone_frame_lists(frames, indices, bones)
    anchor_pos = (0, 0, .76)
    anchor_q = [2**-.5, 0, 0, 2**-.5]
    root_state = MotionRootTrackState(anchor_pos, anchor_q)
    debug = PlaybackUiDebugState()
    recorder = PlaybackH5Recorder.begin(source_csv=str(csv), runtime_joint_names=names,
                                       compile_joint_names=names, baseline_joint_pos=baseline,
                                       root_anchor_pos=anchor_pos, root_anchor_quat_wxyz=anchor_q,
                                       max_frame=count-1, has_hand_data=True)
    for frame in range(count):
        q, pos, quat, _, _, _ = compute_targets_for_motion_frame(
            frame, frames, bone_frames, bones, names, baseline, 1., .1, None, root_state, debug)
        recorder.record_frame(frame, q, root_pos=anchor_pos if pos is None else pos,
                              root_quat_wxyz=anchor_q if quat is None else quat)
    raw = run.directory/'input'/'legacy_channels.h5'
    recorder.write(str(raw), retarget_mode='legacy')
    motion = read_motion(raw)
    delta, upper = solve_upper_body_arrays(motion.delta, names, baseline, FPS, str(source), ROBOT)
    run.report('upper_body_ik', upper)
    q, limits = constrain_joint_positions(delta+baseline, names, FPS, ROBOT)
    run.report('ik_seed_joint_limits', limits)
    upper_path = raw.with_name('upper_ik.h5')
    write_motion(motion, upper_path, q-baseline, attributes=dict(
        retarget_mode='new', upper_body_retarget_method=upper['method'], source_leg_mode=leg_mode))
    seed = read_motion(upper_path)

    run.phase('grounded_ik')
    folder = run.directory/'grounding'
    targets = pipeline.prepare(upper_path, source, ROBOT, folder, source_leg_mode=leg_mode,
                               character='anon', minimum_leg_skin_weight=.99)
    q = seed.q.copy()
    for side in ('left', 'right'):
        pipeline.solve(folder, side)
        with np.load(folder/f'{side}_solved.npz') as solved:
            q[:, solved['cols']] = solved['q']
    desired = np.column_stack([targets['left_desired'], targets['right_desired']])
    soles = robot_sole_vertices(ROBOT)
    root, orientation = targets['root_pos'], targets['root_rot']
    corrected, adjusted_root, projection = project_contact(q, names, FPS, root, orientation, desired, ROBOT, soles)
    run.report('sole_projection', projection)
    if not projection['passed']:
        raise RuntimeError('Visible-sole projection failed acceptance; see sole_projection.json')
    path = folder/'fixed_ik.h5'
    write_motion(seed, path, corrected-baseline, root_pos=adjusted_root, root_rot=orientation, attributes=dict(
        lower_body_retarget_method='semantic-sole-visible-contact-ik', collision_optimization_applied=False,
        without_collision_validated=False))
    reference = read_motion(path)
    contact_problem = GroundProblem(reference, ROBOT, sole_vertices=soles, desired=desired)
    report = source_contact_report(contact_problem.reference_heights[:count], desired)
    contact = (desired[contact_problem.lo] == 0) & (desired[contact_problem.hi] == 0)
    visible = contact_problem.reference_heights[contact]
    report.update(joint_constraints=validate_joint_positions(reference.q, names, FPS, ROBOT),
                  visible_contact_samples=int(contact.sum()),
                  visible_contact_range_m=[float(visible.min()), float(visible.max())] if visible.size else None)
    run.report('grounded_ik', report)
    if (not report['joint_constraints']['passed'] or report['false_floating_on_source_stance'] or
            report['source_flight_frames'] != report['source_flight_retained'] or
            np.any(visible < -.0003-3e-7) or np.any(visible > .003+3e-7)):
        raise RuntimeError('Encoded grounded IK failed joint, stance or source-flight acceptance')
    return reference, desired, soles


def select_device(request):
    import torch

    torch.set_num_threads(4)
    if request == 'cpu' or (request == 'auto' and not torch.cuda.is_available()):
        return 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was selected, but this isaac61 Python cannot use a CUDA device')
    free, total = torch.cuda.mem_get_info(0)
    if request == 'auto' and free < 2.5*1024**3:
        return 'cpu'
    torch.cuda.set_per_process_memory_fraction(min(.20, max(.05, (free-1024**3)/total)), 0)
    return 'cuda:0'


def publish_training_motion(run, reference, baseline, final, input_hashes):
    import numpy as np
    from source.train_workflow.utils.format.hdf5 import load_hdf5_motion
    from source.train_workflow.utils.grounding.motion import sha256

    run.phase('publish_h5')
    for original, expected in input_hashes.items():
        if sha256(original) != expected:
            raise RuntimeError(f'Input changed during motion preparation: {original}')
    pending = run.directory/'delivery_pending'
    pending.mkdir()
    outputs = {'fixed_ik': reference.path, 'collision_baseline': baseline, 'arm_stabilized': final}
    for path in outputs.values():
        shutil.copyfile(path, pending/path.name)
    hashes = {key: sha256(pending/path.name) for key, path in outputs.items()}
    for stage in ('collision_baseline', 'arm_stabilized'):
        evidence = json.loads((run.reports/f'{stage}_validation.json').read_text(encoding='utf-8'))
        if hashes[stage] != evidence['sha256']:
            raise RuntimeError(f'{stage} changed after motion validation')

    motion = load_hdf5_motion(str(pending/final.name))
    if (motion.fps != reference.fps or motion.joint_names != reference.names or
            motion.joint_pos_delta.shape != reference.delta.shape or
            not np.array_equal(motion.frames, np.arange(len(reference.delta))) or not motion.root_valid.all()):
        raise RuntimeError('Validated H5 failed the training motion format check')

    delivery = run.directory/'deliverables'
    pending.rename(delivery)
    run.data.update(deliverables={key: str(delivery/path.name) for key, path in outputs.items()},
                    deliverable_sha256=hashes, training_h5=str(delivery/final.name), training_h5_ready=True)
    run.status('running')
    print(f'[dance] Training reference ready  {delivery/final.name}', flush=True)
    return delivery/baseline.name, delivery/final.name


def record_review_video(run, args, vmd, source, baseline, final, count):
    from source.train_workflow.g1_vmd_pipeline import find_ffmpeg
    from source.train_workflow.utils.grounding.dance_media import LABELS, bake_anon, record_comparison
    from source.train_workflow.utils.grounding.motion import sha256

    run.data.update(video_state='running', comparison_order=LABELS)
    run.phase('video_preflight')
    audio = select_audio(vmd, args.audio)
    run.data['audio'] = str(audio) if audio else None
    media_hashes = {str(ANON): run.data['input_sha256'][str(ANON)], str(source): sha256(source)}
    if audio:
        media_hashes[str(audio)] = sha256(audio)
    if importlib.util.find_spec('isaacsim') is None:
        raise RuntimeError('Missing isaacsim; record from the project Isaac Sim 6.1 environment')
    find_ffmpeg()
    try:
        run.python('isaac_loopback', '-c', LOOPBACK_PROBE, timeout=6)
    except RuntimeError as error:
        raise RuntimeError('Isaac Sim local loopback failed; run in the local isaac61 terminal. '
                           f'See {run.logs / "isaac_loopback.log"}') from error
    run.phase('bake_anon')
    cache = bake_anon(run, source, ANON, count)
    run.phase('record_comparison')
    video = record_comparison(run, baseline, final, cache, audio, count, args.width, args.height)
    for original, expected in media_hashes.items():
        if sha256(original) != expected:
            raise RuntimeError(f'Input changed during video rendering: {original}')
    destination = final.parent/video.name
    pending = destination.with_suffix('.tmp')
    shutil.copyfile(video, pending)
    video_hash = sha256(pending)
    pending.rename(destination)
    run.data['deliverables']['video'] = str(destination)
    run.data['deliverable_sha256']['video'] = video_hash
    if audio:
        run.data['input_sha256'][str(audio)] = media_hashes[str(audio)]
    run.data.update(video_state='complete', pmx_cache=str(cache), full_video_decode=True, audio_decoded=audio is not None)


def execute(run, args, vmd):
    from source.train_workflow.utils.grounding.dance_optimize import optimize
    from source.train_workflow.utils.grounding.motion import sha256
    from source.train_workflow.utils.retarget.upper_body_ik import SourceSkeleton

    if not vmd.is_file() or vmd.suffix.casefold() != '.vmd':
        raise FileNotFoundError(f'Expected an existing .vmd source: {vmd}')
    hashes = {str(path): sha256(path) for path in (vmd, ANON, ROBOT, ROBOT_GEOMETRY)}
    run.data.update(vmd=str(vmd), input_sha256=hashes, retries=args.retries)
    for module in ('fcl', 'clarabel'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f'Missing {module}; launch from the project Isaac Sim 6.1 environment')
    device = select_device(args.device)
    run.data['solver_device'] = device
    run.phase('import')
    source, csv, count, leg_mode = import_dance(run, vmd, args.seconds, args.leg_mode)
    run.phase('retarget')
    reference, desired, soles = retarget(run, csv, source, count, leg_mode)
    baseline, final = optimize(run, reference, SourceSkeleton(source), desired, soles, ROBOT, device, args.retries)
    baseline, final = publish_training_motion(run, reference, baseline, final, hashes)
    if not args.no_video:
        if device != 'cpu':
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        record_review_video(run, args, vmd, source, baseline, final, count)
    run.phases[run.stage] = round(time.monotonic()-run.stage_started, 3)
    run.stage = 'complete'
    run.status('complete')
    print(f'[dance] Complete  {run.directory / "result.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vmd', type=Path, required=True)
    parser.add_argument('--audio', type=Path, help='Audio override; by default use same-stem audio next to the VMD')
    parser.add_argument('--output-dir', type=Path, help='New directory for the run and its evidence')
    parser.add_argument('--seconds', type=float, default=90, help='Maximum duration at 30 fps, starting at frame 0')
    parser.add_argument('--leg-mode', choices=('auto', 'ik', 'fk'), default='auto', help='Source PMX foot mode')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--retries', type=int, default=3, help='Maximum retries per optimization stage')
    parser.add_argument('--no-video', action='store_true', help='Finish after publishing the validated training reference H5')
    parser.add_argument('--width', type=int, default=2400)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--accept-eula', action='store_true', help='Confirm acceptance of the installed Isaac Sim EULA')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < .3:
        parser.error('--seconds must be at least 0.3')
    if args.retries < 0:
        parser.error('--retries must be nonnegative')
    if not args.no_video and (min(args.width, args.height) <= 0 or args.width % 2 or args.height % 2):
        parser.error('Video dimensions must be positive even numbers')
    vmd = resolve_dance_asset(args.vmd)
    directory = args.output_dir or ROOT/'output/dance_auto'/f'{vmd.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    try:
        run = Run(directory.resolve(), args.accept_eula)
    except OSError as error:
        parser.exit(1, f'Could not create a fresh run directory: {error}\n')
    run.data.update(training_h5_ready=False, video_state='skipped' if args.no_video else 'not_started')
    run.status('running')
    try:
        execute(run, args, vmd)
    except (Exception, KeyboardInterrupt) as error:
        log = run.logs/'failure.log'
        log.write_text(traceback.format_exc(), encoding='utf-8')
        if run.data['video_state'] == 'running':
            run.data['video_state'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        run.status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error=str(error) or type(error).__name__, error_type=type(error).__name__, failure_log=str(log))
        print(f'[dance] Failed at {run.stage}. {error}\nSee {run.directory / "result.json"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
