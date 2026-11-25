"""Render complete grouped acceptance rollouts with one worker per active group."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT, SOURCE_DIR
from diagnostics.anon_acceptance_video import FULL_VIDEO_NAME, full_acceptance_chapters


RECORDER = Path(__file__).with_name('record_gait_diagnostic.py')
DEFAULTS = SOURCE_DIR / 'configs/recording/anon_acceptance_video.json'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
FPS = 30
RESOLUTION = [1920, 1080]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def recording_settings(report):
    settings = read(DEFAULTS) if DEFAULTS.is_file() else {}
    checkpoint = report.parent.parent
    match = re.fullmatch(r'checkpoint_(\d+)', checkpoint.name)
    trial = None
    if match:
        experiment = checkpoint.parent / 'video_recording.json'
        if experiment.is_file():
            local = read(experiment)
            settings.update(local)
            trial = local.get('trials', {}).get(str(int(match.group(1))))
    workers = settings.get('workers', 1) if trial is None else trial
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError('Recording worker count must be 1, 2, or 4')
    return workers, trial is not None


def grouped_chapters(rollout, report):
    with np.load(rollout) as trace:
        starts = trace['sequence_start_frames'] if 'sequence_start_frames' in trace else [0]
        chapters = full_acceptance_chapters(trace['labels'], report['segments'], FPS, starts)
    return [[chapter for chapter in chapters if chapter['sequence'] == group]
            for group in range(1, len(starts) + 1)]


def recorder_command(args, output, sequence=None, threads=None, audit=False, cache=None):
    command = [sys.executable, '-u', str(RECORDER), '--rollout', str(args.rollout), '--acceptance-report',
               str(args.acceptance_report), '--output', str(output), '--title', args.title, '--headless']
    if sequence is not None:
        command += ['--sequence', str(sequence), '--worker-threads', str(threads)]
    if audit:
        command.append('--audit-physics')
    if cache is not None:
        command += ['--secondary-cache', str(cache)]
    return command


def check_render(output, chapters, rollout_hash, sequence=None):
    metadata = read(output / 'video_report.json')
    frames = sum(chapter['source_frame_end'] - chapter['source_frame_start'] for chapter in chapters)
    video = output / FULL_VIDEO_NAME
    if (metadata['chapters'] != chapters or metadata['coverage'] != ('full' if sequence is None else 'sequence')
            or metadata.get('sequence') != sequence or metadata['rollout_sha256'] != rollout_hash
            or metadata['frames'] != frames or metadata['fps'] != FPS or metadata['resolution'] != RESOLUTION
            or not metadata['secondary_physics'] or Path(metadata['video']).resolve() != video.resolve()
            or not video.is_file() or video.stat().st_size == 0):
        raise ValueError(f'Incomplete or mismatched recorded acceptance at {output}')
    return metadata


def run_serial(args, output, chapters, rollout_hash, audit=False):
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    frame_count = chapters[-1]['source_frame_end']
    with (output / 'serial.log').open('w', encoding='utf-8') as log:
        subprocess.run(recorder_command(args, output, audit=audit), cwd=PROJECT_ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, creationflags=NO_WINDOW,
                       timeout=max(300, 60 + 6 * frame_count / FPS))
    metadata = check_render(output, chapters, rollout_hash)
    return dict(metadata, wall_seconds=perf_counter() - started)


def ffmpeg_path():
    return shutil.which('ffmpeg') or str(next((Path.home() / 'AppData/Roaming/Python').glob(
        'Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))


def join_groups(directory, metadata, expected_frames):
    concat = directory / 'concat.txt'
    concat.write_text(''.join(f"file group_{item['sequence']:02d}/{FULL_VIDEO_NAME}\n" for item in metadata),
                      encoding='ascii')
    ffmpeg = ffmpeg_path()
    target = directory / FULL_VIDEO_NAME
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-f', 'concat', '-safe', '1', '-i', str(concat),
                    '-map', '0:v:0', '-c', 'copy', '-movflags', '+faststart', str(target)],
                   check=True, cwd=directory, creationflags=NO_WINDOW, timeout=120)
    decode = subprocess.run([ffmpeg, '-v', 'info', '-xerror', '-nostdin', '-i', str(target),
                             '-map', '0:v:0', '-f', 'null', '-'],
                            check=True, capture_output=True, text=True, errors='replace',
                            creationflags=NO_WINDOW, timeout=120)
    frame_counts = re.findall(r'frame=\s*(\d+)', decode.stderr)
    video = next((line for line in decode.stderr.splitlines() if 'Video:' in line), '')
    if (not frame_counts or int(frame_counts[-1]) != expected_frames
            or '1920x1080' not in video or '30 fps' not in video):
        raise ValueError('Merged video failed full-frame decoding or has the wrong format')
    return target


def run_parallel(args, directory, groups, rollout_hash, workers, audit=False):
    directory.mkdir()
    started = perf_counter()
    total_frames = groups[-1][-1]['source_frame_end']
    budget = max(240, 60 + 3 * total_frames / FPS)
    cache = directory / 'secondary_cache'
    with (directory / 'bake.log').open('w', encoding='utf-8') as log:
        command = recorder_command(args, cache) + ['--bake-secondary']
        subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, creationflags=NO_WINDOW, timeout=budget)
    baked = read(cache / 'secondary_cache.json')
    if baked['frames'] != total_frames or baked['rollout_sha256'] != rollout_hash:
        raise ValueError('Baked secondary motion differs from the measured trajectory')
    bake_seconds = perf_counter() - started
    print(f'PARALLEL_VIDEO_BAKE {total_frames} frames {bake_seconds:.2f} seconds', flush=True)
    deadline = started + budget
    threads = max(1, min(8, (os.cpu_count() or 16) // workers))
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    running, finished = {}, {}
    next_group = 1
    try:
        while next_group <= len(groups) or running:
            while len(running) < workers and next_group <= len(groups):
                number = next_group
                output = directory / f'group_{number:02d}'
                output.mkdir()
                with (output / 'record.log').open('w', encoding='utf-8') as log:
                    worker_started = perf_counter()
                    process = subprocess.Popen(recorder_command(args, output, number, threads, audit, cache),
                                               cwd=PROJECT_ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
                running[number] = (process, output, worker_started)
                next_group += 1
            for number, (process, output, worker_started) in list(running.items()):
                result = process.poll()
                if result is None:
                    continue
                if result:
                    raise RuntimeError(f'Group {number} exited with {result}; see {output / "record.log"}')
                data = check_render(output, groups[number - 1], rollout_hash, number)
                finished[number] = dict(data, worker_wall_seconds=perf_counter() - worker_started)
                del running[number]
                print(f'PARALLEL_VIDEO_GROUP {number}/{len(groups)} {data["frames"]} frames', flush=True)
            if (next_group <= len(groups) or running) and perf_counter() >= deadline:
                raise TimeoutError('Parallel acceptance video exceeded the rendering budget')
            if running:
                sleep(.2)
    finally:
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        for process, _, _ in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    ordered = [finished[number] for number in range(1, len(groups) + 1)]
    frames = sum(row['frames'] for row in ordered)
    for key in ('secondary_profile', 'appearance', 'secondary_physics', 'secondary_cache'):
        if any(row[key] != ordered[0][key] for row in ordered):
            raise ValueError(f'Grouped render differs in {key}')
    video = join_groups(directory, ordered, frames)
    return dict(ordered[0], video=str(video), sequence=None, coverage='full', frames=frames,
                duration_seconds=frames / FPS, chapters=[chapter for group in groups for chapter in group],
                wall_seconds=perf_counter() - started, bake_seconds=bake_seconds, worker_count=workers, groups=ordered)


def physics_comparison(serial, parallel):
    keys = ('frames', 'positions', 'rotations')
    with np.load(serial / 'physics_samples.npz') as data:
        original = {key: data[key] for key in keys}
    parts = []
    for directory in sorted(parallel.glob('group_*')):
        with np.load(directory / 'physics_samples.npz') as data:
            parts.append({key: data[key] for key in keys})
    grouped = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    np.testing.assert_array_equal(original['frames'], grouped['frames'])
    np.testing.assert_equal(original['positions'].shape, grouped['positions'].shape)
    np.testing.assert_equal(original['rotations'].shape, grouped['rotations'].shape)
    first = original['rotations'].astype(np.float64)
    second = grouped['rotations'].astype(np.float64)
    for field in ('positions', 'rotations'):
        if not np.isfinite(original[field]).all() or not np.isfinite(grouped[field]).all():
            raise ValueError(f'Invalid sampled skeleton {field}')
    for rotations in (first, second):
        norms = np.linalg.norm(rotations, axis=-1, keepdims=True)
        if (norms < 1e-6).any():
            raise ValueError('Invalid zero-length skeleton quaternion')
        rotations /= norms
    difference = np.minimum(np.linalg.norm(first - second, axis=-1), np.linalg.norm(first + second, axis=-1))
    angular_gap = np.rad2deg(4 * np.arcsin(np.clip(difference / 2, 0., 1.))).max()
    displacement = np.linalg.norm(original['positions'] - grouped['positions'], axis=-1).max() * 1000
    return dict(samples=len(original['frames']), max_local_translation_mm=float(displacement),
                max_local_rotation_deg=float(angular_gap), passed=bool(displacement <= .5 and angular_gap <= .1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('rollout', 'acceptance-report', 'output'):
        parser.add_argument('--' + flag, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--title', required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    report = read(args.acceptance_report)
    groups = grouped_chapters(args.rollout, report)
    chapters = [chapter for group in groups for chapter in group]
    data_hash = digest(args.rollout)
    workers, trial = recording_settings(args.acceptance_report)
    if not report.get('complete_scope') or report.get('acceptance_environments') != 4 or len(groups) != 4:
        workers, trial = 1, False
    profile = dict(requested_workers=workers, trial=trial, started_at=datetime.now().astimezone().isoformat(),
                   dispatcher_sha256=digest(Path(__file__)), recorder_sha256=digest(RECORDER))
    parallel = None
    if workers > 1:
        try:
            parallel = run_parallel(args, args.output / 'parallel', groups, data_hash, workers, audit=trial)
            profile['parallel_seconds'] = parallel['wall_seconds']
            profile['secondary_bake_seconds'] = parallel['bake_seconds']
        except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError, StopIteration, ValueError, KeyError) as error:
            profile['parallel_error'] = str(error)
            profile['parallel_attempt_seconds'] = perf_counter() - started
            print('PARALLEL_VIDEO_RETRY_SERIAL ' + str(error), flush=True)
    serial = None
    if parallel is None or trial:
        serial = run_serial(args, args.output / 'serial', chapters, data_hash, audit=bool(parallel and trial))
        profile['serial_seconds'] = serial['wall_seconds']
    if parallel and trial:
        try:
            profile['physics_comparison'] = physics_comparison(args.output / 'serial', args.output / 'parallel')
        except (OSError, ValueError, AssertionError) as error:
            profile['physics_comparison'] = dict(passed=False, error=str(error))
        profile['speedup'] = serial['wall_seconds'] / parallel['wall_seconds']
        profile['trial_passed'] = profile['physics_comparison']['passed'] and profile['speedup'] > 1.
    elif trial:
        profile['trial_passed'] = False
    use_parallel = parallel is not None and (not trial or profile['trial_passed'])
    final = parallel if use_parallel else serial
    profile['selected_mode'] = 'parallel' if use_parallel else 'serial'
    target = args.output / FULL_VIDEO_NAME
    shutil.copyfile(final['video'], target)
    output = dict(final, video=str(target), recording=profile, wall_seconds=perf_counter() - started)
    write(args.output / 'video_report.json', output)
    print('ACCEPTANCE_RECORDING_MODE ' + json.dumps(profile), flush=True)


if __name__ == '__main__':
    main()
