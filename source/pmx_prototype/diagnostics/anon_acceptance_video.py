"""Render videos from measured Anon acceptance rollouts after Isaac exits."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

from source_layout import PROJECT_ROOT, resolve_source


HEAD_TITLES = {
    'stand': 'Initial stand',
    'stop_after_walk': 'Walk and stop',
    'stop_after_pivot_left': 'Left pivot and stop',
    'stand_end': 'Right pivot and stop',
}


def acceptance_shots(labels, segments, fps):
    """Select up to four short excerpts from the actual scored sequence."""
    head_shots = []
    other_shots = []
    start = 0
    previous_label = None
    for segment in segments:
        end = start + round(segment['seconds'] * fps)
        label = segment['label']
        if end > len(labels) or end <= start or not all(value == label for value in labels[start:end]):
            raise ValueError('Acceptance report segments differ from the rollout')
        response = segment.get('response_steady', {})
        head = response.get('idle_head')
        body_status = 'passed' if segment.get('passed') else 'failed'
        body = response.get('body_status', body_status if 'passed' in segment else 'unscored')
        if label in HEAD_TITLES:
            begin = start if label == 'stand' else max(0, start - 2 * fps)
            finish = min(end, start + (4 if label == 'stand' else 5) * fps)
            title = HEAD_TITLES[label]
            if label == 'stand_end' and previous_label != 'pivot_right':
                title = 'Final stand'
            head_shots.append((begin, finish, title, head, body))
        else:
            title = label.replace('_', ' ').capitalize()
            other_shots.append((max(start, end - 5 * fps), end, title, head, body))
        start = end
        previous_label = label
    if start != len(labels):
        raise ValueError('Acceptance report does not cover the complete rollout')
    # A lite screen or a targeted check can omit some of the four head scenes.
    # Fill open slots from the measured movement cases, favoring failed ones.
    other_shots.sort(key=lambda shot: shot[4] == 'passed')
    shots = sorted((head_shots + other_shots)[:4], key=lambda shot: shot[0])
    if not shots:
        raise ValueError('Acceptance report contains no recorded scenes')
    return shots


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def _video_directory(acceptance_dir):
    attempt = 0
    while True:
        suffix = f'_{attempt:02d}' if attempt else ''
        output = acceptance_dir / f'video_preview{suffix}'
        try:
            output.mkdir()
            return output
        except FileExistsError:
            attempt += 1


def record_acceptance_video(report, title='Policy acceptance'):
    report = Path(report).resolve()
    rollout = report.with_name('rollout.npz')
    output = _video_directory(report.parent)
    log_path = output / 'record.log'
    command = [
        sys.executable, '-u', str(resolve_source('diagnostics/record_gait_diagnostic.py')),
        '--rollout', str(rollout), '--acceptance-report', str(report),
        '--output', str(output), '--title', title, '--headless',
    ]
    started = time.perf_counter()
    result = dict(report=str(report), recording_log=str(log_path), status='failed')
    try:
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=300)
        result['exit_code'] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f'Recorder exited with code {completed.returncode}')
        metadata_path = output / 'video_report.json'
        metadata = _read_json(metadata_path)
        video = output / 'acceptance_preview_1080p.mp4'
        with rollout.open('rb') as stream:
            rollout_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if (Path(metadata['video']).resolve() != video or not video.is_file() or not video.stat().st_size
                or metadata['rollout_sha256'] != rollout_hash or metadata['frames'] <= 0
                or metadata['resolution'] != [1920, 1080] or metadata['fps'] != 30
                or not metadata.get('secondary_physics')):
            raise ValueError('Recording does not match the measured acceptance rollout')
        result.update(status='recorded', video=str(video), video_report=str(metadata_path))
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    result['wall_seconds'] = time.perf_counter() - started
    _write_json(output / 'timing.json', result)
    print('ANON_ACCEPTANCE_VIDEO ' + json.dumps(result), flush=True)
    return result


def _file_signature(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _uses_unified_walk_run(args):
    config = {}
    if args.checkpoint:
        checkpoint_config = args.checkpoint.parent / 'env_config.json'
        if not checkpoint_config.is_file():
            return False
        config = _read_json(checkpoint_config)
    if args.config_override:
        config.update(_read_json(args.config_override))
    return bool(config.get('unified_walk_run'))


def _evaluate_and_record(command, output):
    report = output.resolve() / 'report.json'
    paths = (report, report.with_name('rollout.npz'))
    original = [_file_signature(path) for path in paths]
    started = time.perf_counter()
    child = subprocess.run(command)
    elapsed = time.perf_counter() - started
    if child.returncode:
        return child.returncode
    fresh = all(_file_signature(path) not in (None, previous) for path, previous in zip(paths, original))
    video = (record_acceptance_video(report) if fresh else
             dict(status='missing_acceptance', wall_seconds=0.,
                  error='Evaluator did not write a new report and rollout'))
    timing = dict(evaluation_seconds=elapsed, recording_seconds=video['wall_seconds'], acceptance_video=video)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'acceptance_timing.json', timing)
    print('ANON_ACCEPTANCE_TIMING ' + json.dumps(timing), flush=True)
    return child.returncode


def supervise_acceptance_video(args, entry, argv):
    """Run evaluation and ensure every full scratch checkpoint has a video."""
    command = [sys.executable, '-u', str(entry), *argv, '--no-acceptance-video']
    if args.mode == 'eval' and args.checkpoint and not args.zero_policy and _uses_unified_walk_run(args):
        return _evaluate_and_record(command, args.output)
    if args.mode != 'train' or args.resume_audit_only or args.iterations <= 0:
        return None

    scratch = args.from_scratch or args.scratch_continuation
    if scratch:
        if args.scratch_target != 'latest_unified':
            return None
        history_path = args.experiment_output.resolve() / 'history.json'
        history_length = len(_read_json(history_path)) if history_path.is_file() else 0
        command.append('--scratch-record-full')
    elif not _uses_unified_walk_run(args):
        return None

    run = args.run_dir
    if run is None:
        run = PROJECT_ROOT / 'logs/rsl_rl/anon_locomotion' / datetime.now().strftime('%Y%m%d_%H%M%S')
        command += ['--run-dir', str(run)]
    child = subprocess.run(command)
    if child.returncode or (run / 'stop.request').is_file():
        return child.returncode
    if not scratch:
        # Standalone training has no checkpoint acceptance controller.
        # Reuse the existing lite screen to measure the final checkpoint.
        output = run.resolve() / 'acceptance'
        output.mkdir(parents=True, exist_ok=True)
        config = output / 'screen_config.json'
        _write_json(config, dict(unified_eval_lite=True, unified_eval_short=False,
                                 unified_eval_cases=None, unified_eval_initial_command=None))
        screen = [sys.executable, '-u', str(entry), '--mode', 'eval', '--headless', '--num-envs', '1',
                  '--checkpoint', str(run.resolve() / 'policy_final.pt'), '--seed', str(args.seed),
                  '--config-override', str(config), '--output', str(output), '--no-acceptance-video']
        return _evaluate_and_record(screen, output)
    if history_path.is_file():
        recordings = []
        for row in _read_json(history_path)[history_length:]:
            path = row.get('full_report')
            if path:
                iteration = row['completed_iterations']
                video = row.get('full_video', {})
                if video.get('status') != 'recorded':
                    print(f'ANON_ACCEPTANCE_VIDEO_START {iteration}', flush=True)
                    previous_seconds = video.get('wall_seconds', 0.)
                    video = record_acceptance_video(path, title=f'Policy {iteration} acceptance')
                    video['previous_attempt_seconds'] = previous_seconds
                recordings.append(dict(completed_iterations=iteration, **video))
        summary = dict(recording_seconds=sum(row['wall_seconds'] + row.get('previous_attempt_seconds', 0.)
                                             for row in recordings), videos=recordings)
        _write_json(run / 'acceptance_videos.json', summary)
        print('ANON_TRAIN_ACCEPTANCE_VIDEOS ' + json.dumps(summary), flush=True)
    return child.returncode
