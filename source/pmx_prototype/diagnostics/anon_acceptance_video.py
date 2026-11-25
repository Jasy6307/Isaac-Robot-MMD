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
FULL_VIDEO_NAME = 'acceptance_full_1080p.mp4'
PREVIEW_VIDEO_NAME = 'acceptance_preview_1080p.mp4'


def acceptance_ranges(labels, segments, fps):
    """Locate each report segment in the recorded trajectory."""
    start = 0
    for segment in segments:
        end = start + round(segment['seconds'] * fps)
        label = segment['label']
        if end > len(labels) or end <= start or not all(value == label for value in labels[start:end]):
            raise ValueError('Acceptance report segments differ from the rollout')
        yield start, end, segment
        start = end
    if start != len(labels):
        raise ValueError('Acceptance report does not cover the complete rollout')


def acceptance_shots(labels, segments, fps, sequence_start_frames=()):
    """Select up to four short excerpts for a diagnostic replay."""
    head_shots = []
    other_shots = []
    sequence_start = 0
    boundaries = set(sequence_start_frames)
    previous_label = None
    for start, end, segment in acceptance_ranges(labels, segments, fps):
        if start in boundaries:
            sequence_start = start
        label = segment['label']
        response = segment.get('response_steady', {})
        head = response.get('idle_head')
        body_status = 'passed' if segment.get('passed') else 'failed'
        body = response.get('body_status', body_status if 'passed' in segment else 'unscored')
        if label in HEAD_TITLES:
            begin = start if label == 'stand' else max(sequence_start, start - 2 * fps)
            finish = min(end, start + (4 if label == 'stand' else 5) * fps)
            title = HEAD_TITLES[label]
            if label == 'stand_end' and previous_label != 'pivot_right':
                title = 'Final stand'
            head_shots.append((begin, finish, title, head, body))
        else:
            title = label.replace('_', ' ').capitalize()
            other_shots.append((max(start, end - 5 * fps), end, title, head, body))
        previous_label = label
    # Targeted diagnostics can omit some of the four head scenes.
    # Fill open slots from the measured movement cases, favoring failed ones.
    other_shots.sort(key=lambda shot: shot[4] == 'passed')
    shots = sorted((head_shots + other_shots)[:4], key=lambda shot: shot[0])
    if not shots:
        raise ValueError('Acceptance report contains no recorded scenes')
    return shots


def full_acceptance_chapters(labels, segments, fps, sequence_start_frames):
    """Cover every measured frame, with one chapter per acceptance segment."""
    starts = tuple(int(frame) for frame in sequence_start_frames)
    if not starts or starts[0] != 0 or tuple(sorted(set(starts))) != starts:
        raise ValueError('Invalid acceptance sequence boundaries')
    chapters = []
    sequence = 0
    previous = None
    for index, (start, end, row) in enumerate(acceptance_ranges(labels, segments, fps), 1):
        if start in starts:
            sequence += 1
            previous = None
        label = row['label']
        title = HEAD_TITLES.get(label, label.replace('_', ' ').capitalize())
        if label == 'stand_end' and previous != 'pivot_right':
            title = 'Final stand'
        if label == 'stop' and previous:
            title = 'Stop after ' + previous.replace('_', ' ')
        chapters.append(dict(
            segment=index, label=label, sequence=sequence, title=title,
            source_frame_start=start, source_frame_end=end, video_start_seconds=start / fps,
            passed=row['passed'], falls=row['falls'], counts_toward_gate=row.get('counts_toward_gate', True),
        ))
        previous = label
    boundaries = {chapter['source_frame_start'] for chapter in chapters}
    if not chapters or not set(starts).issubset(boundaries):
        raise ValueError('An acceptance sequence starts outside a report segment boundary')
    return chapters


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def _video_directory(acceptance_dir, full):
    attempt = 0
    while True:
        suffix = f'_{attempt:02d}' if attempt else ''
        directory = 'video_full' if full else 'video_preview'
        output = acceptance_dir / f'{directory}{suffix}'
        try:
            output.mkdir()
            return output
        except FileExistsError:
            attempt += 1


def record_acceptance_video(report, title='Policy acceptance'):
    report = Path(report).resolve()
    rollout = report.with_name('rollout.npz')
    started = time.perf_counter()
    output = None
    result = dict(report=str(report), status='failed')
    try:
        acceptance = _read_json(report)
        full = bool(acceptance.get('complete_scope'))
        output = _video_directory(report.parent, full)
        log_path = output / 'record.log'
        result['recording_log'] = str(log_path)
        grouped = full and acceptance.get('acceptance_environments') == 4
        recorder = ('diagnostics/dispatch_acceptance_video.py' if grouped
                    else 'diagnostics/record_gait_diagnostic.py')
        command = [
            sys.executable, '-u', str(resolve_source(recorder)),
            '--rollout', str(rollout), '--acceptance-report', str(report),
            '--output', str(output), '--title', title, '--headless',
        ]
        segments = acceptance['segments'] if full else []
        expected_frames = sum(round(segment['seconds'] * 30) for segment in segments)
        timeout = max(300, 60 + 6 * expected_frames / 30)
        if grouped:
            # A paired trial can record the candidate and serial control on the same paused training job.
            timeout *= 2
        with log_path.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=timeout)
        result['exit_code'] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f'Recorder exited with code {completed.returncode}')
        metadata_path = output / 'video_report.json'
        metadata = _read_json(metadata_path)
        video = output / (FULL_VIDEO_NAME if full else PREVIEW_VIDEO_NAME)
        with rollout.open('rb') as stream:
            rollout_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if (Path(metadata['video']).resolve() != video or not video.is_file() or not video.stat().st_size
                or metadata['rollout_sha256'] != rollout_hash or metadata['frames'] <= 0
                or metadata['resolution'] != [1920, 1080] or metadata['fps'] != 30
                or not metadata.get('secondary_physics')):
            raise ValueError('Recording does not match the measured acceptance rollout')
        if full:
            chapters = metadata['chapters']
            if metadata.get('coverage') != 'full' or metadata['frames'] != expected_frames or len(chapters) != len(segments):
                raise ValueError('Full acceptance video is missing measured frames or segments')
            start = 0
            for index, (chapter, segment) in enumerate(zip(chapters, segments, strict=True), 1):
                end = start + round(segment['seconds'] * 30)
                if ((chapter['segment'], chapter['label'], chapter['passed'],
                     chapter['source_frame_start'], chapter['source_frame_end'],
                     chapter.get('counts_toward_gate', True))
                        != (index, segment['label'], segment['passed'], start, end,
                            segment.get('counts_toward_gate', True))):
                    raise ValueError('Full acceptance video chapter differs from the scored report')
                start = end
        result.update(status='recorded', video=str(video), video_report=str(metadata_path))
        if 'recording' in metadata:
            result['recording'] = metadata['recording']
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    result['wall_seconds'] = time.perf_counter() - started
    if output is not None:
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
    """Run evaluation or enable full acceptance recording on scheduled training checkpoints."""
    command = [sys.executable, '-u', str(entry), *argv, '--no-acceptance-video']
    if args.mode == 'eval' and args.checkpoint and not args.zero_policy and _uses_unified_walk_run(args):
        return _evaluate_and_record(command, args.output)
    if args.mode != 'train' or args.resume_audit_only or args.iterations <= 0:
        return None

    scratch = args.from_scratch or args.scratch_continuation
    if (scratch and args.scratch_target != 'latest_unified') or (not scratch and not _uses_unified_walk_run(args)):
        return None

    run = args.run_dir
    if run is None:
        run = PROJECT_ROOT / 'logs/rsl_rl/anon_locomotion' / datetime.now().strftime('%Y%m%d_%H%M%S')
        command += ['--run-dir', str(run)]
    acceptance = args.experiment_output.resolve() if scratch else run.resolve() / 'acceptance'
    history_path = acceptance / 'history.json'
    history_length = len(_read_json(history_path)) if history_path.is_file() else 0
    command.append('--record-scheduled-acceptance')
    child = subprocess.run(command)
    if child.returncode or (run / 'stop.request').is_file():
        return child.returncode
    if history_path.is_file():
        recordings = []
        for row in _read_json(history_path)[history_length:]:
            path = row.get('full_report')
            if path and row.get('video_scheduled', True):
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
