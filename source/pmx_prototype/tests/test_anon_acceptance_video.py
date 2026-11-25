"""Verify video acceptance scheduling without starting Kit or training."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics import anon_acceptance_video as video


class AnonAcceptanceVideoTests(unittest.TestCase):
    def test_diagnostic_recording_uses_available_scenes_and_includes_failed_movement(self):
        segments = [
            dict(label='stand', seconds=4, passed=True),
            dict(label='run_forward', seconds=4, passed=True),
            dict(label='run_backward', seconds=5, passed=False),
            dict(label='walk_before_stop', seconds=5, passed=True),
            dict(label='stop_after_walk', seconds=7, passed=False),
            dict(label='pivot_right', seconds=5, passed=True),
            dict(label='stand_end', seconds=7, passed=True),
        ]
        labels = [row['label'] for row in segments for _ in range(row['seconds'] * 30)]
        shots = video.acceptance_shots(labels, segments, 30)
        self.assertEqual([shot[2] for shot in shots],
                         ['Initial stand', 'Run backward', 'Walk and stop', 'Right pivot and stop'])
        self.assertEqual(shots[1][4], 'failed')
        with self.assertRaises(ValueError):
            video.acceptance_shots(labels[:-30], segments, 30)

    def test_parallel_head_excerpt_stays_in_its_own_sequence(self):
        segments = [dict(label='walk', seconds=4, passed=True),
                    dict(label='stop_after_walk', seconds=7, passed=True)]
        labels = ['walk'] * 120 + ['stop_after_walk'] * 210
        shots = video.acceptance_shots(labels, segments, 30, [0, 120])
        self.assertEqual(next(shot for shot in shots if shot[2] == 'Walk and stop')[:2], (120, 270))

    def test_full_recording_covers_preparation_and_scored_segments(self):
        segments = [
            dict(label=label, seconds=seconds, passed=passed, falls=falls)
            for label, seconds, passed, falls in (
                ('walk', 4, True, 0), ('stop', 7, False, 1),
                ('run_forward', 6, True, 0), ('stop', 7, True, 0),
                ('pivot_right', 6, True, 0), ('stand_end', 12, True, 0),
            )
        ]
        segments[0]['counts_toward_gate'] = False
        labels = [row['label'] for row in segments for _ in range(row['seconds'] * 30)]
        chapters = video.full_acceptance_chapters(labels, segments, 30, [0, 330])
        self.assertEqual(len(chapters), len(segments))
        self.assertEqual([chapter['sequence'] for chapter in chapters], [1, 1, 2, 2, 2, 2])
        self.assertEqual([chapter['source_frame_end'] - chapter['source_frame_start'] for chapter in chapters],
                         [row['seconds'] * 30 for row in segments])
        self.assertEqual(chapters[-1]['source_frame_end'], len(labels))
        self.assertFalse(chapters[0]['counts_toward_gate'])
        self.assertTrue(chapters[1]['counts_toward_gate'])
        self.assertEqual(chapters[1]['title'], 'Stop after walk')
        self.assertFalse(chapters[1]['passed'])
        self.assertEqual(chapters[1]['falls'], 1)
        with self.assertRaises(ValueError):
            video.full_acceptance_chapters(labels, segments, 30, [0, 331])

    def test_complete_failed_acceptance_still_records_full_video(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / 'report.json'
            segments = [dict(label='stand', seconds=12, passed=True, falls=0),
                        dict(label='direction_sweep', seconds=40, passed=False, falls=1)]
            report.write_text(json.dumps(dict(complete_scope=True, passed_unified_gate=False, segments=segments)),
                              encoding='utf-8')
            rollout = report.with_name('rollout.npz')
            rollout.write_bytes(b'measured rollout')
            labels = [row['label'] for row in segments for _ in range(row['seconds'] * 30)]
            chapters = video.full_acceptance_chapters(labels, segments, 30, [0])

            def child(command, **kwargs):
                output = Path(command[command.index('--output') + 1])
                self.assertEqual(output, root / 'video_full')
                self.assertGreater(kwargs['timeout'], 300)
                path = output / video.FULL_VIDEO_NAME
                path.write_bytes(b'encoded frames')
                result = dict(video=str(path), coverage='full', frames=len(labels), chapters=chapters,
                              fps=30, resolution=[1920, 1080], secondary_physics=True,
                              rollout_sha256=hashlib.sha256(rollout.read_bytes()).hexdigest())
                (output / 'video_report.json').write_text(json.dumps(result), encoding='utf-8')
                return SimpleNamespace(returncode=0)

            with patch.object(video.subprocess, 'run', side_effect=child), patch('builtins.print'):
                result = video.record_acceptance_video(report)
            self.assertEqual(result['status'], 'recorded')
            self.assertEqual(Path(result['video']), root / 'video_full' / video.FULL_VIDEO_NAME)

    def test_standalone_training_uses_scheduled_full_acceptance_and_retries_recording(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            override = root / 'override.json'
            override.write_text('{"unified_walk_run": true}', encoding='utf-8')
            args = SimpleNamespace(mode='train', checkpoint=None, config_override=override,
                                   resume_audit_only=False, iterations=2000, from_scratch=False,
                                   scratch_continuation=False, run_dir=run, seed=42)
            events = []

            def child(command):
                self.assertIn('--no-acceptance-video', command)
                self.assertIn('--record-scheduled-acceptance', command)
                self.assertEqual(command[command.index('--mode') + 1], 'train')
                events.append('train')
                acceptance = run / 'acceptance'
                acceptance.mkdir(parents=True)
                report = acceptance / 'final.json'
                report.write_text('{"complete_scope": true, "passed_unified_gate": false}', encoding='utf-8')
                history = [dict(completed_iterations=1000, full_report='first.json',
                                full_video=dict(status='recorded',wall_seconds=1.)),
                           dict(completed_iterations=2000, full_report=str(report),
                                full_video=dict(status='failed',wall_seconds=2.))]
                (acceptance / 'history.json').write_text(json.dumps(history), encoding='utf-8')
                return SimpleNamespace(returncode=0)

            def record(report, title):
                self.assertEqual(title, 'Policy 2000 acceptance')
                events.append('record')
                self.assertTrue(json.loads(Path(report).read_text(encoding='utf-8'))['complete_scope'])
                return dict(status='recorded', wall_seconds=3.)

            with patch.object(video.subprocess, 'run', side_effect=child), patch.object(
                    video, 'record_acceptance_video', side_effect=record), patch('builtins.print'):
                result = video.supervise_acceptance_video(args, root / 'train.py', ['--mode', 'train'])
            self.assertEqual(result, 0)
            self.assertEqual(events, ['train', 'record'])
            summary = json.loads((run / 'acceptance_videos.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['recording_seconds'], 6.)
            self.assertEqual([row['completed_iterations'] for row in summary['videos']], [1000, 2000])

    def test_scratch_continuation_records_only_new_full_acceptances(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            run.mkdir()
            history = root / 'history.json'
            previous = [dict(completed_iterations=500, report='historical_screen.json')]
            history.write_text(json.dumps(previous), encoding='utf-8')
            args = SimpleNamespace(mode='train', checkpoint=None, resume_audit_only=False,
                                   iterations=500, from_scratch=False, scratch_continuation=True,
                                   scratch_target='latest_unified', experiment_output=root, run_dir=run)

            def child(command):
                self.assertIn('--no-acceptance-video', command)
                self.assertIn('--record-scheduled-acceptance', command)
                rows = [*previous,
                        dict(completed_iterations=750, full_report='numeric_only.json', video_scheduled=False),
                        dict(completed_iterations=1000, full_report='full.json', video_scheduled=True),
                        dict(completed_iterations=1500, full_report='recorded.json',
                             full_video={'status': 'recorded', 'wall_seconds': 2.})]
                history.write_text(json.dumps(rows), encoding='utf-8')
                return SimpleNamespace(returncode=0)

            with patch.object(video.subprocess, 'run', side_effect=child), patch.object(
                    video, 'record_acceptance_video', return_value=dict(status='recorded', wall_seconds=2.)) as record, patch('builtins.print'):
                self.assertEqual(video.supervise_acceptance_video(args, root / 'train.py', []), 0)
            self.assertEqual([call.args[0] for call in record.call_args_list], ['full.json'])
            summary = json.loads((run / 'acceptance_videos.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['recording_seconds'], 4.)
            self.assertEqual([row['completed_iterations'] for row in summary['videos']], [1000, 1500])

    def test_recording_failure_keeps_quantitative_report(self):
        with TemporaryDirectory() as folder:
            report = Path(folder) / 'report.json'
            report.write_text('{"complete_scope": false, "diagnostic": true}', encoding='utf-8')
            with patch.object(video.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), patch('builtins.print'):
                result = video.record_acceptance_video(report)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((report.parent / 'video_preview/timing.json').is_file())
            self.assertEqual(report.read_text(encoding='utf-8'), '{"complete_scope": false, "diagnostic": true}')


if __name__ == '__main__':
    unittest.main()
