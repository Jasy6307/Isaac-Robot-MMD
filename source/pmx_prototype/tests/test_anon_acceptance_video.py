"""Verify video acceptance scheduling without starting Kit or training."""

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
    def test_lite_recording_uses_available_scenes_and_includes_failed_movement(self):
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

    def test_standalone_training_screens_the_final_checkpoint_then_records(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            override = root / 'override.json'
            override.write_text('{"unified_walk_run": true}', encoding='utf-8')
            args = SimpleNamespace(mode='train', checkpoint=None, config_override=override,
                                   resume_audit_only=False, iterations=10, from_scratch=False,
                                   scratch_continuation=False, run_dir=run, seed=42)
            events = []

            def child(command):
                self.assertIn('--no-acceptance-video', command)
                mode = command[command.index('--mode') + 1]
                events.append(mode)
                if mode == 'train':
                    run.mkdir()
                    (run / 'policy_final.pt').touch()
                else:
                    output = Path(command[command.index('--output') + 1])
                    cfg = Path(command[command.index('--config-override') + 1])
                    self.assertTrue(json.loads(cfg.read_text(encoding='utf-8'))['unified_eval_lite'])
                    self.assertEqual(Path(command[command.index('--checkpoint') + 1]), run / 'policy_final.pt')
                    (output / 'report.json').write_text('{"passed_lite_gate": false}', encoding='utf-8')
                    (output / 'rollout.npz').touch()
                return SimpleNamespace(returncode=0)

            def record(report):
                events.append('record')
                self.assertTrue(report.with_name('rollout.npz').is_file())
                self.assertFalse(json.loads(report.read_text(encoding='utf-8'))['passed_lite_gate'])
                return dict(status='recorded', wall_seconds=3.)

            with patch.object(video.subprocess, 'run', side_effect=child), patch.object(
                    video, 'record_acceptance_video', side_effect=record), patch('builtins.print'):
                result = video.supervise_acceptance_video(args, root / 'train.py', ['--mode', 'train'])
            self.assertEqual(result, 0)
            self.assertEqual(events, ['train', 'eval', 'record'])
            timing = json.loads((run / 'acceptance/acceptance_timing.json').read_text(encoding='utf-8'))
            self.assertEqual(timing['recording_seconds'], 3.)
            self.assertGreaterEqual(timing['evaluation_seconds'], 0.)

    def test_scratch_continuation_records_only_new_full_acceptances(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            run.mkdir()
            history = root / 'history.json'
            previous = [dict(completed_iterations=500, report='previous.json')]
            history.write_text(json.dumps(previous), encoding='utf-8')
            args = SimpleNamespace(mode='train', checkpoint=None, resume_audit_only=False,
                                   iterations=500, from_scratch=False, scratch_continuation=True,
                                   scratch_target='latest_unified', experiment_output=root, run_dir=run)

            def child(command):
                self.assertIn('--no-acceptance-video', command)
                self.assertIn('--scratch-record-full', command)
                rows = [*previous, dict(completed_iterations=750, report='lite_only.json'),
                        dict(completed_iterations=1000, report='lite.json', full_report='full.json'),
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
            report.write_text('{"passed_short_gate": false}', encoding='utf-8')
            with patch.object(video.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), patch('builtins.print'):
                result = video.record_acceptance_video(report)
            self.assertEqual(result['status'], 'failed')
            self.assertTrue((report.parent / 'video_preview/timing.json').is_file())
            self.assertEqual(report.read_text(encoding='utf-8'), '{"passed_short_gate": false}')


if __name__ == '__main__':
    unittest.main()
