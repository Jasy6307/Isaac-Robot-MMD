"""Old CLI contracts resolve to shared implementations without launching Kit."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.runtime.comparison_options import parse_options, actor_layout, frame_range
from source.train_workflow.utils.grounding import legacy_refinement
from source.train_workflow.utils.motion.mesh_audit import build_parser


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.file = Path(self.directory.name)/'input.data'
        self.file.touch()
        self.arguments = ['--g1-h5', str(self.file), '--pmx-cache', str(self.file), '--g1-usd', str(self.file)]

    def test_layouts_and_clip_clock_preserve_historical_presets(self):
        for columns in range(2, 6):
            arguments = self.arguments.copy()
            for flag in ['--corrected-h5', '--fidelity-h5', '--smooth-h5'][:columns-2]:
                arguments += [flag, str(self.file)]
            args = parse_options(arguments)
            layout = actor_layout(args)
            self.assertEqual(len(layout), columns)
            self.assertAlmostEqual(sum(item['offset_x'] for item in layout), 0.)
            self.assertAlmostEqual(layout[-1]['offset_x']-layout[0]['offset_x'], (columns-1)*1.9)
        args = parse_options(arguments, default_preset='girls-segment')
        self.assertEqual(frame_range(args, 30., 2400), (480, 630))
        self.assertEqual(actor_layout(args)[-2]['root'], '/World/G1Smoothed')
        args = parse_options(self.arguments+['--corrected-h5', str(self.file)], default_preset='girls-full')
        self.assertEqual([a['label'] for a in actor_layout(args)], ['G1 - Motion preserved', 'G1 - Full smooth', 'Anon PMX'])

    def test_legacy_wrapper_and_explicit_preset_make_same_plan(self):
        arguments = self.arguments+['--corrected-h5', str(self.file), '--dry-run']
        outputs = []
        for script, extra in [('compare_girls_full_smooth.py', []), ('play_g1_anon_comparison.py', ['--preset', 'girls-full'])]:
            result = subprocess.run([sys.executable, '-B', str(ROOT/'source/pmx_prototype'/script), *arguments, *extra],
                                    cwd=self.directory.name, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(json.loads(result.stdout))
        self.assertEqual(*outputs)

    def test_existing_video_is_rejected_before_simulator_start(self):
        with self.assertRaises(SystemExit):
            parse_options(self.arguments+['--record-video', str(self.file)])

    def test_mesh_scope_defaults_are_explicit(self):
        flags = ['--h5', str(self.file), '--output', self.directory.name]
        self.assertEqual(build_parser('hands').parse_args(flags).pair_scope, 'hands')
        self.assertEqual(build_parser('all').parse_args(flags).pair_scope, 'all')

    def test_refinement_modes_keep_their_original_weights_and_resume_flags(self):
        flags = ['--reference', str(self.file), '--seed', str(self.file), '--output', str(self.file),
                 '--resume-checkpoint', str(self.file), '--window-limit', '2']
        for mode, expected in [('fidelity', (0., 0.)), ('smooth', (.002, .000003))]:
            with patch.object(legacy_refinement, 'run') as run:
                self.assertEqual(legacy_refinement.main(flags, default_method=mode), 0)
            args = run.call_args.args[0]
            self.assertEqual((args.accel_weight, args.jerk_weight), expected)
            self.assertEqual(args.resume_checkpoint, self.file)
            self.assertEqual(args.window_limit, 2)

    def test_migrated_tests_are_discoverable_without_tool_directory_copies(self):
        self.assertFalse(list((ROOT/'source/train_workflow/scripts').glob('test_*.py')))
        self.assertGreaterEqual(len(list((ROOT/'source/train_workflow/tests').glob('test_*.py'))), 33)

    def test_atomic_publication_only_retries_bounded_windows_file_occupancy(self):
        from source.train_workflow.utils import deployment
        busy = PermissionError('Windows file occupied')
        busy.winerror = 32
        with patch.object(deployment.os, 'replace', side_effect=[busy, None]) as replace, \
             patch.object(deployment.time, 'sleep'):
            deployment._atomic_replace('source', 'target')
            self.assertEqual(replace.call_count, 2)
        with patch.object(deployment.os, 'replace', side_effect=busy) as replace, \
             patch.object(deployment.time, 'sleep'):
            with self.assertRaises(PermissionError):
                deployment._atomic_replace('source', 'target')
            self.assertEqual(replace.call_count, 6)
        with patch.object(deployment.os, 'replace', side_effect=PermissionError('permission')) as replace:
            with self.assertRaises(PermissionError):
                deployment._atomic_replace('source', 'target')
            self.assertEqual(replace.call_count, 1)


if __name__ == '__main__':
    unittest.main()
