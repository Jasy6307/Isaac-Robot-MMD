"""Compatibility of moved commands, archived parameters and source snapshots."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from source_bundle import copy_sources
from source_layout import SOURCE_DIR


class SourceLayoutTests(unittest.TestCase):
    def run_command(self, script, arguments, cwd):
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        result = subprocess.run(
            [sys.executable, str(script), *map(str, arguments)],
            cwd=cwd, env=environment, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_legacy_import_keeps_function_globals_and_project_root(self):
        legacy = importlib.import_module('audit_anon_gait')
        self.assertIs(legacy.audit.__globals__, vars(legacy))
        self.assertEqual(legacy.ROOT, SOURCE_DIR.parents[1])

    def test_direct_and_unified_commands_compute_identical_slip(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            poses = np.zeros((1000, 1, 2, 7))
            poses[..., 6] = 1.
            poses[:, 0, :, 0] = np.arange(1000)[:, None] * .01 / 30
            trace = folder / 'trace.npz'
            motion = folder / 'motion.npz'
            np.savez(trace, poses=poses, dt=1/30,
                     link_names=['left_ankle_roll', 'right_ankle_roll'])
            np.savez(motion, foot_corners=np.zeros((2, 4, 3)))
            results = []
            for script, prefix in ((resolve_source('measure_gait_slip.py'), []),
                                   (SOURCE_DIR / 'pmx.py', ['diagnostics', 'measure_gait_slip'])):
                output = folder / f'report_{len(results)}.json'
                self.run_command(script,
                                 [*prefix, '--baseline', trace, '--candidate', trace,
                                  '--motion', motion, '--output', output], folder)
                results.append(json.loads(output.read_text(encoding='utf-8')))
            self.assertEqual(*results)
            self.assertAlmostEqual(results[0]['candidate']['geometric_ground_slip_mean_m_s'], .01)

    def test_snapshot_runs_without_original_source_on_pythonpath(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            files = copy_sources(folder, ['train_anon_locomotion.py',
                                         'measure_gait_slip.py', 'anon_v5_config.json'])
            for required in ('training/reference_asset_builder.py', 'runtime/physical_character.py',
                             'training/build_anon_v4.py', 'diagnostics/measure_gait_slip.py',
                             'archive/configs/anon_v5_config.json'):
                self.assertIn(required, files)
            probe = folder / 'probe.py'
            probe.write_text(
                "import source_layout\nimport build_anon_v3, build_anon_v4, source_bundle\n"
                "from pathlib import Path\n"
                "assert Path(build_anon_v3.__file__).parent == source_layout.SOURCE_DIR / 'training'\n"
                "assert callable(build_anon_v4.ensure_asset)\n",
                encoding='utf-8',
            )
            self.run_command(probe, [], folder)
            self.run_command(folder / 'train_anon_locomotion.py', ['--help'], folder)
            self.run_command(folder / 'diagnostics/measure_gait_slip.py', ['--help'], folder)

    def test_archived_and_compatibility_config_bytes_match(self):
        catalog = json.loads((SOURCE_DIR / 'archive/catalog.json').read_text(encoding='utf-8'))
        for item in catalog['configs']:
            with self.subTest(config=item['legacy_path']):
                original = (SOURCE_DIR / item['legacy_path']).read_bytes()
                archived = (SOURCE_DIR / item['archive_path']).read_bytes()
                self.assertEqual(original, archived)
                self.assertEqual(hashlib.sha256(archived).hexdigest(), item['sha256'])


if __name__ == '__main__':
    unittest.main()
