"""Check historical behavior and safe I/O without starting training or Kit."""
import ast
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.diagnostics import locomotion_tools as tools
from source.pmx_prototype.diagnostics.workspace import Workspace
from source.pmx_prototype.experiments.compat import controller_module
from source.pmx_prototype.experiments import __main__ as experiments

ARCHIVE = ROOT/'source/pmx_prototype/archive/experiments/locomotion_20261116_17'


def historical_function(category, script, name, **namespace):
    tree = ast.parse((ARCHIVE/category/(script+'.py')).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<historical-function>', 'exec'), namespace)
    return namespace[name]


class LocomotionConsolidationTests(unittest.TestCase):
    def test_frozen_snapshots_are_exact(self):
        rows = json.loads((ARCHIVE/'catalog.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 23)
        for row in rows:
            with self.subTest(path=row['original']):
                self.assertEqual(hashlib.sha256((ROOT/row['snapshot']).read_bytes()).hexdigest(), row['sha256'])
                self.assertIn('compat' if '/training/' in row['original'] else 'locomotion_tools',
                              (ROOT/row['original']).read_text(encoding='utf-8'))

    def test_controller_main_and_acceptance_functions_preserved(self):
        for row in json.loads((ARCHIVE/'catalog.json').read_text(encoding='utf-8')):
            if '/training/' not in row['original'] or row['original'].endswith('run_transition_retention.py'):
                continue
            before = ast.parse((ROOT/row['snapshot']).read_text(encoding='utf-8'))
            after = ast.parse((ROOT/'source/pmx_prototype/experiments/workflows'/Path(row['original']).name).read_text(encoding='utf-8'))
            current = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
            for node in before.body:
                if isinstance(node, ast.FunctionDef) and node.name not in ('execute', 'read', 'write', 'launch'):
                    with self.subTest(script=row['original'], function=node.name):
                        self.assertEqual(ast.dump(node), ast.dump(current[node.name]))

    def test_transition_recipe_does_not_mutate_default(self):
        default = controller_module('continue_sampling_pivot')
        before = (default.OUT, default.CONFIG, default.STAGE_ITERATIONS[:], default.CUMULATIVE_BASE)
        alternate = controller_module('run_transition_retention')
        self.assertEqual(alternate.STAGE_ITERATIONS, [100, 100, 100])
        self.assertEqual(alternate.CUMULATIVE_BASE, 48054.58603930008)
        self.assertEqual(before, (default.OUT, default.CONFIG, default.STAGE_ITERATIONS, default.CUMULATIVE_BASE))
        self.assertIsNot(alternate, controller_module('run_transition_retention'))

    def test_old_controller_import_retains_mutable_module_identity(self):
        expected = controller_module('run_split_scratch')
        self.assertIs(importlib.import_module('run_split_scratch'), expected)
        self.assertIs(controller_module('resume_split_scratch').experiment, expected)
        self.assertIs(controller_module('limit_split_scratch').experiment, expected)

    def test_all_diagnostic_factories_import_without_jobs(self):
        import torch  # Windows DLL discovery may query the OS through subprocess.
        with patch('subprocess.Popen', side_effect=AssertionError('unexpected job')):
            for profile in tools.catalog():
                with self.subTest(profile=profile):
                    task = tools.build_task(profile, Workspace())
                    self.assertTrue(callable(getattr(task, 'main', getattr(task, 'replay', None))))

    def test_all_diagnostic_dry_runs_are_lazy(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(tools, 'build_task', side_effect=AssertionError('not lazy')):
            output = Path(folder)/'absent'
            for profile, row in tools.catalog().items():
                with self.subTest(profile=profile), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(tools.main([row['family'], row['action'], '--output-dir', str(output), '--dry-run']), 0)
            self.assertFalse(output.exists())

    def test_experiment_preview_and_active_duplicate_guard(self):
        with patch('subprocess.run', side_effect=AssertionError('unexpected job')):
            for recipe in experiments.catalog():
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(experiments.main(['run', recipe, '--dry-run']), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiments.main(['run', 'run_balanced_scratch'])
        # The umbrella CLI must propagate the recipe launcher's failure code.
        sys.path.insert(0, str(ROOT/'source/pmx_prototype'))
        with patch.object(sys, 'argv', ['pmx.py', 'experiments', 'run', 'run_split_scratch']), patch.object(experiments, 'main', return_value=7):
            with self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT/'source/pmx_prototype/pmx.py'), run_name='__main__')
            self.assertEqual(stopped.exception.code, 7)

    def test_workspace_preserves_inputs_and_refuses_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'input', Path(folder)/'output'
            source.mkdir(); output.mkdir()
            (source/'config.json').write_text('{"a": 1}')
            workspace = Workspace(source, output, inference=True)
            workspace.directory(source)
            self.assertEqual(workspace.read_json(output/'config.json'), {'a': 1})
            workspace.write_json(source/'config.json', {'a': 2})
            self.assertEqual(json.loads((source/'config.json').read_text()), {'a': 1})
            self.assertEqual(workspace.read_json(output/'config.json'), {'a': 2})
            with self.assertRaises(FileExistsError):
                Workspace(source, output).write_json(output/'config.json', {})
            with self.assertRaises(ValueError):
                workspace.write_json(Path(folder)/'escape.json', {})

    def test_offline_workspace_cannot_start_process(self):
        with self.assertRaises(ValueError), patch('subprocess.Popen') as child:
            Workspace().execute('forbidden', [])
        child.assert_not_called()

    def test_inference_maps_config_but_preserves_new_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/'input', Path(folder)/'output'
            source.mkdir(); output.mkdir()
            (source/'config.json').write_text('{}')
            workspace = Workspace(source, output, inference=True)
            child = Mock(pid=123); child.wait.return_value = 0
            with patch('subprocess.Popen', return_value=child) as launch:
                workspace.execute('eval', ['evaluate.py', '--mode', 'eval', '--config-override', output/'config.json', '--output', output/'eval'])
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--output')+1], str(output/'eval'))
            self.assertEqual(Path(command[command.index('--config-override')+1]).resolve(), (source/'config.json').resolve())
            self.assertNotIn('train', command)

    def test_tracking_adapter_preserves_eval_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Workspace(folder, Path(folder)/'new', inference=True)
            workspace.execute = Mock(return_value=0)
            task = tools.build_task('diagnose_tracking_margin', workspace)
            task.execute('repeat', 'policy.pt', 'config.json')
            arguments = workspace.execute.call_args.args[1]
            self.assertEqual(arguments[arguments.index('--mode')+1], 'eval')
            self.assertEqual(arguments[arguments.index('--checkpoint')+1], 'policy.pt')
            self.assertEqual(arguments[arguments.index('--num-envs')+1], '1')

    def test_first_dwell_matches_historical_math(self):
        old = historical_function('diagnostics', 'audit_response_windows', 'first_dwell', np=np)
        new = tools.build_task('audit_response_windows', Workspace()).first_dwell
        flags = np.random.default_rng(47).random((100, 5)) > .15
        flags[:, 0] = True; flags[:, 1] = False
        self.assertEqual(old(flags, .1), new(flags, .1))

    def test_sampling_flags_keep_boundary_semantics(self):
        import torch
        old = historical_function('diagnostics', 'audit_low_speed_sampling', 'flags')
        new = tools.build_task('audit_low_speed_sampling', Workspace()).flags
        commands = torch.tensor([[0., 0., 0.], [0., 0., .6], [.02, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [.4, 0., 0.], [.85, 0., 0.]])
        for key, value in old(commands).items():
            self.assertTrue(torch.equal(value, new(commands)[key]), key)

    def test_response_analysis_matches_historical_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output, baseline = (Path(folder)/name for name in ('input', 'output', 'baseline'))
            for path in (source/'transitions', output, baseline/'transitions'):
                path.mkdir(parents=True)
            schedule = {'walk_stop': [[6, [.3, 0, 0]], [6, [0, 0, 0]]]}
            for base in (source, baseline):
                (base/'transition_schedule.json').write_text(json.dumps(schedule))
                for name in ('old', 'new'):
                    velocity = np.zeros((120, 2, 6)); velocity[:60, :, 1] = -.3
                    np.savez(base/'transitions'/f'{name}_walk_stop.npz', time=np.arange(120)*.1,
                        headings=np.zeros((120, 2)), root_velocity=velocity,
                        ref_commands=np.concatenate((np.tile([.3, 0, 0], (60, 2, 1)), np.zeros((60, 2, 3)))), done=np.zeros((120, 2), dtype=bool))
            namespace = {'__name__': 'historical_response', '__file__': str(ROOT/'source/pmx_prototype/diagnostics/audit_response_windows.py')}
            exec(compile((ARCHIVE/'diagnostics/audit_response_windows.py').read_text(encoding='utf-8'), '<historical-response>', 'exec'), namespace)
            namespace['OUT'] = baseline
            with contextlib.redirect_stdout(io.StringIO()):
                namespace['main']()
                tools.build_task('audit_response_windows', Workspace(source, output)).main()
            self.assertEqual(json.loads((baseline/'response_window_audit.json').read_text()), json.loads((output/'response_window_audit.json').read_text()))
            self.assertFalse((source/'response_window_audit.json').exists())


if __name__ == '__main__':
    unittest.main()
