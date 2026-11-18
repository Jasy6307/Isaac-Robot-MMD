"""Historical numerical parity and compatibility of the third cleanup."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'source/pmx_prototype'))
from source.train_workflow.utils import jump_comparison as jumps, turn_analysis as turns
from source.train_workflow.utils import jump_reports
from source.train_workflow.utils.jump_analysis import load_telemetry, new_diagnostic_output, require_new_files
from source.train_workflow.utils.motion import legacy_leg_full as full, legacy_leg_local as local

ARCHIVE = ROOT/'source/archive/script_round3_20261117'


def original(path):
    with zipfile.ZipFile(ARCHIVE/'sources.zip') as archive:
        return archive.read(path).decode('utf-8-sig')


def old_module(name):
    path = 'source/train_workflow/scripts/'+name+'.py'
    namespace = dict(__name__='frozen_'+name, __file__=str(ROOT/path))
    exec(compile(original(path), path, 'exec'), namespace)
    return SimpleNamespace(**namespace)


def old_main(name, arguments):
    path = 'source/train_workflow/scripts/'+name+'.py'
    with patch.object(sys, 'argv', [name, *map(str, arguments)]):
        exec(compile(original(path), path, 'exec'), dict(__name__='__main__', __file__=str(ROOT/path)))


def telemetry():
    time = np.arange(0, 55, 1/120)
    data = dict(time_s=time, ref_time_s=np.floor(time*30+1e-8)/30,
        sample_dt_s=np.full_like(time, 1/120), ref_root_vz_m_s=.3*np.cos(time),
        root_z_m=.8+.02*np.sin(time), ref_root_z_m=.81+.01*np.sin(time),
        both_feet_no_contact=(np.sin(time*12)>.5).astype(float),
        root_yaw_rad=.1*time, ref_root_delta_yaw_rad=.1*time)
    names = []
    for side in ('left', 'right'):
        data[side+'_contact_N'] = 15+15*np.cos(time*12)
        data[side+'_contact'] = (data[side+'_contact_N']>10).astype(float)
        for name in ('foot_vx_m_s', 'foot_vy_m_s', 'foot_wz_rad_s', 'foot_yaw_rad'):
            data[side+'_'+name] = .1*np.sin(time)
        for part in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'):
            name = f'{side}_{part}_joint'; names.append(name)
            for kind, offset in (('ref', 0), ('target', .01), ('actual', .02)):
                data[name+'_'+kind+'_rad'] = .1*np.sin(time*8)+offset
            data[name+'_velocity_rad_s'] = .8*np.cos(time*8)
            data[name+'_limit_Nm'] = np.full_like(time, 100)
            data[name+'_applied_Nm'] = 5*np.cos(time)
    return data, dict(seed=42, control_hz=30, leg_joint_names=names, contact_threshold_N=10,
        contact_source='ground-only filtered ankle normal forces', end_reason='completed', dance='fixture')


def save_episode(path, data, metadata):
    path.mkdir(parents=True)
    np.savez(path/'samples.npz', columns=np.array(list(data)), values=np.stack(list(data.values()), axis=1))
    (path/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')


class Round3Tests(unittest.TestCase):
    def test_snapshot_integrity_and_fifteen_compatibility_entries(self):
        rows = json.loads((ARCHIVE/'catalog.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 15)
        with zipfile.ZipFile(ARCHIVE/'sources.zip') as archive:
            for row in rows:
                with self.subTest(path=row['path']):
                    self.assertEqual(hashlib.sha256(archive.read(row['path'])).hexdigest(), row['sha256'])
                    self.assertLess(len((ROOT/row['path']).read_text(encoding='utf-8').splitlines()), 20)

    def test_all_legacy_help_and_canonical_dispatch_without_simulation(self):
        rows = json.loads((ARCHIVE/'catalog.json').read_text(encoding='utf-8'))
        commands = [[str(ROOT/r['path']), '--help'] for r in rows]
        commands += [[str(ROOT/'source/train_workflow/scripts/g1_ground_motion.py'), 'legacy-legs', name, '--help'] for name in ('full','window','spikes')]
        commands += [[str(ROOT/'source/train_workflow/scripts/g1_diagnostics.py'), 'jump', 'summary', '--help']]
        with tempfile.TemporaryDirectory() as folder:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, *command], cwd=folder, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_jump_summaries_match_frozen_numerics(self):
        data, metadata = telemetry()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder); save_episode(path/'episode_0', data, metadata)
            self.assertEqual(old_module('compare_jump_training').summarize(path), jumps.compare_jump_training_summary(path))
            self.assertEqual(old_module('summarize_multi_jump').summarize(path), jumps.summarize_multi_jump_summary(path))

    def test_jump_plots_and_reports_match_history_in_separate_outputs(self):
        data, meta = telemetry(); data['ref_root_vz_m_s'][:] = .3
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(folder); report=folder/'episode'; save_episode(report,data,meta)
            annotations = folder/'annotations.json'
            annotations.write_text(json.dumps(dict(dance='fixture',events=[dict(id='fixture',context=[16,18],
                phases=[dict(phase='prepare_open',start=16.1,end=16.9)])])),encoding='utf-8')
            cases=[('decompose_jump_tracking',jump_reports.decompose_jump_tracking_main,'decomposition.json',[]),
                ('mark_jump_phases',jump_reports.mark_jump_phases_main,'phase_review.json',['--annotations',str(annotations)])]
            for name,function,filename,extra in cases:
                old_main(name,[report,*extra])
                output=folder/name
                function([str(report),*extra,'--output-dir',str(output)])
                self.assertEqual(json.loads((report/filename).read_text(encoding='utf-8')),json.loads((output/filename).read_text(encoding='utf-8')))
                self.assertTrue(list(output.glob('*.png')))
                digest=hashlib.sha256((report/filename).read_bytes()).hexdigest()
                with self.assertRaises(FileExistsError): function([str(report),*extra,'--output-dir',str(report)])
                self.assertEqual(hashlib.sha256((report/filename).read_bytes()).hexdigest(),digest)
            before,after=folder/'old.png',folder/'new.png'
            old_main('plot_jump_reference_change',[report,report,before])
            jump_reports.plot_jump_reference_change_main([str(report),str(report),str(after)])
            self.assertEqual(json.loads(before.with_suffix('.json').read_text()),json.loads(after.with_suffix('.json').read_text()))

    def test_turn_report_and_comparison_match_history(self):
        data,meta=telemetry()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            folder=Path(folder); baseline=folder/'baseline'
            save_episode(baseline/'seed_42/episode_0',data,meta)
            old_main('analyze_turn',[baseline])
            output=folder/'analysis'
            turns.analyze_turn_main([str(baseline),'--output-dir',str(output)])
            self.assertEqual(json.loads((baseline/'summary.json').read_text()),json.loads((output/'summary.json').read_text()))
            for name in ('control','treatment'):
                directory=folder/name; (directory/'diagnostics').mkdir(parents=True)
                (directory/'diagnostics/summary.json').write_bytes((baseline/'summary.json').read_bytes())
                validation=dict(passed=True,records=[dict(seed=42,acceptance_passed=True,acceptance_score=1,
                    results=[dict(acceptance=dict(failures=[]))])])
                (directory/'validation.json').write_text(json.dumps(validation),encoding='utf-8')
            args=['--baseline',str(baseline),'--control',str(folder/'control'),'--treatment',str(folder/'treatment')]
            old_main('compare_turn_smoothing',[*args,'--output',folder/'old_comparison'])
            turns.compare_turn_smoothing_main([*args,'--output',str(folder/'new_comparison')])
            self.assertEqual(json.loads((folder/'old_comparison/comparison.json').read_text()),json.loads((folder/'new_comparison/comparison.json').read_text()))

    def test_mismatched_contact_measurement_is_still_rejected(self):
        data, meta = telemetry(); meta['contact_source'] = 'unfiltered'
        with tempfile.TemporaryDirectory() as folder:
            save_episode(Path(folder)/'episode_0', data, meta)
            for function in (jumps.compare_jump_training_summary, jumps.summarize_multi_jump_summary):
                with self.assertRaises(ValueError): function(Path(folder))

    def test_missing_secondary_jump_windows_remain_missing(self):
        data, meta = telemetry(); data = {key:value[:2200] for key,value in data.items()}
        with tempfile.TemporaryDirectory() as folder:
            save_episode(Path(folder)/'episode_0', data, meta)
            current = jumps.summarize_multi_jump_summary(Path(folder))
            self.assertEqual(current, old_module('summarize_multi_jump').summarize(Path(folder)))
            self.assertFalse(current['48s']['episodes'])

    def test_turn_metrics_and_control_interval_selection_match_history(self):
        before = old_module('analyze_turn'); data, meta = telemetry()
        np.testing.assert_array_equal(before.control_rows(data, 30), turns.control_rows(data, 30))
        self.assertEqual(before.summarize(data, meta, 12, 14), turns.summarize(data, meta, 12, 14))
        values = np.random.default_rng(13).normal(size=100)
        self.assertEqual(before.spectral_rms(values, 30), turns.spectral_rms(values, 30))

    def test_spectral_metrics_retain_small_sample_rejection(self):
        with self.assertRaises(ValueError): turns.spectral_rms(np.zeros(7), 30)

    def test_telemetry_loader_preserves_saved_column_order(self):
        data, meta = telemetry()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'episode'; save_episode(path, data, meta)
            loaded = load_telemetry(path)
            self.assertEqual(list(loaded), list(data))
            for name in data: np.testing.assert_array_equal(loaded[name], data[name])

    def test_existing_reports_and_images_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder); existing = output/'plot.png'; existing.write_bytes(b'original')
            with self.assertRaises(FileExistsError): require_new_files(output/'new.json', existing)
            self.assertEqual(existing.read_bytes(), b'original')
            self.assertFalse((output/'new.json').exists())
            self.assertEqual(new_diagnostic_output('test', output).resolve(), output.resolve())

    def test_leg_metrics_match_frozen_math_and_gradients(self):
        class Model:
            def fk(self, q):
                return None, {'left_ankle_roll_link': q[:, :3], 'right_ankle_roll_link': q[:, 3:6]}
        q = torch.randn(32, 8, generator=torch.Generator().manual_seed(93), requires_grad=True)
        old = old_module('smooth_h5_full_legs')
        before, feet_before = old.foot_metrics(Model(), q, [0,1,2,3,4,5])
        after, feet_after = full.foot_metrics(Model(), q, [0,1,2,3,4,5])
        self.assertEqual(before, after)
        self.assertEqual(old_module('smooth_h5_leg_window').metrics(Model(), q, [0,1]), local.metrics(Model(), q, [0,1]))
        torch.testing.assert_close(feet_before, feet_after)
        full.foot_positions(Model(), q).square().sum().backward()
        self.assertIsNotNone(q.grad)

    def test_optimizer_bodies_and_hard_gates_still_match_history(self):
        pairs = [('smooth_h5_full_legs',full,'main'),('smooth_h5_leg_window',local,'smooth_h5_leg_window_main'),('repair_h5_leg_spikes',local,'repair_h5_leg_spikes_main')]
        import inspect
        for name, module, entry in pairs:
            old_tree = ast.parse(original('source/train_workflow/scripts/'+name+'.py'))
            old_main = next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            current = ast.parse(inspect.getsource(getattr(module, entry))).body[0]
            old_loops = [ast.dump(n) for n in ast.walk(old_main) if isinstance(n,ast.For) and 'optimizer.step' in ast.unparse(n)]
            current_loops = [ast.dump(n) for n in ast.walk(current) if isinstance(n,ast.For) and 'optimizer.step' in ast.unparse(n)]
            self.assertEqual(old_loops, current_loops, name)
            self.assertEqual([ast.dump(n) for n in ast.walk(old_main) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call) and isinstance(n.exc.func,ast.Name) and n.exc.func.id=='RuntimeError'],
                             [ast.dump(n) for n in ast.walk(current) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call) and isinstance(n.exc.func,ast.Name) and n.exc.func.id=='RuntimeError'], name)

    def test_h5_export_preserves_root_other_channels_and_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = Path(folder)/'source.h5', Path(folder)/'result.h5'
            with h5py.File(source,'w') as stream:
                stream['joint_names'] = np.array(['a','b'], dtype=h5py.string_dtype())
                stream['joint_pos_delta'] = np.zeros((24,2))
                stream['root_pos_delta'] = np.arange(72).reshape(24,3)
                stream['root_quat_delta_wxyz'] = np.tile([1.,0,0,0],(24,1))
                stream['extra_channel'] = np.arange(24)
                stream.attrs['fps'] = 30.; stream.attrs['original_metadata'] = 'kept'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            full.export_joint_positions(source,result,np.ones((24,2)),{'method':'test'})
            with h5py.File(source) as before, h5py.File(result) as after:
                for name in ('root_pos_delta','root_quat_delta_wxyz','extra_channel'):
                    np.testing.assert_array_equal(before[name][:],after[name][:])
                np.testing.assert_array_equal(after['joint_pos_delta'][:],1.)
                self.assertEqual(after.attrs['original_metadata'],'kept')
            self.assertEqual(full.load_motion(result)[0], ['a','b'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError): full.export_joint_positions(source,result,np.zeros((24,2)),{})

    def test_character_cache_preview_is_lazy(self):
        from diagnostics import character_cache
        with patch.object(character_cache,'open_skeleton',side_effect=AssertionError('opened USD')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(character_cache.main(['temporal','--cache','unused.usdc','--output','unused.json','--dry-run']),0)


if __name__ == '__main__':
    unittest.main()
