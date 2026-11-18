"""Read-only parity on local approved previews, USD caches and legacy leg state."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'source/pmx_prototype'), str(ROOT/'source/pmx_prototype/tools')]
from diagnostics import character_cache
from tools import dance_preview
from source.train_workflow.utils.motion import legacy_leg_full, legacy_leg_local


def source(path):
    with zipfile.ZipFile(ROOT/'source/archive/script_round3_20261117/sources.zip') as archive:
        return archive.read(path).decode('utf-8-sig')


def run_old(path, arguments, *, main_guard=True):
    namespace = dict(__name__='__main__' if main_guard else 'historical', __file__=str(ROOT/path))
    with patch.object(sys,'argv',[str(ROOT/path),*map(str,arguments)]), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(source(path),path,'exec'),namespace)
    return output.getvalue(), SimpleNamespace(**namespace)


class Round3AssetTests(unittest.TestCase):
    def test_live_preview_arguments_match_both_original_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = [
                ('preview_mutsumi', True, ['--dance','Girls','--resolution','1k','--render-fps','60']),
                ('preview_mutsumi', True, ['--no-secondary-physics']),
                ('preview_dance', False, ['--model','anon','--dance','girls','--physics-detail','sparse','--render-fps','60','--interpolate-60']),
                ('preview_dance', False, ['--model','mutsumi','--dance','irisout']),
            ]
            config = json.loads((ROOT/'source/pmx_prototype/configs/playback/mutsumi_deepbluetown.json').read_text(encoding='utf-8'))
            config['id'] = 'custom_preserved'
            custom = Path(folder)/'custom.json'; custom.write_text(json.dumps(config),encoding='utf-8')
            cases.append(('preview_mutsumi',True,['--config',str(custom)]))
            for name, preset_only, arguments in cases:
                with self.subTest(name=name,arguments=arguments):
                    arguments = [*arguments,'--dry-run','--output',str(Path(folder)/'result')]
                    before,_ = run_old('source/pmx_prototype/tools/'+name+'.py',arguments)
                    with contextlib.redirect_stdout(io.StringIO()) as after:
                        dance_preview.main(arguments,preset_only=preset_only)
                    self.assertEqual(json.loads(before),json.loads(after.getvalue()))
            self.assertFalse((Path(folder)/'result').exists())

    def test_cloth_and_temporal_metrics_match_real_cache(self):
        cache = ROOT/'output/mutsumi_validation_120hz_bright/girls/physics.usdc'
        digest = hashlib.sha256(cache.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            for action,name in [('cloth','analyze_mutsumi_cloth_cache'),('temporal','analyze_mutsumi_skirt_jitter')]:
                with self.subTest(action=action):
                    before,after = Path(folder)/(action+'_old.json'),Path(folder)/(action+'_new.json')
                    args = ['--cache',str(cache),'--frames','90']
                    if action=='cloth': args += ['--stride','15']
                    run_old('source/pmx_prototype/diagnostics/'+name+'.py',[*args,'--output',str(before)])
                    with contextlib.redirect_stdout(io.StringIO()):
                        character_cache.main([action,*args,'--output',str(after)])
                    self.assertEqual(json.loads(before.read_text(encoding='utf-8')),json.loads(after.read_text(encoding='utf-8')))
        self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(),digest)

    def test_reference_world_coordinates_and_hard_gates_match(self):
        from pxr import Usd
        body = ROOT/'output/mutsumi_validation_120hz_bright/girls/body.usdc'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            paths = []
            for label in ('previous','current'):
                path = folder/(label+'.usda')
                stage = Usd.Stage.CreateNew(str(path)); stage.GetRootLayer().subLayerPaths = [str(body)]
                stage.SetEndTimeCode(59); stage.GetRootLayer().Save(); paths.append(path)
            sidecar = paths[1].with_suffix('.json')
            sidecar.write_text(json.dumps(dict(leg_clearance=None)),encoding='utf-8')
            args = ['--previous',str(paths[0]),'--current',str(paths[1])]
            before,after = folder/'old.json',folder/'new.json'
            run_old('source/pmx_prototype/diagnostics/audit_mutsumi_reference_motion.py',[*args,'--output',str(before)])
            with contextlib.redirect_stdout(io.StringIO()): character_cache.main(['reference',*args,'--output',str(after)])
            self.assertEqual(json.loads(before.read_text(encoding='utf-8')),json.loads(after.read_text(encoding='utf-8')))
            sidecar.write_text(json.dumps(dict(leg_clearance='modified')),encoding='utf-8')
            with self.assertRaises(AssertionError): character_cache.main(['reference',*args,'--output',str(folder/'must_not_publish.json')])
            self.assertFalse((folder/'must_not_publish.json').exists())

    def test_legacy_leg_one_step_and_rejection_match_real_geometry(self):
        reference = ROOT/'output/girls_ik_ground_fix_20261116/girls_ik_ground_fixed_0_80.h5'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); sample = folder/'sample.h5'
            with h5py.File(reference) as before, h5py.File(sample,'w') as after:
                frames = len(before['joint_pos_delta'])
                for name,item in before.items():
                    if isinstance(item, h5py.Group):
                        before.copy(item, after, name=name)
                        continue
                    data = item[()]
                    if data.ndim and len(data)==frames: data=data[:32]
                    after.create_dataset(name,data=data,dtype=item.dtype)
                for name,value in before.attrs.items(): after.attrs[name]=value
            digest = hashlib.sha256(sample.read_bytes()).hexdigest()
            cases = [('smooth_h5_full_legs',legacy_leg_full.main,[]),
                ('smooth_h5_leg_window',legacy_leg_local.smooth_h5_leg_window_main,['--reference',str(sample),'--start','0','--end','32']),
                ('repair_h5_leg_spikes',legacy_leg_local.repair_h5_leg_spikes_main,['--original',str(sample),'--smooth_seed',str(sample)])]
            for name,current,extra in cases:
                with self.subTest(strategy=name):
                    _,historical = run_old('source/train_workflow/scripts/'+name+'.py',[],main_guard=False)
                    failures=[]; outputs=[]
                    for label,function in [('old',historical.main),('new',current)]:
                        output=folder/(name+'_'+label+'.h5');outputs.append(output)
                        args=['--input',str(sample),'--output',str(output),'--iterations','1',*extra]
                        with patch.object(sys,'argv',[name,*args]),contextlib.redirect_stdout(io.StringIO()):
                            try: function()
                            except RuntimeError as error: failures.append((label,str(error)))
                    self.assertEqual(outputs[0].exists(),outputs[1].exists())
                    self.assertEqual([x[1] for x in failures[:1]],[x[1] for x in failures[1:]])
                    if outputs[0].exists():
                        with h5py.File(outputs[0]) as a,h5py.File(outputs[1]) as b:
                            np.testing.assert_array_equal(a['joint_pos_delta'][:],b['joint_pos_delta'][:])
                    states=[p.with_suffix('.state.npz') for p in outputs]
                    if states[0].exists():
                        with np.load(states[0]) as a,np.load(states[1]) as b:
                            np.testing.assert_array_equal(a['leg_q'],b['leg_q'])
                    reports=[p.with_suffix('.smooth.json') for p in outputs]
                    if all(p.exists() for p in reports):
                        a,b=[json.loads(p.read_text(encoding='utf-8')) for p in reports]
                        for key in ('before','after'): self.assertEqual(a[key],b[key])
            self.assertEqual(hashlib.sha256(sample.read_bytes()).hexdigest(),digest)


if __name__ == '__main__':
    unittest.main()
