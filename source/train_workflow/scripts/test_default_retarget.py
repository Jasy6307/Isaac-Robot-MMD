"""Default source discovery/cache behavior; legacy remains an explicit choice."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.retarget import source_animation as module


class SourceTests(unittest.TestCase):
    def test_discovery_requires_real_vmd(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            root=Path(d);csv=root/'dance.csv'
            with self.assertRaisesRegex(ValueError,'source VMD'):
                module.resolve_vmd(csv)
            vmd=root/'original.vmd';vmd.write_bytes(b'vmd')
            csv.with_suffix('.motion.json').write_text(json.dumps({'vmd':'original.vmd'}))
            self.assertEqual(module.resolve_vmd(csv),vmd)
            self.assertEqual(module.resolve_vmd(csv,explicit=vmd),vmd)

    def test_cache_respects_vmd_content_and_source_mode(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'output') as d:
            root=Path(d);vmd=root/'dance.vmd';vmd.write_bytes(b'v1')
            pmx=root/'model.pmx';pmx.write_bytes(b'rig')
            asset=root/'model.usdc';asset.write_bytes(b'asset')
            baker=root/'source/pmx_prototype/load_vmd.py';baker.parent.mkdir(parents=True);baker.write_text('# baker')
            def bake(command,**kw):
                target=Path(command[command.index('--output')+1]);target.write_bytes(b'animation')
                target.with_suffix('.json').write_text(json.dumps({'vmd':str(vmd),'leg_mode':command[command.index('--leg-mode')+1]}))
            with patch.object(module,'ROOT',root),patch.object(module.subprocess,'run',side_effect=bake) as run:
                a=module.prepare_source_animation(root/'dance.csv',pmx=pmx,asset=asset)
                self.assertEqual(a,module.prepare_source_animation(root/'dance.csv',pmx=pmx,asset=asset))
                self.assertEqual(run.call_count,1)
                vmd.write_bytes(b'v2')
                b=module.prepare_source_animation(root/'dance.csv',pmx=pmx,asset=asset)
                c=module.prepare_source_animation(root/'dance.csv',pmx=pmx,asset=asset,leg_mode='ik')
                self.assertEqual(len({a,b,c}),3)
                self.assertEqual(run.call_count,3)

if __name__=='__main__':unittest.main()
