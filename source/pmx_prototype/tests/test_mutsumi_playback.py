"""The approved playback preset must resolve complete production dependencies."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT,resolve_source
from preview_mutsumi import playback_arguments,select_dance,dance_catalog
from source_bundle import source_files
import json,unittest


class PlaybackTests(unittest.TestCase):
    def setUp(self):self.preset=json.loads(resolve_source('mutsumi_deepbluetown_playback.json').read_text(encoding='utf-8'))

    def test_resolution_and_live_pipeline(self):
        for resolution,size in [('1k',('1080','1350')),('2k',('1440','1800'))]:
            args=playback_arguments(self.preset,resolution,PROJECT_ROOT/'output/playback_test',30)
            self.assertEqual((args[args.index('--width')+1],args[args.index('--height')+1]),size)
            self.assertIn('--secondary-worker',args)
            self.assertNotIn('--startup-settle-seconds',args)

    def test_no_physics_does_not_launch_physics_worker(self):
        args=playback_arguments(self.preset,'2k',PROJECT_ROOT/'output/playback_test',60,no_physics=True)
        self.assertIn('--no-secondary-physics',args);self.assertNotIn('--secondary-worker',args)
        with self.assertRaises(ValueError):playback_arguments(self.preset,'2k',PROJECT_ROOT/'output/playback_test',30,2,True)

    def test_all_registered_dances_use_approved_assets(self):
        registry=json.loads((PROJECT_ROOT/'output/mutsumi_validation_references/manifest.json').read_text(encoding='utf-8'))['dances']
        self.assertEqual(set(dance_catalog()),set(registry))
        self.assertNotIn('galaxias',dance_catalog())
        for dance in dance_catalog():
            preset=select_dance(self.preset,dance)
            self.assertEqual((PROJECT_ROOT/preset['rig_data']).resolve(),Path(registry[dance]['rig_recipe']).resolve())
            for resolution in ['1k','2k']:
                args=playback_arguments(preset,resolution,PROJECT_ROOT/'output/playback_test',30)
                self.assertEqual(args[args.index('--usd')+1],str((PROJECT_ROOT/preset['usd']).resolve()))
                self.assertIn('--no-obstacles',args)
                self.assertIn('--secondary-worker',args)
                self.assertEqual(preset['look'],'mmd-toon')

    def test_aliases_and_unknown_dance(self):
        for alias,canonical in [('irisout','IRIS_OUT'),('jilejingtu','ji_le_jing_tu'),('Girls','girls')]:
            self.assertEqual(select_dance(self.preset,alias),select_dance(self.preset,canonical))
        with self.assertRaises(ValueError):select_dance(self.preset,'galaxias')

    def test_source_snapshot_includes_worker_and_preset(self):
        files=source_files(['preview_mutsumi.py'])
        for name in ['physx_dance.py','runtime/dance_secondary_stream.py','runtime/dance_secondary_worker.py','runtime/camera_bounds_cache.py','configs/playback/mutsumi_deepbluetown.json','configs/physics/secondary_default.json']:
            self.assertIn(name,files)
        self.assertIn('cel_studio.mdl', files)
        self.assertNotIn('mmd_cel.mdl', files)
        self.assertIn('configs/playback/mutsumi_dances.json',files)


if __name__=='__main__':unittest.main()
