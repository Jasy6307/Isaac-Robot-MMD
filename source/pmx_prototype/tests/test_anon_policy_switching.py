"""Hot-switch input boundaries and model/scene identity contracts."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import unittest
from anon_inspector_controls import InspectorCommands
from anon_policy_catalog import PRESETS,identify,physics_signature

WALK=dict(max_speed=.4,max_yaw=.6)
RUN=dict(amp_forward_only=True,amp_forward_speed=1.2,max_speed=1.2,max_yaw=0.)

class SwitchingTests(unittest.TestCase):
    def test_unified_walk_run_has_continuous_speed_and_direction(self):
        c=InspectorCommands(dict(unified_walk_run=True,max_speed=1.2,max_yaw=.6,amp_forward_only=False))
        for _ in range(40):c.event('EQUAL','KEY_PRESS')
        self.assertEqual(c.speed,1.2)
        c.event('S','KEY_PRESS');c.event('A','KEY_PRESS');c.event('Q','KEY_PRESS')
        command=c.command()
        self.assertAlmostEqual(sum(v*v for v in command[:2]),1.2**2)
        self.assertLess(command[0],0);self.assertGreater(command[1],0);self.assertGreater(command[2],0)
        c.event('MINUS','KEY_PRESS');self.assertEqual(c.speed,1.15)
        c.event('SPACE','KEY_PRESS');self.assertEqual(c.command(),[0.,0.,0.])

    def test_switch_clears_keys_but_preserves_preferences(self):
        c=InspectorCommands(WALK);c.speed=.3;c.yaw_speed=.5;c.view=2
        c.event('W','KEY_PRESS');c.event('Q','KEY_PRESS');c.reset_pending=True
        c.switching=True
        self.assertEqual(c.command(),[0.,0.,0.])
        c.event('A','KEY_PRESS');c.configure(RUN);c.switching=False
        self.assertFalse(c.consume_reset());self.assertEqual(c.command(),[0.,0.,0.])
        self.assertEqual(c.view,2);self.assertEqual(c.speed,1.2)
        c.event('W','KEY_REPEAT');self.assertEqual(c.command(),[0.,0.,0.])
        c.event('W','KEY_PRESS');self.assertEqual(c.command(),[1.2,0.,0.])
        c.configure(WALK)
        self.assertEqual(c.command(),[0.,0.,0.]);self.assertEqual((c.speed,c.yaw_speed),(.3,.5))
        c.enabled=False;c.configure(WALK);self.assertFalse(c.enabled)

    def test_catalog_has_one_unified_default(self):
        self.assertIn('unified_walk_run',str(PRESETS['walk']))
        self.assertEqual(len(set(PRESETS.values())),1)
        self.assertTrue(PRESETS['walk'].is_file())
        for key,path in PRESETS.items():
            self.assertEqual(identify(path),key)

    def test_reference_differences_allowed_but_scene_changes_rejected(self):
        cfg=dict(asset='output/anon_physical_v4/anon_physical.usda')
        self.assertEqual(physics_signature(cfg),physics_signature(dict(cfg,full_motion_path='different.npz',residual_scale=.1)))
        for key,value in [('upper_drive_scale',.5),('simulation_device','cpu'),('physics_hz',60),('articulated_wrists',True)]:
            self.assertNotEqual(physics_signature(cfg),physics_signature(dict(cfg,**{key:value})))

if __name__=='__main__':unittest.main()
