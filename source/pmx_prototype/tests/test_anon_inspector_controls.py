"""Input invariants that prevent misleading commands and stuck movement."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import math
import unittest
from anon_inspector_controls import InspectorCommands,keyboard_event_names
from evaluate_unified_walk_run import suite
from locomotion_acceptance import CURRENT_PROTOCOL, inspector_planar_speed_limit
from types import SimpleNamespace

class InspectorInputTests(unittest.TestCase):
    def test_character_events_and_native_enum_adapter(self):
        self.assertIsNone(keyboard_event_names(SimpleNamespace(type=SimpleNamespace(name='CHAR'),input='w')))
        self.assertEqual(keyboard_event_names(SimpleNamespace(type=SimpleNamespace(name='KEY_PRESS'),input=SimpleNamespace(name='W'))),('W','KEY_PRESS'))
        self.assertEqual(keyboard_event_names(SimpleNamespace(type='KEY_RELEASE',input='W')),('W','KEY_RELEASE'))
    def walk(self):return InspectorCommands(dict(max_speed=.4,max_yaw=.6))
    def test_diagonal_and_opposing_keys(self):
        c=self.walk();c.set_speed(.4)
        for k in ['W','A']:c.event(k,'KEY_PRESS')
        x,y,_=c.command();self.assertAlmostEqual(math.hypot(x,y),.4)
        c.event('S','KEY_PRESS');self.assertAlmostEqual(c.command()[0],0.)

    def test_unified_direction_caps_match_full_acceptance(self):
        config = dict(max_speed=1.2, unified_backward_diagonal_speed_cap=.859)
        cases = {label: command for label, _, command in suite(protocol=CURRENT_PROTOCOL)}
        keys_by_case = {
            'run_forward': 'W', 'run_left': 'A', 'run_right': 'D', 'run_backward': 'S',
            'run_diagonal_45': 'WA', 'run_diagonal_135': 'SA',
            'run_diagonal_225': 'SD', 'run_diagonal_315': 'WD',
        }
        for label, keys in keys_by_case.items():
            with self.subTest(keys=keys):
                controls = InspectorCommands(config)
                controls.set_speed(1.2)
                for key in keys:
                    controls.event(key, 'KEY_PRESS')
                request = controls.command()
                self.assertAlmostEqual(math.hypot(*request[:2]), 1.2)
                self.assertAlmostEqual(inspector_planar_speed_limit(request, config), math.hypot(*cases[label][:2]))
        backward = [-1.2, 0., .6]
        self.assertEqual(inspector_planar_speed_limit(backward, dict(config, max_speed=.4)), .4)
        self.assertEqual(inspector_planar_speed_limit(backward, dict(config, unified_eval_backward_diagonal_run_speed=.8)), .8)
    def test_release_and_short_tap(self):
        c=self.walk();c.event('W','KEY_PRESS');self.assertGreater(c.command()[0],0)
        c.event('W','KEY_RELEASE');self.assertEqual(c.command(),[0.,0.,0.])
        c.event('D','KEY_PRESS');c.event('D','KEY_RELEASE')
        self.assertLess(c.command()[1],0);self.assertEqual(c.command(),[0.,0.,0.])
    def test_speed_clamps_and_numpad(self):
        c=self.walk()
        for _ in range(20):c.event('NUMPAD_ADD','KEY_PRESS')
        self.assertEqual(c.speed,.4)
        for _ in range(20):c.event('MINUS','KEY_PRESS')
        self.assertEqual(c.speed,0.)
    def test_stop_blocks_key_repeat_and_reset(self):
        c=self.walk();c.event('W','KEY_PRESS');c.event('SPACE','KEY_PRESS')
        c.event('W','KEY_REPEAT');self.assertEqual(c.command(),[0.,0.,0.])
        self.assertFalse(c.consume_reset())
        c.event('R','KEY_PRESS');self.assertTrue(c.consume_reset());self.assertFalse(c.consume_reset())
    def test_yaw_and_disabled_input(self):
        c=self.walk();c.event('Q','KEY_PRESS');self.assertAlmostEqual(c.command()[2],.35)
        c.enabled=False;self.assertEqual(c.command(),[0.,0.,0.])
        c.event('R','KEY_PRESS');self.assertFalse(c.consume_reset())
        c.request_reset();self.assertTrue(c.consume_reset());self.assertFalse(c.consume_reset())
    def test_fixed_running_cannot_pretend_variable_or_sideways(self):
        c=InspectorCommands(dict(amp_forward_only=True,amp_forward_speed=1.2,max_speed=1.2,max_yaw=0.))
        c.event('W','KEY_PRESS');self.assertEqual(c.command(),[1.2,0.,0.])
        c.event('EQUAL','KEY_PRESS');self.assertEqual(c.speed,1.2)
        c.event('A','KEY_PRESS');self.assertEqual(c.command(),[0.,0.,0.])

if __name__=='__main__':unittest.main()
