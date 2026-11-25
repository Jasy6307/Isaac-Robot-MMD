"""Pushes must be physical, one-shot, and confined to the selected actor."""

import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_character_push import push_character
from anon_inspector_controls import InspectorCommands
from anon_inspector_ui import PolicyInspector


class CharacterPushTests(unittest.TestCase):
    def test_force_scales_with_mass_and_timestep_and_targets_selected_chest(self):
        for hz in (60, 120):
            robot = SimpleNamespace(max_links=3, get_masses=lambda:torch.tensor([[10., 20., 10.], [5., 10., 5.]]),
                                    apply_forces_and_torques_at_position=Mock())
            env = SimpleNamespace(evaluation=True, num_envs=2, robot=robot, device='cpu',
                                  links=['pelvis','waist_pitch','head'], pelvis_id=0,
                                  ids=torch.arange(2,dtype=torch.int32), cfg=dict(physics_hz=hz))
            state = random.getstate()
            result = push_character(env, 1, rng=random.Random(12))
            self.assertEqual(random.getstate(), state)
            forces, torque, position, ids, world = robot.apply_forces_and_torques_at_position.call_args.args
            self.assertEqual(forces.shape, (2, 3, 3))
            self.assertEqual(torch.count_nonzero(forces[0]), 0)
            self.assertEqual(torch.count_nonzero(forces[1, [0, 2]]), 0)
            self.assertEqual(forces[1, 1, 2], 0)
            self.assertEqual(ids.tolist(), [1])
            self.assertIsNone(torque)
            self.assertIsNone(position)
            self.assertTrue(world)
            self.assertAlmostEqual(float(forces.norm()) / hz / 20, result['delta_speed'], places=5)
            self.assertTrue(.50 <= result['delta_speed'] <= .55)

    def test_random_directions_cover_the_circle(self):
        robot = SimpleNamespace(max_links=1, get_masses=lambda:torch.ones(1,1),
                                apply_forces_and_torques_at_position=Mock())
        env = SimpleNamespace(evaluation=True,num_envs=1,robot=robot,device='cpu',
                              links=['pelvis'],pelvis_id=0,ids=torch.tensor([0],dtype=torch.int32),cfg=dict(physics_hz=120))
        rng = random.Random(5)
        forces = [push_character(env,rng=rng)['force'] for _ in range(50)]
        self.assertEqual({(math.copysign(1,x),math.copysign(1,y)) for x,y,_ in forces},
                         {(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)})

    def test_push_exits_acceptance_and_is_sent_once_without_automatic_reset(self):
        ui = PolicyInspector.__new__(PolicyInspector)
        ui.controls = InspectorCommands({})
        ui.pending = ''
        ui.playback_frames = iter([([.4,0.,0.], True, 'Acceptance')])
        ui.playback_name = 'Acceptance'
        ui.show_input_controls = Mock()
        ui.refresh_sliders = Mock()
        ui.controls.request_reset()
        ui.push_character()
        self.assertIsNone(ui.playback_frames)
        with patch('anon_inspector_ui.has_foreground', return_value=True):
            self.assertEqual(ui.next_request(), dict(command=[0.,0.,0.],reset=False,push_character=True))
            self.assertEqual(ui.next_request(), dict(command=[0.,0.,0.],reset=False))
        ui.controls.request_push()
        ui.reset()
        self.assertTrue(ui.controls.consume_reset())
        self.assertFalse(ui.controls.consume_push())


if __name__ == '__main__':
    unittest.main()
