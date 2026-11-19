"""Preparation and augmentation-only switches; no simulator or optimization."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
import prepare_mirror_scratch as prepared
import run_balanced_scratch as controller
from evaluate_unified_walk_run import LOW_SPEED_SECONDS,suite
from lateral_symmetry import configure
from locomotion_acceptance import CURRENT_PROTOCOL


class PreparationTests(unittest.TestCase):
    def test_default_and_prepare_never_request_start(self):
        self.assertFalse(prepared.arguments([]).start)
        self.assertFalse(prepared.arguments(['--prepare']).start)
        self.assertTrue(prepared.arguments(['--start']).start)

    def test_default_main_only_prepares(self):
        result=dict(status='prepared_waiting_user_start',num_envs=8192)
        with patch.object(prepared,'prepare',return_value=result) as prepare,patch.object(prepared,'launch') as launch:
            with contextlib.redirect_stdout(io.StringIO()):prepared.main([])
            prepare.assert_called_once();launch.assert_not_called()

    def test_fresh_recipe_has_no_pretrained_teacher_or_weak_loss(self):
        config=prepared.configuration()
        self.assertTrue(config['train_mirror_augmentation'])
        self.assertTrue(config['amp_mirror_augmentation'])
        self.assertEqual(config['train_mirror_loss_coeff'],0.)
        self.assertEqual(config['train_leg_std'],.08)
        self.assertFalse(config.get('train_retention_weight'))
        self.assertEqual(config['unified_acceptance_protocol'],CURRENT_PROTOCOL)

    def test_latest_suite_explicitly_covers_backward_walk(self):
        config=dict(unified_acceptance_protocol=CURRENT_PROTOCOL,
                    unified_idle_head_acceptance=True,unified_eval_backward_walk=True)
        durations=controller.acceptance_durations(config)
        for lite,duration in [(True,durations['lightweight_suite_seconds']),
                              (False,durations['complete_suite_seconds'])]:
            cases=suite(lite=lite,protocol=config['unified_acceptance_protocol'],idle_head=True,backward_walk=True)
            self.assertIn(('walk_backward',8,[-.4,0,0]),cases)
            self.assertIn('run_backward',[label for label,_,_ in cases])
            self.assertEqual(sum(seconds for _,seconds,_ in cases)+(0 if lite else LOW_SPEED_SECONDS),duration)

    def test_zero_coefficient_explicitly_disables_loss(self):
        env=SimpleNamespace(cfg=dict(train_lateral_mirror=True,train_mirror_augmentation=True,train_mirror_loss_coeff=0.))
        config={'algorithm':{}}
        with patch('lateral_symmetry.AnatomicalMirror.from_env',return_value=object()):configure(config,env)
        settings=config['algorithm']['symmetry_cfg']
        self.assertTrue(settings['use_data_augmentation'])
        self.assertFalse(settings['use_mirror_loss'])
        self.assertEqual(settings['mirror_loss_coeff'],0.)

    def test_nonzero_coefficient_keeps_existing_weak_loss(self):
        env=SimpleNamespace(cfg=dict(train_lateral_mirror=True,train_mirror_loss_coeff=.05))
        config={'algorithm':{}}
        with patch('lateral_symmetry.AnatomicalMirror.from_env',return_value=object()):configure(config,env)
        self.assertTrue(config['algorithm']['symmetry_cfg']['use_mirror_loss'])

    def test_invalid_loss_coefficients_are_rejected(self):
        for coefficient in [-.1,float('nan'),float('inf')]:
            with self.subTest(coefficient=coefficient),self.assertRaises(ValueError):
                configure({'algorithm':{}},SimpleNamespace(cfg=dict(train_lateral_mirror=True,train_mirror_loss_coeff=coefficient)))


if __name__=='__main__':unittest.main()
