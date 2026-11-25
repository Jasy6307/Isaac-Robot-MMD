"""Preparation and augmentation-only switches; no simulator or optimization."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
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

    def test_mass_profile_freezes_actual_asset_and_dependencies(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            new_body=root/'output/anon_physical_v4_plagenhoef_female_v1'
            new_body.mkdir(parents=True)
            body_files=[new_body/name for name in ('anon_physical.usda','rig.usda','manifest.json')]
            old_asset=root/'output/anon_physical_v4/anon_physical.usda'
            baseline=root/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json'
            baseline.parent.mkdir(parents=True)
            motion=root/'motion.npz'
            expert=root/'expert.npz'
            for path in (baseline,motion,expert,*body_files):
                path.write_text(path.name,encoding='utf-8')
            config=dict(full_body_reference=True,articulated_wrists=True,asset=str(old_asset),
                        body_mass_profile='plagenhoef_female_v1',full_motion_path=str(motion),
                        amp_expert_paths=[str(expert)],train_lateral_mirror=True,
                        train_mirror_augmentation=True,amp_mirror_augmentation=True,
                        train_mirror_loss_coeff=0.,train_leg_std=.08,
                        unified_acceptance_protocol=CURRENT_PROTOCOL)
            recipe=root/'recipe.json'
            recipe.write_text(json.dumps(config),encoding='utf-8')
            output=root/'experiment'
            with patch.object(prepared,'ROOT',root), \
                    patch('reference_asset_builder.ensure_reference_asset',return_value=body_files[0]) as resolve:
                plan=prepared.prepare(output,root/'run',recipe,ledger=None)
            resolve.assert_called_once_with(root,version=4,mass_profile='plagenhoef_female_v1')
            self.assertEqual(plan['asset'],str(body_files[0]))
            self.assertEqual(json.loads((output/'config.json').read_text())['asset'],plan['asset'])
            self.assertNotIn(str(old_asset),plan['input_hashes'])
            for path in body_files:
                self.assertEqual(plan['input_hashes'][str(path)],prepared.digest(path))

    def test_latest_suite_explicitly_covers_backward_walk(self):
        config=dict(unified_acceptance_protocol=CURRENT_PROTOCOL,
                    unified_idle_head_acceptance=True,unified_eval_backward_walk=True,unified_eval_low_speed=True)
        durations=controller.acceptance_durations(config)
        cases=suite(protocol=config['unified_acceptance_protocol'],idle_head=True,backward_walk=True)
        self.assertIn(('walk_backward',8,[-.4,0,0]),cases)
        self.assertIn('run_backward',[label for label,_,_ in cases])
        self.assertEqual(sum(seconds for _,seconds,_ in cases)+LOW_SPEED_SECONDS,durations['complete_suite_seconds'])

    def test_routine_full_suite_shortens_only_the_two_standing_holds(self):
        options = dict(protocol=CURRENT_PROTOCOL, idle_head=True, backward_walk=True)
        routine = suite(**options)
        previous = suite(**options, stand_seconds=40)
        self.assertEqual(len(routine), 40)
        self.assertEqual([a[0] for a, b in zip(routine, previous, strict=True) if a != b], ['stand', 'stand_end'])
        self.assertEqual([duration for label, duration, _ in routine if label in ('stand', 'stand_end')], [12, 12])
        self.assertEqual(sum(duration for _, duration, _ in routine) + LOW_SPEED_SECONDS, 441.5)
        self.assertEqual(controller.acceptance_durations(dict(
            unified_acceptance_protocol=CURRENT_PROTOCOL, unified_idle_head_acceptance=True,
            unified_eval_backward_walk=True))['complete_suite_seconds'], 321.5)

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
