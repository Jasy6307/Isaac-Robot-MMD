import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from secondary_profiles import apply_secondary_mode,DEFAULT_SECONDARY_PROFILE,crowd_secondary_modes


class SecondaryModeTests(unittest.TestCase):
    def test_mixed_modes_follow_controlled_actor_and_override_uniform_budget(self):
        self.assertEqual(crowd_secondary_modes(4,0,'full',True),['light','light3','light3','light3'])
        self.assertEqual(crowd_secondary_modes(4,2,'light2',True),['light3','light3','light','light3'])
        self.assertEqual(crowd_secondary_modes(4,2,'full',True,controlled_mode='light2'),['light3','light3','light2','light3'])
        self.assertEqual(crowd_secondary_modes(4,2,'light2'),['light2']*4)
        for count,selected in ((0,0),(4,-1),(4,4)):
            with self.assertRaises(ValueError):crowd_secondary_modes(count,selected)
        with self.assertRaises(ValueError):crowd_secondary_modes(4,0,mixed=True,controlled_mode='invalid')

    def test_full_keeps_approved_settings_untouched(self):
        settings=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        before=json.dumps(settings,sort_keys=True)
        self.assertIs(apply_secondary_mode(settings),settings)
        self.assertEqual(json.dumps(settings,sort_keys=True),before)

    def test_light_changes_only_solver_budget(self):
        settings=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        light=apply_secondary_mode(settings,'light')
        self.assertEqual(light['physics_hz'],120)
        self.assertEqual(light['native_solver_position_iterations'],64)
        self.assertEqual(light['hair_constraint_solver']['iterations'],48)
        self.assertEqual(light['hair_constraint_solver']['contact_refine_iterations'],8)
        for key in settings:
            if key!='hair_constraint_solver':self.assertEqual(light[key],settings[key],key)
        for key in settings['hair_constraint_solver']:
            if key not in ('iterations','contact_refine_iterations'):
                self.assertEqual(light['hair_constraint_solver'][key],settings['hair_constraint_solver'][key])
        light['appearance']['hair_radius_m']=10
        self.assertEqual(settings['appearance']['hair_radius_m'],.003)

    def test_invalid_mode_is_explicit(self):
        with self.assertRaises(ValueError):apply_secondary_mode({},'invalid')

    def test_light2_is_separate_and_keeps_collision_settings(self):
        settings=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        lighter=apply_secondary_mode(settings,'light2')
        self.assertEqual(lighter['secondary_mode'],'light2')
        self.assertEqual(lighter['native_solver_position_iterations'],32)
        self.assertEqual(lighter['native_solver_velocity_iterations'],8)
        self.assertEqual(lighter['hair_constraint_solver']['iterations'],16)
        self.assertEqual(lighter['hair_constraint_solver']['contact_refine_iterations'],8)
        self.assertEqual(lighter['appearance'],settings['appearance'])
        self.assertEqual(lighter['physics_hz'],120)
        self.assertEqual(apply_secondary_mode(settings,'light')['hair_constraint_solver']['iterations'],48)

    def test_light3_changes_only_solve_budget_and_preserves_source(self):
        settings=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        before=json.dumps(settings,sort_keys=True)
        lighter=apply_secondary_mode(settings,'light3')
        self.assertEqual(lighter['secondary_mode'],'light3')
        self.assertEqual(lighter['native_solver_position_iterations'],8)
        self.assertEqual(lighter['native_solver_velocity_iterations'],8)
        self.assertEqual(lighter['hair_constraint_solver']['iterations'],4)
        self.assertEqual(lighter['hair_constraint_solver']['contact_refine_iterations'],2)
        for key,value in settings.items():
            if key!='hair_constraint_solver':self.assertEqual(lighter[key],value,key)
        for key,value in settings['hair_constraint_solver'].items():
            if key not in ('iterations','contact_refine_iterations'):
                self.assertEqual(lighter['hair_constraint_solver'][key],value,key)
        self.assertEqual(json.dumps(settings,sort_keys=True),before)

if __name__=='__main__':unittest.main()
