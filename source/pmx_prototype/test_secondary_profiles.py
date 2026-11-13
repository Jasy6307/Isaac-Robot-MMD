import json
import unittest
from pathlib import Path
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE, resolve_skirt_grid


class SecondaryProfilesTest(unittest.TestCase):
    def test_default_preserves_accepted_parameters_at_requested_frequency(self):
        default=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        accepted=json.loads(Path(__file__).with_name('secondary_fabric_waist_sidehair_v5_120hz.json').read_text(encoding='utf-8'))
        self.assertEqual(default['physics_hz'],120)
        self.assertEqual(default['quality_profile'],'cloth')
        self.assertEqual(default['profile_id'],'fabric_waist_sidehair_v5')
        for key in ('appearance','skirt_contact_smoothing','skirt_contact_release','cloth_translation_only'):
            self.assertEqual(default[key],accepted[key],key)
        self.assertNotIn('rows',default)
        self.assertNotIn('bones',default)

    def test_sidehair_promotion_preserves_all_v4_physics(self):
        default=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        previous=json.loads(Path(__file__).with_name('secondary_fabric_waist_v4_120hz.json').read_text(encoding='utf-8'))
        support=default['appearance'].pop('local_hair_support')
        self.assertTrue(support['pin_roots'])
        self.assertEqual(support['bone_prefixes'],['rclh_','lclh_','rch_','lch_'])
        for key,value in previous.items():
            if key not in ('profile_id','description'):self.assertEqual(default[key],value,key)

    def test_models_use_their_own_grid_dimensions(self):
        for rows in (9,16):
            settings={'skirt_max_stretch':1.02}
            resolve_skirt_grid(settings,[f'qun_{r}_{c}' for r in range(rows) for c in range(21)])
            self.assertEqual((settings['rows'],settings['columns']),(rows,21))

    def test_cross_model_dimensions_and_incomplete_rings_are_rejected(self):
        names=[f'qun_{r}_{c}' for r in range(16) for c in range(21)]
        with self.assertRaises(ValueError):resolve_skirt_grid({'skirt_max_stretch':1.02,'rows':9},names)
        with self.assertRaises(ValueError):resolve_skirt_grid({'skirt_max_stretch':1.02},names[:-1])

    def test_historical_profiles_do_not_require_new_stretch_settings(self):
        settings={'cloth_translation_only':True}
        resolve_skirt_grid(settings,[])
        self.assertEqual(settings,{'cloth_translation_only':True})


if __name__=='__main__':unittest.main()
