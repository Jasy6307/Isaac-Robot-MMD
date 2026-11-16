
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
import unittest
from pathlib import Path
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE, resolve_skirt_grid


class SecondaryProfilesTest(unittest.TestCase):
    def test_final_baseline(self):
        default=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
        self.assertEqual(default['physics_hz'],120)
        self.assertEqual(default['quality_profile'],'cloth')
        self.assertEqual(default['profile_id'],'hair_contact_drape_v1')
        self.assertEqual(default['hair_max_stretch'],1.015)
        self.assertIn('hair_constraint_solver',default)
        self.assertNotIn('rows',default)
        self.assertNotIn('bones',default)
        for rows in (9,16):
            settings=json.loads(DEFAULT_SECONDARY_PROFILE.read_text(encoding='utf-8'))
            resolve_skirt_grid(settings,[f'qun_{r}_{c}' for r in range(rows) for c in range(21)])
            actual=settings['appearance']['skirt_waist_support']['shape_gain_by_row']
            if rows==16:self.assertEqual(actual,[1.,12.,12.,11.,8.,5.5,3.5,2.,1.25,1.])

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
