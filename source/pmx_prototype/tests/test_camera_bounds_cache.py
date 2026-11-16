"""Camera cache must preserve bounds and invalidate changed geometry layers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import tempfile,unittest
from types import SimpleNamespace
import numpy as np
from camera_bounds_cache import cached_camera_bounds


class CacheTests(unittest.TestCase):
    def test_hit_exact_bounds_and_layer_content_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);layer=folder/'asset.usda';layer.write_text('geometry-v1')
            stage=SimpleNamespace(GetUsedLayers=lambda:[SimpleNamespace(realPath=str(layer))])
            calls=[]
            def compute():
                calls.append(1);return np.array([-.123456789123,-2.,0.]),np.array([1.,2.,3.])
            first=cached_camera_bounds(stage,compute,folder/'cache','mesh')
            second=cached_camera_bounds(stage,compute,folder/'cache','mesh')
            self.assertFalse(first[2]['hit']);self.assertTrue(second[2]['hit'])
            np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
            self.assertEqual(len(calls),1)
            layer.write_text('geometry-v2')
            self.assertFalse(cached_camera_bounds(stage,compute,folder/'cache','mesh')[2]['hit'])
            self.assertEqual(len(calls),2)

    def test_corrupt_cache_and_disabled_cache_recompute(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);layer=folder/'asset.usda';layer.write_text('geometry')
            stage=SimpleNamespace(GetUsedLayers=lambda:[SimpleNamespace(realPath=str(layer))])
            def compute():return np.zeros(3),np.ones(3)
            result=cached_camera_bounds(stage,compute,folder/'cache','mesh')
            Path(result[2]['path']).write_text('invalid-json')
            self.assertFalse(cached_camera_bounds(stage,compute,folder/'cache','mesh')[2]['hit'])
            self.assertFalse(cached_camera_bounds(stage,compute,folder/'cache','mesh',enabled=False)[2]['hit'])


if __name__=='__main__':unittest.main()
