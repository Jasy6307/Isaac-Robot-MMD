"""Camera fit must retain feet, outstretched arms and jump peaks outside labels."""
import sys
from pathlib import Path
from itertools import product
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.validation_camera import replica_offsets,fit_camera,reference_camera_track


class CameraTests(unittest.TestCase):
    def test_close_offsets_preserve_left_to_right_seed_order(self):
        offsets=replica_offsets(3,1.25)
        self.assertTrue(np.allclose(offsets[:,0],[1.25,0,-1.25]))
        self.assertTrue(np.all(offsets[:,1:]==0))

    def test_bounds_project_inside_the_visible_area_for_all_aspect_ratios(self):
        bounds=np.array([[-2.,-.6,-.08],[2.,.5,2.1]])
        for width,height in ((1280,720),(1920,1080),(1000,1000)):
            header=round(height*.11);footer=40
            camera=fit_camera(bounds,width=width,height=height,header_pixels=header,footer_pixels=footer)
            eye=np.array(camera["eye"]);target=np.array(camera["target"])
            forward=(target-eye)/np.linalg.norm(target-eye)
            right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
            up=np.cross(right,forward)
            p=np.array(list(product(*zip(bounds[0],bounds[1]))))-eye
            depth=p@forward
            x=(p@right)/depth/(20.955/52)
            y=(p@up)/depth/(20.955/52*height/width)
            pixels=(1-y)*height/2
            self.assertTrue(np.all(depth>0))
            self.assertLess(float(abs(x).max()),1.)
            self.assertGreater(float(pixels.min()),header)
            self.assertLess(float(pixels.max()),height-footer)

    def test_closer_spacing_reduces_required_distance(self):
        def bounds(spacing):return [[-spacing-.7,-.4,0.],[spacing+.7,.4,1.9]]
        wide=fit_camera(bounds(2.8),width=1280,height=720,header_pixels=80,footer_pixels=34)
        close=fit_camera(bounds(1.25),width=1280,height=720,header_pixels=80,footer_pixels=34)
        self.assertLess(close["distance_m"],wide["distance_m"])
        self.assertTrue(close["fixed_for_entire_recording"])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):replica_offsets(3,0)
        with self.assertRaises(ValueError):fit_camera([[0,0,0],[0,1,1]],width=100,height=100,header_pixels=20,footer_pixels=10)

    def test_camera_follows_reference_xy_but_not_jump_height(self):
        root=np.column_stack((np.linspace(0,1,60),np.linspace(0,2,60),np.sin(np.arange(60))))
        track=reference_camera_track(dict(tables=dict(_root_pos_delta=root)),70)
        self.assertTrue(np.all(track[:,2]==0))
        self.assertTrue(np.allclose(track[-1],[1.,2.,0.]))
        self.assertLess(float(np.max(abs(track[:60,:2]-root[:,:2]))),.04)


if __name__=="__main__":unittest.main()
