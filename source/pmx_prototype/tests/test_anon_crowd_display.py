import math,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_crowd_display import CrowdInterpolatedDisplay


class CrowdDisplayTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.draws=[]
        self.display=CrowdInterpolatedDisplay(lambda poses:self.draws.append(poses),lambda:self.now,self.sleep)

    def sleep(self,seconds):self.now+=seconds

    def push(self,frame,xs,restarts=(False,False),headings=(0.,0.)):
        poses=[(np.array([[x,0.,0.]]),np.array([[0.,0.,0.,1.]])) for x in xs]
        centers=[np.array([x,0.,1.],dtype=np.float32) for x in xs]
        self.display.push(frame,poses,centers,headings,restarts)
        return poses

    def test_two_actor_midpoints_are_real_interpolated_poses(self):
        self.push(0,[0.,10.]);self.display.present_if_due()
        self.now=1/30;source=self.push(1,[2.,12.])
        self.now+=1/60;self.display.present_if_due()
        for i,x in enumerate((1.,11.)):
            np.testing.assert_allclose(self.draws[-1][i]['p'],[[x,0.,0.]])
            np.testing.assert_allclose(self.draws[-1][i]['center'],[x,0.,1.])
            self.assertEqual(self.draws[-1][i]['center'].dtype,np.dtype(float))
            self.assertAlmostEqual(self.draws[-1][i]['frame'],.5)
        np.testing.assert_array_equal(source[0][0],[[2.,0.,0.]])
        self.assertEqual(self.display.interpolated_frames,1)

    def test_only_reset_actor_snaps_and_other_actor_keeps_interpolating(self):
        self.push(0,[0.,10.]);self.now=1/30
        self.push(1,[100.,12.],restarts=(True,False))
        self.now+=1/60;self.display.present_if_due()
        self.assertEqual(self.draws[-1][0]['p'][0,0],100.)
        self.assertEqual(self.draws[-1][0]['frame'],1)
        self.assertAlmostEqual(self.draws[-1][1]['p'][0,0],11.)
        self.assertAlmostEqual(self.draws[-1][1]['frame'],.5)

    def test_camera_heading_uses_short_rotation_and_quaternions_stay_unit(self):
        self.push(0,[0.,10.],headings=(math.pi-.1,0.));self.now=1/30
        self.push(1,[2.,12.],headings=(-math.pi+.1,0.))
        self.display.poses[0].current['q']=np.array([[0.,0.,1.,0.]])
        self.now+=1/60;self.display.present_if_due()
        self.assertAlmostEqual(self.draws[-1][0]['heading'],math.pi)
        np.testing.assert_allclose(np.linalg.norm(self.draws[-1][0]['q'],axis=-1),1.)
        np.testing.assert_allclose(self.draws[-1][0]['q'][0,2:],np.sqrt(.5),atol=1e-12)

    def test_render_clock_does_not_repaint_early_and_finish_reaches_endpoint(self):
        self.push(0,[0.,10.]);self.display.present_if_due()
        self.now=.001;self.assertFalse(self.display.present_if_due())
        self.now=1/30;self.push(1,[2.,12.]);self.display.finish()
        self.assertEqual(self.draws[-1][0]['p'][0,0],2.)
        self.assertEqual(self.draws[-1][1]['p'][0,0],12.)
        self.assertEqual(self.draws[-1][1]['frame'],1)

    def test_rejects_actor_count_changes(self):
        self.push(0,[0.,10.])
        with self.assertRaises(ValueError):self.push(1,[0.,10.,20.],restarts=[False]*3,headings=[0.]*3)

if __name__=='__main__':unittest.main()
