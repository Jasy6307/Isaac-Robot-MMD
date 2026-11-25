"""Independent seeded commands and the actual body-to-PMX coordinate mapping."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_crowd_commands import RandomCrowdCommands,BodyPoseMapper


class CrowdCommandTests(unittest.TestCase):
    def test_seeded_independent_schedules_stand_first_and_respect_speed_limits(self):
        a=RandomCrowdCommands(4,42);b=RandomCrowdCommands(4,42)
        positions=np.zeros((4,2));headings=np.zeros(4)
        np.testing.assert_array_equal(a.sample(0.,positions,headings),np.zeros((4,3)))
        history=[]
        for step in range(300):
            result=a.sample(step/30,positions,headings)
            np.testing.assert_array_equal(result,b.sample(step/30,positions,headings))
            self.assertLessEqual(np.linalg.norm(result[:,:2],axis=1).max(),.550001)
            self.assertLessEqual(np.abs(result[:,2]).max(),.400001)
            history.append(result)
        self.assertTrue(any(not np.array_equal(row[0],row[1]) for row in history))
        a.reset();np.testing.assert_array_equal(a.sample(0.,positions,headings),np.zeros((4,3)))

    def test_boundary_guard_requests_world_inward_motion_in_the_heading_frame(self):
        positions=np.array([[7.,0.],[0.,-7.]])
        headings=np.array([np.pi/2,-np.pi/3])
        commands=RandomCrowdCommands(2).sample(2.,positions,headings)
        f,l=commands[:,:2].T;c=np.cos(headings);s=np.sin(headings)
        world=np.stack([s*f+c*l,-c*f+s*l],axis=1)
        self.assertTrue(np.all((positions*world).sum(axis=1)<0.))
        np.testing.assert_array_equal(commands[:,2],np.zeros(2))

    def test_pause_returns_zero_and_small_speed_is_supported(self):
        sampler=RandomCrowdCommands(1,max_speed=.05)
        pos=np.zeros((1,2));head=np.zeros(1)
        self.assertLessEqual(np.linalg.norm(sampler.sample(2.,pos,head)[0,:2]),.050001)
        np.testing.assert_array_equal(sampler.sample(3.,pos,head,paused=True),np.zeros((1,3)))

    def test_bone_locals_follow_world_body_transforms_without_double_translation(self):
        from scipy.spatial.transform import Rotation
        manifest=dict(bone_owners=['pelvis','chest'],rest_positions=[[0,0,0],[0,0,1]],
            links={'pelvis':{'origin':[0,0,0]},'chest':{'origin':[0,0,1]}},parents=[-1,0])
        quat=Rotation.from_euler('z',90,degrees=True).as_quat()
        poses=np.array([[3.,4.,.8,*quat],[3.,4.,1.8,*quat]])
        p,q,wp,wq=BodyPoseMapper(manifest,['pelvis','chest'])(poses)
        np.testing.assert_allclose(p,[[3,4,.8],[0,0,1]],atol=1e-12)
        np.testing.assert_allclose(q[1],[0,0,0,1],atol=1e-12)
        np.testing.assert_allclose(wp,poses[:,:3],atol=1e-12)


if __name__=='__main__':unittest.main()
