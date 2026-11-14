"""Regression: contact correction must survive the inextensible chain solve."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import unittest
import numpy as np
from hair_constraints import project_hair
from hair_contact_velocity import correct_velocity


class HairContactTest(unittest.TestCase):
    def test_contact_velocity_blocks_inward_motion_preserving_sliding(self):
        velocity=np.array([[-1.,2.,0.]])
        corrected=correct_velocity.py_func(velocity,np.array([[.051,0.,0.]]),np.array([0]),
            np.array([.003]),np.zeros((1,3)),np.zeros((1,3)),np.array([.05]),
            np.ones((1,1),dtype=bool),np.array([[.3,0.,0.,0.,0.,0.]]),1/120)
        self.assertGreaterEqual(corrected[0,0],.3)
        self.assertLessEqual(corrected[0,0],.38)
        self.assertEqual(corrected[0,1],2.)

    def test_chain_avoids_body_without_stretching_or_moving_skirt(self):
        target=np.array([[0.,0.,0.],[.06,0.,-.08],[.06,0.,-.18],[1.,0.,-1.]])
        p=target.copy();p[1]=[.01,0.,-.1];p[2]=[.01,0.,-.23]
        result=project_hair.py_func(p,target,np.array([0.,1.,1.,1.]),
            np.array([[0.,1.,.1015],[1.,2.,.1015]]),np.array([[0,1,2]]),
            np.array([1,2]),np.empty(0,dtype=np.int64),np.empty((0,2),dtype=np.int64),
            np.full(4,.003),np.array([[0.,0.,-.1]]),np.array([[0.,0.,-.1]]),
            np.array([.04]),np.ones((2,1),dtype=bool),np.empty((0,2),dtype=np.int64),1.015,32,.12)
        self.assertTrue(np.all(np.linalg.norm(np.diff(result[:3],axis=0),axis=1)<=.101501))
        self.assertGreater(np.linalg.norm(result[1]-[0,0,-.1]),.041)
        np.testing.assert_array_equal(result[0],target[0])
        np.testing.assert_array_equal(result[3],target[3])


if __name__=='__main__':unittest.main()
