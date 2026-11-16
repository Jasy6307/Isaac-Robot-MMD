"""Regression: contact correction must survive the inextensible chain solve."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import unittest
import numpy as np
from hair_constraints import project_hair, project_hair_fast, bound_hair_lengths
from hair_contact_velocity import correct_velocity, limit_chain_velocity


class HairContactTest(unittest.TestCase):
    def test_moving_edge_uses_reference_solver(self):
        p=np.array([[0.,0.,0.],[.01,0.,0.],[.02,0.,0.],[0.,0.,.02]])
        args=(p.copy(),np.array([1.,1.,1.,0.]),np.array([[0.,1.,.02],[1.,2.,.02]]),
              np.array([[0,1,2]]),np.array([0,1,2]),np.array([2,3]),np.array([[2,3]]),
              np.full(4,.01),np.empty((0,3)),np.empty((0,3)),np.empty(0),
              np.empty((3,0),dtype=bool),np.empty((0,2),dtype=np.int64),1.015,5,.06)
        np.testing.assert_array_equal(project_hair_fast(p.copy(),*args),project_hair(p.copy(),*args))

    def test_edge_blocks_preserve_ordered_contacts(self):
        rng=np.random.default_rng(513)
        for _ in range(30):
            p=rng.normal(0,.035,(45,3));target=p.copy()
            weights=np.zeros(45);weights[:5]=1
            args=(target,weights,np.array([[i,i+1,.04] for i in range(4)]),
                  np.array([[i,i+1,i+2] for i in range(3)]),np.arange(5),np.arange(5,45),
                  np.array([[i,i+1] for i in range(5,44)]),np.full(45,.006),
                  np.empty((0,3)),np.empty((0,3)),np.empty(0),np.empty((5,0),dtype=bool),
                  np.array([[0,2],[1,3]],dtype=np.int64),1.015,12,.06)
            expected=project_hair(p.copy(),*args)
            actual=project_hair_fast(p.copy(),*args)
            np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-15)
            np.testing.assert_array_equal(actual.astype(np.float32),expected.astype(np.float32))

    def test_compiled_ordered_length_cap_matches_reference(self):
        rng=np.random.default_rng(42)
        for _ in range(20):
            points=rng.normal(size=(70,3))
            links=np.array([[i,i+1,.2+i*.001] for i in range(69)])
            expected=points.copy();changed=np.zeros(70,dtype=bool)
            for a,b,length in links:
                a=int(a);b=int(b);delta=expected[b]-expected[a];distance=np.linalg.norm(delta)
                if distance>length:
                    expected[b]=expected[a]+delta*(length/distance);changed[b]=True
            result,mask=bound_hair_lengths(points.copy(),np.zeros(70,dtype=bool),links)
            np.testing.assert_allclose(result,expected,rtol=0,atol=1e-14)
            np.testing.assert_array_equal(mask,changed)

    def test_edge_broadphase_keeps_grazing_contacts(self):
        for axis in (0,1):
            for sign in (-1.,1.):
                for offset in (-1e-12,0.,1e-12):
                    p=np.array([[0.,0.,0.],[0.,0.,-.1],[0.,0.,.1]])
                    p[0,axis]=sign*(.009+offset)
                    args=(p.copy(),np.array([1.,0.,0.]),np.empty((0,3)),np.empty((0,3),dtype=np.int64),np.array([0]),np.array([1,2]),np.array([[1,2]]),np.full(3,.003),np.empty((0,3)),np.empty((0,3)),np.empty(0),np.empty((1,0),dtype=bool),np.empty((0,2),dtype=np.int64),1.015,48,0.)
                    np.testing.assert_array_equal(project_hair_fast(p.copy(),*args),project_hair(p.copy(),*args))

    def test_contact_broadphase_matches_original_ordered_solver(self):
        rng=np.random.default_rng(22)
        for bend in (0.,.06):
            for _ in range(12):
                target=rng.normal(0,.05,(11,3));target[9:]=[[2.,0.,-.1],[2.,0.,.1]]
                p=target+rng.normal(0,.01,target.shape)
                args=(target,np.array([0.]+[1.]*5+[0.]*5),
                    np.array([[i,i+1,.08] for i in range(5)]),
                    np.array([[i,i+1,i+2] for i in range(4)]),
                    np.arange(1,6),np.arange(6,11),np.array([[6,7],[7,8],[9,10]]),
                    np.full(11,.003),np.array([[0.,0.,-.1]]),np.array([[0.,0.,.1]]),
                    np.array([.04]),np.ones((5,1),dtype=bool),np.array([[1,3],[2,4],[3,5]]),1.015,48,bend)
                fast=project_hair_fast(p.copy(),*args);reference=project_hair(p.copy(),*args)
                # Scalar FP64 expressions differ only by rounding from BLAS
                # 3-vector norms; the PhysX FP32 state must stay identical.
                np.testing.assert_allclose(fast,reference,rtol=2e-14,atol=2e-15)
                np.testing.assert_array_equal(fast.astype(np.float32),reference.astype(np.float32))
                np.testing.assert_array_equal(fast[6:],reference[6:])

    def test_compiled_chain_velocity_preserves_pin_and_tangential_sliding(self):
        particles=np.array([[0.,0.,0.],[1.,0.,0.]])
        velocity=np.array([[.2,.3,0.],[2.,3.,0.]])
        result=limit_chain_velocity(velocity,particles,np.array([0.,1.]),np.array([[0.,1.,1.]]))
        np.testing.assert_allclose(result,[[.2,.3,0.],[.2,3.,0.]],atol=1e-14)

    def test_compiled_chain_velocity_leaves_slack_strand_free(self):
        velocity=np.array([[0.,0.,0.],[2.,3.,0.]])
        result=limit_chain_velocity(velocity.copy(),np.array([[0.,0.,0.],[.5,0.,0.]]),np.array([0.,1.]),np.array([[0.,1.,1.]]))
        np.testing.assert_array_equal(result,velocity)

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
