"""Pose interpolation preserves endpoints and never mutates physics states."""
import sys, unittest, time, threading
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from dance_pose_interpolation import PoseInterpolator,interpolate_pose
from dance_secondary_stream import DanceSecondaryStream

class PoseInterpolationTest(unittest.TestCase):
    def test_midpoint_and_quaternion_sign(self):
        a=dict(frame=0.,p=np.array([[0.,0.,0.]]),q=np.array([[0.,0.,0.,1.]]))
        b=dict(frame=1.,p=np.array([[2.,0.,0.]]),q=np.array([[0.,0.,-1.,0.]]))
        p,q=interpolate_pose(a,b,.5)
        np.testing.assert_allclose(p,[[1.,0.,0.]])
        np.testing.assert_allclose(np.abs(q),[[0.,0.,2**-.5,2**-.5]],atol=1e-12)
        np.testing.assert_array_equal(a['p'],[[0.,0.,0.]])
        np.testing.assert_array_equal(interpolate_pose(a,b,1.)[0],b['p'])
    def test_late_state_holds_and_reset_does_not_blend(self):
        a=dict(frame=0.,p=np.zeros((1,3)),q=np.array([[0.,0.,0.,1.]]),physics_steps=0)
        b=dict(frame=1.,p=np.ones((1,3)),q=a['q'],physics_steps=4)
        x=PoseInterpolator(a,0.);x.push(b,1.)
        self.assertAlmostEqual(x.sample(1.+1/60)['frame'],.5)
        self.assertEqual(x.sample(2.)['frame'],1.)
        x.push(dict(a,restart=True),3.)
        self.assertEqual(x.sample(3.01)['frame'],0.)

    def test_pump_paces_fast_worker_and_handles_reset(self):
        class Channel:
            def __init__(self):self.ready=deque();self.requests=[];self.lock=threading.Lock()
            def receive(self,seq):
                with self.lock:
                    if not self.ready:return None
                    value=self.ready.popleft();assert value['seq']==seq;return value
            def send(self,request):
                with self.lock:
                    self.requests.append(request)
                    frame=0. if request.get('reset') else float(request['seq'])
                    self.ready.append(dict(seq=request['seq'],frame=frame,p=np.zeros((1,3)),q=np.array([[0.,0.,0.,1.]]),restart=bool(request.get('reset'))))
        stream=DanceSecondaryStream.__new__(DanceSecondaryStream)
        stream.channel=Channel();stream.process=SimpleNamespace(poll=lambda:None);stream.seq=1;stream.waits=[];stream.last_state=None
        initial=dict(seq=0,frame=0.,p=np.zeros((1,3)),q=np.array([[0.,0.,0.,1.]]),restart=False)
        stream.channel.send(dict(seq=1));stream.start_interpolation(initial)
        try:
            start=time.perf_counter()
            while time.perf_counter()-start<.2:
                stream.next_interpolated();time.sleep(.004)
            self.assertLessEqual(len(stream.channel.requests),8)
            stream.next_interpolated(reset=True)
            deadline=time.perf_counter()+1.
            while time.perf_counter()<deadline:
                state=stream.next_interpolated()
                if state.get('restart'):break
                time.sleep(.004)
            self.assertTrue(state['restart']);self.assertEqual(state['frame'],0.)
        finally:
            stream.interpolation_stop.set();stream.interpolation_thread.join(1.)
            self.assertFalse(stream.interpolation_thread.is_alive())

if __name__=='__main__':unittest.main()
