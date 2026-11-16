"""Bounded prefetch and reset ordering for the live physics pipeline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import unittest
from collections import deque
from types import SimpleNamespace
from dance_secondary_stream import DanceSecondaryStream


class Channel:
    def __init__(self):
        self.ready=deque([dict(seq=0,frame=0,restart=False)])
        self.frame=0;self.requests=[]

    def receive(self,seq):
        if not self.ready:return None
        value=self.ready.popleft()
        assert value['seq']==seq
        return value

    def send(self,request):
        # A second future frame would violate the real-time prefetch bound.
        assert not self.ready
        self.requests.append(request)
        self.frame=0 if request.get('reset') else self.frame+1
        self.ready.append(dict(seq=request['seq'],frame=self.frame,restart=bool(request.get('reset'))))


class StreamTests(unittest.TestCase):
    def client(self):
        value=DanceSecondaryStream.__new__(DanceSecondaryStream)
        value.channel=Channel();value.process=SimpleNamespace(poll=lambda:None)
        value.seq=0;value.waits=[];value.last_state=None
        return value

    def test_prefetch_is_bounded_without_skipping_poses(self):
        stream=self.client()
        for i in range(20):
            self.assertEqual(stream.next()['frame'],i)
            self.assertEqual(len(stream.channel.ready),1)
        self.assertEqual(len(stream.channel.requests),20)

    def test_reset_discards_old_prefetch_and_displays_reset_pose(self):
        stream=self.client();stream.next()
        state=stream.next(reset=True)
        self.assertEqual(state['frame'],0);self.assertTrue(state['restart'])
        self.assertEqual(stream.channel.requests[1],dict(seq=2,reset=True))
        self.assertEqual(stream.next()['frame'],1)
        self.assertEqual(len(stream.channel.ready),1)

    def test_deferred_ready_requests_initial_pose_only_once(self):
        stream=self.client();stream.metadata=None
        stream.channel.ready=deque([dict(seq=-1,metadata=dict(physics_hz=120))])
        first=stream.ready(None);second=stream.ready(None)
        self.assertIs(first,second)
        self.assertEqual(stream.channel.requests,[dict(seq=0)])


if __name__=='__main__':unittest.main()
