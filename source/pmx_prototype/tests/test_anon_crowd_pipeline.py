import sys,unittest,threading
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_crowd_pipeline import CrowdPlaybackPipeline
from anon_secondary_stream import AnonSecondaryStream


class CrowdPipelineTests(unittest.TestCase):
    def run_pipeline(self,prefetch,async_body=False,mixed=False):
        events=[];planned=[];sent=[]
        owner=threading.get_ident()
        class Stream:
            def __init__(self,index):self.index=index;self.queue=[]
            def send(self,seq,state):
                events.append(('secondary_send',seq,self.index));self.queue.append((seq,state))
                assert len(self.queue)<=2
            def receive(self,seq,on_wait=None):
                events.append(('secondary_receive',seq,self.index))
                actual,state=self.queue.pop(0);assert actual==seq
                return seq,state
        streams=[Stream(0),Stream(1)]
        def plan(seq,previous):
            self.assertEqual(threading.get_ident(),owner)
            planned.append((seq,previous['seq']))
            request=dict(seq=seq,commands=[seq,seq+1],controlled_actor=seq%2,reset_ids=[1] if seq==2 else [])
            if mixed:request['secondary_modes']=['light' if i==seq%2 else 'light3' for i in range(2)]
            return request
        def send(request):events.append(('body_send',request['seq']));sent.append(request)
        def receive(seq):
            request=sent[-1];assert request['seq']==seq
            return dict(seq=seq,poses=[seq*10,seq*10+1],resets=[False,seq==2])
        pipeline=CrowdPlaybackPipeline(dict(seq=-1),plan,send,receive,streams,prefetch=prefetch,
            receive_async=receive if async_body else None)
        try:frames=[pipeline.advance(has_next=seq<4) for seq in range(5)]
        finally:pipeline.close()
        self.assertEqual(planned,[(seq,seq-1) for seq in range(5)])
        self.assertTrue(all(not stream.queue for stream in streams))
        self.assertIsNone(pipeline.pending)
        return frames,events

    def test_prefetch_keeps_all_frames_requests_and_reset_routing(self):
        ahead,events=self.run_pipeline(True);serial,_=self.run_pipeline(False)
        self.assertEqual(ahead,serial)
        self.assertLess(events.index(('secondary_receive',0,0)),events.index(('body_send',1)))
        self.assertLess(events.index(('body_send',2)),events.index(('secondary_receive',1,0)))
        self.assertEqual([frame[0]['seq'] for frame in ahead],list(range(5)))
        self.assertTrue(ahead[2][2][1][1]['restart'])
        self.assertFalse(ahead[2][2][0][1]['restart'])

    def test_body_only_lookahead_is_bounded(self):
        sent=[]
        def receive(seq):return dict(seq=seq,poses=[],resets=[])
        pipeline=CrowdPlaybackPipeline(dict(seq=-1),lambda n,p:dict(seq=n,commands=[]),sent.append,receive,[])
        self.assertEqual(pipeline.advance(has_next=False)[0]['seq'],0)
        self.assertEqual(len(sent),1)

    def test_async_dispatch_keeps_all_states_resets_and_planning_on_main_thread(self):
        ahead,events=self.run_pipeline(True,async_body=True);serial,_=self.run_pipeline(False)
        self.assertEqual(ahead,serial)
        self.assertLess(events.index(('secondary_receive',0,0)),events.index(('body_send',1)))
        self.assertEqual([event[1] for event in events if event[0]=='body_send'],list(range(5)))

    def test_mixed_budgets_are_routed_with_the_same_ordered_actor_request(self):
        frames,_=self.run_pipeline(True,async_body=True,mixed=True)
        for seq,(_,request,secondary) in enumerate(frames):
            self.assertEqual(request['controlled_actor'],seq%2)
            for actor,(_,payload) in enumerate(secondary):
                self.assertEqual(payload['secondary_mode'],'light' if actor==seq%2 else 'light3')

    def test_secondary_transport_preserves_optional_budget_without_changing_uniform_requests(self):
        sent=[];stream=SimpleNamespace(channel=SimpleNamespace(send=sent.append))
        AnonSecondaryStream.send(stream,0,dict(poses='pose0',restart=False))
        self.assertNotIn('secondary_mode',sent[-1])
        AnonSecondaryStream.send(stream,1,dict(poses='pose1',restart=True,secondary_mode='light3'),audit=True)
        self.assertEqual(sent[-1],dict(seq=1,poses='pose1',reset=True,audit=True,secondary_mode='light3'))

    def test_slow_next_body_does_not_block_current_secondary_result(self):
        started=threading.Event();release=threading.Event();owner=threading.get_ident();wait_threads=[]
        def receive(seq):return dict(seq=seq,poses=[],resets=[])
        def receive_async(seq):
            self.assertNotEqual(threading.get_ident(),owner);started.set()
            if not release.wait(2):raise TimeoutError('Test did not release body worker')
            return receive(seq)
        pipeline=CrowdPlaybackPipeline(dict(seq=-1),lambda n,p:dict(seq=n,commands=[]),lambda r:None,
            receive,[],receive_async=receive_async)
        try:
            self.assertEqual(pipeline.advance(has_next=True)[0]['seq'],0)
            self.assertFalse(started.is_set())
            self.assertEqual(pipeline.advance(has_next=True)[0]['seq'],1)
            self.assertTrue(started.wait(1));self.assertFalse(pipeline.pending.done())
            release.set()
            self.assertEqual(pipeline.advance(has_next=False,on_wait=lambda:wait_threads.append(threading.get_ident()))[0]['seq'],2)
            self.assertTrue(all(thread==owner for thread in wait_threads))
        finally:release.set();pipeline.close()
        pipeline.close()

    def test_background_body_failure_propagates_and_executor_closes(self):
        def receive(seq):return dict(seq=seq,poses=[],resets=[])
        def failed(seq):raise RuntimeError('Body worker failed')
        pipeline=CrowdPlaybackPipeline(dict(seq=-1),lambda n,p:dict(seq=n,commands=[]),lambda r:None,
            receive,[],receive_async=failed)
        try:
            pipeline.advance(has_next=True);pipeline.advance(has_next=True)
            with self.assertRaisesRegex(RuntimeError,'Body worker failed'):pipeline.advance(has_next=False)
        finally:pipeline.close()
        self.assertIsNone(pipeline.executor)

if __name__=='__main__':unittest.main()
