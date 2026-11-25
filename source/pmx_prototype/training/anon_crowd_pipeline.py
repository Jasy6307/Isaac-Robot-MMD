"""Bounded one-step lookahead; consume every body and secondary state in order."""


class CrowdPlaybackPipeline:
    def __init__(self,initial,plan,send,receive,streams,*,prefetch=True,receive_async=None):
        self.initial=initial;self.plan=plan;self.send=send;self.receive=receive
        self.streams=streams;self.prefetch=prefetch;self.pending=None;self.sequence=0
        self.receive_async=receive_async;self.executor=None

    def _solve(self,sequence,previous):
        request=self.plan(sequence,previous)
        if request['seq']!=sequence:raise ValueError('Crowd request sequence mismatch')
        return self._dispatch(request,self.receive)

    def _dispatch(self,request,receive):
        sequence=request['seq']
        self.send(request)
        state=receive(sequence)
        if int(state['seq'])!=sequence:raise ValueError('Crowd body sequence mismatch')
        state['requested_commands']=request['commands']
        modes=request.get('secondary_modes')
        if modes is not None and len(modes)!=len(self.streams):raise ValueError('Secondary mode actor count mismatch')
        for index,stream in enumerate(self.streams):
            secondary=dict(poses=state['poses'][index],restart=bool(state['resets'][index]))
            if modes is not None:secondary['secondary_mode']=modes[index]
            stream.send(sequence,secondary)
        return state,request

    def advance(self,*,has_next,on_wait=None):
        sequence=self.sequence
        if self.pending is not None and self.executor is not None:
            import time
            while not self.pending.done():
                if on_wait is not None:on_wait()
                time.sleep(.0005)
            state,request=self.pending.result()
        else:state,request=self.pending if self.pending is not None else self._solve(sequence,self.initial)
        self.pending=None
        # Body N+1 runs alongside secondary N. Secondary N+1 is then queued
        # before rendering N, so both worker stages overlap the frontend.
        # Let the first pose initialize the skinning/Fabric render buffers
        # while workers are idle, before queuing another native solve.
        if self.prefetch and sequence>0 and has_next:
            if self.receive_async is None:self.pending=self._solve(sequence+1,state)
            else:
                # Planning reads Inspector input and must stay on the UI thread.
                # Only transport and secondary dispatch run in the background.
                request_next=self.plan(sequence+1,state)
                if request_next['seq']!=sequence+1:raise ValueError('Crowd request sequence mismatch')
                if self.executor is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='anon-body-lookahead')
                self.pending=self.executor.submit(self._dispatch,request_next,self.receive_async)
        secondary=[stream.receive(sequence,on_wait=on_wait) for stream in self.streams]
        self.initial=state;self.sequence+=1
        return state,request,secondary

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True,cancel_futures=True)
            self.executor=None
