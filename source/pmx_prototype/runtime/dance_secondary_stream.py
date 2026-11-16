"""One-frame live physics pipeline, authenticated localhost transport."""
from pathlib import Path
import subprocess,sys,time,threading,queue
_source_dir=str(Path(__file__).resolve().parents[1])
if _source_dir not in sys.path:sys.path.insert(0,_source_dir)
from source_layout import PROJECT_ROOT,resolve_source
from anon_policy_transport import PolicyServer


class DanceSecondaryStream:
    def __init__(self,args,app=None,wait=True):
        self.app=app;self.output=args.output/'secondary_worker';self.output.mkdir(parents=True,exist_ok=True)
        self.channel=PolicyServer(self.output)
        self.log=(self.output/'worker.log').open('w',encoding='utf-8')
        self.process=None;self.seq=0;self.last_state=None;self.waits=[];self.metadata=None;self.closed=False
        self.launched_at=time.perf_counter()
        command=[sys.executable,str(resolve_source('dance_secondary_worker.py')),
            '--usd',str(args.usd.resolve()),'--pmx',str(args.pmx.resolve()),
            '--output',str(self.output.resolve()),'--quality',args.quality,
            '--render-fps',str(30 if getattr(args,'interpolate_60',False) else args.render_fps),'--physics-threads',str(args.physics_threads),*self.channel.worker_args]
        command+=['--startup-settle-seconds',str(args.startup_settle_seconds)]
        command+=['--skirt-edge-stride',str(getattr(args,'skirt_edge_stride',1))]
        for option,value in [('--rig-data',args.rig_data),('--secondary-profile',args.secondary_profile)]:
            if value:command += [option,str(value.resolve())]
        if args.runtime_audit:command+=['--runtime-audit']
        try:
            self.process=subprocess.Popen(command,cwd=PROJECT_ROOT,stdout=self.log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if wait:self.ready(app)
        except BaseException:
            self.close();raise

    def ready(self,app):
        self.app=app
        if self.metadata is None:
            self.metadata=self._receive(-1,startup=True)['metadata']
            self.channel.send(dict(seq=0))
        return self.metadata

    def _receive(self,seq,startup=False):
        deadline=time.perf_counter()+120;ui_tick=0.
        while time.perf_counter()<deadline:
            value=self.channel.receive(seq)
            if value is not None:return value
            if self.process.poll() is not None:
                raise RuntimeError('Secondary physics worker stopped; see '+str(self.output/'worker.log'))
            if startup and time.perf_counter()>ui_tick:
                self.app.update();ui_tick=time.perf_counter()+.05
            time.sleep(.0005)
        raise TimeoutError('Secondary physics worker timed out')

    def next(self,reset=False):
        tick=time.perf_counter();state=self._receive(self.seq);self.seq+=1
        if reset:
            # Drain the single in-flight packet before asking for a reset.
            self.channel.send(dict(seq=self.seq,reset=True))
            state=self._receive(self.seq);self.seq+=1
        # Exactly one request may be in flight while this pose is rendered.
        self.channel.send(dict(seq=self.seq))
        self.last_state=state;self.waits.append((time.perf_counter()-tick)*1000)
        return state

    def start_interpolation(self,state):
        from dance_pose_interpolation import PoseInterpolator
        self.interpolator=PoseInterpolator(state,time.perf_counter())
        self.interpolation_queue=queue.Queue(maxsize=1)
        self.interpolation_stop=threading.Event()
        self.interpolation_reset=threading.Event()
        self.interpolation_reset_waiting=False
        self.interpolation_error=None
        self.interpolation_last_received=time.perf_counter()
        self.interpolation_thread=threading.Thread(target=self._interpolation_pump,daemon=True)
        self.interpolation_thread.start()

    def _interpolation_pump(self):
        # One packet in flight and at most one waiting for display. The pump
        # advances the 30Hz physical stream without waiting for a 60Hz UI tick.
        deadline=time.perf_counter()
        try:
            while not self.interpolation_stop.is_set():
                state=self.channel.receive(self.seq)
                if state is None:
                    if self.process.poll() is not None:raise RuntimeError('Secondary physics worker stopped')
                    self.interpolation_stop.wait(.001)
                    continue
                self.seq+=1
                self.interpolation_last_received=time.perf_counter()
                if not self.interpolation_reset.is_set():
                    while not self.interpolation_stop.is_set() and not self.interpolation_reset.is_set():
                        try:
                            self.interpolation_queue.put(state,timeout=.02)
                            break
                        except queue.Full:
                            pass
                if self.interpolation_stop.is_set():break
                if self.interpolation_reset.is_set():
                    while True:
                        try:self.interpolation_queue.get_nowait()
                        except queue.Empty:break
                    self.interpolation_reset.clear()
                    self.channel.send(dict(seq=self.seq,reset=True))
                    deadline=time.perf_counter()
                    continue
                if state.get('restart'):deadline=time.perf_counter()
                deadline+=1./30.
                if self.interpolation_stop.wait(max(0.,deadline-time.perf_counter())):break
                if self.interpolation_reset.is_set():
                    while True:
                        try:self.interpolation_queue.get_nowait()
                        except queue.Empty:break
                    self.interpolation_reset.clear()
                    self.channel.send(dict(seq=self.seq,reset=True))
                    deadline=time.perf_counter()
                else:
                    self.channel.send(dict(seq=self.seq))
        except Exception as error:
            self.interpolation_error=error

    def next_interpolated(self,reset=False):
        from dance_pose_interpolation import PoseInterpolator
        if reset:
            self.interpolation_reset_waiting=True
            self.interpolation_reset.set()
        if self.interpolation_error is not None:
            raise RuntimeError('Secondary physics stream failed') from self.interpolation_error
        if time.perf_counter()-self.interpolation_last_received>120:
            raise TimeoutError('Secondary physics worker timed out')
        try:state=self.interpolation_queue.get_nowait()
        except queue.Empty:state=None
        if state is not None and (not self.interpolation_reset_waiting or state.get('restart')):
            now=time.perf_counter()
            if state.get('restart'):
                self.interpolation_reset_waiting=False
                self.interpolator=PoseInterpolator(state,now)
                self.last_state=state
                return state
            self.interpolator.push(state,now)
        result=self.interpolator.sample(time.perf_counter())
        self.last_state=result
        self.waits.append(0.)
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        if hasattr(self,'interpolation_stop'):
            self.interpolation_stop.set()
            self.interpolation_thread.join(timeout=2.)
        if self.process is not None and self.process.poll() is None:
            try:
                if self.channel.connection:self.channel.send(dict(stop=True))
                self.process.wait(timeout=15)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate();self.process.wait(timeout=10)
        self.channel.close();self.log.close()
