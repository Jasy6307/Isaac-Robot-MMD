"""One-frame lookahead for policy-driven hair and skirt simulation."""
from pathlib import Path
import subprocess
import sys
import time
import os

_source_dir=str(Path(__file__).resolve().parents[1])
if _source_dir not in sys.path:sys.path.insert(0,_source_dir)
from source_layout import PROJECT_ROOT,resolve_source
from anon_policy_transport import PolicyServer

class AnonSecondaryStream:
    def __init__(self,args,initial=None,manifest=None,links=None,app=None,*,on_wait=None):
        self.output=args.output/'secondary_worker';self.output.mkdir(parents=True,exist_ok=True)
        self.channel=PolicyServer(self.output)
        self.log=(self.output/'worker.log').open('w',encoding='utf-8')
        self.process=None;self.app=app;self.on_wait=on_wait
        self.metadata=None
        cmd=[sys.executable,str(resolve_source('runtime/anon_secondary_worker.py')),'--pmx',str(args.pmx.resolve()),
             '--quality',args.quality,'--physics-threads',str(args.physics_threads if args.physics_threads is not None else 16),
             '--task-threads',os.environ.get('ANON_SECONDARY_TASK_THREADS','8'),
             '--skirt-edge-stride',str(args.skirt_edge_stride),
             '--secondary-mode',getattr(args,'secondary_mode','full'),
             '--output',str(self.output.resolve()),*self.channel.worker_args]
        if args.secondary_profile and not args.legacy_secondary_physics:cmd+=['--profile',str(args.secondary_profile.resolve())]
        if args.reference_hair_compute:cmd+=['--reference-hair-compute']
        if args.legacy_compute:cmd+=['--legacy-compute']
        try:
            self.process=subprocess.Popen(cmd,cwd=PROJECT_ROOT,stdout=self.log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if initial is not None:self.initialize(initial,manifest,links)
        except BaseException:
            self.close();raise

    def initialize(self,initial,manifest,links):
        """Send the unchanged initial pose after overlapping worker Kit startup."""
        deadline=time.perf_counter()+120
        while self.channel.connection is None:
            if self.channel.error:raise self.channel.error
            if self.process.poll() is not None:raise RuntimeError('Secondary worker failed; see '+str(self.output/'worker.log'))
            if time.perf_counter()>deadline:raise TimeoutError('Secondary worker connection timed out')
            if self.on_wait:self.on_wait()
            else:self.app.update()
            time.sleep(.01)
        self.channel.send(dict(poses=initial,manifest=manifest,links=links))
        self.metadata=self.receive(-1,True,on_wait=self.on_wait)
        self.on_wait=None

    def send(self,seq,state,audit=False):
        request=dict(seq=seq,poses=state['poses'],reset=bool(state['restart']),audit=audit)
        if 'secondary_mode' in state:request['secondary_mode']=state['secondary_mode']
        self.channel.send(request)

    def receive(self,seq,startup=False,on_wait=None):
        deadline=time.perf_counter()+120;ui_tick=0.
        while time.perf_counter()<deadline:
            value=self.channel.receive(seq)
            if value is not None:return value
            if self.process.poll() is not None:raise RuntimeError('Secondary worker failed; see '+str(self.output/'worker.log'))
            if on_wait is not None:on_wait()
            elif startup and time.perf_counter()>=ui_tick:self.app.update();ui_tick=time.perf_counter()+.05
            time.sleep(.0005)
        raise TimeoutError('Secondary worker response timed out')

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                if self.channel.connection:self.channel.send(dict(stop=True))
                self.process.wait(timeout=15)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate();self.process.wait(timeout=10)
        self.channel.close();self.log.close()
