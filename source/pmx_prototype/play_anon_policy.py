"""Live policy inference with native PhysX hair/skirt physics (enabled by default).

Separate processes keep the policy's GPU articulation scene and the secondary
CPU rigid-node scene independent. Body motion is obtained from new policy steps,
never a saved rollout. Secondary forces do not feed back into the policy.
"""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
from anon_policy_catalog import PRESETS
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,default=PRESETS['walk'])
parser.add_argument('--pmx',type=Path,default=ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
parser.add_argument('--headless',action='store_true')
parser.add_argument('--steps',type=int,default=0,help='Finite 30 Hz steps; 0 keeps the interactive viewer open')
parser.add_argument('--demo',action='store_true',help='Autonomous forward, reverse, sidestep, turn and stop sequence')
parser.add_argument('--inspector',action='store_true',help='Visual keyboard inspector; WASD/QE, +/- speed, Space stop, R reset')
parser.add_argument('--record-video',type=Path)
parser.add_argument('--no-secondary-physics',action='store_true')
parser.add_argument('--secondary-profile',type=Path,default=DEFAULT_SECONDARY_PROFILE,help='JSON appearance physics settings')
parser.add_argument('--quality',choices=['hair','cloth'],default=None)
parser.add_argument('--legacy-secondary-physics',action='store_true',help='Use the previous thick collision and rigid skirt profile for comparison')
parser.add_argument('--policy-config-override',type=Path,help='Explicit controller configuration override for controlled comparisons')
parser.add_argument('--appearance',choices=['default','bright-soft'],default='default',help='Use the shared default or explicitly lock the bright soft-light look')
parser.add_argument('--video-width',type=int,default=1920)
parser.add_argument('--video-height',type=int,default=1080)
parser.add_argument('--anti-aliasing',type=int,choices=[0,1,2,3,4],default=1,help='Isaac modes: 1 native-resolution TAA; 3 DLSS')
parser.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/live_secondary')
parser.add_argument('--body-device',choices=['cpu','cuda:0'],default='cuda:0',help='Body physics and policy inference device; secondary physics stays on CPU')
parser.add_argument('--transport',choices=['memory','files'],default='memory')
parser.add_argument('--legacy-compute',action='store_true',help='A/B check: retain redundant training diagnostics and substep pose reconstruction')
parser.add_argument('--physics-threads',type=int,help='CPU PhysX worker threads; 0 runs synchronously')
parser.add_argument('--worker-physics-threads',type=int,default=0,help='The single-body inference worker needs no parallel PhysX task pool')
parser.add_argument('--benchmark-warmup',type=int,default=30,help='Frames excluded from steady-state timing')
parser.add_argument('--runtime-audit',action='store_true',help='Save body and rendered skeleton arrays for behavior equivalence checks')
parser.add_argument('--async-render',action='store_true',help='Overlap live rendering with simulation; recording remains synchronous')
parser.add_argument('--save-preview',action='store_true',help='Save one final viewport image after the measured loop')
parser.add_argument('--control-script',type=Path,help='Reproducible switch/command events: JSON list of frame, switch_to, command, reset')
args=parser.parse_args()
if args.inspector and (args.headless or args.demo):parser.error('--inspector needs a visible window and cannot be combined with --demo')
args.quality=args.quality or ('hair' if args.legacy_secondary_physics else DEFAULT_SECONDARY_QUALITY)
W,H=args.video_width,args.video_height
if W%2 or H%2:parser.error('Video dimensions must be even')
args.checkpoint=args.checkpoint.resolve(); args.output=args.output.resolve()
policy_config=json.loads((args.checkpoint.parent/'env_config.json').read_text(encoding='utf-8'))
if args.policy_config_override:policy_config.update(json.loads(args.policy_config_override.read_text(encoding='utf-8')))
from anon_config_compat import clean_legacy_config
policy_config=clean_legacy_config(policy_config)
version=policy_config.get('natural_gait_version',5)
args.output.mkdir(parents=True,exist_ok=True)
ipc=args.output/('ipc_'+uuid.uuid4().hex[:8]); ipc.mkdir()
worker_log=(args.output/'policy_worker.log').open('w',encoding='utf-8')
worker=None
from anon_policy_transport import PolicyServer
channel=PolicyServer(ipc,args.transport)
from isaacsim import SimulationApp
extra=['--/app/asyncRendering=false']
if args.async_render and not args.record_video:
    extra=['--/app/asyncRendering=true','--/app/asyncRenderingLowLatency=true','--/app/omni.usd/asyncHandshake=true','--/omni/replicator/asyncRendering=true']
if args.physics_threads is not None:extra.append(f'--/persistent/physics/numThreads={args.physics_threads}')
app=SimulationApp({'headless':args.headless,'width':W,'height':H,'renderer':'RaytracedLighting',
                   'anti_aliasing':args.anti_aliasing,'extra_args':extra})
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
import omni.usd
import omni.ui as ui
from pxr import Gf,UsdGeom,UsdLux,UsdSkel,Vt
from omni.kit.viewport.utility import get_active_viewport
from physx_secondary import BodyMotion,PhysxSecondary
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb


class LiveBodyMotion(BodyMotion):
    live=None
    def pose(self,frame):
        return super().pose(frame) if self.live is None else self.live


def atomic_request(data):
    channel.send(data)


def receive(seq,timeout=120):
    deadline=time.perf_counter()+timeout
    next_ui_update=time.perf_counter()+.05
    while time.perf_counter()<deadline:
        if (ipc/'error.txt').exists(): raise RuntimeError((ipc/'error.txt').read_text(encoding='utf-8'))
        if worker.poll() is not None: raise RuntimeError('Policy worker exited; see '+str(args.output/'policy_worker.log'))
        value=channel.receive(seq)
        if value is not None:return value
        if time.perf_counter()>=next_ui_update:
            app.update(); next_ui_update=time.perf_counter()+.05
        time.sleep(.001)
    raise TimeoutError('Policy worker response timed out')


physics=encoder=inspector=None
queued_switch='';switch_inflight=False
def request_switch(preset):
    global queued_switch,switch_inflight
    queued_switch=preset;switch_inflight=True
result=0
try:
    worker_args=[sys.executable,str(resolve_source('anon_policy_worker.py')),'--checkpoint',str(args.checkpoint),'--ipc',str(ipc),'--body-device',args.body_device,*channel.worker_args]
    if args.legacy_compute:worker_args+=['--legacy-compute']
    worker_args+=['--physics-threads',str(args.worker_physics_threads)]
    if args.policy_config_override:worker_args+=['--config-override',str(args.policy_config_override.resolve())]
    if args.inspector or args.control_script:worker_args+=['--preload-presets']
    hand_trace=[];command_trace=[];body_trace=[];skeleton_p=[];skeleton_q=[];policy_trace=[];switches=[];fall_events=[];identities=set()
    worker=subprocess.Popen(worker_args,
                            stdout=worker_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    state=receive(-1)
    active_policy=str(state['policy_id']);active_checkpoint=str(state['checkpoint'])
    catalog=json.loads(str(state['catalog_json']))
    policy_config=json.loads(str(state['config_json']))
    motion=LiveBodyMotion(ROOT/'output/pmx_prototype/character.usdc',args.pmx,secondary_profile=args.secondary_profile if not args.legacy_secondary_physics else None)
    motion.rig_settings['max_node_distance_from_target_m']=2.
    manifest=json.loads((Path(policy_config['asset']).parent/'manifest.json').read_text(encoding='utf-8'))
    assert list(manifest['bone_paths'])==motion.joints,'Skeleton order mismatch'
    links=list(state['link_names']); pelvis=links.index('pelvis')
    owners=np.array([links.index(n) for n in manifest['bone_owners']])
    rest=np.array(manifest['rest_positions']); origins=np.array([manifest['links'][n]['origin'] for n in manifest['bone_owners']])
    parents=np.array(manifest['parents']); children=np.flatnonzero(parents>=0)
    def body_pose(transforms):
        t=transforms[owners]; rotations=Rotation.from_quat(t[:,3:])
        wp=t[:,:3]+rotations.apply(rest-origins); wq=t[:,3:].copy()
        p,q=wp.copy(),wq.copy(); parent=parents[children]
        inverse=Rotation.from_quat(wq[parent]).inv()
        p[children]=inverse.apply(wp[children]-wp[parent])
        q[children]=(inverse*Rotation.from_quat(wq[children])).as_quat()
        return p,q,wp,wq
    motion.live=body_pose(state['poses'])
    context=omni.usd.get_context(); context.new_stage(); stage=context.get_stage()
    UsdGeom.SetStageUpAxis(stage,'Z'); UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.DefinePrim('/World/Character').GetReferences().AddReference(str(ROOT/'output/pmx_prototype/character.usdc'))
    animation=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
    def publish(p,q):
        animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
    publish(*motion.live[:2])
    if not args.no_secondary_physics: physics=PhysxSecondary(stage,motion,obstacles=False,quality=args.quality)
    if physics:
        floor=UsdGeom.Xformable(stage.GetPrimAtPath('/World/PhysxGround'))
        next(op for op in floor.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(1000,1000,.05))
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900)
    light=UsdLux.DistantLight.Define(stage,'/World/Key'); light.CreateIntensityAttr(1600)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(25,-25,-35))
    ground=UsdGeom.Mesh.Define(stage,'/World/VisibleGround')
    ground.CreatePointsAttr([(-50,-50,-.006),(50,-50,-.006),(50,50,-.006),(-50,50,-.006)])
    ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0,1,2,3]); ground.CreateSubdivisionSchemeAttr('none')
    ground.CreateDisplayColorAttr([(.24,.28,.32)])
    from character_appearance import DefaultCharacterLook,apply_bright_character_look
    import inspect,hashlib
    look_source=inspect.getsource(apply_bright_character_look)
    appearance=None if args.appearance=='bright-soft' else DefaultCharacterLook(stage,state['poses'][pelvis,:3])
    face_softbox=apply_bright_character_look(stage,state['poses'][pelvis,:3]) if appearance is None else appearance.face_light
    secondary_source_hash=hashlib.sha256(Path(sys.modules[PhysxSecondary.__module__].__file__).read_bytes()).hexdigest()
    camera=UsdGeom.Camera.Define(stage,'/World/Camera'); camera.CreateFocalLengthAttr(42)
    camera.CreateHorizontalApertureAttr(36); camera.CreateVerticalApertureAttr(36*H/W)
    camera_op=camera.AddTransformOp()
    viewport=get_active_viewport(); viewport.camera_path='/World/Camera'; viewport.set_texture_resolution((W,H))
    for _ in range(30): app.update()
    if physics:
        physics.start(context.get_stage_id())
        for _ in range(physics.physics_hz): physics.step(0,produce_pose=args.legacy_compute,diagnostics=args.legacy_compute)
        physics.runtime_costs[:]=0;physics.runtime_calls=0
    models=[ui.SimpleFloatModel(0) for _ in range(3)]
    if args.inspector:
        from anon_inspector_ui import PolicyInspector
        inspector=PolicyInspector(policy_config,args.checkpoint,args.output,request_switch,catalog,active_policy)
    elif not args.headless:
        panel=ui.Window('Anon live policy | Hair and skirt PhysX',width=390,height=230)
        with panel.frame:
            with ui.VStack(spacing=7):
                ui.Label(f'Live v{version} policy / native PhysX hair and skirt')
                maximum=policy_config.get('max_speed',.4);yaw_max=policy_config.get('max_yaw',.6)
                for name,model,low,high in zip(['Forward m/s','Left m/s','Yaw rad/s'],models,[-maximum,-maximum,-yaw_max],[maximum,maximum,yaw_max]):
                    ui.Label(name); ui.FloatSlider(model=model,min=low,max=high)
                def stop():
                    for model in models: model.set_value(0.)
                ui.Button('Stop',clicked_fn=stop)
    schedule=[(2,(0,0,0)),(6,(.35,0,0)),(2,(0,0,0)),(4,(-.25,0,0)),(4,(0,.3,0)),(4,(0,-.3,0)),
              (4,(.2,0,.6)),(4,(.2,0,-.6)),(4,(0,0,.6)),(3,(0,0,0))]
    if policy_config.get('motion_stage') in ['forward_speeds','forward_speeds_aligned']:
        schedule=[(3,(0,0,0)),(6,(.4,0,0)),(6,(.6,0,0)),(6,(.79330927,0,0)),(3,(0,0,0)),(6,(.4,0,0)),(4,(0,0,0))]
    elif policy_config.get('motion_stage')=='native_forward':schedule=[(30,(.79330927,0,0))]
    elif policy_config.get('motion_stage') in ['amp_forward','amp_running']:schedule=[(3,(0,0,0)),(30,(policy_config.get('amp_forward_speed',.4),0,0)),(4,(0,0,0))]
    elif policy_config.get('motion_stage')=='complete_motion_transitions':
        schedule=[(3,(0,0,0)),(6,(.79330927,0,0)),(6,(.6,0,0)),(8,(.4,0,0)),(4,(0,0,0)),
                  (6,(.4,0,0)),(6,(.6,0,0)),(6,(.79330927,0,0)),(4,(0,0,0)),
                  (6,(-.4,0,0)),(2,(0,0,0)),(6,(0,.35,0)),(6,(0,-.35,0)),(3,(0,0,0)),
                  (4,(.15,0,.6)),(4,(.15,0,-.6)),(4,(0,0,.6)),(4,(0,0,-.6)),(3,(0,0,0))]
    if policy_config.get('imitation_command_scope')=='translation_040_yaw_060':
        schedule=[(3,(0,0,0)),(10,(.4,0,0)),(4,(0,0,0)),(6,(.4,0,0)),(6,(-.4,0,0)),
                  (3,(0,0,0)),(6,(0,.35,0)),(6,(0,-.35,0)),(3,(0,0,0)),(4,(.15,0,.6)),
                  (4,(.15,0,-.6)),(4,(0,0,.6)),(4,(0,0,-.6)),(3,(0,0,0))]
    def command_at(t):
        if policy_config.get('unified_walk_run'):
            from unified_command_suite import demo_command
            return demo_command(t)
        if policy_config.get('continuous_commands') and not policy_config.get('amp_forward_only'):
            from continuous_command_suite import demo_command
            return demo_command(t,pivots=policy_config.get('demo_pivots',False))
        for seconds,cmd in schedule:
            if t<seconds:return cmd
            t-=seconds
        return (0,0,0)
    if args.record_video:
        ff=shutil.which('ffmpeg') or str(next((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))
        args.record_video=args.record_video.resolve(); args.record_video.parent.mkdir(parents=True,exist_ok=True)
        encoder=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','30','-i','pipe:0',
                                  '-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(args.record_video)],stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',round(23*H/720))
    deviations={'hair':[],'skirt':[]}; costs=[]; details=[];frame_intervals=[]; completed=falls=0; started=time.perf_counter(); last=state['poses']
    limit=args.steps or (sum(x[0] for x in schedule)*30 if args.headless and args.demo else 0)
    script_events={} if not args.control_script else {int(e['frame']):e for e in json.loads(args.control_script.read_text(encoding='utf-8'))}
    script_command=[0.,0.,0.];script_reset=False
    if not args.steps and args.headless and args.demo and policy_config.get('continuous_commands') and not policy_config.get('amp_forward_only'):
        if policy_config.get('unified_walk_run'):
            from unified_command_suite import demo_seconds
            limit=round(demo_seconds()*30)
        else:
            from continuous_command_suite import segments
            limit=round(sum(s for _,s in segments(True,pivots=policy_config.get('demo_pivots',False)))*30)
    def choose_command(index):
        global script_command,script_reset
        if index in script_events:
            event=script_events[index]
            if event.get('switch_to'):
                script_command=[0.,0.,0.];request_switch(event['switch_to'])
            if 'command' in event:script_command=event['command']
            script_reset=bool(event.get('reset',False))
        cmd=script_command if args.control_script else inspector.command() if inspector else command_at(index/30) if args.demo else tuple(m.get_value_as_float() for m in models)
        if switch_inflight:cmd=[0.,0.,0.]
        cmd=np.array(cmd,dtype=float); cmd[:2]*=min(1.,policy_config.get('max_speed',.4)/max(np.linalg.norm(cmd[:2]),1e-6)); cmd[2]=np.clip(cmd[2],-policy_config.get('max_yaw',.6),policy_config.get('max_yaw',.6))
        return cmd.tolist()
    def send_command(seq,command):
        global queued_switch,script_reset
        data=dict(seq=seq,command=command,reset=inspector.controls.consume_reset() if inspector else script_reset)
        if queued_switch:data['switch_to']=queued_switch;queued_switch=''
        atomic_request(data);script_reset=False
    pending_command=choose_command(0)
    send_command(0,pending_command)
    while app.is_running() and (not limit or completed<limit):
        tick=time.perf_counter(); cmd=pending_command
        state=receive(completed,30)
        identities.add((int(state['worker_pid']),int(state['physics_stage_id']),int(context.get_stage_id()),id(physics)))
        if str(state['policy_id'])!=active_policy:
            switches.append(dict(frame=completed,previous=active_policy,policy=str(state['policy_id']),generation=int(state['generation'])))
            active_policy=str(state['policy_id']);active_checkpoint=str(state['checkpoint'])
            policy_config=json.loads(str(state['config_json']))
            switch_inflight=False
            if inspector:inspector.apply_policy(policy_config,active_policy)
        if str(state.get('switch_error','')):
            switch_inflight=False
            switches.append(dict(frame=completed,error=str(state['switch_error']),policy=active_policy))
        if str(state.get('switch_complete','')):
            switch_inflight=False
            if inspector:inspector.apply_policy(policy_config,active_policy)
        policy_trace.append(active_policy)
        if 'command' in state:cmd=state['command'].tolist()
        policy_done=time.perf_counter()
        # The next policy step can overlap this frame's secondary physics/render.
        # The display still consumes every state in sequence; none are skipped.
        if not limit or completed+1<limit:
            pending_command=choose_command(completed+1)
            send_command(completed+1,pending_command)
        current=state['poses']; falls+=int(state['done'])
        if bool(state['done']):fall_events.append(dict(frame=completed,policy=active_policy))
        hand_world=current[[links.index('left_hand'),links.index('right_hand')],:3]-current[pelvis,:3]
        hand_trace.append(Rotation.from_quat(current[pelvis,3:]).inv().apply(hand_world))
        command_trace.append(cmd)
        body_started=time.perf_counter()
        if physics:
            if bool(state['done']) or bool(state.get('reset',False)):
                motion.live=body_pose(current); physics.initial_nodes=motion.live[2][motion.nodes].copy(); physics.reset(); last=current.copy()
            qa,qb=last[:,3:],current[:,3:].copy(); qb[(qa*qb).sum(-1)<0]*=-1
            substeps=physics.physics_hz//30
            for substep in range(substeps):
                alpha=(substep+1)/substeps
                interpolated=last*(1-alpha)+current*alpha
                interpolated[:,3:]=qa*(1-alpha)+qb*alpha
                interpolated[:,3:]/=np.linalg.norm(interpolated[:,3:],axis=-1,keepdims=True)
                motion.live=body_pose(interpolated)
                p,q=physics.step(0,produce_pose=args.legacy_compute or substep==substeps-1,diagnostics=args.legacy_compute)
            publish(p,q)
            target=motion.live[2][motion.nodes][physics.dynamic_order]
            distance=np.linalg.norm(physics.last_positions-target,axis=-1)
            groups=motion.system['groups'][physics.dynamic_order]
            for group,name in [(14,'hair'),(8,'skirt')]:deviations[name].append(float(np.sqrt(np.mean(distance[groups==group]**2))))
        else:
            motion.live=body_pose(current); publish(*motion.live[:2])
            p,q=motion.live[:2]
        if args.runtime_audit:
            body_trace.append(current.copy());skeleton_p.append(p.copy());skeleton_q.append(q.copy())
        secondary_finished=time.perf_counter()
        center=current[pelvis,:3].astype(float); center[2]=.88
        UsdGeom.Xformable(face_softbox).GetOrderedXformOps()[0].Set(
            Gf.Vec3d(float(center[0]),float(center[1]-2.5),1.7))
        offset=inspector.camera_offset() if inspector else [2.3,-4,1.15]
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center+np.array(offset))),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse())
        if inspector:inspector.update(state,falls,completed+1,time.perf_counter()-started)
        look_started=time.perf_counter()
        if appearance is not None:appearance.update(completed)
        secondary_done=time.perf_counter()
        details.append([(secondary_finished-body_started)*1000,(look_started-secondary_finished)*1000,(secondary_done-look_started)*1000])
        app.update()
        if completed==0:
            for _ in range(5):app.update()
        render_done=time.perf_counter()
        if encoder:
            rgb=_capture_viewport_rgb()
            if rgb is None:raise RuntimeError('Viewport capture failed')
            im=Image.fromarray(rgb); draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,W,round(85*H/720)),fill=(17,23,33))
            draw.text((round(24*W/1280),round(10*H/720)),f'爱音 · {catalog[active_policy]["label"]} + 头发 / 裙摆 PhysX',font=font,fill=(245,247,250))
            draw.text((round(24*W/1280),round(47*H/720)),f'前后 {cmd[0]:+.2f} m/s  左右 {cmd[1]:+.2f} m/s  转向 {cmd[2]:+.2f} rad/s',font=font,fill=(120,218,255))
            draw.rectangle((0,round(681*H/720),W,H),fill=(17,23,33))
            draw.text((round(24*W/1280),round(687*H/720)),f'在线策略推理 | {physics.physics_hz if physics else 0} Hz 头发与裙摆物理 | 外观物理单向驱动',font=font,fill=(220,227,235))
            encoder.stdin.write(np.asarray(im).tobytes())
            if completed%150==0:im.save(args.output/f'frame_{completed:04d}.jpg')
        last=current.copy(); completed+=1
        costs.append([(policy_done-tick)*1000,(secondary_done-policy_done)*1000,(render_done-secondary_done)*1000,(time.perf_counter()-render_done)*1000])
        if completed%150==0:print('ANON_LIVE_SECONDARY',completed,flush=True)
        if not args.headless and not encoder:time.sleep(max(0,1/30-(time.perf_counter()-tick)))
        frame_intervals.append((time.perf_counter()-tick)*1000)
    if encoder:
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Video encoder failed')
        encoder=None
    np.savez_compressed(args.output/'policy_trace.npz',hands=hand_trace,commands=command_trace,policy_ids=policy_trace,dt=1/30)
    if args.runtime_audit:np.savez_compressed(args.output/'runtime_audit.npz',body=body_trace,translations=skeleton_p,rotations=skeleton_q)
    if args.save_preview:
        for _ in range(5):app.update()
        rgb=_capture_viewport_rgb()
        if rgb is None:raise RuntimeError('Final preview capture failed')
        Image.fromarray(rgb).save(args.output/'preview.png')
    measured=np.sum(np.asarray(costs)[min(args.benchmark_warmup,max(0,len(costs)-1)):],axis=1)
    report=dict(method='Live PPO inference in isolated PhysX worker; native CPU PhysX secondary rigid-node simulation; no rollout input',
                body_device=args.body_device,transport=args.transport,legacy_compute=args.legacy_compute,physics_threads=args.physics_threads,
                worker_physics_threads=args.worker_physics_threads,
                asynchronous_rendering=bool(args.async_render and not args.record_video),
                benchmark=dict(warmup_frames=min(args.benchmark_warmup,max(0,len(costs)-1)),frames=len(measured),fps=1000/float(np.mean(measured)),frame_ms_p50=float(np.percentile(measured,50)),frame_ms_p95=float(np.percentile(measured,95)),frame_ms_p99=float(np.percentile(measured,99))),
                delivered_fps=1000/float(np.mean(frame_intervals[min(args.benchmark_warmup,max(0,len(frame_intervals)-1)):])),
                secondary_cost_ms=dict(zip(['prepare','simulate','fetch','fabric','reconstruct','contact_feedback_and_diagnostics'],(physics.runtime_costs/max(1,physics.runtime_calls)).tolist())) if physics else {},
                secondary_diagnostics_collected=args.legacy_compute,
                frame_detail_ms=dict(zip(['secondary_and_skeleton','camera_and_ui','appearance'],np.mean(details[min(args.benchmark_warmup,max(0,len(details)-1)):],axis=0).tolist())),
                character_appearance='bright_soft' if args.appearance=='bright-soft' else 'shared_default',
                appearance_source_sha256=hashlib.sha256(look_source.encode()).hexdigest(),secondary_source_sha256=secondary_source_hash,
                runtime_compensation_removed=True,
                checkpoint=str(args.checkpoint),final_checkpoint=active_checkpoint,models=catalog,policy_switches=switches,
                fall_events=fall_events,
                scene_identities=[list(identity) for identity in identities],
                frames=completed,simulated_seconds=completed/30,falls=falls,
                video_resolution=[W,H],video_crf=16,
                anti_aliasing=args.anti_aliasing,
                wall_seconds=time.perf_counter()-started,secondary_physics=physics is not None,
                mean_stage_cost_ms=dict(zip(['policy_wait','secondary','render','capture'],np.mean(costs,axis=0).tolist() if costs else [0.]*4)),
                inspector_mode=args.inspector,
                inference_pipeline='One step ahead; every body state consumed in order',
                secondary_steps=physics.steps if physics else 0,quality=args.quality,
                physics_hz=physics.physics_hz if physics else None,
                secondary_substeps_per_frame=physics.physics_hz//30 if physics else 0,
                dynamic_nodes=physics.dynamic_view.count if physics else 0,
                secondary_profile=str(args.secondary_profile.resolve()) if args.secondary_profile and not args.legacy_secondary_physics else 'previous_default',
                secondary_settings=motion.rig_settings,
                hand_collision_body_indices=motion.hand_body_indices if physics else [],
                palm_proxy_dimensions_m={str(i):(2*shape[0]).tolist() for i,shape in motion.hand_proxy_shapes.items()} if physics else {},
                skirt_edge_capsules=len(physics.edge_nodes) if physics else 0,
                hand_skirt_contact_steps=physics.hand_contact_steps if physics else 0,
                edge_hand_contact_steps=physics.edge_hand_contact_steps if physics else 0,
                hand_skirt_contact_peak_n=physics.hand_contact_peak if physics else 0,
                hair_mean_deviation_m=float(np.mean(deviations['hair'])) if physics else 0,
                skirt_mean_deviation_m=float(np.mean(deviations['skirt'])) if physics else 0,
                body_contact_peak_n=physics.body_contact_peak if physics else 0,
                steps_with_body_contact=physics.body_contact_steps if physics else 0,
                limitations=['Rigid-node approximation of PMX hair/skirt; not triangle cloth',
                             'One-way body-to-secondary coupling; no secondary force feedback into RL articulation'])
    if not args.legacy_compute:
        for key in ['hand_skirt_contact_steps','edge_hand_contact_steps','hand_skirt_contact_peak_n','body_contact_peak_n','steps_with_body_contact']:
            report[key]=None
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
except Exception:
    import traceback
    traceback.print_exc(); result=1
finally:
    if inspector:inspector.close()
    if encoder:encoder.kill();encoder.wait()
    if physics:physics.close()
    if worker:
        try:atomic_request(dict(stop=True))
        except (OSError,AttributeError,EOFError):pass
        try:worker.wait(timeout=15)
        except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=10)
    channel.close()
    worker_log.close()
    app.close(exit_code=result)
