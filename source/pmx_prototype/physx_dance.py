"""PhysX-driven PMX hair/skirt, scene collider interaction, 60 FPS rendering."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source, config_path
from pathlib import Path
import argparse
import asyncio
import json
import sys
import time
import traceback
import gc
import math,atexit,os
from collections import deque

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd', type=Path, default=ROOT/'output/pmx_prototype/ji_le_jing_tu.usdc')
parser.add_argument('--pmx', type=Path, default=ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
parser.add_argument('--output', type=Path, default=ROOT/'output/pmx_prototype/physx')
parser.add_argument('--rig-data', type=Path, help='Supplementary model-specific physics rig JSON')
parser.add_argument('--secondary-profile', type=config_path, default=DEFAULT_SECONDARY_PROFILE, help='Physics settings applied before topology/skin initialization; preserves native PMX rig')
parser.add_argument('--legacy-secondary-physics',action='store_true',help='Disable the default secondary profile for historical comparisons')
parser.add_argument('--music', type=Path, default=ROOT/'media/dance/ji_le_jing_tu/ji_le_jing_tu.wav')
parser.add_argument('--record-video', type=Path)
parser.add_argument('--ffmpeg', type=Path)
parser.add_argument('--bake-usd', type=Path)
parser.add_argument('--render-fps', type=int, choices=[30,60], default=60)
parser.add_argument('--width', type=int, default=1000)
parser.add_argument('--height', type=int, default=1000)
parser.add_argument('--reference-clock', action='store_true', help='Record source frame zero as the first video sample; requires benchmark')
parser.add_argument('--fit-camera', action='store_true', help='Fit a fixed frontal camera to this complete body animation')
parser.add_argument('--fit-camera-mesh',action='store_true',help='Fit skinned mesh bounds, excluding unrendered control bones')
parser.add_argument('--look',choices=LOOK_CHOICES, default=DEFAULT_LOOK,help='Default: mmd-toon with the shared cel-studio look')
parser.add_argument('--steps', type=int, default=0)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--no-music', action='store_true')
parser.add_argument('--music-sync-tolerance',type=float,default=.15,help='Maximum estimated audio drift in seconds before resync')
parser.add_argument('--no-obstacles', action='store_true')
parser.add_argument('--no-secondary-physics', action='store_true', help='Preview source body animation without live hair/skirt simulation')
parser.add_argument('--secondary-worker',action='store_true',help='Pipeline live 120Hz secondary physics in a separate process with one-frame prefetch')
parser.add_argument('--interpolate-60',action='store_true',help='Render 60Hz interpolated poses from a 30Hz live physics stream')
parser.add_argument('--startup-settle-seconds',type=float,default=None,help='Hold source frame zero for physics settling before playback; worker default 2s, 0 disables')
parser.add_argument('--serial-startup',action='store_true',help='Startup timing comparison: defer worker launch until the renderer is ready')
parser.add_argument('--no-camera-cache',action='store_true',help='Recompute camera-fit bounds instead of using the USD-content cache')
parser.add_argument('--require-scene-contact', action='store_true')
parser.add_argument('--scene', type=Path, help='Optional USD environment with ordinary Physics CollisionAPI colliders')
parser.add_argument('--capture-frames', type=int, nargs='*', default=[])
parser.add_argument('--legacy-compute', action='store_true', help='A/B check: reconstruct and publish every substep and collect all diagnostics')
parser.add_argument('--collect-diagnostics', action='store_true', help='Collect quality/contact statistics; physical contact feedback always remains enabled')
parser.add_argument('--physics-threads', type=int, default=16, help='CPU PhysX worker threads, matching the walking viewer')
parser.add_argument('--skirt-edge-stride',type=int,choices=[1,2,3],default=1,help='Experimental collision proxy density')
parser.add_argument('--runtime-audit', action='store_true', help='Save per-display-frame particles and skeleton for equivalence checks')
parser.add_argument('--reference-hair-compute', action='store_true', help='A/B check: use previous V10 contact computation with identical parameters')
parser.add_argument('--async-render', action='store_true', help='Overlap live rendering; video recording remains synchronous')
parser.add_argument('--quality', choices=['legacy', 'smooth', 'hair', 'cloth'], default=None, help='Default: cloth; legacy-secondary-physics uses the previous hair default')
args = parser.parse_args()
args.startup_settle_seconds=(2. if args.secondary_worker else 0.) if args.startup_settle_seconds is None else args.startup_settle_seconds
if not math.isfinite(args.startup_settle_seconds) or args.startup_settle_seconds<0:parser.error('startup settle seconds must be finite and nonnegative')
if not math.isfinite(args.music_sync_tolerance) or args.music_sync_tolerance<=0:parser.error('music sync tolerance must be finite and positive')
if args.startup_settle_seconds and not args.secondary_worker:parser.error('startup settling currently requires --secondary-worker')
if args.interpolate_60 and (not args.secondary_worker or args.render_fps!=60):parser.error('interpolate-60 requires a 60 FPS secondary worker preview')
if args.fit_camera_mesh:args.fit_camera=True
args.quality=args.quality or ('hair' if args.legacy_secondary_physics else DEFAULT_SECONDARY_QUALITY)
if args.legacy_secondary_physics:args.secondary_profile=None
if args.require_scene_contact and args.no_obstacles:
    parser.error('require-scene-contact measures the built-in test spheres; do not combine it with no-obstacles')
if args.scene and not args.scene.is_file():
    parser.error(f'Scene USD does not exist: {args.scene}')
if args.reference_clock and not args.benchmark: parser.error('reference-clock requires benchmark')
if args.no_secondary_physics and (args.require_scene_contact or args.bake_usd):
    parser.error('no-secondary-physics cannot perform contact validation or bake a physics cache')
if args.secondary_worker and (args.no_secondary_physics or args.legacy_compute or args.reference_hair_compute or args.collect_diagnostics or args.record_video or args.bake_usd or args.require_scene_contact):
    parser.error('secondary-worker is for live preview/runtime audit; use the inline path for legacy comparisons, recording or contact diagnostics')
if args.secondary_worker and (args.scene or not args.no_obstacles):
    parser.error('secondary-worker currently previews the original ground without test spheres; use --no-obstacles and omit --scene')
args.output.mkdir(parents=True, exist_ok=True)
startup_started=time.perf_counter();startup_phases={};camera_cache_report={};initial_stream_state=None
secondary_worker=None
if args.secondary_worker and not args.serial_startup:
    from dance_secondary_stream import DanceSecondaryStream
    secondary_worker=DanceSecondaryStream(args,wait=False)
    atexit.register(secondary_worker.close)
    startup_phases['worker_launched']=time.perf_counter()-startup_started

from isaacsim import SimulationApp
render_settings=['--/app/asyncRendering=false']
if args.async_render and not args.record_video:
    render_settings=['--/app/asyncRendering=true','--/app/asyncRenderingLowLatency=true','--/app/omni.usd/asyncHandshake=true','--/omni/replicator/asyncRendering=true']
app = SimulationApp({'headless': args.headless, 'width': args.width, 'height': args.height,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3,
                     'limit_cpu_threads':int(os.environ.get('DANCE_FRONTEND_TASK_THREADS','16')),
                     'extra_args': [*render_settings, f'--/persistent/physics/numThreads={args.physics_threads}']})
startup_phases['frontend_app_ready']=time.perf_counter()-startup_started

import numpy as np
import omni.usd
import omni.ui as ui
import omni.kit.app
from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdLux, Vt
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from physx_secondary import BodyMotion, PhysxSecondary

physics = music = encoder = encoder_log = baked_stage = None
gc_probe=None;gc_events=[];gc_ticks={}
try:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.tensors', True)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    motion = BodyMotion(args.usd, args.pmx, rig_data=args.rig_data, secondary_profile=args.secondary_profile)
    motion.rig_settings['preview_skirt_edge_stride']=args.skirt_edge_stride
    startup_phases['frontend_model_ready']=time.perf_counter()-startup_started
    root = UsdSkel.Root.Define(stage, '/World/Character')
    root.GetPrim().GetReferences().AddReference(str(args.usd.resolve()))
    anim = UsdSkel.Animation.Get(stage, '/World/Character/Animation')
    anim.CreateJointsAttr(motion.joints)
    anim.CreateScalesAttr([Gf.Vec3h(1)]*len(motion.joints))
    def publish(p, q):
        anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
        anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    lp, lq, wp, wq = motion.pose(0)
    publish(lp, lq)
    if not (args.no_secondary_physics or args.secondary_worker):
        physics = PhysxSecondary(stage, motion, obstacles=not args.no_obstacles, quality=args.quality)
    else:
        # Match the original ground authored by PhysxSecondary, without
        # creating a physics scene, secondary bodies or their constraints.
        floor=UsdGeom.Cube.Define(stage,'/World/PhysxGround');floor.CreateSizeAttr(1)
        xf=UsdGeom.Xformable(floor)
        xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-.04));xf.AddScaleOp().Set(Gf.Vec3f(10,10,.05))
        floor.CreateDisplayColorAttr([(.18,.21,.26)])
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    if args.scene:
        stage.DefinePrim('/World/Environment').GetReferences().AddReference(str(args.scene.resolve()))
    dome = UsdLux.DomeLight.Define(stage, '/World/Light'); dome.CreateIntensityAttr(800)
    key = UsdLux.DistantLight.Define(stage, '/World/Key'); key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    from character_appearance import DefaultCharacterLook
    appearance=DefaultCharacterLook(stage) if args.look==DEFAULT_LOOK else None
    startup_phases['appearance_ready']=time.perf_counter()-startup_started
    camera = UsdGeom.Camera.Define(stage, '/World/Camera'); camera.CreateFocalLengthAttr(50)
    set_camera_view(eye=(2.8, -6.3, 2.35), target=(0, -.25, .95), camera_prim_path='/World/Camera')
    if args.fit_camera:
        def compute_camera_bounds():
            if args.fit_camera_mesh:
                camera_cache=UsdSkel.Cache();camera_cache.Populate(UsdSkel.Root.Get(motion.source,'/Character'),Usd.PrimDefaultPredicate)
                camera_query=camera_cache.GetSkelQuery(motion.skeleton);camera_skin=camera_cache.GetSkinningQuery(motion.source.GetPrimAtPath('/Character/Body'))
                camera_rest=UsdGeom.Mesh.Get(motion.source,'/Character/Body').GetPointsAttr().Get();lo=np.full(3,np.inf);hi=-lo.copy()
                for frame in range(0,motion.end_frame+1,15):
                    points=Vt.Vec3fArray(camera_rest)
                    if not camera_skin.ComputeSkinnedPoints(camera_query.ComputeSkinningTransforms(frame),points,frame):raise RuntimeError('Camera bounds skinning failed')
                    xyz=np.asarray(points);lo=np.minimum(lo,xyz.min(0));hi=np.maximum(hi,xyz.max(0))
                lo-=np.array([.22,.22,.12]);hi+=np.array([.22,.22,.22])
            else:
                samples=np.array([motion.pose(f)[2][:147] for f in range(0,motion.end_frame+1,15)])
                lo=samples.min(axis=(0,1))-np.array([.22,.22,.12]);hi=samples.max(axis=(0,1))+np.array([.22,.22,.22])
            return lo,hi
        from camera_bounds_cache import cached_camera_bounds
        lo,hi,camera_cache_report=cached_camera_bounds(motion.source,compute_camera_bounds,ROOT/"output/pmx_camera_cache","mesh" if args.fit_camera_mesh else "joints",enabled=not args.no_camera_cache)
        startup_phases["camera_ready"]=time.perf_counter()-startup_started
        center=(lo+hi)/2; half=(hi-lo)/2
        distance=max(half[0]/(20.955/100),half[2]/(20.955/100*args.height/args.width))+half[1]
        eye_height=max(float(center[2]),.55) if args.fit_camera_mesh else float(center[2])
        if args.fit_camera_mesh:distance+=eye_height-float(center[2])
        camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*args.height/args.width)
        set_camera_view(eye=(float(center[0]),float(center[1]-distance),eye_height),target=center.tolist(),camera_prim_path='/World/Camera')
    if args.look=='soft-pbr':
        from character_appearance import apply_bright_character_look
        apply_bright_character_look(stage,center if args.fit_camera else (0.,-.25,.95))
    viewport = get_active_viewport()
    viewport.camera_path = '/World/Camera'; viewport.set_texture_resolution((args.width, args.height))
    print(f'PHYSX_BUILD nodes={len(motion.nodes)} joints={physics.joint_count}' if physics else 'SECONDARY_PHYSICS_WORKER_STARTING' if args.secondary_worker else 'SECONDARY_PHYSICS_DISABLED', flush=True)
    root.CreateVisibilityAttr('invisible')
    loading_window=ui.Window('Loading character',width=300,height=80)
    with loading_window.frame:ui.Label('Preparing character and physics...')
    stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    for _ in range(15):
        app.update()
    if physics:
        physics.start(context.get_stage_id())
        print('PHYSX_STARTED', flush=True)
    for _ in range(60):
        app.update()  # Warm the Fabric/render path without advancing manual physics.
    if args.secondary_worker:
        from dance_secondary_stream import DanceSecondaryStream
        if secondary_worker is None:
            secondary_worker=DanceSecondaryStream(args,app=None,wait=False)
            startup_phases['worker_launched']=time.perf_counter()-startup_started
        secondary_worker.ready(app)
        initial_stream_state=secondary_worker.next()
        publish(initial_stream_state['p'],initial_stream_state['q'])
    if appearance:appearance.update(0)
    loading_visibility_verified=root.GetVisibilityAttr().Get()=='invisible'
    if not loading_visibility_verified:raise RuntimeError('Character visibility changed before physics was ready')
    root.GetVisibilityAttr().Set('inherited');loading_window.visible=False
    app.update()
    stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    startup_phases['character_revealed']=time.perf_counter()-startup_started
    (args.output/'startup_report.json').write_text(json.dumps(dict(phases=startup_phases,parallel_worker=bool(args.secondary_worker and not args.serial_startup),camera_cache=camera_cache_report,hidden_until_ready=loading_visibility_verified,initial_source_frame=initial_stream_state['frame'] if initial_stream_state else 0,resolution=[args.width,args.height],settle_seconds=args.startup_settle_seconds,worker=secondary_worker.metadata if secondary_worker else None),ensure_ascii=False,indent=2),encoding='utf-8')
    print('STARTUP_READY',json.dumps(startup_phases),flush=True)
    if not args.no_music and not args.benchmark:
        from live_music import LiveMusic
        music = LiveMusic(args.music,drift_tolerance=args.music_sync_tolerance)
    hz = secondary_worker.metadata['physics_hz'] if secondary_worker else physics.physics_hz if physics else 120
    panel = ui.Window('Dance Physics Monitor', width=340, height=300)
    panel.frame.style={
        'Label':{'color':0xffe8edf4,'font_size':13},
        'Button':{'background_color':0xff34465b,'color':0xfff4f7fb,'border_radius':5,'font_size':13},
    }
    reset_requested = [False]
    def request_reset():
        reset_requested[0] = True
    with panel.frame:
        with ui.VStack(spacing=7):
            ui.Label('DANCE  /  PHYSICS MONITOR',height=22,style={'font_size':16,'color':0xfff4f7fb})
            ui.Label('Native PhysX  ·  '+('CPU worker' if secondary_worker else 'CPU') if physics or secondary_worker else 'Source animation  ·  secondary physics off',height=18,style={'font_size':11,'color':0xff9eafc2})
            ui.Rectangle(height=1,style={'background_color':0xff354354})
            with ui.HStack(height=68,spacing=8):
                with ui.VStack(spacing=2):
                    ui.Label('PHYSICS  /  ACTUAL',height=17,style={'font_size':10,'color':0xffa9bacd})
                    physics_rate_label=ui.Label('Measuring',height=30,style={'font_size':21,'color':0xff76d7b2})
                    ui.Label(f'target  {hz} Hz',height=16,style={'font_size':10,'color':0xff9eafc2})
                with ui.VStack(spacing=2):
                    ui.Label('DISPLAY  /  PRESENTED',height=17,style={'font_size':10,'color':0xffa9bacd})
                    display_fps_label=ui.Label('0.0 FPS',height=30,style={'font_size':21,'color':0xff82b9ff})
                    ui.Label('rendered frames / wall time',height=16,style={'font_size':10,'color':0xff9eafc2})
            ui.Rectangle(height=1,style={'background_color':0xff354354})
            with ui.HStack(height=39,spacing=8):
                pose_rate_label=ui.Label('Solved poses  ·  measuring',width=ui.Fraction(1),style={'font_size':12,'color':0xffe8edf4})
                speed_label=ui.Label('0.00×',width=ui.Fraction(1),alignment=ui.Alignment.RIGHT_CENTER,style={'font_size':15,'color':0xffffd27a})
            work_label=ui.Label('Frame work  ·  measuring',height=19,style={'font_size':11,'color':0xff9eafc2})
            contact_label=ui.Label('Collisions  ·  '+('active' if physics or secondary_worker else 'off'),height=19,style={'font_size':11,'color':0xff9eafc2})
            if args.interpolate_60:
                ui.Label('60 FPS display uses interpolated 30 Hz solved poses',height=18,style={'font_size':10,'color':0xffe6b96f})
            with ui.HStack(height=28):
                music_model = ui.CheckBox(width=24, enabled=music is not None).model
                music_model.set_value(music is not None)
                ui.Label('Play accompaniment',style={'font_size':12})
            ui.Button('Reset dance and PhysX bodies', clicked_fn=request_reset)
            ui.Label(('Hair + skirt self-contact  ·  '+args.quality) if (physics or secondary_worker) else 'Hair/skirt follow bones; dynamic collision off',height=17,style={'font_size':10,'color':0xff9eafc2})

    async def dock_monitor_right_bottom():
        """Dock beside the right-side Property panel, using the standard Kit layout."""
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
            prop=ui.Workspace.get_window('Property')
            if prop is None:continue
            prop.visible=True
            stage_window=ui.Workspace.get_window('Stage')
            if stage_window is not None:stage_window.visible=True
            if not prop.docked:
                viewport_window=ui.Workspace.get_window('Viewport')
                if viewport_window is None or stage_window is None:continue
                if not stage_window.docked:
                    stage_window.dock_in(viewport_window,ui.DockPosition.RIGHT,.30)
                    await omni.kit.app.get_app().next_update_async()
                prop.dock_in(stage_window,ui.DockPosition.BOTTOM,.50)
                await omni.kit.app.get_app().next_update_async()
            panel.deferred_dock_in('Property',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            return
        viewport_window=ui.Workspace.get_window('Viewport')
        if viewport_window is not None:panel.dock_in(viewport_window,ui.DockPosition.RIGHT,.30)
    monitor_dock_task=asyncio.ensure_future(dock_monitor_right_bottom())
    steps_per_motion_frame = hz//30
    costs, phys_costs, captures = [], [], []
    completed = loops = tick = total_physics_steps = 0
    motion_steps = total_motion_steps = 0
    collect_diagnostics = bool(args.legacy_compute or args.collect_diagnostics or args.record_video or args.bake_usd or args.require_scene_contact)
    audit_particles, audit_p, audit_q, phase_costs = [], [], [], []
    frame_intervals=[]
    started = deadline = time.perf_counter()
    loop_started = started
    p, q = lp, lq
    physics_rate_points=deque()
    pose_rate_points=deque()
    rate_baseline_time=started
    rate_baseline_steps=(initial_stream_state['physics_steps'] if initial_stream_state else 0)
    rate_baseline_pose=(initial_stream_state.get('seq',0) if initial_stream_state else 0)
    physics_rate_points.append((rate_baseline_time,rate_baseline_steps))
    pose_rate_points.append((rate_baseline_time,rate_baseline_pose))

    def rolling_rate(points,now,value):
        if value>points[-1][1]:points.append((now,value))
        while len(points)>2 and now-points[1][0]>1.25:points.popleft()
        elapsed=points[-1][0]-points[0][0]
        return (points[-1][1]-points[0][1])/elapsed if elapsed>.15 else 0.
    def report():
        elapsed = time.perf_counter()-started
        if secondary_worker:
            r=dict(secondary_worker.metadata,status='complete' if args.steps and completed==args.steps else 'running',
                physics_steps=total_physics_steps,simulated_seconds=total_physics_steps/hz,wall_seconds=elapsed,
                actual_physics_hz=total_physics_steps/max(elapsed,1e-9),
                loops=loops,
                gc_pause_ms=dict(count=len(gc_events),maximum=max(gc_events,default=0.),total=sum(gc_events)),
                observed_fps=completed/max(elapsed,1e-9),render_frames=completed,
                display_pose_interpolation=bool(args.interpolate_60),
                presented_motion_seconds=secondary_worker.last_state['frame']/30 if secondary_worker.last_state is not None else 0.,
                target_render_fps=args.render_fps,resolution=[args.width,args.height],
                frontend_task_threads=int(os.environ.get('DANCE_FRONTEND_TASK_THREADS','16')),
                look=args.look,ground_appearance='original',music=music.status() if music else None,
                steady_mean_work_ms=float(np.mean(costs[30:] or costs)) if costs else 0,
                steady_frame_interval_ms=dict(mean=float(np.mean(frame_intervals[30:] or frame_intervals)),p95=float(np.percentile(frame_intervals[30:] or frame_intervals,95)),p99=float(np.percentile(frame_intervals[30:] or frame_intervals,99)),maximum=float(np.max(frame_intervals[30:] or frame_intervals))) if frame_intervals else {},
                steady_phase_ms=dict(zip(['worker_wait','publish_appearance_music','render'],np.mean(phase_costs[30:] or phase_costs,axis=0).tolist())) if phase_costs else {},
                worker_report=str(secondary_worker.output/'worker_report.json'))
            (args.output/'physx_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
            return r
        if not physics:
            r=dict(status='complete' if args.steps and completed==args.steps else 'running',
                backend='source animation, no secondary simulation',secondary_physics_enabled=False,
                physics_hz=None,physics_steps=0,dynamic_bodies=0,joints=0,
                animation_seconds=total_motion_steps/hz,wall_seconds=elapsed,
                observed_fps=completed/max(elapsed,1e-9),render_frames=completed,
                target_render_fps=args.render_fps,resolution=[args.width,args.height],
                look=args.look,ground_appearance='original',music=music.status() if music else None,
                steady_mean_work_ms=float(np.mean(costs[30:] or costs)) if costs else 0,
                steady_phase_ms=dict(zip(['animation','publish_appearance_music','render'],np.mean(phase_costs[30:] or phase_costs,axis=0).tolist())) if phase_costs else {})
            (args.output/'physx_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
            return r
        r = dict(backend='native omni.physx CPU', status='complete' if args.steps and completed == args.steps else 'running',
                 secondary_physics_enabled=True,
                 quality_profile=args.quality,
                 resolution=[args.width,args.height],
                 legacy_compute=args.legacy_compute, secondary_diagnostics_collected=collect_diagnostics,
                 optimized_hair_contacts=not (args.legacy_compute or args.reference_hair_compute),
                 asynchronous_rendering=bool(args.async_render and not args.record_video),
                 physics_threads=args.physics_threads,
                 steady_mean_work_ms=float(np.mean(costs[30:] or costs)) if costs else 0,
                 secondary_runtime_ms_per_substep=(physics.runtime_costs/max(physics.runtime_calls,1)).tolist(),
                 steady_phase_ms=dict(zip(['physics','publish_appearance_music','render'],np.mean(phase_costs[30:] or phase_costs,axis=0).tolist())) if phase_costs else {},
                 look=args.look,ground_appearance='original',
                 camera_fit='skinned_mesh' if args.fit_camera_mesh else 'joints' if args.fit_camera else 'fixed',
                 secondary_profile=str(args.secondary_profile.resolve()) if args.secondary_profile else None,
                 curvature_force_triplets=len(physics.curvature_triplets) if args.quality != 'legacy' else 0,
                 hair_self_collision=args.quality in ('hair', 'cloth'), tip_limit_joints=physics.tip_limit_count,
                 skirt_length_cap_joints=physics.skirt_length_cap_count,
                 hair_length_cap_joints=physics.hair_length_cap_count,
                 hair_length_projection_links=len(physics.hair_length_links),
                 skirt_bend_limit_joints=physics.skirt_bend_joint_count,
                 skirt_edge_capsules=len(physics.edge_nodes),
                 hand_contact_peak_newtons=physics.hand_contact_peak,
                 hand_contact_steps=physics.hand_contact_steps,
                 hair_cloth_contacts=physics.hair_cloth_contacts,
                 hair_cloth_contact_peak_newtons=physics.hair_cloth_contact_peak,
                 hair_cloth_contact_steps=physics.hair_cloth_contact_steps,
                 cloth_self_collision=args.quality == 'cloth', cloth_limit_joints=physics.cloth_limit_count,
                 quality_sample_count=len(physics.quality_samples),
                 quality_metrics={key:float(np.mean([v[key] for v in physics.quality_samples]))
                                  for key in physics.quality_samples[0]} if physics.quality_samples else {},
                 reference_clock=args.reference_clock, source_first_frame=0 if args.reference_clock else 30/args.render_fps, render_frames=completed, simulated_seconds=total_physics_steps/hz, wall_seconds=elapsed,
                 physics_steps=total_physics_steps, actual_physics_hz=total_physics_steps/max(elapsed, 1e-9),
                 observed_fps=completed/max(elapsed, 1e-9), target_render_fps=args.render_fps, physics_hz=hz,
                 mean_work_ms=float(np.mean(costs)) if costs else 0,
                 p95_work_ms=float(np.percentile(costs, 95)) if costs else 0,
                 mean_physics_ms_per_render=float(np.mean(phys_costs)) if phys_costs else 0,
                 dynamic_bodies=physics.dynamic_view.count, kinematic_anchors=physics.pin_view.count,
                 joints=physics.joint_count, solver='PGS', solver_position_iterations=255 if motion.rig_settings.get('skirt_max_stretch') else 96 if args.quality == 'cloth' else 40 if args.quality == 'hair' else 24,
                 solver_velocity_iterations=16 if args.quality == 'cloth' else 8 if args.quality == 'hair' else 4,
                 body_contact_peak_newtons=physics.body_contact_peak, physics_steps_with_body_contact=physics.body_contact_steps,
                 scene_contact_tracking_enabled=physics.track_obstacles,
                 scene_contact_peak_newtons=physics.contact_peak if physics.track_obstacles else None,
                 tracked_scene_colliders=['/World/Obstacles/SkirtSphere', '/World/Obstacles/HairSphere'] if physics.track_obstacles else [],
                 physics_steps_with_scene_contact=physics.contact_steps, loops=loops, captures=captures,
                 music=music.status() if music else None,
                 limitations=['Rigid-node collision proxy, not triangle cloth',
                              ('Nonlocal hair/cloth self-contact and hair-cloth node/edge proxy contact enabled' if physics.hair_cloth_contacts else 'Nonlocal hair and cloth node self-contact; hair-cloth contact disabled') if args.quality == 'cloth' else 'Nonlocal hair node self-contact only; cloth/hair-cloth self-contact disabled' if args.quality in ('hair', 'cloth') else 'No internal self-collision',
                              'Animated body is kinematic and does not react to contact forces'])
        if not collect_diagnostics:
            for key in ('body_contact_peak_newtons','physics_steps_with_body_contact','hand_contact_peak_newtons','hand_contact_steps','hair_cloth_contact_peak_newtons','hair_cloth_contact_steps','scene_contact_peak_newtons','physics_steps_with_scene_contact'):
                r[key]=None
        (args.output/'physx_report.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')
        return r
    if args.bake_usd:
        from pxr import Usd
        args.bake_usd.parent.mkdir(parents=True,exist_ok=True)
        baked_stage = Usd.Stage.CreateNew(str(args.bake_usd.resolve()))
        baked_stage.GetRootLayer().subLayerPaths=[str(args.usd.resolve()).replace('\\','/')]
        baked_stage.SetDefaultPrim(baked_stage.GetPrimAtPath('/Character'))
        baked_stage.SetTimeCodesPerSecond(30); baked_stage.SetStartTimeCode(0);baked_stage.SetEndTimeCode(motion.end_frame)
        baked_stage.GetDefaultPrim().SetCustomDataByKey('secondaryMotionSolver','native PhysX rigid-node hair and skirt')
        baked_anim=UsdSkel.Animation.Get(baked_stage,'/Character/Animation')
        baked_anim.GetJointsAttr().Set(motion.joints);baked_anim.GetScalesAttr().Set([Gf.Vec3h(1)]*len(motion.joints))
        baked_anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(lq.astype(np.float32)),0)
        baked_anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(lp.astype(np.float32)),0)
    if args.record_video:
        import subprocess
        from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
        if not args.ffmpeg: raise ValueError('--record-video requires --ffmpeg')
        args.record_video.parent.mkdir(parents=True,exist_ok=True)
        encoder_log=(args.output/'encoder.log').open('w')
        command=[str(args.ffmpeg),'-y','-hide_banner','-f','rawvideo','-pix_fmt','rgb24','-s',f'{args.width}x{args.height}','-r',str(args.render_fps),'-i','pipe:0']
        audio_filter='volume=.25,'+(f'apad=whole_dur={args.steps/args.render_fps:.9f}' if args.steps else 'apad')
        if args.music.is_file(): command+=['-i',str(args.music),'-map','0:v','-map','1:a','-af',audio_filter,'-c:a','aac','-shortest']
        if args.steps: command+=['-t',str(args.steps/args.render_fps)]
        command+=['-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.record_video)]
        encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=encoder_log,stderr=encoder_log)
    def gc_probe(phase,info):
        generation=info['generation']
        if phase=='start':gc_ticks[generation]=time.perf_counter()
        elif generation in gc_ticks:gc_events.append((time.perf_counter()-gc_ticks.pop(generation))*1000)
    gc.callbacks.append(gc_probe)
    if secondary_worker and args.interpolate_60:
        secondary_worker.start_interpolation(initial_stream_state)
    started=deadline=loop_started=time.perf_counter()
    print('PHYSX_LIVE_READY', flush=True)
    while app.is_running() and (not args.steps or completed < args.steps):
        restart = False
        clock_steps=physics.steps if physics else motion_steps
        if not secondary_worker and (clock_steps >= (motion.end_frame+1)*steps_per_motion_frame or reset_requested[0]):
            tick = 0; loops += 1; reset_requested[0] = False
            if physics:physics.reset()
            motion_steps=clock_steps=0;restart = True
            loop_started = time.perf_counter()
            p, q, _, _ = motion.pose(0)
        begin = time.perf_counter()
        # Fixed 1/120s steps follow wall time in normal viewing, independently
        # of rendering. Never enlarge dt or skip collision steps on slow frames.
        due = (0 if args.reference_clock and completed==0 else hz//args.render_fps) if args.benchmark else min(hz//15, max(0, int((begin-loop_started)*hz)-clock_steps))
        due = min(due, (motion.end_frame+1)*steps_per_motion_frame-clock_steps)
        frame = min(clock_steps/steps_per_motion_frame, motion.end_frame)
        for substep in range(due if physics else 0):
            frame = min((physics.steps+1)/steps_per_motion_frame, motion.end_frame)
            # Preserve every native contact/damping/projection step; only
            # reconstruct and publish the final visible endpoint.
            p, q = physics.step(frame, produce_pose=args.legacy_compute or substep==due-1,
                                diagnostics=collect_diagnostics,compiled_hair=not args.legacy_compute,
                                optimized_contacts=not (args.legacy_compute or args.reference_hair_compute))
            total_physics_steps += 1
        if secondary_worker:
            state=initial_stream_state if initial_stream_state is not None and not reset_requested[0] else (secondary_worker.next_interpolated(reset=reset_requested[0]) if args.interpolate_60 else secondary_worker.next(reset=reset_requested[0]))
            initial_stream_state=None;reset_requested[0]=False
            p,q,frame=state['p'],state['q'],state['frame']
            restart=state['restart'];loops=state['loops']
            total_physics_steps=total_motion_steps=state['physics_steps']
            motion_steps=round(frame*steps_per_motion_frame)
        elif not physics:
            motion_steps+=due;frame=min(motion_steps/steps_per_motion_frame,motion.end_frame)
            p,q,_,_=motion.pose(frame)
        if not secondary_worker:total_motion_steps+=due
        phys_ms = (time.perf_counter()-begin)*1000
        publish(p, q)
        if appearance is not None:appearance.update(frame)
        if music:
            was_enabled = music.enabled
            music.set_enabled(music_model.get_value_as_bool(), frame/30)
            music.update(frame/30, restart=restart and was_enabled)
        if completed % 30 == 0:
            now=time.perf_counter()
            measured_physics_hz=rolling_rate(physics_rate_points,now,total_physics_steps)
            pose_counter=(state.get('seq',0) if secondary_worker else completed) if (secondary_worker or physics) else 0
            measured_pose_hz=rolling_rate(pose_rate_points,now,pose_counter)
            elapsed=max(now-started,1e-9)
            display_fps_label.text=f'{completed/elapsed:.1f} FPS'
            physics_rate_label.text=f'{measured_physics_hz:.0f} / {hz} Hz' if physics or secondary_worker else 'OFF'
            pose_rate_label.text=f'Solved poses  ·  {measured_pose_hz:.1f} Hz' if physics or secondary_worker else 'Solved poses  ·  source animation'
            speed_label.text=f'{total_motion_steps/hz/elapsed:.2f}×'
            work_label.text=f'Frame work  ·  {np.mean(costs[-120:]):.1f} ms' if costs else 'Frame work  ·  measuring'
            contact_label.text=('Collisions  ·  live worker' if secondary_worker else
                                f'Collisions  ·  peak {physics.contact_peak:.2f} N' if physics and collect_diagnostics else
                                'Collisions  ·  active' if physics else 'Collisions  ·  off')
        before_render=time.perf_counter()
        app.update()
        after_render=time.perf_counter()
        phase_costs.append([phys_ms,(before_render-begin)*1000-phys_ms,(after_render-before_render)*1000])
        if args.runtime_audit:
            audit_particles.append(state['particles'].copy() if secondary_worker else physics.last_positions.copy() if physics else np.empty((0,3)));audit_p.append(p.copy());audit_q.append(q.copy())
        if baked_stage:
            baked_anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)),frame)
            baked_anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)),frame)
        if encoder:
            rgb=_capture_viewport_rgb(pre_render=None)
            if rgb is None: raise RuntimeError('PhysX video capture failed')
            encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
        costs.append((time.perf_counter()-begin)*1000); phys_costs.append(phys_ms)
        if not args.benchmark:
            deadline += 1/args.render_fps
            if args.interpolate_60:
                # Do not issue a burst of catch-up renders after a long frame;
                # those loops do not represent evenly presented 60Hz output.
                deadline=max(deadline,time.perf_counter())
            time.sleep(max(0, deadline-time.perf_counter()))
        if tick in args.capture_frames and not loops:
            for _ in range(8): app.update()
            path = args.output/f'render_{tick:05d}.png'
            if path.exists(): path.unlink()
            capture = capture_viewport_to_file(viewport, str(path.resolve()))
            wait_until = time.perf_counter()+5
            while time.perf_counter() < wait_until and (not path.is_file() or not path.stat().st_size):
                app.update(); time.sleep(.01)
            if not path.is_file(): raise RuntimeError('PhysX screenshot failed')
            captures.append(str(path.resolve()))
        completed += 1; tick += 1
        frame_intervals.append((time.perf_counter()-begin)*1000)
        if completed % 300 == 0:
            r = report()
            print(f'PHYSX_PROGRESS frames={completed} work_ms={np.mean(costs):.2f} scene_contact_N={physics.contact_peak if physics else 0.:.3f}', flush=True)
    if args.steps and completed != args.steps:
        raise RuntimeError('PhysX test closed early')
    if args.require_scene_contact and physics.contact_peak <= 1e-6:
        raise RuntimeError('No native contact force was measured against the scene test objects')
    if baked_stage: baked_stage.GetRootLayer().Save()
    if args.runtime_audit:
        np.savez_compressed(args.output/'runtime_audit.npz',particles=np.asarray(audit_particles),p=np.asarray(audit_p),q=np.asarray(audit_q))
    if secondary_worker:
        np.savez_compressed(args.output/'performance_trace.npz',interval_ms=frame_intervals,phase_ms=phase_costs,work_ms=costs)
    if encoder:
        encoder.stdin.close()
        if encoder.wait()!=0: raise RuntimeError('Video encoder failed')
        encoder=None;encoder_log.close()
    print('PHYSX_LIVE_PASS ' + json.dumps(report(), ensure_ascii=False), flush=True)
except Exception as error:
    traceback.print_exc()
    (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error)), indent=2), encoding='utf-8')
    print('PHYSX_LIVE_FAIL '+repr(error), flush=True)
    raise
finally:
    if gc_probe in gc.callbacks:gc.callbacks.remove(gc_probe)
    if encoder:
        encoder.stdin.close();encoder.wait()
    if music: music.close()
    if physics: physics.close()
    if secondary_worker:secondary_worker.close()
    app.close()
