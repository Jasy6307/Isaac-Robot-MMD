"""PhysX-driven PMX hair/skirt, scene collider interaction, 60 FPS rendering."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd', type=Path, default=ROOT/'output/pmx_prototype/ji_le_jing_tu.usdc')
parser.add_argument('--pmx', type=Path, default=ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
parser.add_argument('--output', type=Path, default=ROOT/'output/pmx_prototype/physx')
parser.add_argument('--rig-data', type=Path, help='Supplementary model-specific physics rig JSON')
parser.add_argument('--secondary-profile', type=Path, default=DEFAULT_SECONDARY_PROFILE, help='Physics settings applied before topology/skin initialization; preserves native PMX rig')
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
parser.add_argument('--look',choices=LOOK_CHOICES, default=DEFAULT_LOOK,help='Default: approved look 08, preserving original ground')
parser.add_argument('--steps', type=int, default=0)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--no-music', action='store_true')
parser.add_argument('--no-obstacles', action='store_true')
parser.add_argument('--require-scene-contact', action='store_true')
parser.add_argument('--scene', type=Path, help='Optional USD environment with ordinary Physics CollisionAPI colliders')
parser.add_argument('--capture-frames', type=int, nargs='*', default=[])
parser.add_argument('--quality', choices=['legacy', 'smooth', 'hair', 'cloth'], default=None, help='Default: cloth; legacy-secondary-physics uses the previous hair default')
args = parser.parse_args()
if args.fit_camera_mesh:args.fit_camera=True
args.quality=args.quality or ('hair' if args.legacy_secondary_physics else DEFAULT_SECONDARY_QUALITY)
if args.legacy_secondary_physics:args.secondary_profile=None
if args.require_scene_contact and args.no_obstacles:
    parser.error('require-scene-contact measures the built-in test spheres; do not combine it with no-obstacles')
if args.scene and not args.scene.is_file():
    parser.error(f'Scene USD does not exist: {args.scene}')
if args.reference_clock and not args.benchmark: parser.error('reference-clock requires benchmark')
args.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'width': args.width, 'height': args.height,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3,
                     'extra_args': ['--/app/asyncRendering=false']})

import numpy as np
import omni.usd
import omni.ui as ui
import omni.kit.app
from pxr import Gf, Usd, UsdGeom, UsdSkel, UsdLux, Vt
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from physx_secondary import BodyMotion, PhysxSecondary

physics = music = encoder = encoder_log = baked_stage = None
try:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.tensors', True)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    motion = BodyMotion(args.usd, args.pmx, rig_data=args.rig_data, secondary_profile=args.secondary_profile)
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
    physics = PhysxSecondary(stage, motion, obstacles=not args.no_obstacles, quality=args.quality)
    if args.scene:
        stage.DefinePrim('/World/Environment').GetReferences().AddReference(str(args.scene.resolve()))
    dome = UsdLux.DomeLight.Define(stage, '/World/Light'); dome.CreateIntensityAttr(800)
    key = UsdLux.DistantLight.Define(stage, '/World/Key'); key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    from character_appearance import DefaultCharacterLook
    appearance=DefaultCharacterLook(stage) if args.look==DEFAULT_LOOK else None
    camera = UsdGeom.Camera.Define(stage, '/World/Camera'); camera.CreateFocalLengthAttr(50)
    set_camera_view(eye=(2.8, -6.3, 2.35), target=(0, -.25, .95), camera_prim_path='/World/Camera')
    if args.fit_camera:
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
    print(f'PHYSX_BUILD nodes={len(motion.nodes)} joints={physics.joint_count}', flush=True)
    stage.GetRootLayer().Export(str(args.output/'scene.usda'))
    for _ in range(15):
        app.update()
    physics.start(context.get_stage_id())
    print('PHYSX_STARTED', flush=True)
    for _ in range(60):
        app.update()  # Warm the Fabric/render path without advancing manual physics.
    if not args.no_music and not args.benchmark:
        from live_music import LiveMusic
        music = LiveMusic(args.music)
    panel = ui.Window('Native PhysX | PMX Live', width=370, height=230)
    reset_requested = [False]
    def request_reset():
        reset_requested[0] = True
    with panel.frame:
        with ui.VStack(spacing=5):
            ui.Label(f'PhysX CPU | {physics.physics_hz} Hz physics | {args.render_fps} FPS render')
            ui.Label('Shape-preserving profile: '+args.quality)
            fps_label = ui.Label('Measuring...')
            contact_label = ui.Label('Scene contact force: -- N')
            with ui.HStack(height=22):
                music_model = ui.CheckBox(width=24, enabled=music is not None).model
                music_model.set_value(music is not None)
                ui.Label('Play music (accompaniment)')
            ui.Button('Reset dance and PhysX bodies', clicked_fn=request_reset)
            ui.Label('Blue/orange spheres are regular scene colliders')
            ui.Label('Hair and skirt node self-contact enabled' if args.quality == 'cloth' else 'Hair node self-contact enabled' if args.quality == 'hair' else 'Internal node self-contact disabled')
    hz = physics.physics_hz
    steps_per_motion_frame = hz//30
    costs, phys_costs, captures = [], [], []
    completed = loops = tick = total_physics_steps = 0
    started = deadline = time.perf_counter()
    loop_started = started
    p, q = lp, lq
    def report():
        elapsed = time.perf_counter()-started
        r = dict(backend='native omni.physx CPU', status='complete' if args.steps and completed == args.steps else 'running',
                 quality_profile=args.quality,
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
    print('PHYSX_LIVE_READY', flush=True)
    while app.is_running() and (not args.steps or completed < args.steps):
        restart = False
        if physics.steps >= (motion.end_frame+1)*steps_per_motion_frame or reset_requested[0]:
            tick = 0; loops += 1; reset_requested[0] = False
            physics.reset(); restart = True
            loop_started = time.perf_counter()
            p, q, _, _ = motion.pose(0)
        begin = time.perf_counter()
        # Fixed 1/120s steps follow wall time in normal viewing, independently
        # of rendering. Never enlarge dt or skip collision steps on slow frames.
        due = (0 if args.reference_clock and completed==0 else hz//args.render_fps) if args.benchmark else min(hz//15, max(0, int((begin-loop_started)*hz)-physics.steps))
        due = min(due, (motion.end_frame+1)*steps_per_motion_frame-physics.steps)
        frame = min(physics.steps/steps_per_motion_frame, motion.end_frame)
        for substep in range(due):
            frame = min((physics.steps+1)/steps_per_motion_frame, motion.end_frame)
            p, q = physics.step(frame)
            total_physics_steps += 1
        phys_ms = (time.perf_counter()-begin)*1000
        publish(p, q)
        if appearance is not None:appearance.update(frame)
        if music:
            was_enabled = music.enabled
            music.set_enabled(music_model.get_value_as_bool(), frame/30)
            music.update(frame/30, restart=restart and was_enabled)
        if completed % 30 == 0:
            fps_label.text = (f'FPS: {completed/max(time.perf_counter()-started, 1e-9):.1f} | '
                              f'work: {np.mean(costs[-120:]):.2f} ms') if costs else 'Warming up'
            contact_label.text = f'Scene contact peak: {physics.contact_peak:.3f} N'
        app.update()
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
        if completed % 300 == 0:
            r = report()
            print(f'PHYSX_PROGRESS frames={completed} work_ms={r["mean_work_ms"]:.2f} scene_contact_N={physics.contact_peak:.3f}', flush=True)
    if args.steps and completed != args.steps:
        raise RuntimeError('PhysX test closed early')
    if args.require_scene_contact and physics.contact_peak <= 1e-6:
        raise RuntimeError('No native contact force was measured against the scene test objects')
    if baked_stage: baked_stage.GetRootLayer().Save()
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
    if encoder:
        encoder.stdin.close();encoder.wait()
    if music: music.close()
    if physics: physics.close()
    app.close()
