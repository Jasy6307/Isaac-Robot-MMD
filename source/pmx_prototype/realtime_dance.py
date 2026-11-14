"""Live XPBD + VMD dance in Isaac Sim, with rendered frame timings and probes."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES,resolve_appearance_args
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd', type=Path, default=ROOT/'output/pmx_prototype/ji_le_jing_tu.usdc')
parser.add_argument('--pmx', type=Path, default=ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
parser.add_argument('--output', type=Path, default=ROOT/'output/pmx_prototype/realtime')
parser.add_argument('--steps', type=int, default=0, help='Rendered simulation frames; 0 loops until closed')
parser.add_argument('--render-fps', type=int, default=60, help='Render rate: a positive multiple of the 30Hz physics batch rate')
parser.add_argument('--benchmark', action='store_true', help='Remove render pacing; advance simulation by 1/render-fps per rendered frame')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--capture-frames', type=int, nargs='*', default=[])
parser.add_argument('--substeps', type=int, default=4)
parser.add_argument('--iterations', type=int, default=12)
parser.add_argument('--gravity', type=float, default=9.81)
parser.add_argument('--music', type=Path, default=ROOT/'media/dance/ji_le_jing_tu/ji_le_jing_tu.wav')
parser.add_argument('--no-music', action='store_true', help='Start with accompaniment unchecked')
parser.add_argument('--look', choices=LOOK_CHOICES, default=DEFAULT_LOOK)
parser.add_argument('--environment-scale',type=float,default=None)
parser.add_argument('--bangs-shadow-level',type=float,default=None)
parser.add_argument('--cel-brightness-scale',type=float,default=None)
parser.add_argument('--face-shaping',action=argparse.BooleanOptionalAction,default=None)
args = resolve_appearance_args(parser.parse_args())
if not 0 < args.cel_brightness_scale <= 2:
    parser.error('cel-brightness-scale must be in (0,2]')
if not 0 <= args.environment_scale <= 2 or not 0 <= args.bangs_shadow_level <= 1:
    parser.error('environment-scale must be in [0,2]; bangs-shadow-level must be in [0,1]')
if args.substeps < 1 or args.iterations < 1 or args.steps < 0:
    parser.error('substeps/iterations must be positive; steps must be nonnegative')
if args.render_fps < 30 or args.render_fps % 30:
    parser.error('render-fps must be a multiple of 30 (for example 30, 60 or 120)')
render_ratio = args.render_fps//30
args.output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'width': 1000, 'height': 1000,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3,
                     'extra_args': ['--/app/asyncRendering=false']})

import numpy as np
import omni.usd
import omni.timeline
import omni.ui as ui
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdSkel, Vt
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from realtime_xpbd import RealtimeXPBD, InterpolatedXPBD
from secondary_motion import penetration


def distribution(values):
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                maximum=float(np.max(values))) if values else {}


music = None
music_error = None
try:
    solver = RealtimeXPBD(args.usd, args.pmx, args.substeps, args.iterations, args.gravity)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    character = UsdSkel.Root.Define(stage, '/World/Character')
    character.GetPrim().GetReferences().AddReference(str(args.usd.resolve()))
    animation = UsdSkel.Animation.Get(stage, '/World/Character/Animation')
    animation.CreateJointsAttr(solver.joints)
    animation.CreateScalesAttr([Gf.Vec3h(1)]*len(solver.joints))
    rotations, translations = animation.GetRotationsAttr(), animation.GetTranslationsAttr()
    stage.SetTimeCodesPerSecond(30)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(solver.end_frame)
    timeline = omni.timeline.get_timeline_interface()
    dome = UsdLux.DomeLight.Define(stage, '/World/Light')
    dome.CreateIntensityAttr(800)
    key = UsdLux.DistantLight.Define(stage, '/World/Key')
    key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    floor = UsdGeom.Mesh.Define(stage, '/World/Floor')
    floor.CreatePointsAttr([(-5, -5, -.015), (5, -5, -.015), (5, 5, -.015), (-5, 5, -.015)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr('none')
    floor.CreateDisplayColorAttr([(.18, .21, .26)])
    from character_appearance import apply_bright_character_look
    if args.look!='original':
        apply_bright_character_look(stage)
    environment=UsdLux.DomeLight.Get(stage,'/World/Light')
    environment.GetIntensityAttr().Set(float(environment.GetIntensityAttr().Get())*args.environment_scale)
    bangs_shadow=None
    if args.look==DEFAULT_LOOK:
        from character_appearance import apply_cel_look
        apply_cel_look(stage,brightness_scale=args.cel_brightness_scale,face_shaping=args.face_shaping)
        from bangs_shadow import BangsShadow
        try:
            bangs_shadow=BangsShadow(stage,shadow_level=args.bangs_shadow_level)
        except ValueError as error:
            print(f"[Appearance] Bangs occlusion unavailable: {error}",flush=True)
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(50)
    set_camera_view(eye=(2.8, -6.3, 2.35), target=(0, -.25, .95), camera_prim_path='/World/Camera')
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('A rendering viewport is required for FPS validation')
    viewport.camera_path = '/World/Camera'
    viewport.set_texture_resolution((1000, 1000))

    def publish(p, q, frame):
        # Vt's NumPy quaternion buffer order is XYZW.
        rotations.Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
        translations.Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        timeline.set_current_time(frame/30)
        if bangs_shadow is not None:
            bangs_shadow.update(frame)

    solver.advance(0)
    publish(solver.pose_p, solver.pose_q, 0)
    if rotations.GetNumTimeSamples() or translations.GetNumTimeSamples():
        raise RuntimeError('Live pose did not hide referenced animation samples')
    for _ in range(60):
        app.update()
    stream = InterpolatedXPBD(solver)
    try:
        from live_music import LiveMusic
        music = LiveMusic(args.music, volume=.2)
    except Exception as error:
        music_error = str(error)
        print('LIVE_MUSIC_UNAVAILABLE ' + music_error, flush=True)

    panel = ui.Window('Live XPBD | Anon - Gokuraku Jodo', width=360, height=260)
    reset_requested = [False]
    def reset_motion():
        reset_requested[0] = True
    with panel.frame:
        with ui.VStack(spacing=6):
            ui.Label(f'Render: {args.render_fps} FPS | XPBD: {30*args.substeps} Hz')
            fps_label = ui.Label('Measuring frame rate...')
            cost_label = ui.Label('XPBD: -- ms | frame cost: -- ms')
            frame_label = ui.Label('Dance: 0.0 s | no physics cache loaded')
            with ui.HStack(height=22):
                music_model = ui.CheckBox(width=24, enabled=music is not None and not args.benchmark).model
                music_model.set_value(music is not None and not args.no_music and not args.benchmark)
                ui.Label('Play music (accompaniment)')
            music_status = ui.Label('Music disabled for accelerated benchmark' if args.benchmark else
                                    ('Audio unavailable: '+music_error if music_error else 'Music follows dance time'))
            with ui.HStack():
                ui.Label('Gravity (m/s2)', width=145)
                gravity_model = ui.FloatDrag(min=0, max=20, step=.1).model
                gravity_model.set_value(args.gravity)
            ui.Button('Reset dance and particles', clicked_fn=reset_motion)
            ui.Label('Capsule collisions; mesh self-collision is not enabled')

    source_cache, live_cache = UsdSkel.Cache(), UsdSkel.Cache()
    source_cache.Populate(UsdSkel.Root.Get(solver.source, '/Character'), Usd.PrimDefaultPredicate)
    live_cache.Populate(character, Usd.PrimDefaultPredicate)
    source_query = source_cache.GetSkelQuery(solver.skeleton)
    source_skin = source_cache.GetSkinningQuery(solver.source.GetPrimAtPath('/Character/Body'))
    live_query = live_cache.GetSkelQuery(UsdSkel.Skeleton.Get(stage, '/World/Character/Skeleton'))
    live_skin = live_cache.GetSkinningQuery(stage.GetPrimAtPath('/World/Character/Body'))
    sample_frames = {0, 150, 300, 600, 900, 1500, 2400, 3000, 4500, 6000, solver.end_frame}
    metrics, captures = {}, []
    def probe(frame):
        values = []
        for query, skin, timecode in [(source_query, source_skin, frame), (live_query, live_skin, frame)]:
            points = Vt.Vec3fArray.FromNumpy(solver.rest_points.astype(np.float32))
            if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(timecode), points, timecode):
                raise RuntimeError('Live CPU skinning probe failed')
            points = np.asarray(points)
            if not np.isfinite(points).all():
                raise RuntimeError('Live skinned mesh is non-finite')
            values.append(penetration(points[solver.weighted_vertices], solver.ca, solver.cb, solver.system['radii']))
        metrics[frame] = dict(before=values[0], after=values[1])

    frame_costs, solve_costs, state_costs = [], [], []
    paced_costs, parameter_changes = [], []
    presentation_intervals, last_present = [], None
    frame, render_tick, completed, loops, physics_batches = 0, 0, 0, 0, 0
    started = time.perf_counter()
    next_deadline = started
    mode = 'live XPBD; no baked secondary motion read'
    def report():
        before = sum(v['before']['sum_m'] for v in metrics.values())
        after = sum(v['after']['sum_m'] for v in metrics.values())
        wall_seconds = time.perf_counter()-started
        result = dict(mode=mode, body_animation=str(args.usd.resolve()), frames=completed, loops=loops,
                      status='complete' if args.steps and completed == args.steps else 'running',
                      viewport_resolution=[1000, 1000], headless=args.headless, paced=not args.benchmark,
                      target_render_fps=args.render_fps, physics_batch_hz=30, physics_batches=physics_batches,
                      solver_hz=30*args.substeps, substeps=args.substeps, iterations=args.iterations,
                      gravity=solver.gravity, parameter_changes=parameter_changes,
                      music=music.status() if music else dict(error=music_error),
                      nodes=len(solver.nodes), constraints=len(solver.system['edge_array']),
                      render_frame_cost_ms=distribution(frame_costs), xpbd_ms=distribution(solve_costs),
                      body_solve_upload_ms=distribution(state_costs),
                      capacity_fps=1000/np.mean(frame_costs) if frame_costs else None,
                      presentation_fps=1000/np.mean(presentation_intervals) if presentation_intervals else None,
                      paced_fps=1000/np.mean(paced_costs) if paced_costs else None,
                      over_33ms_fraction=float(np.mean(np.array(frame_costs)>1000/30)) if frame_costs else None,
                      over_render_budget_fraction=float(np.mean(np.array(frame_costs)>1000/args.render_fps)) if frame_costs else None,
                      wall_seconds=wall_seconds, observed_wall_fps=completed/max(wall_seconds, 1e-9),
                      simulated_seconds=completed/args.render_fps,
                      vertex_capsule_metrics=metrics, captures=captures,
                      sampled_penetration_sum_reduction_fraction=1-after/before if before else None,
                      limitations=['Custom live XPBD, not native PhysX', 'Capsule proxies, no triangle self-collision',
                                   'Capacity excludes startup, diagnostic skinning probes and screenshot stalls'])
        (args.output/'realtime_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    print('LIVE_XPBD_READY ' + str(args.usd.resolve()), flush=True)
    (args.output/'scene_ready.json').write_text(json.dumps(dict(mode=mode, source=str(args.usd.resolve()),
        look=args.look,environment_scale=args.environment_scale,dome_intensity=float(environment.GetIntensityAttr().Get()),
        bangs_shadow_level=args.bangs_shadow_level,
        cel_brightness_scale=args.cel_brightness_scale,face_shaping=args.face_shaping,
        render_fps=args.render_fps, solver_hz=30*args.substeps,
        music_checked=music_model.get_value_as_bool()), indent=2), encoding='utf-8')
    while app.is_running() and (not args.steps or completed < args.steps):
        music_restart = False
        if reset_requested[0] or render_tick >= (solver.end_frame+1)*render_ratio:
            reset_requested[0] = False
            render_tick = 0
            loops += 1
            stream.reset()
            music_restart = True
        frame = min(render_tick/render_ratio, solver.end_frame)
        begin = time.perf_counter()
        gravity = gravity_model.get_value_as_float()
        if abs(gravity-solver.gravity) > 1e-5:
            parameter_changes.append(dict(frame=frame, gravity=gravity))
            solver.gravity = gravity
        old_updates = stream.updates
        p, q = stream.sample(frame)
        updates = stream.updates-old_updates
        physics_batches += updates
        publish(p, q, frame)
        if music is not None:
            was_enabled = music.enabled
            music.set_enabled(music_model.get_value_as_bool(), frame/30)
            music.update(frame/30, restart=music_restart and was_enabled)
        state_ms = (time.perf_counter()-begin)*1000
        if completed % 15 == 0:
            recent = presentation_intervals[-args.render_fps:]
            observed = 1000/np.mean(recent) if recent else 0
            fps_label.text = f'FPS: {observed:.1f} | target: {args.render_fps}'
            cost_label.text = f'XPBD: {solver.solve_ms:.2f} ms | frame: {np.mean(frame_costs[-60:]):.2f} ms' if frame_costs else 'Warming up'
            frame_label.text = f'Dance: {frame/30:.1f} s | loop: {loops}'
            if music is not None and not args.benchmark:
                music_status.text = 'Music: playing, following dance' if music.enabled else 'Music: off'
        app.update()
        presented = time.perf_counter()
        if last_present is not None:
            presentation_intervals.append((presented-last_present)*1000)
        last_present = presented
        cost = (time.perf_counter()-begin)*1000
        frame_costs.append(cost)
        if updates:
            solve_costs.append(stream.solve_ms/updates)
        state_costs.append(state_ms)
        if not args.benchmark:
            next_deadline += 1/args.render_fps
            time.sleep(max(0, next_deadline-time.perf_counter()))
        paced_costs.append((time.perf_counter()-begin)*1000)
        if render_tick % render_ratio == 0 and frame in sample_frames:
            probe(int(frame))
        if render_tick % render_ratio == 0 and frame in args.capture_frames and not loops:
            for _ in range(8):
                app.update()  # Hold pose for temporal AA; do not advance physics.
            path = args.output/f'frame_{int(frame):05d}.png'
            if path.exists():
                path.unlink()
            capture = capture_viewport_to_file(viewport, str(path.resolve()))
            deadline = time.perf_counter()+5
            while time.perf_counter() < deadline and (not path.is_file() or not path.stat().st_size):
                app.update()
                time.sleep(.01)
            if not path.is_file():
                raise RuntimeError(f'Missing live screenshot: {path}')
            captures.append(str(path.resolve()))
        completed += 1
        render_tick += 1
        if completed % 300 == 0:
            r = report()
            print(f'LIVE_XPBD_PROGRESS frames={completed} capacity_fps={r["capacity_fps"]:.1f} paced_fps={r["paced_fps"]:.1f}', flush=True)
    if args.steps and completed != args.steps:
        raise RuntimeError(f'Viewer closed early: {completed}/{args.steps}')
    result = report()
    (args.output/'frame_timings.csv').write_text('render_frame,frame_ms,body_solve_upload_ms,paced_ms\n' +
        ''.join(f'{i},{a:.5f},{c:.5f},{d:.5f}\n' for i,(a,c,d) in
                enumerate(zip(frame_costs, state_costs, paced_costs))), encoding='utf-8')
    print('LIVE_XPBD_PASS ' + json.dumps({k:v for k,v in result.items() if k not in ('vertex_capsule_metrics',)}, ensure_ascii=False), flush=True)
except Exception as error:
    import traceback
    traceback.print_exc()
    print('LIVE_XPBD_FAIL ' + repr(error), flush=True)
    (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error), frame=locals().get('frame')), indent=2), encoding='utf-8')
    raise
finally:
    if music is not None:
        music.close()
    app.close()
