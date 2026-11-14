"""Open the converted PMX in Isaac Sim. Default: repeat the eight-second demo."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd', type=Path, default=ROOT / 'output/pmx_prototype/character.usdc')
parser.add_argument('--steps', type=int, default=0, help='0: keep window open; otherwise render this many frames')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--captures', action='store_true', help='Save rest/arm/elbow/hip/knee PNGs')
parser.add_argument('--output', type=Path, default=ROOT / 'output/pmx_prototype')
parser.add_argument('--start-frame', type=int, default=0)
parser.add_argument('--capture-frames', type=int, nargs='+', help='Override capture frames for a dance')
args = parser.parse_args()
if not args.usd.is_file():
    parser.error(f'Convert a PMX first: {args.usd} does not exist')

from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'width': 1000, 'height': 1000,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3,
                     'extra_args': ['--/app/asyncRendering=false']})

import omni.usd
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdSkel
from isaacsim.core.utils.viewports import set_camera_view

try:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    character = UsdSkel.Root.Define(stage,'/World/Character').GetPrim()
    character.GetReferences().AddReference(str(args.usd.resolve()))
    asset_stage = Usd.Stage.Open(str(args.usd.resolve()))
    if not asset_stage.GetDefaultPrim() or not stage.GetPrimAtPath('/World/Character/Body'):
        raise RuntimeError('Character reference did not resolve; check USD defaultPrim')
    secondary_solver = asset_stage.GetDefaultPrim().GetCustomDataByKey('secondaryMotionSolver')
    playback_mode = (f'UsdSkel playback; baked secondary motion ({secondary_solver})' if secondary_solver else
                     'UsdSkel animation; no physics')
    fps = asset_stage.GetTimeCodesPerSecond()
    first, last = int(asset_stage.GetStartTimeCode()), int(asset_stage.GetEndTimeCode())
    period = max(last - first, 1)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(first)
    stage.SetEndTimeCode(last)
    dome = UsdLux.DomeLight.Define(stage, '/World/Light')
    dome.CreateIntensityAttr(800)
    key = UsdLux.DistantLight.Define(stage, '/World/Key')
    key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    floor = UsdGeom.Mesh.Define(stage, '/World/Floor')
    floor.CreatePointsAttr([(-5, -5, -0.015), (5, -5, -0.015), (5, 5, -0.015), (-5, 5, -0.015)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr('none')
    floor.CreateDisplayColorAttr([(0.18, 0.21, 0.26)])
    from character_appearance import DefaultCharacterLook
    appearance=DefaultCharacterLook(stage)
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(50)
    if period > 300:
        # Leave room for raised hands and the dance's forward root travel.
        eye, target = (2.8, -6.3, 2.35), (0, -0.25, 0.95)
    else:
        eye, target = (2.0, -4.2, 1.8), (0, 0, 0.85)
    set_camera_view(eye=eye, target=target, camera_prim_path='/World/Camera')
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('No rendering viewport available')
    viewport.camera_path = '/World/Camera'
    viewport.set_texture_resolution((1000, 1000))
    timeline = omni.timeline.get_timeline_interface()
    # Long secondary-motion caches contain hundreds of animated joints. Sample
    # the read-only asset with USD, and publish only the current pose into the
    # live scene. Strong defaults hide the referenced time samples from Hydra.
    source_animation = UsdSkel.Animation.Get(asset_stage, '/Character/Animation')
    live_animation = UsdSkel.Animation.Get(stage, '/World/Character/Animation')
    def set_frame(frame):
        if secondary_solver:
            live_animation.GetRotationsAttr().Set(source_animation.GetRotationsAttr().Get(frame))
            live_animation.GetTranslationsAttr().Set(source_animation.GetTranslationsAttr().Get(frame))
        timeline.set_current_time(frame / fps)
        appearance.update(frame,time_code=Usd.TimeCode(frame))
    args.output.mkdir(parents=True, exist_ok=True)
    # Warm shaders/textures before recording any evidence.
    set_frame(args.start_frame)
    # Export a lightweight, independently playable scene reference, before the
    # live streaming pose overrides (which are only for viewer performance).
    if secondary_solver:
        live_animation.GetRotationsAttr().Clear()
        live_animation.GetTranslationsAttr().Clear()
    stage.GetRootLayer().Export(str(args.output / 'scene.usda'))
    set_frame(args.start_frame)
    for _ in range(90):
        app.update()
    captures = []
    if args.captures:
        samples = ([(f'frame_{f:05d}', f) for f in args.capture_frames] if args.capture_frames else
                   [('rest', 0), ('arm', 30), ('elbow', 90), ('hip', 150), ('knee', 210)])
        for label, frame in samples:
            set_frame(frame)
            for _ in range(25):
                app.update()
            path = args.output / f'{label}.png'
            capture = capture_viewport_to_file(viewport, str(path.resolve()))
            for _ in range(30):
                app.update()
            if not path.is_file():
                raise RuntimeError(f'Capture missing: {path}')
            captures.append(str(path))
    print('PMX_SCENE_READY ' + str(args.usd.resolve()), flush=True)
    ready = {'usd': str(args.usd), 'first_frame': first, 'last_frame': last,
             'fps': fps, 'captures': captures, 'mode': playback_mode}
    (args.output / 'scene_ready.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')
    count = 0
    started = time.perf_counter()
    while app.is_running() and (not args.steps or count < args.steps):
        elapsed_frames = count if args.headless else (time.perf_counter() - started) * fps
        set_frame(first + (args.start_frame - first + elapsed_frames) % period)
        app.update()
        count += 1
        if not args.headless:
            time.sleep(max(0, 1 / fps - (time.perf_counter() - started - elapsed_frames / fps)))
    report = {'frames': count, 'usd': str(args.usd), 'captures': captures,
              'playback_wall_seconds': time.perf_counter() - started,
              'mode': playback_mode, 'credits': asset_stage.GetDefaultPrim().GetCustomDataByKey('credits') or ''}
    (args.output / 'viewer_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PMX_VIEWER_PASS ' + json.dumps(report, ensure_ascii=False), flush=True)
finally:
    app.close()
