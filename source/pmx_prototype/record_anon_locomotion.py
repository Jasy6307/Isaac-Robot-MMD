"""Render a saved, measured policy rollout with Anon's original skin and telemetry.

Rendering replays link poses; no new control, root correction or motion cleanup.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rollout', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--env-index', type=int, default=0)
parser.add_argument('--start-frame', type=int, default=0)
parser.add_argument('--max-frames', type=int, default=0)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--ffmpeg', type=Path)
parser.add_argument('--title', default='爱音 · PPO 残差行走策略')
args = parser.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720,
                     'renderer': 'RaytracedLighting', 'anti_aliasing': 3,
                     'extra_args': ['--/app/asyncRendering=false']})
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from omni.kit.viewport.utility import get_active_viewport
from physical_character import SkinBridge
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb


def main():
    ffmpeg = str(args.ffmpeg) if args.ffmpeg else shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            candidates = list((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe'))
            if candidates: ffmpeg = str(sorted(candidates)[-1])
    if not ffmpeg: raise RuntimeError('FFmpeg not found; specify --ffmpeg')
    data = np.load(args.rollout)
    is_mocap = 'kind' in data.files and str(data['kind']) == 'mocap_reference'
    is_reference = is_mocap or ('kind' in data.files and str(data['kind']) == 'procedural_reference')
    metadata_path = args.rollout.parent/'report.json'
    config = json.loads(metadata_path.read_text()).get('env_config', {}) if metadata_path.exists() else {}
    asset = Path(config.get('asset', ROOT/'output/anon_physical/anon_physical.usda'))
    poses = data['poses'][:, args.env_index].copy()
    names = list(data['link_names'])
    pelvis = names.index('pelvis')
    origin = poses[0, pelvis, :2].copy()
    poses[:, :, :2] -= origin
    fps = round(1/float(data['dt']))
    count = len(poses)-args.start_frame
    if args.max_frames: count = min(count, args.max_frames)
    if count <= 0: raise ValueError('No trajectory frames requested')
    context = omni.usd.get_context(); context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.DefinePrim('/World/Anon').GetReferences().AddReference(str(asset.resolve()))
    stage.GetPrimAtPath('/World/Anon/Bodies').SetActive(False)
    manifest_file = asset.parent/'manifest.json'
    if not manifest_file.exists(): manifest_file = ROOT/'output/anon_physical/stand/manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    bridge = SkinBridge(stage, manifest, names)
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(900.)
    key = UsdLux.DistantLight.Define(stage, '/World/Key'); key.CreateIntensityAttr(1600.)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    floor = UsdGeom.Mesh.Define(stage, '/World/Floor')
    floor.CreatePointsAttr([(-30, -30, -.006), (30, -30, -.006), (30, 30, -.006), (-30, 30, -.006)])
    floor.CreateFaceVertexCountsAttr([4]); floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr('none'); floor.CreateDisplayColorAttr([(.24, .28, .32)])
    grid = UsdGeom.BasisCurves.Define(stage, '/World/Grid')
    points = []
    for x in range(-20, 21):
        points.extend([(x, -20, -.003), (x, 20, -.003), (-20, x, -.003), (20, x, -.003)])
    grid.CreatePointsAttr(points); grid.CreateCurveVertexCountsAttr([2]*(len(points)//2))
    grid.CreateTypeAttr('linear'); grid.CreateWidthsAttr([.006]); grid.SetWidthsInterpolation('constant')
    grid.CreateDisplayColorAttr([(.4, .46, .5)])
    camera = UsdGeom.Camera.Define(stage, '/World/Camera'); camera.CreateFocalLengthAttr(42.)
    camera.CreateHorizontalApertureAttr(36.); camera.CreateVerticalApertureAttr(36.*720/1280)
    camera_op = camera.AddTransformOp()
    vp = get_active_viewport(); vp.camera_path = '/World/Camera'; vp.set_texture_resolution((1280, 720))
    from character_appearance import DefaultCharacterLook
    appearance=DefaultCharacterLook(stage,character_root='/World/Anon/Visual')
    def publish(frame):
        bridge.update(poses[frame])
        appearance.update(frame)
        center = poses[frame, pelvis, :3].astype(float); center[2] = .88
        eye = center+np.array([2.3, -4., 1.15])
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1)).GetInverse())
    publish(args.start_frame)
    for _ in range(45): app.update()
    vp.camera_path = '/World/Camera'
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 25)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    labels = dict(stand='站立', forward='前进', stop='停步', backward='后退', left='向左侧走', right='向右侧走',
                  diagonal='斜向行走', turn_left='行走并左转', turn_right='行走并右转', turn_in_place='原地左转',
                  turn_in_place_right='原地右转', final_stop='最终停步')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.encoder.log').open('wb')
    encoder = subprocess.Popen([ffmpeg, '-y', '-hide_banner', '-f', 'rawvideo',
                  '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(fps), '-i', 'pipe:0',
                  '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                  '-movflags', '+faststart', str(args.output.resolve())], stdin=subprocess.PIPE, stdout=log, stderr=log)
    first = None
    changes = []
    try:
        for video_frame in range(count):
            frame = video_frame+args.start_frame
            publish(frame)
            for _ in range(3): app.update()
            rgb = _capture_viewport_rgb()
            if rgb is None or rgb.shape != (720, 1280, 3): raise RuntimeError('Invalid viewport capture')
            if first is None: first = rgb.copy()
            if video_frame % 150 == 0:
                changes.append(float(np.abs(rgb.astype(float)-first).mean()))
            image = Image.fromarray(rgb); draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1280, 108), fill=(17, 23, 33))
            label = str(data['labels'][frame])
            caption = labels.get(label, ('方向 '+label.removeprefix('direction_')+'°') if label.startswith('direction_') else label)
            draw.text((24, 10), args.title+' / '+caption, font=font, fill=(245, 247, 250))
            command = data['commands'][frame, args.env_index]; vel = data['velocities'][frame, args.env_index]
            draw.text((24, 48), f'指令  前后 {command[0]:+.2f} m/s   左右 {command[1]:+.2f} m/s   转向 {command[2]:+.2f} rad/s', font=small, fill=(120, 218, 255))
            if is_mocap:
                draw.rectangle((0, 43, 1280, 76),fill=(17,23,33))
                draw.text((24,48),'来源：100STYLE Neutral 真人动捕，经骨架比例与关节范围适配',font=small,fill=(120,218,255))
            measured = '参考' if is_reference else '实际'
            draw.text((24, 77), f'{measured}  前后 {vel[0]:+.2f} m/s   左右 {vel[1]:+.2f} m/s   转向 {vel[2]:+.2f} rad/s', font=small, fill=(220, 227, 235))
            draw.rectangle((0, 681, 1280, 720), fill=(17, 23, 33))
            footer = '程序生成参考动画（非动捕、非物理策略结果）' if is_reference else '真实物理轨迹回放  |  无根节点辅助力'
            if is_mocap: footer = '100STYLE Neutral 动捕重定向 | Ian Mason 等 | CC BY 4.0 | 非物理策略结果'
            draw.text((24, 687), f'{footer}  |  {fps} FPS  |  {frame/fps:05.2f} s', font=small, fill=(220, 227, 235))
            if data['dones'][frame, args.env_index]:
                draw.text((24, 126), '本帧触发安全重置', font=font, fill=(255, 70, 70), stroke_width=1)
            if video_frame == 0 or (frame > 0 and data['labels'][frame] != data['labels'][frame-1]):
                image.save(args.output.with_name(args.output.stem+f'_{frame:04d}.jpg'), quality=95)
            encoder.stdin.write(np.asarray(image).tobytes())
            if video_frame % 150 == 0: print(f'ANON_VIDEO {video_frame}/{count}', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0: raise RuntimeError('FFmpeg encoding failed')
    except BaseException:
        encoder.kill(); encoder.wait(); raise
    finally:
        log.close()
    meta = dict(video=str(args.output.resolve()), rollout=str(args.rollout.resolve()), fps=fps,
                frames=count, duration_seconds=count/fps, env_index=args.env_index,
                method=('Kinematic authored reference FK replay; not a physical policy rollout' if is_reference else
                        'Unmodified PhysX policy link-pose replay through SkinBridge; camera tracks position only'),
                image_change_checks=changes, rendering_physics_steps=0)
    meta['kind'] = 'procedural_reference_animation' if is_reference else 'physical_policy_rollout'
    if is_mocap:
        meta['kind'] = 'mocap_reference_animation'
        meta['method'] = '100STYLE Neutral captured motion retargeted onto Anon joint chains; kinematic FK replay'
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('ANON_VIDEO_COMPLETE '+json.dumps(meta), flush=True)


result = 0
try:
    main()
except Exception:
    import traceback
    traceback.print_exc(); sys.stderr.flush(); result = 1
finally:
    app.close(exit_code=result)
