"""Render two matched hand-skirt measurement replays at native 1080p."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT
from diagnostics.replay_anon_secondary import sha256
sys.path.insert(0, str(PROJECT_ROOT))


FPS, WIDTH, HEIGHT = 30, 960, 1080
PANELS = ('6700  /  BASELINE', '6800  /  HAND-SKIRT REWARD')


def load_lane(path):
    info = json.loads((path / 'secondary.json').read_text(encoding='utf-8'))
    rollout = Path(info['rollout'])
    if sha256(rollout) != info['rollout_sha256']:
        raise ValueError(f'Frozen measured trajectory changed: {rollout}')
    with np.load(rollout) as trace, np.load(path / 'secondary.npz') as skin, \
            np.load(path / 'hand_skirt_measurements.npz') as measured:
        pelvis = trace['link_names'].tolist().index('pelvis')
        collisions = np.zeros((info['frames'], 2, 2), dtype=bool)
        collisions[measured['frames']] = measured['hits']
        return dict(positions=skin['positions'], rotations=skin['rotations'], info=info,
                    labels=trace['labels'][:info['frames']], commands=trace['requested_commands'][:info['frames']],
                    pelvis=trace['poses'][:info['frames'], 0, pelvis, :3],
                    heading=trace['headings'][:info['frames'], 0], collisions=collisions)


def measure_video(ffmpeg, path, expected_frames):
    result = subprocess.run([ffmpeg, '-v', 'info', '-xerror', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'],
                            capture_output=True, text=True, errors='replace', check=True)
    counts = re.findall(r'frame=\s*(\d+)', result.stderr)
    stream = next(line for line in result.stderr.splitlines() if 'Video:' in line)
    if not counts or int(counts[-1]) != expected_frames or '1920x1080' not in stream or '30 fps' not in stream:
        raise ValueError('Unexpected output video format or frame count')
    return dict(frames=int(counts[-1]), fps=FPS, resolution=[1920, 1080], seconds=expected_frames / FPS)


def record(app, lanes, output, ffmpeg):
    from omni.kit.viewport.utility import get_active_viewport
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, UsdGeom, UsdLux, UsdSkel, Vt
    from scipy.spatial.transform import Rotation
    from character_appearance import DefaultCharacterLook
    from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb

    a, b = lanes
    for key in ('labels', 'commands'):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ('physics_hz', 'edge_nodes', 'dynamic_nodes', 'secondary_profile_sha256', 'pmx_sha256'):
        if a['info'][key] != b['info'][key]:
            raise ValueError(f'Physics replay differs in {key}')
    labels = a['labels']
    walk = np.flatnonzero(labels == 'walk_before_stop')
    stop = np.flatnonzero(labels == 'stop_after_walk')
    stand = np.flatnonzero(labels == 'stand')
    shots = [
        ('Idle', int(stand[0]) + 2 * FPS, int(stand[-1]) + 1),
        ('Forward walk 0.30 m/s', int(walk[0]) + 2 * FPS, int(walk[-1]) + 1),
        ('Walk and stop', int(stop[0]) - FPS, int(stop[-1]) + 1),
    ]
    expected = sum(end - start for _, start, end in shots)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    usd = PROJECT_ROOT / 'output/pmx_prototype/character.usdc'
    stage.DefinePrim('/World/Character').GetReferences().AddReference(str(usd))
    animation = UsdSkel.Animation.Get(stage, '/World/Character/Animation')
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(900.)
    key = UsdLux.DistantLight.Define(stage, '/World/Key')
    key.CreateIntensityAttr(1600.)
    key.AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    floor = UsdGeom.Mesh.Define(stage, '/World/VisibleGround')
    floor.CreatePointsAttr([(-50, -50, -.006), (50, -50, -.006), (50, 50, -.006), (-50, 50, -.006)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr('none')
    floor.CreateDisplayColorAttr([(.24, .28, .32)])
    appearance = DefaultCharacterLook(stage, a['pelvis'][0])
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(42.)
    camera.CreateVerticalApertureAttr(32.)
    camera.CreateHorizontalApertureAttr(32. * WIDTH / HEIGHT)
    camera_transform = camera.AddTransformOp()
    viewport = get_active_viewport()
    viewport.camera_path = '/World/Camera'
    viewport.set_texture_resolution((WIDTH, HEIGHT))
    face = ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf', 33)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 26)
    for _ in range(30):
        app.update()

    def paint(rgb, lane, title, hits):
        picture = Image.fromarray(rgb)
        overlay = ImageDraw.Draw(picture)
        accent = (154, 209, 251) if lane == 0 else (158, 233, 190)
        overlay.rectangle((0, 0, WIDTH, 106), fill=(15, 23, 35))
        overlay.rectangle((0, 0, WIDTH, 5), fill=accent)
        overlay.text((25, 17), PANELS[lane], font=small, fill=accent)
        overlay.text((25, 48), title, font=face, fill='white')
        overlay.rectangle((0, 1009, WIDTH, HEIGHT), fill=(15, 23, 35))
        for x, side, collision in zip((25, WIDTH // 2 + 8), ('Left', 'Right'), hits):
            parts = ' + '.join(name for name, hit in zip(('palm', 'fingers'), collision) if hit)
            label = f'{side}   {parts} overlap' if parts else f'{side}   clear'
            overlay.text((x, 1028), label, font=small, fill=(255, 173, 154) if parts else (165, 238, 197))
        return picture

    with tempfile.TemporaryDirectory(prefix='render_panels_', dir=output.parent) as directory:
        panels = [Path(directory) / f'panel_{lane}.mkv' for lane in (0, 1)]
        for lane, (data, panel) in enumerate(zip(lanes, panels)):
            encoder = subprocess.Popen([ffmpeg, '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-c:v', 'ffv1', '-level', '3',
                '-threads', '4', str(panel)], stdin=subprocess.PIPE)
            frames = 0
            try:
                for title, begin, end in shots:
                    offset = Rotation.from_euler('z', a['heading'][end - 1]).apply([.5, -2.35, .18])
                    for frame in range(begin, end):
                        animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(data['positions'][frame]))
                        animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(data['rotations'][frame]))
                        center = data['pelvis'][frame].astype(float)
                        center[2] = .94
                        camera_transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center + offset)), Gf.Vec3d(*center),
                                                                    Gf.Vec3d(0, 0, 1)).GetInverse())
                        appearance.update(frames)
                        app.update()
                        if frame == begin:
                            for _ in range(10):
                                app.update()
                        rgb = _capture_viewport_rgb()
                        if rgb is None or rgb.shape != (HEIGHT, WIDTH, 3):
                            raise RuntimeError('Invalid viewport capture')
                        picture = paint(rgb, lane, title, data['collisions'][frame])
                        if frame == begin + min(FPS, end - begin - 1):
                            picture.save(output.parent / f'preview_{lane}_{begin}.jpg', quality=94)
                        encoder.stdin.write(np.asarray(picture).tobytes())
                        frames += 1
                    print(f'COMPARISON_SHOT {lane} {title} {frames}/{expected}', flush=True)
                encoder.stdin.close()
                if encoder.wait() or frames != expected:
                    raise RuntimeError('Incomplete comparison panel')
            except BaseException:
                if encoder.poll() is None:
                    encoder.kill()
                encoder.wait()
                raise
        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', str(panels[0]), '-i', str(panels[1]),
                        '-filter_complex', 'hstack=inputs=2', '-an', '-c:v', 'libx264', '-preset', 'fast',
                        '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    out = measure_video(ffmpeg, output, expected)
    return dict(out, video=str(output), sha256=sha256(output), appearance='shared approved look 08',
                bangs_shadow_enabled=appearance.bangs is not None, physics_hz=a['info']['physics_hz'],
                edge_nodes=a['info']['edge_nodes'], secondary_profile_sha256=a['info']['secondary_profile_sha256'],
                rollout_sha256=[d['info']['rollout_sha256'] for d in lanes], shots=shots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    args = parser.parse_args()
    if args.video.exists():
        raise FileExistsError(args.video)
    started = perf_counter()
    lanes = [load_lane(path) for path in (args.baseline, args.candidate)]
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        ffmpeg = str(next((Path.home() / 'AppData/Roaming/Python').glob(
            'Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': WIDTH, 'height': HEIGHT, 'renderer': 'RaytracedLighting',
                         'anti_aliasing': 1, 'extra_args': ['--/app/asyncRendering=false']})
    try:
        metadata = record(app, lanes, args.video.resolve(), ffmpeg)
        metadata['recording_seconds'] = perf_counter() - started
        args.video.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('HAND_SKIRT_VIDEO ' + json.dumps(metadata), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == '__main__':
    main()
