"""Render synced A/B panels from the verified and separately baked test shots."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'source/pmx_prototype')]
import source_layout
from diagnostics.replay_anon_secondary import sha256
from runtime.anon_crowd_recording import find_ffmpeg


WIDTH, HEIGHT, FPS = 960, 1080, 30
HAPERTURE, FOCAL = 31., 42.
LANES = ('base', 'trained')
PANEL = (17, 25, 38)
ACCENTS = ((155, 205, 255), (157, 232, 186))
ORANGE = (255, 186, 99)
GREEN = (157, 232, 186)
RED = (255, 162, 152)


def camera_pose(chapter, camera_xy):
    left = np.array(chapter['camera_left_xy'])
    forward = np.array([left[1], -left[0]])
    center = np.r_[camera_xy, .82]
    distance_scale = chapter['camera_scale']
    eye = center + distance_scale * np.r_[3.75 * forward - 2.65 * left, .92]
    return center, eye


def project(point, center, eye):
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = point - eye
    zoom = WIDTH * FOCAL / HAPERTURE / np.dot(relative, forward)
    return np.array([WIDTH/2 + zoom*np.dot(relative, right),
                     HEIGHT/2 - zoom*np.dot(relative, up)])


def impulse_arrow(draw, center, eye, waist, impulse):
    direction = np.r_[impulse, 0.]
    waist = waist.astype(float) + np.array([0., 0., .09])
    start = project(waist - .64 * direction, center, eye)
    end = project(waist + .05 * direction, center, eye)
    axis = end-start
    length = np.linalg.norm(axis)
    if length < 4:
        return
    unit = axis/length
    normal = np.array([-unit[1], unit[0]])
    points = [end, end-22*unit+11*normal, end-22*unit-11*normal]
    draw.line([tuple(start), tuple(end)], fill=(31, 22, 21), width=13)
    draw.line([tuple(start), tuple(end-3*unit)], fill=ORANGE, width=7)
    draw.polygon([tuple(p) for p in points], fill=ORANGE, outline=(31, 22, 21), width=2)


def render(app, destination, chapters, preview, plan):
    from omni.kit.viewport.utility import get_active_viewport
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, UsdSkel, Vt
    from character_appearance import DefaultCharacterLook
    from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb

    font = 'C:/Windows/Fonts/msyh.ttc'
    title_font = ImageFont.truetype(font, 35)
    body_font = ImageFont.truetype(font, 24)
    indicator_font = ImageFont.truetype(font, 26)
    small_font = ImageFont.truetype(font, 20)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    asset = ROOT / 'output/pmx_prototype/character.usdc'
    stage.DefinePrim('/World/Character').GetReferences().AddReference(str(asset))
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
    appearance = DefaultCharacterLook(stage)
    candidate_label = plan['candidate_label']
    if not plan['clean_passed']:
        candidate_label += '  ·  试验版'
    labels = (plan.get('base_label', '第一轮抗扰续训  ·  对照'), candidate_label)
    face_light = UsdGeom.Xformable(appearance.face_light).GetOrderedXformOps()[0]
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(FOCAL)
    camera.CreateHorizontalApertureAttr(HAPERTURE)
    camera.CreateVerticalApertureAttr(HAPERTURE*HEIGHT/WIDTH)
    camera_transform = camera.AddTransformOp()
    viewport = get_active_viewport()
    viewport.camera_path = '/World/Camera'
    viewport.set_texture_resolution((WIDTH, HEIGHT))
    for _ in range(25):
        app.update()

    def paint(rgb, lane_index, chapter, frame, velocity, center, eye, waist, impulse):
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        lane = LANES[lane_index]
        elapsed = (frame - chapter['impact_frame'] + 1)/FPS
        outcome = chapter['lanes'][lane]
        if frame >= chapter['impact_frame'] and elapsed < .37:
            impulse_arrow(draw, center, eye, waist, impulse)
        accent = ACCENTS[lane_index]
        draw.rectangle((0, 0, WIDTH, 127), fill=PANEL)
        draw.rectangle((0, 0, WIDTH, 5), fill=accent)
        draw.text((28, 14), labels[lane_index], font=body_font, fill=accent)
        number = chapter['start_frame']//chapter['duration_frames']+1
        draw.text((28, 48), f'{number:02d}/05  {chapter["title"]}', font=title_font, fill='white')
        forward, side, yaw = chapter['command']
        draw.text((29, 97), f'指令  前后 {forward:+.2f} m/s   横移 {side:+.2f} m/s   转向 {yaw:+.2f} rad/s',
                  font=small_font, fill=(187, 205, 222))
        draw.rectangle((0, 972, WIDTH, HEIGHT), fill=PANEL)
        if frame < chapter['impact_frame']:
            state, color = '受力前', accent
        elif elapsed < .37:
            state, color = '施加推力 · 8.33 ms', ORANGE
        elif outcome['fell'] and elapsed + .001 >= float(outcome['fall_seconds']):
            state, color = f'跌倒 · {float(outcome["fall_seconds"]):.2f} s', RED
        elif outcome['recovered'] and elapsed + .001 >= outcome['recovery_seconds']:
            state, color = f'已恢复 · {outcome["recovery_seconds"]:.2f} s', GREEN
        else:
            state, color = '受力后 · 继续跟踪原指令', accent
        draw.text((28, 982), state, font=indicator_font, fill=color)
        draw.text((WIDTH-29, 982), f'相对受力 {elapsed:+.2f} s', font=body_font,
                  anchor='ra', fill='white')
        impulse_speed = f'{chapter["impulse_m_s"]:.3f}'.rstrip('0').rstrip('.')
        draw.text((28, 1017),
                  f'向{chapter["direction_cn"]}推  {impulse_speed} m/s（等效速度增量）     '
                  f'瞬时  前后 {velocity[0]:+.2f}  横移 {velocity[1]:+.2f}  转向 {velocity[2]:+.2f}',
                  font=small_font, fill=(197, 211, 227))
        area = WIDTH-56
        draw.rectangle((28, 1055, WIDTH-28, 1060), fill=(60, 77, 95))
        draw.rectangle((28, 1055, 28+round(area*frame/(chapter['duration_frames']-1)), 1060),
                       fill=accent)
        tick = 28+round(area*chapter['impact_frame']/(chapter['duration_frames']-1))
        draw.rectangle((tick-2, 1050, tick+2, 1065), fill=ORANGE)
        return image

    ffmpeg = find_ffmpeg()
    validation = json.loads((OUT/'bake_validation.json').read_text(encoding='utf-8'))
    panels = []
    rendered = 0
    for lane_index, lane in enumerate(LANES):
        path = OUT / lane / 'baked.npz'
        if sha256(path) != validation[lane]['baked']:
            raise ValueError(f'Baked data changed for {lane}')
        handle = np.load(path)
        data = dict(handle)
        handle.close()
        waist = data['link_names'].tolist().index('waist_pitch')
        pelvis = data['link_names'].tolist().index('pelvis')
        if preview:
            encoder = None
        else:
            panel = OUT / f'{lane}_panel.mp4'
            panels.append(panel)
            encoder = subprocess.Popen([ffmpeg, '-n', '-v', 'error', '-nostdin', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14', '-pix_fmt', 'yuv420p',
                '-threads', '4', str(panel)], stdin=subprocess.PIPE)
        total = 0
        try:
            for clip, chapter in enumerate(chapters):
                frames = [chapter['impact_frame'], chapter['duration_frames']-1] if preview else range(chapter['duration_frames'])
                for number, frame in enumerate(frames):
                    animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(data['positions'][clip,frame]))
                    animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(data['rotations'][clip,frame]))
                    center, eye = camera_pose(chapter, data['camera_xy'][clip,frame])
                    camera_transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center),
                                                                Gf.Vec3d(0,0,1)).GetInverse())
                    pelvis_xy = data['poses'][clip,frame,pelvis,:2]
                    face_light.Set(Gf.Vec3d(float(pelvis_xy[0]),float(pelvis_xy[1]) - 2.5,1.7))
                    appearance.update(rendered)
                    app.update()
                    if number == 0 or preview:
                        for _ in range(8):
                            app.update()
                    for attempt in range(4):
                        rgb = _capture_viewport_rgb()
                        if rgb is not None and rgb.shape == (HEIGHT, WIDTH, 3):
                            break
                        print(f'VIEWPORT_RETRY attempt={attempt+1} shape={getattr(rgb, "shape", None)}', flush=True)
                        viewport.camera_path = '/World/Camera'
                        viewport.set_texture_resolution((WIDTH, HEIGHT))
                        for _ in range(12):
                            app.update()
                    else:
                        raise RuntimeError(f'Invalid viewport capture shape: {getattr(rgb, "shape", None)}')
                    image = paint(rgb, lane_index, chapter, frame, data['velocity'][clip,frame],
                                  center, eye, data['poses'][clip,frame,waist,:3], data['arrow_xy'][clip])
                    if preview:
                        image.save(OUT/f'preview_{lane}_{clip+1}_{frame}.jpg', quality=95)
                    else:
                        encoder.stdin.write(np.asarray(image).tobytes())
                    total += 1
                    rendered += 1
                print(f'RENDER {lane} trial={chapter["trial_id"]} frames={total}', flush=True)
            if encoder is not None:
                encoder.stdin.close()
                if encoder.wait() or total != sum(c['duration_frames'] for c in chapters):
                    raise RuntimeError(f'Incomplete panel {lane}')
        except BaseException:
            if encoder is not None and encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            raise
    if preview:
        return
    subprocess.run([ffmpeg,'-n','-v','error','-nostdin','-i',str(panels[0]),'-i',str(panels[1]),
                    '-filter_complex','hstack=inputs=2','-an','-c:v','libx264','-crf','16','-preset','fast',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)], check=True)
    verification = subprocess.run([ffmpeg,'-v','info','-xerror','-nostdin','-i',str(destination),
                                   '-map','0:v:0','-f','null','-'],
                                  check=True, capture_output=True,text=True,errors='replace')
    counts = re.findall(r'frame=\s*(\d+)', verification.stderr)
    stream = next(line for line in verification.stderr.splitlines() if 'Video:' in line)
    frames = sum(c['duration_frames'] for c in chapters)
    if not counts or int(counts[-1]) != frames or '1920x1080' not in stream or '30 fps' not in stream:
        raise ValueError('Unexpected decoded frame count or stream format')
    (OUT/'video_validation.json').write_text(json.dumps(dict(
        video=str(destination), sha256=sha256(destination), frames=frames, fps=FPS,
        seconds=frames/FPS, resolution=[1920,1080], appearance='mmd-toon (cel-studio)',
        secondary_hz=120, lanes=LANES, candidate=plan['chosen'], chapters=chapters), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'VIDEO_VERIFIED {frames} frames {destination}', flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=OUT)
    parser.add_argument('--destination', type=Path,
                        default=ROOT / 'output/anon_push_recovery/acceptance.mp4')
    args = parser.parse_args()
    OUT = args.work_dir.resolve()
    destination = args.destination.resolve()
    if not args.preview and destination.exists():
        raise FileExistsError(destination)
    chapters = json.loads((OUT/'chapters.json').read_text(encoding='utf-8'))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':True,'width':WIDTH,'height':HEIGHT,'renderer':'RaytracedLighting',
                         'anti_aliasing':1,'extra_args':['--/app/asyncRendering=false']})
    exit_code = 0
    try:
        plan = json.loads((OUT/'video_plan.json').read_text(encoding='utf-8'))
        render(app, destination, chapters, args.preview, plan)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
