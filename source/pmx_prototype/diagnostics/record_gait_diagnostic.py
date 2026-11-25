"""Matched side/rear diagnostic replay, with shared approved look and PhysX cloth.

Policy inputs are measured link-pose rollouts; reference inputs are kinematic.
No pose smoothing, joint correction, or policy inference is performed here.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, resolve_source, config_path

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import hashlib
import time

ROOT=PROJECT_ROOT
sys.path.insert(0,str(ROOT))
from diagnostics.anon_acceptance_video import (
    FULL_VIDEO_NAME, PREVIEW_VIDEO_NAME, acceptance_shots, full_acceptance_chapters,
)
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--title',required=True)
p.add_argument('--headless',action='store_true')
p.add_argument('--seconds',type=float,default=10)
p.add_argument('--manifest',type=Path)
p.add_argument('--acceptance-report',type=Path,help='Record measured acceptance footage in 1080p')
p.add_argument('--secondary-profile',type=config_path,default=DEFAULT_SECONDARY_PROFILE)
p.add_argument('--sequence', type=int, help='Record one complete group from a grouped acceptance rollout')
p.add_argument('--worker-threads', type=int, help='CPU thread budget for one parallel render worker')
p.add_argument('--audit-physics', action='store_true', help='Save sampled display skeletons for paired recording checks')
p.add_argument('--bake-secondary', action='store_true', help='Compute full secondary motion without capturing video')
p.add_argument('--secondary-cache', type=Path, help='Render the already computed full secondary motion')
args=p.parse_args()
if args.sequence is not None and (not args.acceptance_report or args.sequence < 1):
    p.error('--sequence requires a positive acceptance group number and --acceptance-report')
if args.worker_threads is not None and args.worker_threads < 1:
    p.error('--worker-threads must be positive')
if (args.bake_secondary or args.secondary_cache) and not args.acceptance_report:
    p.error('A secondary cache requires a complete acceptance report')
if args.bake_secondary and (args.secondary_cache or args.sequence is not None):
    p.error('--bake-secondary computes all acceptance groups in order')
width,height=(1920,1080) if args.acceptance_report else (640,1080)
started=time.perf_counter()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':width,'height':height,'renderer':'RaytracedLighting','anti_aliasing':1,
                   'limit_cpu_threads':args.worker_threads or 16,'disable_viewport_updates':args.bake_secondary,
                   'extra_args':['--/app/asyncRendering=false']})
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Gf,UsdGeom,UsdLux,UsdSkel,Vt
from omni.kit.viewport.utility import get_active_viewport
from physx_secondary import BodyMotion,PhysxSecondary
from character_appearance import DefaultCharacterLook
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb

class ReplayMotion(BodyMotion):
    live=None
    def pose(self,frame):
        return super().pose(frame) if self.live is None else self.live


def advance_pose(physics,motion,body_pose,last,current,*,fast=False):
    qa=last[:,3:];qb=current[:,3:].copy();qb[(qa*qb).sum(-1)<0]*=-1
    substeps=physics.physics_hz//30
    for substep in range(substeps):
        a=(substep+1)/substeps
        state=last*(1-a)+current*a
        state[:,3:]=qa*(1-a)+qb*a
        state[:,3:]/=np.linalg.norm(state[:,3:],axis=-1,keepdims=True)
        motion.live=body_pose(state)
        if fast:
            p,q=physics.step(0,produce_pose=substep==substeps-1,diagnostics=False,
                             compiled_hair=True,optimized_contacts=True)
        else:
            p,q=physics.step(0)
    return p,q


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def physics_resets(data, sequence_starts):
    reset_at = set(sequence_starts.tolist())
    if 'dones' in data:
        reset_at.update((np.flatnonzero(data['dones'][:-1, 0]) + 1).tolist())
    return reset_at


def advance_recording_pose(physics, motion, body_pose, last, current, reset):
    if reset:
        last = current
        motion.live = body_pose(last)
        physics.initial_nodes = motion.live[2][motion.nodes].copy()
        physics.reset()
        for _ in range(physics.physics_hz // 2):
            physics.step(0, produce_pose=False, diagnostics=False, compiled_hair=True, optimized_contacts=True)
    return advance_pose(physics, motion, body_pose, last, current, fast=True)


def bake_acceptance(data, report, poses, pelvis, physics, motion, body_pose):
    if not report.get('complete_scope'):
        raise ValueError('Secondary baking requires a full measured acceptance rollout')
    started = time.perf_counter()
    sequence_starts = data['sequence_start_frames'] if 'sequence_start_frames' in data else np.asarray([0])
    chapters = full_acceptance_chapters(data['labels'], report['segments'], 30, sequence_starts)
    resets = physics_resets(data, sequence_starts)
    frames, bones = len(poses), len(motion.joints)
    positions = np.lib.format.open_memmap(args.output / 'positions.npy', mode='w+',
                                         dtype=np.float32, shape=(frames, bones, 3))
    rotations = np.lib.format.open_memmap(args.output / 'rotations.npy', mode='w+',
                                         dtype=np.float32, shape=(frames, bones, 4))
    last = None
    for chapter in chapters:
        begin, finish = chapter['source_frame_start'], chapter['source_frame_end']
        origin = sequence_starts[chapter['sequence'] - 1]
        clip = poses[begin:finish].copy()
        clip[:, :, :2] -= poses[origin, pelvis, :2]
        for frame, current in enumerate(clip, begin):
            p, q = advance_recording_pose(physics, motion, body_pose, last, current, frame in resets)
            positions[frame], rotations[frame] = p, q
            last = current
        print(f'ACCEPTANCE_BAKE_PROGRESS {finish}/{frames}', flush=True)
    positions.flush()
    rotations.flush()
    info = dict(frames=frames, bones=bones, rollout_sha256=file_digest(args.rollout),
                profile_sha256=file_digest(args.secondary_profile), physics_hz=physics.physics_hz,
                wall_seconds=time.perf_counter() - started)
    (args.output / 'secondary_cache.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


def open_secondary_cache(frames, bones):
    cache = args.secondary_cache.resolve()
    metadata = json.loads((cache / 'secondary_cache.json').read_text(encoding='utf-8'))
    if (metadata['frames'] != frames or metadata['bones'] != bones
            or metadata['rollout_sha256'] != file_digest(args.rollout)
            or metadata['profile_sha256'] != file_digest(args.secondary_profile)):
        raise ValueError('Secondary cache comes from a different measured trajectory or profile')
    positions = np.load(cache / 'positions.npy', mmap_mode='r')
    rotations = np.load(cache / 'rotations.npy', mmap_mode='r')
    if positions.shape != (frames, bones, 3) or rotations.shape != (frames, bones, 4):
        raise ValueError('Secondary cache has an invalid bone or frame count')
    return positions, rotations


def record_acceptance(data,report,poses,pelvis,physics,motion,body_pose,publish,appearance,camera_op,ff):
    if len(data['labels'])!=len(poses):
        raise ValueError('Acceptance rollout labels and poses have different lengths')
    sequence_starts = data['sequence_start_frames'] if 'sequence_start_frames' in data else np.asarray([0])
    full = bool(report.get('complete_scope'))
    chapters = []
    if full:
        chapters = full_acceptance_chapters(data['labels'], report['segments'], 30, sequence_starts)
        shots = []
        for chapter, segment in zip(chapters, report['segments'], strict=True):
            response = segment.get('response_steady', {})
            status = 'passed' if segment['passed'] else 'failed'
            shots.append((chapter['source_frame_start'], chapter['source_frame_end'], chapter['title'],
                          response.get('idle_head'), response.get('body_status', status)))
        # Keep a lane's appearance and physical state across adjacent scenes.
        # A new lane or a recorded episode reset needs a fresh cloth state.
        reset_at = physics_resets(data, sequence_starts)
        if 'headings' in data:
            headings = data['headings'][:, 0]
        else:
            forward = Rotation.from_quat(poses[:, pelvis, 3:]).apply([0., -1., 0.])
            headings = np.arctan2(forward[:, 0], -forward[:, 1])
        camera_offsets = Rotation.from_euler('z', headings[:, None]).apply([1.22, -4.55, .40])
        requests = data['requested_commands'] if 'requested_commands' in data else data['commands'][:, 0]
    else:
        shots = acceptance_shots(data['labels'], report['segments'], 30, sequence_starts)

    selected = list(range(len(shots)))
    if args.sequence is not None:
        if not full:
            raise ValueError('An individual group requires complete acceptance data')
        selected = [index for index, chapter in enumerate(chapters) if chapter['sequence'] == args.sequence]
        if not selected:
            raise ValueError(f'Acceptance group {args.sequence} does not exist')
    expected_frames = sum(shots[index][1] - shots[index][0] for index in selected)
    audited_frames, audited_positions, audited_rotations = [], [], []
    cached = open_secondary_cache(len(poses), len(motion.joints)) if args.secondary_cache else None
    if cached is not None and not full:
        raise ValueError('Secondary caching requires a full acceptance recording')

    face=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',34)
    text=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',24)
    video = args.output / (FULL_VIDEO_NAME if full else PREVIEW_VIDEO_NAME)
    encoder_threads = ['-threads', str(max(1, args.worker_threads // 2))] if args.worker_threads else []
    enc=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','30','-i','pipe:0','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p',
        *encoder_threads,'-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    frames=0
    yaw_pitch=np.rad2deg(data['head_errors'][:,0]) if 'head_errors' in data else None
    last = None
    try:
        for index in selected:
            begin,finish,title,head,body_status = shots[index]
            sequence_start = sequence_starts[np.searchsorted(sequence_starts, begin, side='right') - 1]
            warm_start = begin if full else max(sequence_start, begin - 30)
            clip=poses[warm_start:finish].copy()
            origin = sequence_start if full else warm_start
            clip[:, :, :2] -= poses[origin, pelvis, :2]
            if full:
                chapter = chapters[index]
                banner = f"{args.title}   |   {index + 1:02d}/{len(shots):02d}"
                stage_status = 'passed' if chapter['passed'] else 'failed'
            else:
                if 'headings' in data:
                    heading=float(data['headings'][finish-1,0])
                else:
                    forward=Rotation.from_quat(poses[finish-1,pelvis,3:]).apply([0.,-1.,0.])
                    heading=float(np.arctan2(forward[0],-forward[1]))
                offset=Rotation.from_euler('z',heading).apply([1.22,-4.55,.40])
                chapters.append(dict(title=title,video_start_seconds=frames/30,
                                     source_frame_start=begin,source_frame_end=finish))
            for frame,current in enumerate(clip,warm_start):
                reset = frame in reset_at if full else frame == warm_start
                if cached is None:
                    p, q = advance_recording_pose(physics, motion, body_pose, last, current, reset)
                else:
                    p, q = cached[0][frame], cached[1][frame]
                last=current
                if frame<begin:continue
                if args.audit_physics and (reset or frame % 30 == 0 or frame == finish - 1):
                    audited_frames.append(frame)
                    audited_positions.append(p.astype(np.float32, copy=True))
                    audited_rotations.append(q.astype(np.float32, copy=True))
                publish(p,q)
                center=current[pelvis,:3].astype(float);center[2]=.88
                if full:
                    offset = camera_offsets[frame]
                camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center+offset)),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse())
                appearance.update(frame if full else frames);app.update()
                refresh_viewport = reset if full else frame == begin
                if refresh_viewport:
                    for _ in range(12):app.update()
                rgb=_capture_viewport_rgb()
                if rgb is None or rgb.shape!=(height,width,3):
                    raise RuntimeError('Invalid acceptance viewport capture')
                im=Image.fromarray(rgb)
                draw=ImageDraw.Draw(im)
                panel_right = 790 if full else 585
                draw.rounded_rectangle((48,42,panel_right,221),radius=18,fill=(20,27,40))
                draw.text((72,60),banner if full else args.title,font=text,fill=(159,198,227))
                draw.text((72,97),title,font=face,fill='white')
                statuses=[(72,'Body',body_status)]
                if head and head.get('required', True):
                    statuses=[(72,'Head',head['status']),(300,'Body',body_status)]
                if full:
                    if chapter['counts_toward_gate']:
                        statuses = [(72, 'Stage', stage_status), (300, 'Body', body_status)]
                    else:
                        statuses = [(72, 'Stage', 'preparation')]
                    if chapter['counts_toward_gate'] and head and head.get('required'):
                        statuses.append((516, 'Head', head['status']))
                    elif chapter['falls']:
                        draw.text((516, 160), f"Falls {chapter['falls']}", font=text, fill=(254,156,143))
                for x,label,status in statuses:
                    color = {'passed': (160,235,201), 'preparation': (159,198,227)}.get(status, (254,156,143))
                    draw.text((x,160),f'{label} {status.upper()}',font=text,fill=color)
                draw.rounded_rectangle((48,856,panel_right,1032),radius=18,fill=(20,27,40))
                if yaw_pitch is not None:
                    yaw,pitch=yaw_pitch[frame]
                    draw.text((72,874),f'Head now   {yaw:+.1f}° yaw   {pitch:+.1f}° pitch',font=text,fill='white')
                if head and 'yaw_bias_deg' in head:
                    draw.text((72,915),f"Steady mean   {head['yaw_bias_deg']:+.1f} / {head['pitch_bias_deg']:+.1f} deg",
                              font=text,fill=(159,198,227))
                footer = f'Trajectory {frame/30:06.2f} s'
                if full:
                    forward, lateral, yaw = requests[frame]
                    footer = f"{frame // 1800:02d}:{frame / 30 % 60:04.1f}  |  Request  F {forward:+.2f}  L {lateral:+.2f}  Yaw {yaw:+.2f}"
                draw.text((72,964),footer,font=text,fill=(194,205,223))
                enc.stdin.write(np.asarray(im).tobytes())
                if not full and frame==min(begin+90,finish-1):
                    im.save(args.output/f'preview_{len(chapters)}.jpg',quality=94)
                frames+=1
            if full:
                print(f'ACCEPTANCE_VIDEO_SECTION {index+1}/{len(shots)} {frames}/{expected_frames} {title}', flush=True)
            else:
                print('ACCEPTANCE_VIDEO_SHOT '+title,flush=True)
        enc.stdin.close()
        if enc.wait():raise RuntimeError('Acceptance video encoding failed')
        decode = [ff, '-v', 'info' if full else 'error', '-xerror', '-i', str(video), '-map', '0:v:0', '-f', 'null', '-']
        checked = subprocess.run(decode, check=True, capture_output=full, text=True, errors='replace')
        if full:
            counts = re.findall(r'frame=\s*(\d+)', checked.stderr)
            stream = next((line for line in checked.stderr.splitlines() if 'Video:' in line), '')
            if (frames != expected_frames or not counts or int(counts[-1]) != frames
                    or '1920x1080' not in stream or '30 fps' not in stream):
                raise ValueError('Full acceptance encoding differs from the measured frame count or format')
    except BaseException:
        if enc.poll() is None:enc.kill()
        enc.wait()
        raise
    if args.audit_physics:
        np.savez_compressed(args.output / 'physics_samples.npz', frames=audited_frames,
                            positions=audited_positions, rotations=audited_rotations)
    coverage = 'sequence' if args.sequence is not None else ('full' if full else 'preview')
    return dict(video=str(video.resolve()),frames=frames,fps=30,resolution=[width,height],
                duration_seconds=frames/30,wall_seconds=time.perf_counter()-started,
                chapters=[chapters[index] for index in selected],
                coverage=coverage,sequence=args.sequence,
                secondary_cache=str(args.secondary_cache.resolve()) if args.secondary_cache else None,
                rollout=str(args.rollout.resolve()),acceptance_report=str(args.acceptance_report.resolve()),
                rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
                secondary_physics=True,secondary_profile=str(args.secondary_profile.resolve()),appearance='mmd-toon (cel-studio)')


physics=None
code=0
try:
    d=np.load(args.rollout);poses=d['poses'][:,0].copy();links=list(d['link_names']);pelvis=links.index('pelvis')
    assert abs(float(d['dt'])-1/30)<1e-6
    report=json.loads(args.acceptance_report.read_text(encoding='utf-8')) if args.acceptance_report else None
    count=len(poses) if report else min(len(poses),round(args.seconds*30))
    poses=poses[:count]
    poses[:,:,:2]-=poses[0,pelvis,:2].copy()
    motion=ReplayMotion(ROOT/'output/pmx_prototype/character.usdc',ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx',secondary_profile=args.secondary_profile)
    manifest_path=args.manifest or (Path(report['env_config']['asset']).parent/'manifest.json' if report
                                    else ROOT/'output/anon_physical_v3/manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert list(manifest['bone_paths'])==motion.joints
    owners=np.array([links.index(n) for n in manifest['bone_owners']]);rest=np.array(manifest['rest_positions'])
    origins=np.array([manifest['links'][n]['origin'] for n in manifest['bone_owners']]);parents=np.array(manifest['parents']);children=np.flatnonzero(parents>=0)
    def body_pose(transforms):
        t=transforms[owners];rot=Rotation.from_quat(t[:,3:]);wp=t[:,:3]+rot.apply(rest-origins);wq=t[:,3:].copy()
        p,q=wp.copy(),wq.copy();parent=parents[children];inv=Rotation.from_quat(wq[parent]).inv()
        p[children]=inv.apply(wp[children]-wp[parent]);q[children]=(inv*Rotation.from_quat(wq[children])).as_quat()
        return p,q,wp,wq
    motion.live=body_pose(poses[0]);motion.rig_settings['max_node_distance_from_target_m']=2.
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.DefinePrim('/World/Character').GetReferences().AddReference(str(ROOT/'output/pmx_prototype/character.usdc'))
    animation=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
    def publish(p,q):
        animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
    publish(*motion.live[:2])
    physics=PhysxSecondary(stage,motion,obstacles=False,quality=DEFAULT_SECONDARY_QUALITY)
    floor=UsdGeom.Xformable(stage.GetPrimAtPath('/World/PhysxGround'))
    next(op for op in floor.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(1000,1000,.05))
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900)
    key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(1600);UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25,-25,-35))
    ground=UsdGeom.Mesh.Define(stage,'/World/VisibleGround');ground.CreatePointsAttr([(-50,-50,-.006),(50,-50,-.006),(50,50,-.006),(-50,50,-.006)])
    ground.CreateFaceVertexCountsAttr([4]);ground.CreateFaceVertexIndicesAttr([0,1,2,3]);ground.CreateSubdivisionSchemeAttr('none');ground.CreateDisplayColorAttr([(.24,.28,.32)])
    if not args.bake_secondary:
        appearance=DefaultCharacterLook(stage,poses[0,pelvis,:3])
        camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateFocalLengthAttr(42)
        vertical=36*height/width if report else 36
        camera.CreateHorizontalApertureAttr(vertical*width/height);camera.CreateVerticalApertureAttr(vertical)
        camera_op=camera.AddTransformOp();vp=get_active_viewport();vp.camera_path='/World/Camera';vp.set_texture_resolution((width,height))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        ff=shutil.which('ffmpeg') or str(next((Path.home()/'AppData/Roaming/Python').glob(
            'Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))
    args.output.mkdir(parents=True,exist_ok=True)
    for _ in range(30):app.update()
    if not args.secondary_cache:
        physics.start(context.get_stage_id())
    views=[('side',np.array([2.8,0,.12])),('rear',np.array([0,2.8,.12]))]
    if args.bake_secondary:
        bake_acceptance(d, report, poses, pelvis, physics, motion, body_pose)
    elif report:
        metadata=record_acceptance(d,report,poses,pelvis,physics,motion,body_pose,publish,appearance,camera_op,ff)
        (args.output/'video_report.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('ACCEPTANCE_VIDEO_COMPLETE '+json.dumps(metadata),flush=True)
    for view,offset in ([] if report else views):
        motion.live=body_pose(poses[0]);physics.initial_nodes=motion.live[2][motion.nodes].copy();physics.reset()
        for _ in range(physics.physics_hz):physics.step(0)
        last=poses[0].copy();enc=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x1080','-r','30','-i','pipe:0','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p',str(args.output/(view+'.mp4'))],stdin=subprocess.PIPE)
        try:
            for frame,current in enumerate(poses):
                p,q=advance_pose(physics,motion,body_pose,last,current)
                publish(p,q)
                center=current[pelvis,:3].astype(float).copy();center[2]=.85
                camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center+offset)),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse())
                appearance.update(frame);app.update()
                if frame==0:
                    for _ in range(12):app.update()
                rgb=_capture_viewport_rgb();assert rgb.shape==(1080,640,3)
                im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,90),fill=(17,23,33));draw.text((16,12),args.title,font=font,fill='white')
                speed=float(np.mean(d['commands'][:count,0,0])) if 'commands' in d else .4
                draw.text((16,54),('侧面' if view=='side' else '背面')+f' · 同机位 / {speed:.1f} m/s',font=small,fill=(120,218,255))
                draw.rectangle((0,1010,640,1080),fill=(17,23,33));draw.text((16,1018),'重定向动捕 · 运动学参考' if 'kind' in d and str(d['kind'])=='mocap_reference' else '真实物理策略轨迹 · 无驱动补偿',font=small,fill='white')
                draw.text((16,1047),'外观 08 · 头发 / 裙摆 PhysX · '+f'{frame/30:.2f}s',font=small,fill='white')
                enc.stdin.write(np.asarray(im).tobytes())
                if frame%90==0:im.save(args.output/f'{view}_{frame:04d}.jpg',quality=94)
                last=current.copy()
            enc.stdin.close();assert enc.wait()==0
        except BaseException:
            enc.kill();enc.wait();raise
        print('GAIT_VIEW_COMPLETE',view,flush=True)
    if not report:
        report=dict(rollout=str(args.rollout.resolve()),rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),frames_per_view=count,fps=30,resolution=[640,1080],camera_offsets=dict(side=[2.8,0,.12],rear=[0,2.8,.12]),camera_tracks_position_only=True,pose_processing='Only remove initial planar translation; no angle correction or time warp',secondary_physics=True,secondary_settings=motion.rig_settings,appearance='mmd-toon (cel-studio)',appearance_defaults_sha256=hashlib.sha256((resolve_source('appearance_defaults.py')).read_bytes()).hexdigest(),falls=int(d['dones'][:count].sum()) if 'dones' in d else None)
        report['support_source_sha256']={name:hashlib.sha256((resolve_source(name)).read_bytes()).hexdigest() for name in ['appearance_defaults.py','character_appearance.py','bangs_shadow.py','cel_studio.mdl','physx_secondary.py','secondary_profiles.py']}
        report['commanded_forward_speed']=float(np.mean(d['commands'][:count,0,0])) if 'commands' in d else .4
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:
    if physics:physics.close()
    app.close(exit_code=code)
sys.exit(code)
