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
import shutil
import subprocess
import sys
import hashlib
import time

ROOT=PROJECT_ROOT
sys.path.insert(0,str(ROOT))
from diagnostics.anon_acceptance_video import acceptance_shots
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--title',required=True)
p.add_argument('--headless',action='store_true')
p.add_argument('--seconds',type=float,default=10)
p.add_argument('--manifest',type=Path)
p.add_argument('--acceptance-report',type=Path,help='Record highlights from the measured acceptance report in 1080p')
p.add_argument('--secondary-profile',type=config_path,default=DEFAULT_SECONDARY_PROFILE)
args=p.parse_args()
width,height=(1920,1080) if args.acceptance_report else (640,1080)
started=time.perf_counter()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':width,'height':height,'renderer':'RaytracedLighting','anti_aliasing':1,'extra_args':['--/app/asyncRendering=false']})
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


def record_acceptance(data,report,poses,pelvis,physics,motion,body_pose,publish,appearance,camera_op,ff):
    if len(data['labels'])!=len(poses):
        raise ValueError('Acceptance rollout labels and poses have different lengths')
    shots=acceptance_shots(data['labels'],report['segments'],30)

    face=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',34)
    text=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',24)
    video=args.output/'acceptance_preview_1080p.mp4'
    enc=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r','30','-i','pipe:0','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    frames=0
    chapters=[]
    yaw_pitch=np.rad2deg(data['head_errors'][:,0]) if 'head_errors' in data else None
    try:
        for begin,finish,title,head,body_status in shots:
            warm_start=max(0,begin-30)
            clip=poses[warm_start:finish].copy()
            clip[:,:,:2]-=clip[0,pelvis,:2].copy()
            last=clip[0]
            motion.live=body_pose(last)
            physics.initial_nodes=motion.live[2][motion.nodes].copy()
            physics.reset()
            for _ in range(physics.physics_hz//2):
                physics.step(0,produce_pose=False,diagnostics=False,compiled_hair=True,optimized_contacts=True)
            if 'headings' in data:
                heading=float(data['headings'][finish-1,0])
            else:
                forward=Rotation.from_quat(poses[finish-1,pelvis,3:]).apply([0.,-1.,0.])
                heading=float(np.arctan2(forward[0],-forward[1]))
            offset=Rotation.from_euler('z',heading).apply([1.22,-4.55,.40])
            chapters.append(dict(title=title,video_start_seconds=frames/30,
                                 source_frame_start=begin,source_frame_end=finish))
            for frame,current in enumerate(clip,warm_start):
                p,q=advance_pose(physics,motion,body_pose,last,current,fast=True)
                last=current
                if frame<begin:continue
                publish(p,q)
                center=current[pelvis,:3].astype(float);center[2]=.88
                camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*(center+offset)),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse())
                appearance.update(frames);app.update()
                if frame==begin:
                    for _ in range(12):app.update()
                rgb=_capture_viewport_rgb()
                if rgb is None or rgb.shape!=(height,width,3):
                    raise RuntimeError('Invalid acceptance viewport capture')
                im=Image.fromarray(rgb)
                draw=ImageDraw.Draw(im)
                draw.rounded_rectangle((48,42,585,221),radius=18,fill=(20,27,40))
                draw.text((72,60),args.title,font=text,fill=(159,198,227))
                draw.text((72,97),title,font=face,fill='white')
                statuses=[(72,'Body',body_status)]
                if head:
                    statuses=[(72,'Head',head['status']),(300,'Body',body_status)]
                for x,label,status in statuses:
                    color=(160,235,201) if status=='passed' else (254,156,143)
                    draw.text((x,160),f'{label} {status.upper()}',font=text,fill=color)
                draw.rounded_rectangle((48,856,585,1032),radius=18,fill=(20,27,40))
                if yaw_pitch is not None:
                    yaw,pitch=yaw_pitch[frame]
                    draw.text((72,874),f'Head now   {yaw:+.1f}° yaw   {pitch:+.1f}° pitch',font=text,fill='white')
                if head and 'yaw_bias_deg' in head:
                    draw.text((72,915),f"Steady mean   {head['yaw_bias_deg']:+.1f} / {head['pitch_bias_deg']:+.1f} deg",
                              font=text,fill=(159,198,227))
                draw.text((72,964),f'Trajectory {frame/30:06.2f} s',font=text,fill=(194,205,223))
                enc.stdin.write(np.asarray(im).tobytes())
                if frame==min(begin+90,finish-1):
                    im.save(args.output/f'preview_{len(chapters)}.jpg',quality=94)
                frames+=1
            print('ACCEPTANCE_VIDEO_SHOT '+title,flush=True)
        enc.stdin.close()
        if enc.wait():raise RuntimeError('Acceptance video encoding failed')
        subprocess.run([ff,'-v','error','-xerror','-i',str(video),'-map','0:v:0','-f','null','-'],check=True)
    except BaseException:
        if enc.poll() is None:enc.kill()
        enc.wait()
        raise
    return dict(video=str(video.resolve()),frames=frames,fps=30,resolution=[width,height],
                duration_seconds=frames/30,wall_seconds=time.perf_counter()-started,chapters=chapters,
                rollout=str(args.rollout.resolve()),acceptance_report=str(args.acceptance_report.resolve()),
                rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),
                secondary_physics=True,secondary_profile=str(args.secondary_profile.resolve()),appearance='shared approved look 08')


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
    physics.start(context.get_stage_id())
    views=[('side',np.array([2.8,0,.12])),('rear',np.array([0,2.8,.12]))]
    if report:
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
        report=dict(rollout=str(args.rollout.resolve()),rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),frames_per_view=count,fps=30,resolution=[640,1080],camera_offsets=dict(side=[2.8,0,.12],rear=[0,2.8,.12]),camera_tracks_position_only=True,pose_processing='Only remove initial planar translation; no angle correction or time warp',secondary_physics=True,secondary_settings=motion.rig_settings,appearance='shared approved look 08',appearance_defaults_sha256=hashlib.sha256((resolve_source('appearance_defaults.py')).read_bytes()).hexdigest(),falls=int(d['dones'][:count].sum()) if 'dones' in d else None)
        report['support_source_sha256']={name:hashlib.sha256((resolve_source(name)).read_bytes()).hexdigest() for name in ['appearance_defaults.py','character_appearance.py','bangs_shadow.py','mmd_cel.mdl','physx_secondary.py','secondary_profiles.py']}
        report['commanded_forward_speed']=float(np.mean(d['commands'][:count,0,0])) if 'commands' in d else .4
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:
    if physics:physics.close()
    app.close(exit_code=code)
sys.exit(code)
