"""Matched side/rear diagnostic replay, with shared approved look and PhysX cloth.

Policy inputs are measured link-pose rollouts; reference inputs are kinematic.
No pose smoothing, joint correction, or policy inference is performed here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--title',required=True)
p.add_argument('--headless',action='store_true')
p.add_argument('--seconds',type=float,default=10)
p.add_argument('--manifest',type=Path,default=ROOT/'output/anon_physical_v3/manifest.json')
p.add_argument('--secondary-profile',type=Path,default=DEFAULT_SECONDARY_PROFILE)
args=p.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':640,'height':1080,'renderer':'RaytracedLighting','anti_aliasing':1,'extra_args':['--/app/asyncRendering=false']})
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

physics=None
code=0
try:
    d=np.load(args.rollout);poses=d['poses'][:,0].copy();links=list(d['link_names']);pelvis=links.index('pelvis')
    assert abs(float(d['dt'])-1/30)<1e-6
    count=min(len(poses),round(args.seconds*30));poses=poses[:count]
    poses[:,:,:2]-=poses[0,pelvis,:2].copy()
    motion=ReplayMotion(ROOT/'output/pmx_prototype/character.usdc',ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx',secondary_profile=args.secondary_profile)
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
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
    camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateFocalLengthAttr(42);camera.CreateHorizontalApertureAttr(36*640/1080);camera.CreateVerticalApertureAttr(36)
    camera_op=camera.AddTransformOp();vp=get_active_viewport();vp.camera_path='/World/Camera';vp.set_texture_resolution((640,1080))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    ff=str(next((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))
    args.output.mkdir(parents=True,exist_ok=True)
    for _ in range(30):app.update()
    physics.start(context.get_stage_id())
    for view,offset in [('side',np.array([2.8,0,.12])),('rear',np.array([0,2.8,.12]))]:
        motion.live=body_pose(poses[0]);physics.initial_nodes=motion.live[2][motion.nodes].copy();physics.reset()
        for _ in range(physics.physics_hz):physics.step(0)
        last=poses[0].copy();enc=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x1080','-r','30','-i','pipe:0','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p',str(args.output/(view+'.mp4'))],stdin=subprocess.PIPE)
        try:
            for frame,current in enumerate(poses):
                qa=last[:,3:];qb=current[:,3:].copy();qb[(qa*qb).sum(-1)<0]*=-1
                for substep in range(physics.physics_hz//30):
                    a=(substep+1)/(physics.physics_hz//30);x=last*(1-a)+current*a;x[:,3:]=qa*(1-a)+qb*a;x[:,3:]/=np.linalg.norm(x[:,3:],axis=-1,keepdims=True)
                    motion.live=body_pose(x);p,q=physics.step(0)
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
    report=dict(rollout=str(args.rollout.resolve()),rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest(),frames_per_view=count,fps=30,resolution=[640,1080],camera_offsets=dict(side=[2.8,0,.12],rear=[0,2.8,.12]),camera_tracks_position_only=True,pose_processing='Only remove initial planar translation; no angle correction or time warp',secondary_physics=True,secondary_settings=motion.rig_settings,appearance='shared approved look 08',appearance_defaults_sha256=hashlib.sha256((HERE/'appearance_defaults.py').read_bytes()).hexdigest(),falls=int(d['dones'][:count].sum()) if 'dones' in d else None)
    report['support_source_sha256']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['appearance_defaults.py','character_appearance.py','bangs_shadow.py','mmd_cel.mdl','physx_secondary.py','secondary_profiles.py']}
    report['commanded_forward_speed']=float(np.mean(d['commands'][:count,0,0])) if 'commands' in d else .4
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
except Exception:
    import traceback
    traceback.print_exc();code=1
finally:
    if physics:physics.close()
    app.close(exit_code=code)
sys.exit(code)
