"""Record live native PhysX hair and skirt while playing an existing PMX dance."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES,resolve_appearance_args
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE,DEFAULT_SECONDARY_QUALITY
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--usd',type=Path,required=True)
parser.add_argument('--pmx',type=Path,default=ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--audio',type=Path)
parser.add_argument('--ffmpeg',required=True)
parser.add_argument('--max-frames',type=int,default=0)
parser.add_argument('--save-poses',type=Path,help='Save exact rendered joint poses for controlled look comparisons')
parser.add_argument('--load-poses',type=Path,help='Render exact joint poses from a previous recording')
parser.add_argument('--verify-bangs-shadow',action='store_true')
parser.add_argument('--export-reference-frame',type=int,default=-1,
                    help='Export evaluated mesh and exact camera beside video at this frame for renderer comparisons.')
parser.add_argument('--secondary-profile',type=Path,default=DEFAULT_SECONDARY_PROFILE)
parser.add_argument('--legacy-secondary-physics',action='store_true')
parser.add_argument('--quality',choices=['hair','cloth'],default=None)
parser.add_argument('--video-width',type=int,default=900)
parser.add_argument('--video-height',type=int,default=1000)
parser.add_argument('--environment-scale',type=float,default=None)
parser.add_argument('--bangs-shadow-level',type=float,default=None)
parser.add_argument('--cel-brightness-scale',type=float,default=None)
parser.add_argument('--face-shaping',action=argparse.BooleanOptionalAction,default=None)
parser.add_argument('--look',choices=LOOK_CHOICES, default=DEFAULT_LOOK,
                    help='Character recording appearance; default is approved look 08 with original floor.')
args=resolve_appearance_args(parser.parse_args())
if not 0 <= args.environment_scale <= 2 or not 0 <= args.bangs_shadow_level <= 1 or not 0 < args.cel_brightness_scale <= 2:
    parser.error('Invalid appearance scale or shadow level')
args.quality=args.quality or ('hair' if args.legacy_secondary_physics else DEFAULT_SECONDARY_QUALITY)
if args.legacy_secondary_physics:args.secondary_profile=None
W,H=args.video_width,args.video_height
if min(W,H)<2 or W%2 or H%2:parser.error('Video dimensions must be positive even numbers')
for name in ('usd','pmx','audio'):
    value=getattr(args,name)
    if value is not None:
        value=value.resolve()
        if not value.is_file():parser.error(f'{name} not found: {value}')
        setattr(args,name,value)
args.output=args.output.resolve()
args.output.parent.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True,'width':W,'height':H,'renderer':'RaytracedLighting',
                   'anti_aliasing':3,'extra_args':['--/app/asyncRendering=false']})
import numpy as np
from PIL import Image
import omni.usd
import omni.kit.app
from pxr import Gf,Usd,UsdGeom,UsdLux,UsdSkel,UsdShade,Vt
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from physx_secondary import BodyMotion,PhysxSecondary
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb

physics=encoder=log=None
exit_code=1
temporary=args.output.with_name(args.output.stem+'.partial.mp4')
try:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.tensors',True)
    motion=BodyMotion(args.usd,args.pmx,secondary_profile=args.secondary_profile)
    count=motion.end_frame+1
    if args.max_frames:count=min(count,args.max_frames)
    shared_poses=None
    if args.load_poses:
        shared_poses=np.load(args.load_poses,allow_pickle=False)
        if shared_poses['p'].shape!=(count,len(motion.joints),3) or shared_poses['q'].shape!=(count,len(motion.joints),4):
            raise ValueError('Shared pose shape does not match motion and skeleton')
        if str(shared_poses['source_usd'])!=str(args.usd) or str(shared_poses['pmx'])!=str(args.pmx):
            raise ValueError('Shared poses belong to a different character or animation')
    saved_p=[];saved_q=[]
    cache=UsdSkel.Cache();cache.Populate(UsdSkel.Root.Get(motion.source,'/Character'),Usd.PrimDefaultPredicate)
    query=cache.GetSkelQuery(motion.skeleton)
    skin=cache.GetSkinningQuery(motion.source.GetPrimAtPath('/Character/Body'))
    rest=UsdGeom.Mesh.Get(motion.source,'/Character/Body').GetPointsAttr().Get()
    lo=np.full(3,np.inf);hi=-lo.copy()
    for i in range(motion.end_frame+1):
        points=Vt.Vec3fArray(rest)
        if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(i),points,i):raise RuntimeError('Skinning failed')
        xyz=np.asarray(points);lo=np.minimum(lo,xyz.min(0));hi=np.maximum(hi,xyz.max(0))
    center=(lo+hi)/2;half=(hi-lo)/2+np.array([.25,.25,.25])
    distance=max(half[0]/(20.955/100),half[2]/(20.955/100*H/W))+half[1]
    eye=[float(center[0]),float(center[1]-distance),float(center[2])]
    ctx=omni.usd.get_context();ctx.new_stage();stage=ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    root=UsdSkel.Root.Define(stage,'/World/Character');root.GetPrim().GetReferences().AddReference(str(args.usd.resolve()))
    anim=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
    anim.CreateJointsAttr(motion.joints);anim.CreateScalesAttr([Gf.Vec3h(1)]*len(motion.joints))
    def publish(p,q):
        anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
        anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
    p,q,_,_=motion.pose(0);publish(p,q)
    physics=PhysxSecondary(stage,motion,obstacles=False,quality=args.quality)
    UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(800)
    key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25,-25,-35))
    floor=UsdGeom.Mesh.Define(stage,'/World/VisibleFloor')
    floor.CreatePointsAttr([(-12,-12,-.025),(12,-12,-.025),(12,12,-.025),(-12,12,-.025)])
    floor.CreateFaceVertexCountsAttr([4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3]);floor.CreateSubdivisionSchemeAttr('none')
    floor.CreateDisplayColorAttr([(.25,.28,.33)])
    if args.look=='soft-pbr':
        from character_appearance import apply_bright_character_look
        apply_bright_character_look(stage,center)
    bangs_shadow=None
    if args.look==DEFAULT_LOOK:
        from character_appearance import apply_bright_character_look,apply_cel_look
        apply_bright_character_look(stage,center)
        apply_cel_look(stage,brightness_scale=args.cel_brightness_scale,face_shaping=args.face_shaping)
        from bangs_shadow import BangsShadow
        try:
            bangs_shadow=BangsShadow(stage,shadow_level=args.bangs_shadow_level)
        except ValueError as error:
            print(f"[Appearance] Bangs occlusion unavailable: {error}",flush=True)
    environment=UsdLux.DomeLight.Get(stage,'/World/Light')
    environment.GetIntensityAttr().Set(float(environment.GetIntensityAttr().Get())*args.environment_scale)
    cam=UsdGeom.Camera.Define(stage,'/World/Camera');cam.CreateFocalLengthAttr(50)
    cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(20.955*H/W)
    set_camera_view(eye=eye,target=center.tolist(),camera_prim_path='/World/Camera')
    vp=get_active_viewport();vp.camera_path='/World/Camera';vp.set_texture_resolution((W,H))
    def ground_state():
        return {path:{a.GetName():str(a.Get()) for a in stage.GetPrimAtPath(path).GetAttributes()}
                for path in ('/World/PhysxGround','/World/VisibleFloor')}
    ground_before=ground_state()
    for _ in range(20):app.update()
    physics.start(ctx.get_stage_id())
    for _ in range(60):app.update()
    log=args.output.with_suffix('.encoder.log').open('wb')
    command=[args.ffmpeg,'-y','-hide_banner','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','30','-i','pipe:0']
    if args.audio:
        command+=['-i',str(args.audio.resolve()),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k','-af','volume=0.25,apad']
    command+=['-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-t',str(count/30),'-movflags','+faststart',str(temporary)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
    group_counts={name:int(np.sum(motion.system['groups'][physics.dynamic_order]==group)) for name,group in [('hair',14),('skirt',8)]}
    movement={name:0. for name in group_counts}
    trajectory_hash=hashlib.sha256()
    print('PHYSX_RECORD_START',group_counts,'frames',count,flush=True)
    for i in range(count):
        if i:
            substeps=physics.physics_hz//30
            for sub in range(1,substeps+1):p,q=physics.step(i-1+sub/substeps)
        if shared_poses is not None:
            p,q=shared_poses['p'][i],shared_poses['q'][i]
        if args.save_poses:
            saved_p.append(p.copy());saved_q.append(q.copy())
        publish(p,q)
        trajectory_hash.update(np.asarray(p,dtype=np.float32).tobytes())
        trajectory_hash.update(np.asarray(q,dtype=np.float32).tobytes())
        if bangs_shadow is not None:
            bangs_shadow.update(i)
        steps=physics.steps
        for _ in range(3):app.update()
        rgb=None
        for attempt in range(4):
            rgb=_capture_viewport_rgb(pre_render=None)
            if rgb is not None:break
            for _ in range(30):app.update()
        if rgb is None or rgb.shape!=(H,W,3):raise RuntimeError('Invalid capture')
        if args.verify_bangs_shadow and i==60 and bangs_shadow is not None:
            face_shader=UsdShade.Shader(stage.GetPrimAtPath('/World/Character/Looks/Material_00/CelSurface'))
            face_shader.GetInput('receive_bangs').Set(False)
            for _ in range(4):app.update()
            without=_capture_viewport_rgb(pre_render=None)
            face_shader.GetInput('receive_bangs').Set(True)
            for _ in range(4):app.update()
            rgb=_capture_viewport_rgb(pre_render=None)
            change=np.max(np.abs(rgb.astype(float)-without.astype(float)),axis=2)
            changed=int(np.sum(change>5))
            print('BANGS_RENDER_CHECK pixels_changed',changed,flush=True)
            Image.fromarray(without).save(args.output.with_name('bangs_disabled.png'))
            Image.fromarray(rgb).save(args.output.with_name('bangs_enabled.png'))
            if changed<20:raise RuntimeError('Bangs shadow did not visibly affect the rendered image')
        if physics.steps!=steps:raise RuntimeError('Capture advanced physical clock')
        if i==args.export_reference_frame:
            live_cache=UsdSkel.Cache()
            live_cache.Populate(root,Usd.PrimDefaultPredicate)
            live_skel=UsdSkel.Skeleton.Get(stage,'/World/Character/Skeleton')
            live_query=live_cache.GetSkelQuery(live_skel)
            live_mesh=UsdGeom.Mesh.Get(stage,'/World/Character/Body')
            live_skin=live_cache.GetSkinningQuery(live_mesh.GetPrim())
            points=Vt.Vec3fArray(live_mesh.GetPointsAttr().Get())
            transforms=live_query.ComputeSkinningTransforms(Usd.TimeCode.Default())
            if not live_skin.ComputeSkinnedPoints(transforms,points,Usd.TimeCode.Default()):
                raise RuntimeError('Reference mesh skinning failed')
            normals=Vt.Vec3fArray(live_mesh.GetNormalsAttr().Get())
            bind=UsdSkel.BindingAPI(live_mesh)
            joint_ids=np.asarray(bind.GetJointIndicesPrimvar().Get()).reshape(-1,4)
            joint_weights=np.asarray(bind.GetJointWeightsPrimvar().Get()).reshape(-1,4)
            rotations=np.asarray(transforms)[:,:3,:3]
            posed_normals=np.einsum('vi,vkij,vk->vj',np.asarray(normals),rotations[joint_ids],joint_weights)
            posed_normals/=np.maximum(np.linalg.norm(posed_normals,axis=1,keepdims=True),1e-8)
            np.savez(args.output.with_suffix('.reference.npz'),points=np.asarray(points),
                     normals=posed_normals,
                     faces=np.asarray(live_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),
                     uv=np.asarray(UsdGeom.PrimvarsAPI(live_mesh).GetPrimvar('st').Get()))
            args.output.with_suffix('.reference.json').write_text(json.dumps(dict(
                frame=i,eye=eye,target=center.tolist(),width=W,height=H,
                focal_length=50,horizontal_aperture=20.955,vertical_aperture=20.955*H/W,
                pmx=str(args.pmx),source_usd=str(args.usd),look=args.look),indent=2),encoding='utf-8')
            Image.fromarray(rgb).save(args.output.with_suffix('.reference.png'))
        _,_,wp,_=motion.pose(i)
        delta=np.linalg.norm(physics.last_positions-wp[motion.nodes[physics.dynamic_order]],axis=1)
        for name,group in [('hair',14),('skirt',8)]:
            movement[name]=max(movement[name],float(delta[motion.system['groups'][physics.dynamic_order]==group].max()))
        if i in {0,60,150,270,312,360,count-1}:Image.fromarray(rgb).save(args.output.with_name(args.output.stem+f'_{i:04d}.jpg'),quality=95)
        encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if i%60==0:print('PHYSX_RECORD_FRAME',i,count,'steps',physics.steps,'max_motion_m',movement,flush=True)
    encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('Encoder failed')
    encoder=None;log.close()
    if any(v<=0 for v in movement.values()):raise RuntimeError('Missing hair/skirt dynamics')
    os.replace(temporary,args.output)
    if args.save_poses:
        args.save_poses.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_poses,p=np.asarray(saved_p),q=np.asarray(saved_q),
                            source_usd=str(args.usd),pmx=str(args.pmx),dynamic_mask=motion.dynamic)
    assert ground_state()==ground_before,'Original floor changed during recording'
    report=dict(video=str(args.output),source_usd=str(args.usd.resolve()),pmx=str(args.pmx.resolve()),
        backend='native omni.physx CPU, solved live during recording',quality=args.quality,look=args.look,
        environment_scale=args.environment_scale,bangs_shadow_level=args.bangs_shadow_level,
        cel_brightness_scale=args.cel_brightness_scale,face_shaping=args.face_shaping,
        trajectory_sha256=trajectory_hash.hexdigest(),
        rendered_pose_source=str(args.load_poses.resolve()) if args.load_poses else 'live PhysX',
        secondary_profile=str(args.secondary_profile.resolve()) if args.secondary_profile else None,
        frames=count,fps=30,width=W,height=H,duration_seconds=count/30,physics_hz=physics.physics_hz,physics_steps=physics.steps,
        dynamic_groups=group_counts,max_dynamic_displacement_from_animated_pose_m=movement,
        camera=dict(eye=eye,target=center.tolist()),audio=str(args.audio) if args.audio else None,
        ground_unchanged=True,ground_state=ground_before,bangs_shadow_enabled=bangs_shadow is not None,
        hair_cloth_contacts=physics.hair_cloth_contacts,hair_cloth_contact_steps=physics.hair_cloth_contact_steps,
        hair_length_cap_joints=physics.hair_length_cap_count,
        hair_length_projection_links=len(physics.hair_length_links),
        limitations=['Rigid-node hair/skirt proxies; not triangle cloth','Body animation is kinematic',
                      'Hair self-contact enabled; skirt and hair-cloth self-contact disabled for hair quality'])
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PHYSX_RECORD_PASS',args.output,flush=True)
    exit_code=0
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
finally:
    # All frame writes, encoder finalization and reports are complete before
    # successful shutdown. This batch entry point has no Replicator writer.
    # Avoid the normal profiler/framework teardown, which can hang after Kit
    # prints "Simulation App Shutting Down" on this Windows renderer path.
    for resource,cleanup in (
        (encoder,lambda: (encoder.kill(),encoder.wait())),
        (physics,lambda: physics.close()),
        (log,lambda: log.close()),
    ):
        if resource is not None:
            try:
                cleanup()
            except BaseException:
                import traceback
                traceback.print_exc()
                exit_code=1
    print('PHYSX_RECORD_SHUTDOWN exit_code='+str(exit_code),flush=True)
    sys.stdout.flush();sys.stderr.flush()
    app.close(skip_cleanup=True,exit_code=exit_code)
sys.exit(exit_code)
