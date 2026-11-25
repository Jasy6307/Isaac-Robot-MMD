"""Re-render an immutable UsdSkel physics cache with shared character appearance."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse,json,sys,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--physics-report',type=Path,required=True)
p.add_argument('--video',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--audio',type=Path)
p.add_argument('--ffmpeg',required=True)
p.add_argument('--look',choices=LOOK_CHOICES,default=DEFAULT_LOOK)
p.add_argument('--dlss-quality', action='store_true', help='Fix DLSS to Quality for matched captures')
p.add_argument('--frames',type=int,default=0)
p.add_argument('--capture-frames',type=int,nargs='*',default=[])
p.add_argument('--width',type=int,default=720)
p.add_argument('--height',type=int,default=900)
a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);a.video.parent.mkdir(exist_ok=True,parents=True)
original_report=json.loads(a.physics_report.read_text(encoding='utf-8'))
assert original_report['status']=='complete' and original_report['physics_hz']==120
cache_sha=hashlib.sha256(a.cache.read_bytes()).hexdigest()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':a.width,'height':a.height,'renderer':'RaytracedLighting','anti_aliasing':3,'extra_args':['--/app/asyncRendering=false']})
import numpy as np
import carb.settings
import omni.usd,omni.timeline
from pxr import Usd,UsdGeom,UsdSkel
from PIL import Image
from omni.kit.viewport.utility import get_active_viewport
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
from source.pmx_prototype.character_appearance import DefaultCharacterLook,apply_bright_character_look
encoder=None;started=time.monotonic()
try:
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    stage.GetRootLayer().subLayerPaths=[str(a.scene.resolve()).replace('\\','/')]
    omni.timeline.get_timeline_interface().stop()
    cache=Usd.Stage.Open(str(a.cache.resolve()));source=UsdSkel.Animation.Get(cache,'/Character/Animation')
    live=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
    assert list(source.GetJointsAttr().Get())==list(live.GetJointsAttr().Get())
    assert cache.GetTimeCodesPerSecond()==30
    count=min(a.frames or int(cache.GetEndTimeCode())+1,int(cache.GetEndTimeCode())+1)
    def fingerprint(prim):return {x.GetName():str(x.Get()) for x in prim.GetAttributes()}
    def camera_framing(prim):return {name:str(prim.GetAttribute(name).Get()) for name in ('xformOp:transform','xformOpOrder','projection','focalLength','horizontalAperture','verticalAperture','horizontalApertureOffset','verticalApertureOffset','clippingRange')}
    ground=stage.GetPrimAtPath('/World/PhysxGround');ground_before=fingerprint(ground)
    camera=stage.GetPrimAtPath('/World/Camera');camera_before=camera_framing(camera)
    softbox=stage.GetPrimAtPath('/World/FaceSoftbox')
    pos=softbox.GetAttribute('xformOp:translate').Get() if softbox else None
    center=(float(pos[0]),float(pos[1])+2.5,0.) if pos is not None else (0.,-.25,.95)
    def publish(frame):
        live.GetTranslationsAttr().Set(source.GetTranslationsAttr().Get(frame))
        live.GetRotationsAttr().Set(source.GetRotationsAttr().Get(frame))
    publish(0)
    appearance=DefaultCharacterLook(stage,center) if a.look==DEFAULT_LOOK else None
    if a.look=='soft-pbr':apply_bright_character_look(stage,center)
    if appearance:appearance.update(0)
    assert fingerprint(ground)==ground_before and camera_framing(camera)==camera_before
    viewport=get_active_viewport();viewport.camera_path='/World/Camera';viewport.set_texture_resolution((a.width,a.height))
    for _ in range(75):app.update()
    settings = carb.settings.get_settings()
    if a.dlss_quality:
        settings.set_int('/rtx/post/dlss/execMode', 2)
        for _ in range(10):app.update()
        if settings.get('/rtx/post/dlss/execMode') != 2:
            raise RuntimeError('Could not activate DLSS Quality')
    stage.GetRootLayer().Export(str(a.output/'scene.usda'))
    command=[a.ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}','-r','30','-i','pipe:0']
    if a.audio:command+=['-i',str(a.audio.resolve()),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k','-af',f'volume=.25,apad=whole_dur={count/30}']
    command+=['-c:v','libx264','-crf','18','-preset','fast','-threads','4','-pix_fmt','yuv420p','-t',str(count/30),'-movflags','+faststart',str(a.video)]
    log=(a.output/'encoder.log').open('wb');encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
    costs=[];verified=0
    for frame in range(count):
        begin=time.monotonic();publish(frame)
        if appearance:appearance.update(frame)
        for _ in range(3):app.update()
        rgb=None
        for attempt in range(4):
            rgb=_capture_viewport_rgb(pre_render=None)
            if rgb is not None:break
            print('CACHED_CAPTURE_RETRY',frame,attempt+1,flush=True)
            for _ in range(30):app.update()
        if rgb is None:raise RuntimeError('Cache capture failed')
        assert rgb.shape[:2]==(a.height,a.width),rgb.shape
        encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if frame%30==0 or frame==count-1:
            assert live.GetTranslationsAttr().Get()==source.GetTranslationsAttr().Get(frame)
            assert live.GetRotationsAttr().Get()==source.GetRotationsAttr().Get(frame)
            verified+=1
        if frame in a.capture_frames:Image.fromarray(rgb).save(a.output/f'frame_{frame:05d}.png')
        costs.append(time.monotonic()-begin)
        if (frame+1)%300==0:print('CACHED_RENDER_PROGRESS',frame+1,count,'mean_ms',round(1000*np.mean(costs),2),flush=True)
    encoder.stdin.close();code=encoder.wait();encoder=None;log.close()
    if code:raise RuntimeError(f'Encoder failed: {code}')
    state_changes={label:{k:[before.get(k),after.get(k)] for k in set(before)|set(after) if before.get(k)!=after.get(k)} for label,before,after in [('ground',ground_before,fingerprint(ground)),('camera_framing',camera_before,camera_framing(camera))]}
    (a.output/'state_changes.json').write_text(json.dumps(state_changes,indent=2),encoding='utf-8')
    assert fingerprint(ground)==ground_before and camera_framing(camera)==camera_before,state_changes
    assert hashlib.sha256(a.cache.read_bytes()).hexdigest()==cache_sha
    shadow=appearance.bangs if appearance else None
    from source.pmx_prototype import appearance_defaults as defaults
    report=dict(status='complete',frames=count,fps=30,source_first_frame=0,physics_hz=original_report['physics_hz'],physics_steps_recomputed=0,cache_reused=True,cache_sha256=cache_sha,look=a.look,appearance_parameters=dict(environment_scale=defaults.ENVIRONMENT_SCALE,bangs_shadow_level=defaults.BANGS_SHADOW_LEVEL,cel_brightness_scale=defaults.CEL_BRIGHTNESS_SCALE,face_shaping=defaults.FACE_SHAPING),ground_unchanged=True,camera_unchanged=True,pose_samples_exact=verified,semantic_mapping=appearance.semantic_mapping if appearance else None,bangs_shadow_enabled=shadow is not None,bangs_shadow_updates=shadow.updates if shadow else 0,mean_render_ms=1000*float(np.mean(costs)),wall_seconds=time.monotonic()-started)
    report['resolution']=[a.width,a.height]
    if appearance is not None:
        report['toon_style'] = defaults.TOON_STYLE
        report['colored_shadows'] = True
        report['fabric_shadow_darken'] = defaults.FABRIC_SHADOW_DARKEN
    report['renderer'] = settings.get('/rtx/rendermode')
    report['dlss_exec_mode'] = settings.get('/rtx/post/dlss/execMode')
    (a.output/'render_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CACHED_RENDER_PASS',json.dumps(report,ensure_ascii=False),flush=True)
except Exception:
    import traceback
    traceback.print_exc();raise
finally:
    if encoder:
        encoder.stdin.close();encoder.wait()
    app.close()
