"""Shared two-to-five-column comparison renderer. Kit starts only in main()."""
from pathlib import Path
import time
import json
from source.pmx_prototype.runtime.comparison_options import parse_options, actor_layout, frame_range


def main(argv=None, *, preset='standard'):
    a=parse_options(argv,default_preset=preset)
    if a.dry_run:
        print(json.dumps(dict(preset=a.preset,start_seconds=a.start_seconds,end_seconds=a.end_seconds,
                              layout=actor_layout(a)),ensure_ascii=False,indent=2))
        return 0
    args=a
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':a.headless,'width':a.width,'height':a.height,'renderer':'RaytracedLighting','anti_aliasing':3,'limit_cpu_threads':16})
    import numpy as np,h5py,torch
    import omni.usd,omni.ui as ui
    from pxr import Usd,UsdGeom,UsdSkel,UsdLux,UsdPhysics,PhysxSchema,Gf,Vt
    from scipy.spatial.transform import Rotation
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    from source.train_workflow.utils.motion.self_collision import G1CollisionModel,Joint,_quat_matrix
    from source.pmx_prototype.runtime.character_appearance import DefaultCharacterLook
    from source.pmx_prototype.runtime.live_music import LiveMusic
    music=None
    try:
        with h5py.File(args.g1_h5, "r") as stream:
            names = list(stream["joint_names"].asstr()[:])
            joint_positions = (stream["joint_pos_delta"][:] +
                               stream.attrs["record_baseline_joint_pos"])
            root_positions = (stream["root_pos_delta"][:] +
                              stream.attrs["record_root_anchor_pos"])
            root_delta = stream["root_quat_delta_wxyz"][:]
            root_anchor = stream.attrs["record_root_anchor_quat_wxyz"]
            fps = float(stream.attrs["fps"])
            retarget_mode = str(stream.attrs.get("retarget_mode", "unknown"))
            upper_method = str(stream.attrs.get("upper_body_retarget_method", ""))
            lower_method = str(stream.attrs.get("lower_body_retarget_method", ""))
            policy_checkpoint = str(stream.attrs.get("policy_checkpoint", ""))
            leg_mode = str(stream.attrs.get('source_leg_mode', 'unknown'))
        count = len(joint_positions)
        start_frame,end_frame=frame_range(a,fps,count)
        if count < 2 or abs(fps - 30.0) > 1e-6:
            raise ValueError(f"Expected nonempty 30 FPS G1 H5, got {count} frames at {fps} FPS")
        if retarget_mode not in {"new", "policy"}:
            raise ValueError(f"G1 H5 must be from the updated retarget or trained-policy chain: {retarget_mode}")

        # Build full robot FK from the articulated USD joint tree. The visual robot
        # is made kinematic and has its joint prims removed in this playback stage;
        # its exact H5 poses then share the PMX stage without fighting PhysX.
        g1_model = G1CollisionModel(args.g1_usd, names, "cpu")
        robot_source = Usd.Stage.Open(str(args.g1_usd))
        pending = []
        for prim in robot_source.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parent_paths = joint.GetBody0Rel().GetTargets()
            child_paths = joint.GetBody1Rel().GetTargets()
            if not parent_paths or not child_paths:
                continue
            pending.append(Joint(
                prim.GetName(), parent_paths[0].name, child_paths[0].name,
                np.array(joint.GetLocalPos0Attr().Get()),
                _quat_matrix(joint.GetLocalRot0Attr().Get()),
                np.array(joint.GetLocalPos1Attr().Get()),
                _quat_matrix(joint.GetLocalRot1Attr().Get()),
                prim.GetAttribute("physics:axis").Get(), -np.inf, np.inf,
            ))
        g1_model.joints = []
        seen = {"pelvis"}
        while pending:
            ready = [joint for joint in pending if joint.parent in seen]
            if not ready:
                raise RuntimeError("Disconnected G1 USD joint tree")
            for joint in ready:
                g1_model.joints.append(joint)
                seen.add(joint.child)
                pending.remove(joint)
        g1_model.constants = {
            joint.name: tuple(torch.tensor(value, dtype=torch.float32)
                              for value in (joint.p0, joint.r0, joint.p1, joint.r1))
            for joint in g1_model.joints
        }
        rotations, positions = g1_model.fk(torch.tensor(joint_positions, dtype=torch.float32))
        rotations = {name: value.numpy() for name, value in rotations.items()}
        positions = {name: value.numpy() for name, value in positions.items()}
        root_rotation = (
            Rotation.from_quat(root_delta[:, [1, 2, 3, 0]]) *
            Rotation.from_quat(root_anchor[[1, 2, 3, 0]])
        ).as_matrix()
        align = Rotation.from_euler("z", 180, degrees=True).as_matrix()
        layout=actor_layout(a)
        g1_offset=np.array([layout[0]['offset_x'],0.,a.actor_z_offset])
        anon_offset=np.array([layout[-1]['offset_x'],0.,a.actor_z_offset])

        ctx=omni.usd.get_context();ctx.new_stage();stage=ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        cache_stage=Usd.Stage.Open(str(a.pmx_cache))
        source_anim=UsdSkel.Animation.Get(cache_stage,'/Character/Animation')
        samples=source_anim.GetRotationsAttr().GetTimeSamples()
        cache_fps=cache_stage.GetTimeCodesPerSecond()
        if not samples or samples[0]>0 or samples[-1]/cache_fps<(count-1)/fps-1e-6:
            raise ValueError('Anon cache does not cover the complete G1 timeline')
        character=UsdSkel.Root.Define(stage,'/World/Character')
        character.GetPrim().GetReferences().AddReference(str(a.pmx_cache))
        UsdGeom.Xformable(character).AddTranslateOp().Set(Gf.Vec3d(*anon_offset))
        animation=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
        def publish_anon(frame):
            t=Usd.TimeCode(float(frame)*cache_fps/fps)
            animation.GetTranslationsAttr().Set(source_anim.GetTranslationsAttr().Get(t))
            animation.GetRotationsAttr().Set(source_anim.GetRotationsAttr().Get(t))
        def build_robot(root_path, motion_rotations, motion_positions, root_rotations, root_positions_world, offset):
            g1=stage.DefinePrim(root_path,'Xform');g1.GetReferences().AddReference(str(args.g1_usd))
            joint_paths=[]
            for prim in stage.Traverse():
                if not prim.GetPath().HasPrefix(root_path):continue
                if prim.IsA(UsdPhysics.Joint):joint_paths.append(prim.GetPath());continue
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
                if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            for path in joint_paths:stage.GetPrimAtPath(path).SetActive(False)
            operations={}
            for prim in stage.Traverse():
                if prim.GetPath().HasPrefix(root_path) and prim.GetName() in motion_rotations:
                    xform=UsdGeom.Xformable(prim);xform.ClearXformOpOrder();operations[prim.GetName()]=xform.AddTransformOp()
            if set(operations)!=set(motion_rotations):raise RuntimeError('G1 visual USD is missing FK links')
            def publish(frame):
                for name,op in operations.items():
                    matrix=np.eye(4)
                    matrix[:3,:3]=align@root_rotations[frame]@motion_rotations[name][frame]
                    matrix[:3,3]=align@(root_rotations[frame]@motion_positions[name][frame]+root_positions_world[frame])+offset
                    op.Set(Gf.Matrix4d(matrix.T.tolist()))
            return publish
        publishers=[build_robot('/World/G1',rotations,positions,root_rotation,root_positions,g1_offset)]
        for actor in layout[1:-1]:
            with h5py.File(getattr(a,actor['option'])) as motion:
                if list(motion['joint_names'].asstr()[:])!=names or len(motion['joint_pos_delta'])!=count or float(motion.attrs['fps'])!=fps:
                    raise ValueError('Comparison H5 timeline/joints differ')
                if not motion.attrs.get('without_collision_validated',False):
                    raise ValueError('Comparison H5 collision validation missing')
                q=motion['joint_pos_delta'][:]+motion.attrs['record_baseline_joint_pos']
                rp=motion['root_pos_delta'][:]+motion.attrs['record_root_anchor_pos']
                rw=(Rotation.from_quat(motion['root_quat_delta_wxyz'][:][:,[1,2,3,0]])*
                    Rotation.from_quat(motion.attrs['record_root_anchor_quat_wxyz'][[1,2,3,0]])).as_matrix()
            r,p1=g1_model.fk(torch.tensor(q,dtype=torch.float32))
            publishers.append(build_robot(actor['root'],{k:v.numpy() for k,v in r.items()},
                {k:v.numpy() for k,v in p1.items()},rw,rp,np.array([actor['offset_x'],0.,a.actor_z_offset])))
        def publish_g1(frame):
            for publish in publishers:publish(frame)

        # Match the PMX preview's original ground geometry and material.
        floor=UsdGeom.Cube.Define(stage,'/World/PhysxGround');floor.CreateSizeAttr(1)
        xf=UsdGeom.Xformable(floor)
        xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-.04));xf.AddScaleOp().Set(Gf.Vec3f(10,10,.05))
        floor.CreateDisplayColorAttr([(.18,.21,.26)])
        dome=UsdLux.DomeLight.Define(stage,'/World/Light');dome.CreateIntensityAttr(800)
        key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(1800)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25,-25,-35))
        appearance=DefaultCharacterLook(stage,center=tuple(anon_offset))
        camera=UsdGeom.Camera.Define(stage,'/World/CompareCamera')
        camera.CreateFocalLengthAttr(50);camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955*a.height/a.width)
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(0,-(27.5 if a.smooth_h5 else 25 if a.fidelity_h5 else 15.5 if a.corrected_h5 else 12),2.05),Gf.Vec3d(0,0,.85),Gf.Vec3d(0,0,1)).GetInverse())
        viewport=get_active_viewport();viewport.camera_path='/World/CompareCamera'
        viewport.set_texture_resolution((a.width,a.height))
        publish_g1(start_frame);publish_anon(start_frame)
        for _ in range(60):app.update()
        if a.record_video:
            import subprocess,hashlib
            from PIL import Image
            from source.train_workflow.g1_vmd_pipeline import find_ffmpeg
            from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
            output=a.record_video;output.parent.mkdir(parents=True,exist_ok=True)
            partial=output.with_suffix('.partial.mp4')
            ffmpeg=find_ffmpeg()
            labels="drawbox=x=0:y=0:w=iw:h=48:color=black@0.55:t=fill"
            for i,actor in enumerate(layout):
                title=actor['label']
                if len(layout)==2 and i==0 and not a.labels:title=f'G1 - {leg_mode.upper()} retarget'
                x=(i*a.width//len(layout)+28) if len(layout) in (2,3) and i!=1 else i*a.width//len(layout)+20
                if i==0 and len(layout) in (2,3):x=28
                labels+=",drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"+f"text='{title}':fontcolor=white:fontsize=25:x={x}:y=10"
            if a.preset=='girls-segment':
                labels+=",drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"+f"text='Dance {a.start_seconds:g}-{a.end_seconds:g}s - normal speed':fontcolor=white:fontsize=25:x=20:y=h-40:box=1:boxcolor=black@0.55"
            command=[ffmpeg,'-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}',
                     '-r',str(fps),'-i','pipe:0']
            if a.audio:
                command+=['-ss',str(start_frame/fps),'-i',str(a.audio),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','192k','-af','volume=0.25,apad']
            command+=['-vf',labels,'-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                      '-t',str((end_frame-start_frame)/fps),'-movflags','+faststart',str(partial)]
            with output.with_suffix('.encoder.log').open('xb') as log:
                encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
                try:
                    for frame in range(start_frame,end_frame):
                        publish_g1(frame);publish_anon(frame);appearance.update(frame)
                        for _ in range(3):app.update()
                        rgb=_capture_viewport_rgb(pre_render=None)
                        if rgb is None or rgb.shape!=(a.height,a.width,3):raise RuntimeError(f'Invalid video frame {frame}')
                        if frame in (start_frame,300,900,end_frame-1):Image.fromarray(rgb).save(output.with_name(output.stem+f'_{frame:04d}.jpg'))
                        encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
                        if frame%300==0:print(f'SHARED_RECORD_FRAME {frame}/{count}',flush=True)
                    encoder.stdin.close()
                    if encoder.wait()!=0:raise RuntimeError('FFmpeg encoding failed')
                finally:
                    if encoder.poll() is None:
                        encoder.stdin.close();encoder.wait()
            partial.rename(output)
            report={'video':str(output),'frames':end_frame-start_frame,'source_start_frame':start_frame,'source_end_frame_exclusive':end_frame,'fps':fps,'duration_seconds':(end_frame-start_frame)/fps,
                    'width':a.width,'height':a.height,'same_stage':True,'source_leg_mode':leg_mode,
                    'fidelity_h5':str(a.fidelity_h5) if a.fidelity_h5 else None,'corrected_h5':str(a.corrected_h5) if a.corrected_h5 else None,'g1_h5':str(a.g1_h5),'pmx_cache':str(a.pmx_cache),'audio':str(a.audio) if a.audio else None,
                    'g1_sha256':hashlib.sha256(a.g1_h5.read_bytes()).hexdigest(),
                    'pmx_sha256':hashlib.sha256(a.pmx_cache.read_bytes()).hexdigest(),
                    'appearance':'shared approved DefaultCharacterLook','anon_secondary_motion':'native PhysX cache'}
            report['comparison_order']=[item['label'] for item in layout]
            if a.preset=='girls-full':
                report['comparison_order']=['Original preserved G1','Full-smoothed G1','Anon PMX']
            report['smooth_h5']=str(a.smooth_h5) if a.smooth_h5 else None
            for option in ('corrected_h5','fidelity_h5','smooth_h5'):
                path=getattr(a,option)
                if path is not None:
                    report[option+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('SHARED_RECORD_COMPLETE '+str(output),flush=True)
            raise SystemExit(0)
        if a.audio:music=LiveMusic(a.audio)
        state={'playing':False,'frame':start_frame,'origin':time.perf_counter(),'origin_frame':start_frame,'updating':False}
        def set_playing(value):
            state['playing']=bool(value);state['origin']=time.perf_counter();state['origin_frame']=state['frame']
            if music:music.set_enabled(bool(value),state['frame']/fps)
            print(f'SHARED_LIVE_PLAY_STATE playing={state["playing"]} frame={state["frame"]}',flush=True)
        def seek(frame):
            state['frame']=int(np.clip(frame,start_frame,end_frame-1));state['origin_frame']=state['frame'];state['origin']=time.perf_counter()
            publish_g1(state['frame']);publish_anon(state['frame'])
            if music and state['playing']:music.update(state['frame']/fps,restart=True)
        def slider_changed(model):
            if not state['updating']:seek(model.get_value_as_int())
        window=ui.Window('Girls - G1 IK + Anon',width=460,height=175)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label(' | '.join(item['label'] for item in layout))
                with ui.HStack():
                    ui.Button('Play',clicked_fn=lambda:set_playing(True))
                    ui.Button('Pause',clicked_fn=lambda:set_playing(False))
                    ui.Button('Restart',clicked_fn=lambda:seek(start_frame))
                    ui.Label('Loop',width=35);loop=ui.CheckBox();loop.model.set_value(True)
                slider=ui.IntSlider(min=start_frame,max=end_frame-1);slider.model.add_value_changed_fn(slider_changed)
                label=ui.Label('Ready')
                ui.Label('Space: Play/Pause | R: Restart | Drag slider to seek')
        window.deferred_dock_in('Content',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        import carb.input,omni.appwindow
        input_interface=carb.input.acquire_input_interface()
        keyboard=omni.appwindow.get_default_app_window().get_keyboard()
        def on_key(event):
            if event.type==carb.input.KeyboardEventType.KEY_PRESS:
                if event.input==carb.input.KeyboardInput.SPACE:set_playing(not state['playing'])
                elif event.input==carb.input.KeyboardInput.R:seek(start_frame)
            return True
        keyboard_subscription=input_interface.subscribe_to_keyboard_events(keyboard,on_key)
        if a.snapshot:
            a.snapshot.parent.mkdir(parents=True,exist_ok=True)
            capture_viewport_to_file(viewport,str(a.snapshot.resolve()))
        set_playing(a.auto_play)
        print(f'SHARED_LIVE_READY frames={count} fps={fps} g1={a.g1_h5} anon={a.pmx_cache}',flush=True)
        last_log=time.perf_counter()
        preview_started=last_log
        while app.is_running():
            tick=time.perf_counter()
            if a.preview_seconds>0 and tick-preview_started>=a.preview_seconds:break
            if state['playing']:
                frame=state['origin_frame']+int((tick-state['origin'])*fps)
                if frame>=end_frame:
                    if loop.model.get_value_as_bool():seek(start_frame);frame=start_frame
                    else:frame=end_frame-1;set_playing(False)
                state['frame']=frame
            frame=state['frame']
            publish_g1(frame);publish_anon(frame);appearance.update(frame)
            if music:music.update(frame/fps)
            state['updating']=True;slider.model.set_value(frame);state['updating']=False
            label.text=f'{frame/fps:.2f} / {count/fps:.2f} s  |  frame {frame}  |  '+('Playing' if state['playing'] else 'Paused')
            app.update()
            if tick-last_log>10:
                print(f'SHARED_LIVE_PROGRESS frame={frame} playing={state["playing"]}',flush=True);last_log=tick
            time.sleep(max(0,1/60-(time.perf_counter()-tick)))
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if music:music.close()
        app.close()
