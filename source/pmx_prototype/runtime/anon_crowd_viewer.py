"""Internal multi-character viewer used by inspect_anon_policy.py."""
from source_layout import PROJECT_ROOT,resolve_source
import argparse,asyncio,hashlib,json,math,os,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


def run(argv):
    from anon_policy_catalog import DEFAULT_INSPECTOR_CHECKPOINT
    from secondary_profiles import DEFAULT_SECONDARY_PROFILE,SKIRT_EDGE_STRIDES,SECONDARY_MODES,crowd_secondary_modes
    p=argparse.ArgumentParser(prog='inspect_anon_policy.py',description='Multi-character policy inspector')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_INSPECTOR_CHECKPOINT)
    p.add_argument('--characters',type=int,required=True,help='Characters in the shared scene (provided by the unified inspector)')
    p.add_argument('--controlled-character',type=int,default=1,help='Initially controlled character, numbered from 1')
    p.add_argument('--random-only',action='store_true',help='Previous autonomous mode: all characters receive random commands')
    p.add_argument('--mixed-secondary',action='store_true',help='Separate controlled/background budgets (default: light/light3); overrides --secondary-mode')
    p.add_argument('--controlled-secondary-mode',choices=SECONDARY_MODES,help='Controlled actor budget with --mixed-secondary (default: light)')
    p.add_argument('--inter-character-collisions',action='store_true',help='Enable body collisions between characters (default: off)')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-speed',type=float,default=.55,help='Random planar speed limit, m/s')
    p.add_argument('--max-yaw',type=float,default=.4,help='Random turn limit, rad/s')
    p.add_argument('--arena-radius',type=float,default=7.,help='Soft command-return radius, m')
    p.add_argument('--body-device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--physics-threads',type=int,default=4,help='Threads per secondary worker')
    p.add_argument('--physics-detail',choices=SKIRT_EDGE_STRIDES,default='sparse',help='Skirt-edge proxy density (default: sparse)')
    p.add_argument('--frontend-task-threads',type=int,default=4,help='Kit task threads for the viewer (default: 4)')
    p.add_argument('--pmx',type=Path,default=PROJECT_ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
    p.add_argument('--steps',type=int,default=0,help='Finite 30 Hz control steps; 0 keeps the viewer open')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--real-time',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--render-fps',type=int,choices=[30,60],help='Live display rate (default: 60; recording/unpaced diagnostics: 30)')
    p.add_argument('--interpolate-60',action='store_true',help='Alias for --render-fps 60')
    p.add_argument('--no-secondary-physics',action='store_true',help='Explicit body-only diagnostic')
    p.add_argument('--serial-playback',action='store_true',help='Diagnostic: disable one-step worker prefetch')
    p.add_argument('--secondary-mode',choices=SECONDARY_MODES,help='Hair/skirt solve budget (default: light)')
    p.add_argument('--save-preview',action='store_true')
    p.add_argument('--record-video',type=Path,help='MP4 at 30 FPS in simulation time, independent of wall-clock speed')
    p.add_argument('--video-width',type=int,default=1600)
    p.add_argument('--video-height',type=int,default=1000)
    p.add_argument('--dlss-mode',choices=['performance','balanced','quality','auto'],default='quality',help='DLSS resolution preset (default: quality)')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    a.secondary_mode=a.secondary_mode or ('full' if a.no_secondary_physics else 'light')
    if a.controlled_secondary_mode is not None and not a.mixed_secondary:
        p.error('--controlled-secondary-mode requires --mixed-secondary')
    if a.mixed_secondary and (a.no_secondary_physics or a.random_only or a.characters<2):
        p.error('--mixed-secondary requires 2-8 actors, secondary physics and a manually controlled actor')
    if a.interpolate_60 and a.render_fps==30:p.error('--interpolate-60 requires a 60 FPS display')
    if a.interpolate_60:a.render_fps=60
    a.render_fps=a.render_fps or (60 if a.real_time and not a.record_video and not a.no_secondary_physics else 30)
    if a.render_fps==60 and (a.record_video or not a.real_time):
        p.error('60 FPS interpolation requires real-time preview; recording/unpaced diagnostics use --render-fps 30')
    if a.no_secondary_physics and a.secondary_mode!='full':p.error('Light mode requires secondary physics')
    if not 1<=a.characters<=8 or a.steps<0 or a.seed<0 or a.physics_threads<0 or a.frontend_task_threads<0:
        p.error('Use characters 1-8 and nonnegative steps, seed and thread counts')
    if not 0<a.max_speed<=1.2 or not 0<=a.max_yaw<=.6 or not math.isfinite(a.arena_radius) or a.arena_radius<=2:
        p.error('Use speed (0,1.2], yaw [0,0.6] and arena radius >2')
    if a.headless and not a.steps:p.error('--headless requires finite --steps')
    if not 1<=a.controlled_character<=a.characters:p.error('Controlled character must be between 1 and --characters')
    controlled_mode=a.controlled_secondary_mode or 'light'
    initial_modes=crowd_secondary_modes(a.characters,a.controlled_character-1,a.secondary_mode,a.mixed_secondary,controlled_mode)
    last_secondary_modes=initial_modes.copy()
    if min(a.video_width,a.video_height)<2 or a.video_width%2 or a.video_height%2:
        p.error('Video dimensions must be positive even integers')
    if a.record_video:
        from anon_crowd_recording import find_ffmpeg
        find_ffmpeg()
        if a.record_video.exists():p.error('Choose a new video path; existing videos will not be overwritten')
    a.checkpoint=a.checkpoint.resolve()
    if not a.checkpoint.is_file() or not a.pmx.is_file():p.error('Checkpoint or PMX file is missing')
    for name in ('env_config.json','agent_config.json'):
        if not (a.checkpoint.parent/name).is_file():p.error(f'Missing {name} beside checkpoint')
    a.output=(a.output or PROJECT_ROOT/'output/anon_locomotion/crowd'/f'{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}').resolve()
    if a.output.exists() and any(a.output.iterdir()):p.error('Choose a new empty output directory')
    a.output.mkdir(parents=True,exist_ok=True)
    from anon_crowd_commands import RandomCrowdCommands,BodyPoseMapper
    from anon_policy_transport import PolicyServer
    from anon_secondary_stream import AnonSecondaryStream
    from anon_startup_progress import StartupJournal
    import numpy as np
    started=time.perf_counter();journal=StartupJournal(a.output,'viewer',started=started)
    journal.mark('engine',f'Starting {a.characters} actors')
    channel=PolicyServer(a.output/'policy_ipc')
    worker=app=None;streams=[];body_log=None;trace=None;dock_task=None;recorder=None;video_frames=0;inspector=None;panel=None
    status=None;exit_code=0;playback_started=False;display=None;render_times=[];render_sources=[];solved_times=[];pipeline=None
    try:
        body_log=(a.output/'policy_worker.log').open('w',encoding='utf-8')
        worker_options=['--controlled-actor',str(a.controlled_character-1)]
        if not a.random_only:worker_options.append('--manual-control')
        if a.inter_character_collisions:worker_options.append('--inter-character-collisions')
        worker=subprocess.Popen([sys.executable,'-u',str(resolve_source('anon_crowd_worker.py')),
            '--checkpoint',str(a.checkpoint),'--characters',str(a.characters),'--seed',str(a.seed),
            '--body-device',a.body_device,'--output',str(a.output),*worker_options,*channel.worker_args],
            cwd=PROJECT_ROOT,stdout=body_log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        os.environ.setdefault('ANON_SECONDARY_TASK_THREADS','2')
        if not a.no_secondary_physics:
            for i in range(a.characters):
                config=SimpleNamespace(output=a.output/f'actor_{i:02d}',pmx=a.pmx,quality='cloth',
                    physics_threads=a.physics_threads,skirt_edge_stride=SKIRT_EDGE_STRIDES[a.physics_detail],secondary_profile=DEFAULT_SECONDARY_PROFILE,
                    legacy_secondary_physics=False,reference_hair_compute=False,legacy_compute=False,secondary_mode=initial_modes[i])
                streams.append(AnonSecondaryStream(config))
        from isaacsim import SimulationApp
        width,height=a.video_width,a.video_height
        app=SimulationApp(dict(headless=a.headless,hide_ui=False,width=width,height=height,
            window_width=1600,window_height=1000,renderer='RaytracedLighting',anti_aliasing=3,
            limit_cpu_threads=a.frontend_task_threads,fast_shutdown=True,extra_args=['--/app/asyncRendering=false']))
        journal.mark('imports')
        import omni.usd,omni.ui as ui
        from pxr import Gf,Sdf,UsdGeom,UsdSkel,UsdLux,Vt
        from omni.kit.viewport.utility import get_active_viewport
        flags=dict(paused=False,reset=False)
        panel=ui.Window('Crowd policy',width=1000,height=230)
        with panel.frame:
            with ui.VStack(spacing=5):
                status=ui.Label('Loading shared body physics...',height=34,word_wrap=True,style={'font_size':16})
                with ui.HStack(height=34,spacing=8):
                    def toggle():
                        flags['paused']=not flags['paused']
                        pause.text='Resume random commands' if flags['paused'] else 'Stop random commands'
                    pause=ui.Button('Stop random commands',width=250,clicked_fn=toggle)
                    ui.Button('Reset all',width=140,clicked_fn=lambda:flags.update(reset=True))
                    ui.Label('Shared checkpoint | Independent commands | Auto reset on fall')
                actor_labels=[ui.Label(f'Anon {i+1:02d}',height=25,style={'font_size':15}) for i in range(a.characters)]
        async def dock():
            import omni.kit.app
            for _ in range(120):
                await omni.kit.app.get_app().next_update_async()
                content=ui.Workspace.get_window('Content');viewport_window=ui.Workspace.get_window('Viewport')
                if content is None or viewport_window is None:continue
                content.visible=True
                if not content.docked:
                    content.dock_in(viewport_window,ui.DockPosition.BOTTOM,.25)
                    await omni.kit.app.get_app().next_update_async()
                panel.deferred_dock_in('Content',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
                return
        dock_task=asyncio.ensure_future(dock())
        next_ui=0.
        def note_render(frame):
            render_times.append(time.perf_counter());render_sources.append(frame)
            if inspector is not None:inspector.record_render()
        def pump(message=None):
            nonlocal next_ui
            if not app.is_running():
                if not playback_started:raise KeyboardInterrupt
                return  # Finish in-flight worker results and write the session report.
            if message is not None:status.text=message
            if playback_started and display is not None:
                display.present_if_due();return
            if time.perf_counter()>=next_ui:
                app.update();next_ui=time.perf_counter()+.02
                if playback_started:note_render(completed-1)
        def receive(seq,*,update_ui=True):
            deadline=time.perf_counter()+180
            while time.perf_counter()<deadline:
                value=channel.receive(seq)
                if value is not None:return value
                if worker.poll() is not None:raise RuntimeError('Body worker exited; see policy_worker.log')
                if update_ui:pump()
                time.sleep(.001)
            raise TimeoutError('Shared body policy timed out')
        journal.mark('body');state=receive(-1)
        config=state['config'];links=state['links'];pelvis=links.index('pelvis')
        env_groups=state['collision_groups'];actor_paths=state['actor_paths'];body_stage_id=state['physics_stage_id']
        if not a.random_only:
            from anon_crowd_inspector import CrowdInspector
            from anon_policy_catalog import label_for
            catalog={'crowd':dict(label=label_for('custom',config,a.checkpoint),checkpoint=str(a.checkpoint),available=True)}
            inspector=CrowdInspector(config,a.output,catalog,'crowd',a.render_fps,count=a.characters,
                selected=a.controlled_character-1,actor_collisions=a.inter_character_collisions)
            if dock_task is not None and not dock_task.done():dock_task.cancel()
            panel.destroy();panel=None;status=inspector.crowd_status_label
        manifest=json.loads((Path(config['asset']).parent/'manifest.json').read_text(encoding='utf-8'))
        mapper=BodyPoseMapper(manifest,links)
        scheduler=RandomCrowdCommands(a.characters,a.seed,max_speed=a.max_speed,max_yaw=a.max_yaw,radius=a.arena_radius)
        journal.mark('assets');context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        animations=[];looks=[];faces=[]
        from character_appearance import DefaultCharacterLook
        for i in range(a.characters):
            pump(f'Preparing Anon {i+1}/{a.characters}: materials and skeleton')
            root=f'/World/Anon_{i:02d}'
            stage.DefinePrim(root).GetReferences().AddReference(str(PROJECT_ROOT/'output/pmx_prototype/character.usdc'))
            animations.append(UsdSkel.Animation.Get(stage,root+'/Animation'))
            look=DefaultCharacterLook(stage,state['poses'][i,pelvis,:3],character_root=root);looks.append(look)
            # Each character receives the approved face light, linked only to
            # its own meshes so that multiplying actors doesn't multiply light.
            light_path=f'/World/FaceSoftbox_{i:02d}'
            Sdf.CopySpec(stage.GetRootLayer(),'/World/FaceSoftbox',stage.GetRootLayer(),light_path)
            light=UsdLux.SphereLight.Get(stage,light_path)
            collection=UsdLux.LightAPI(light.GetPrim()).GetLightLinkCollectionAPI()
            collection.CreateIncludeRootAttr(False);collection.CreateIncludesRel().SetTargets([Sdf.Path(root)])
            look.face_light=light;faces.append(UsdGeom.Xformable(light).GetOrderedXformOps()[0])
        stage.RemovePrim('/World/FaceSoftbox')
        floor=UsdGeom.Cube.Define(stage,'/World/PhysxGround');floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.04));floor.AddScaleOp().Set(Gf.Vec3f(1000,1000,.05))
        floor.CreateDisplayColorAttr([(.18,.21,.26)])
        ground=UsdGeom.Mesh.Define(stage,'/World/VisibleGround')
        ground.CreatePointsAttr([(-50,-50,-.006),(50,-50,-.006),(50,50,-.006),(-50,50,-.006)])
        ground.CreateFaceVertexCountsAttr([4]);ground.CreateFaceVertexIndicesAttr([0,1,2,3])
        ground.CreateSubdivisionSchemeAttr('none');ground.CreateDisplayColorAttr([(.24,.28,.32)])
        camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateFocalLengthAttr(35)
        camera.CreateHorizontalApertureAttr(36);camera.CreateVerticalApertureAttr(36*height/width)
        radius=a.arena_radius
        overview_eye=np.array([radius*1.2,-radius*1.65,radius*1.25]);overview_target=np.array([0.,0.,.6])
        overview_matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*overview_eye),Gf.Vec3d(*overview_target),Gf.Vec3d(0,0,1)).GetInverse()
        camera_op=camera.AddTransformOp();camera_op.Set(overview_matrix)
        viewport=get_active_viewport();viewport.camera_path='/World/Camera';viewport.set_texture_resolution((width,height))
        def publish(i,pose):
            pos,quat=pose[:2]
            animations[i].GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
            animations[i].GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(quat.astype(np.float32)))
        def draw_interpolated(samples):
            if not app.is_running():return False
            for i,pose in enumerate(samples):
                publish(i,(pose['p'],pose['q']))
                center=pose['center']
                faces[i].Set(Gf.Vec3d(float(center[0]),float(center[1]-2.5),1.7))
                looks[i].update(pose['frame'])
            if inspector is not None and inspector.follow_selected:
                pose=samples[inspector.selected_actor]
                target,offset=inspector.camera_target(pose['center'],pose['heading'],time.perf_counter())
                eye=target+np.asarray(offset)
                camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
            else:camera_op.Set(overview_matrix)
            app.update();note_render(min(pose['frame'] for pose in samples))
            return True
        for i in range(a.characters):publish(i,mapper(state['poses'][i]))
        # Establish initial skinning and face-occlusion buffers while secondary
        # workers are still waiting for their initial poses, as in single view.
        if inspector is not None and inspector.follow_selected:
            target,offset=inspector.follow_camera(state['poses'],pelvis,time.perf_counter())
            eye=target+np.asarray(offset)
            camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())
        for look in looks:look.update(-1)
        journal.mark('warmup','Preparing initial display buffers')
        for update in range(30):
            if not app.is_running():raise KeyboardInterrupt
            app.update()
            if (update+1)%10==0:journal.mark('warmup',f'Viewport {update+1}/30 updates')
        # Apply after renderer initialization so Kit cannot replace the default.
        import carb.settings
        render_settings=carb.settings.get_settings()
        render_settings.set_int('/rtx/post/dlss/execMode',{'performance':0,'balanced':1,'quality':2,'auto':3}[a.dlss_mode])
        journal.mark('secondary')
        deadline=time.perf_counter()+180
        for i,stream in enumerate(streams):
            stream.app=app
            while stream.channel.connection is None:
                if stream.channel.error:raise stream.channel.error
                if stream.process.poll() is not None:raise RuntimeError(f'Secondary worker {i+1} exited; see its worker.log')
                if time.perf_counter()>deadline:raise TimeoutError('Crowd secondary worker connection timed out')
                pump(f'Connecting hair/skirt physics: Anon {i+1}/{a.characters}');time.sleep(.005)
            # Dispatch all initial poses before waiting for rig construction.
            stream.channel.send(dict(poses=state['poses'][i],manifest=manifest,links=links))
        for i,stream in enumerate(streams):
            stream.metadata=stream.receive(-1,True,
                on_wait=lambda n=i:pump(f'Initializing hair/skirt physics: Anon {n+1}/{a.characters}'))
        if a.render_fps==60:
            from anon_crowd_display import CrowdInterpolatedDisplay
            from scipy.spatial.transform import Rotation
            display=CrowdInterpolatedDisplay(draw_interpolated)
        journal.mark('ready');startup_seconds=time.perf_counter()-started
        falls=np.zeros(a.characters,int);resets=falls.copy();max_distance=np.zeros(a.characters)
        distinct=[set() for _ in range(a.characters)]
        initial=state['roots'][:,:2].copy();completed=0;clock_start=0
        trace=(a.output/'commands.jsonl').open('w',encoding='utf-8');loop_started=time.perf_counter()
        frame_deadline=loop_started
        if a.record_video:
            from anon_crowd_recording import CrowdVideoRecorder
            from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
            recorder=CrowdVideoRecorder(a.record_video,(width,height),a.output,radius=a.arena_radius)
        playback_started=True
        from anon_crowd_pipeline import CrowdPlaybackPipeline
        def plan_request(sequence,previous):
            nonlocal clock_start
            manual=inspector.next_request() if inspector is not None else None
            controlled=inspector.selected_actor if inspector is not None else a.controlled_character-1
            reset=inspector.consume_reset_all() if inspector is not None else flags['reset'];flags['reset']=False
            if reset:scheduler.reset();clock_start=sequence
            paused=inspector.background_paused if inspector is not None else flags['paused']
            requested=scheduler.sample((sequence-clock_start)/30,previous['roots'][:,:2],previous['headings'],paused=paused)
            reset_ids=[]
            if manual is not None:
                requested[controlled]=manual['command']
                if manual['reset']:reset_ids=[controlled]
            request=dict(seq=sequence,commands=requested,reset=reset,reset_ids=reset_ids,controlled_actor=controlled,
                push_character=manual.get('push_character',False) if manual is not None else False,
                acceptance_inputs=bool(manual and manual.get('acceptance_inputs')),background_paused=paused)
            if a.mixed_secondary:
                request['secondary_modes']=crowd_secondary_modes(a.characters,controlled,a.secondary_mode,True,controlled_mode)
            return request
        pipeline=CrowdPlaybackPipeline(state,plan_request,channel.send,receive,streams,prefetch=not a.serial_playback,
            receive_async=lambda seq:receive(seq,update_ui=False))
        while app.is_running() and (not a.steps or completed<a.steps):
            tick=time.perf_counter()
            state,request,secondary_batch=pipeline.advance(has_next=not a.steps or completed+1<a.steps,on_wait=pump)
            solved_times.append(time.perf_counter());solved_poses=[]
            requested=request['commands'];controlled=request['controlled_actor'];paused=request['background_paused']
            for i in range(a.characters):
                if streams:
                    secondary=secondary_batch[i]
                    last_secondary_modes[i]=secondary.get('secondary_mode',initial_modes[i])
                    if a.mixed_secondary and last_secondary_modes[i]!=request['secondary_modes'][i]:
                        raise RuntimeError(f'Unexpected secondary mode for actor {i+1}')
                    pose=(secondary['p'],secondary['q'])
                else:pose=mapper(state['poses'][i])
                solved_poses.append((pose[0],pose[1]))
                center=state['poses'][i,pelvis,:3]
                if display is None:
                    publish(i,pose)
                    faces[i].Set(Gf.Vec3d(float(center[0]),float(center[1]-2.5),1.7));looks[i].update(completed)
                falls[i]+=int(state['falls'][i]);resets[i]+=int(state['resets'][i])
                distinct[i].add(tuple(np.round(requested[i],3)))
                max_distance[i]=max(max_distance[i],float(np.linalg.norm(state['roots'][i,:2]-initial[i])))
                cmd=state['commands'][i];vel=state['velocity'][i]
                if inspector is None:
                    actor_labels[i].text=(f'Anon {i+1:02d}  |  Command {cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}'
                        f'  |  Actual {vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}  |  Falls {falls[i]}')
                if state['falls'][i]:print(f'ANON_CROWD_FALL actor={i+1} frame={completed}',flush=True)
            trace_row=dict(frame=completed,requested=requested.tolist(),commands=state['commands'].tolist(),
                actual=state['velocity'].tolist(),roots=state['roots'].tolist(),falls=state['falls'].tolist(),
                resets=state['resets'].tolist(),controlled_character=controlled+1 if inspector is not None else None)
            if a.mixed_secondary:trace_row['secondary_modes']=last_secondary_modes.copy()
            trace.write(json.dumps(trace_row,separators=(',',':'))+'\n')
            camera_eye,camera_target=overview_eye,overview_target
            if inspector is not None:
                if state['resets'][inspector.selected_actor]:inspector.reset_camera()
                inspector.update_crowd(state,falls,resets,completed,acceptance_inputs=request['acceptance_inputs'])
                if display is None and inspector.follow_selected:
                    camera_target,offset=inspector.follow_camera(state['poses'],pelvis,time.perf_counter())
                    camera_eye=camera_target+np.asarray(offset)
                    camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera_eye),Gf.Vec3d(*camera_target),Gf.Vec3d(0,0,1)).GetInverse())
                elif display is None:camera_op.Set(overview_matrix)
            if display is not None:
                forward=Rotation.from_quat(state['poses'][:,pelvis,3:]).apply(np.tile([0.,-1.,0.],(a.characters,1)))
                headings=np.where(np.linalg.norm(forward[:,:2],axis=1)>1e-4,np.arctan2(forward[:,0],-forward[:,1]),0.)
                display.push(completed,solved_poses,state['poses'][:,pelvis,:3],headings,state['resets'])
            completed+=1
            fps=completed/max(time.perf_counter()-loop_started,1e-6)
            mode_label=f'{controlled_mode} / BG light3' if a.mixed_secondary else a.secondary_mode
            status.text=(f'{a.characters} Anons | Lab Direct / {a.body_device} | {completed/30:.1f}s simulated'
                         f' | {fps:.1f} control FPS | Falls {int(falls.sum())} | Secondary {mode_label}')
            if inspector is not None:
                status.text+=f'\nBackground: {"stopped" if paused else "random"} | Actor collisions: {"on" if a.inter_character_collisions else "off"}'
            if display is not None:
                display.present_if_due()
            elif app.is_running():
                app.update()
                note_render(completed-1)
                # A fresh pose was just rendered. Do not immediately repaint
                # it again from a worker-wait callback on the next iteration.
                next_ui=time.perf_counter()+1/30
            if recorder is not None and app.is_running():
                rgb=_capture_viewport_rgb()
                if rgb is None:raise RuntimeError('Crowd video frame capture failed')
                recorder.set_camera(camera_eye,camera_target)
                recorder.write(rgb,completed-1,state['poses'][:,pelvis,:3],falls,
                    controlled_actor=controlled if inspector is not None else None)
            if completed%30==0:trace.flush();print(f'ANON_CROWD_PROGRESS frames={completed} falls={falls.tolist()}',flush=True)
            if a.real_time and recorder is None:
                # A shared clock avoids accumulating a sleep overshoot every
                # frame. Preserve every state; resync after a long interruption.
                frame_deadline+=1/30
                if time.perf_counter()-frame_deadline>.1:frame_deadline=time.perf_counter()
                if display is not None:display.wait_until(frame_deadline)
                else:
                    while app.is_running() and time.perf_counter()<frame_deadline:
                        pump();time.sleep(min(.001,max(0.,frame_deadline-time.perf_counter())))
        if display is not None and app.is_running():display.finish()
        if recorder is not None:
            recorder.finish();video_frames=recorder.frames;recorder=None
            if a.steps and video_frames!=completed:raise RuntimeError('Video is missing control-step frames')
        if a.save_preview and app.is_running():
            from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
            from PIL import Image
            for _ in range(5):app.update()
            rgb=_capture_viewport_rgb()
            if rgb is None:raise RuntimeError('Crowd preview capture failed')
            Image.fromarray(rgb).save(a.output/'preview.png')
        start_render=int(np.searchsorted(render_sources,30)) if render_sources else 0
        intervals=np.diff(render_times)[min(start_render,max(0,len(render_times)-2)):]*1000
        solve_intervals=np.diff(solved_times)[min(30,max(0,len(solved_times)-2)):]*1000
        solve_windows=(150/np.maximum(np.asarray(solved_times[180:])-np.asarray(solved_times[30:-150]),1e-6)
            if len(solved_times)>180 else np.array([]))
        report=dict(checkpoint=str(a.checkpoint),seed=a.seed,characters=a.characters,frames=completed,
            anti_aliasing_effective=render_settings.get('/rtx/post/aa/op'),
            dlss_mode=a.dlss_mode,dlss_mode_effective=render_settings.get('/rtx/post/dlss/execMode'),
            simulated_seconds=completed/30,startup_seconds=startup_seconds,wall_seconds=time.perf_counter()-loop_started,
            control_fps=completed/max(time.perf_counter()-loop_started,1e-6),
            target_render_fps=a.render_fps,display_pose_interpolation=display is not None,render_frames=len(render_times),
            delivered_fps=float(1000/intervals.mean()) if len(intervals) else None,
            render_frame_ms_p95=float(np.percentile(intervals,95)) if len(intervals) else None,
            solved_pose_fps=float(1000/solve_intervals.mean()) if len(solve_intervals) else None,
            solved_frame_ms_p95=float(np.percentile(solve_intervals,95)) if len(solve_intervals) else None,
            solved_frame_ms_p99=float(np.percentile(solve_intervals,99)) if len(solve_intervals) else None,
            solved_rate_5s_min=float(solve_windows.min()) if len(solve_windows) else None,
            solved_rate_5s_max=float(solve_windows.max()) if len(solve_windows) else None,
            interpolated_render_frames=display.interpolated_frames if display is not None else 0,
            backend='lab-direct',body_device=a.body_device,shared_body_physics=True,
            inference_pipeline=('serial' if a.serial_playback else
                'one-step lookahead; asynchronous body receipt and secondary dispatch' if pipeline.receive_async is not None else
                'one-step lookahead; ordered body and secondary states'),
            asynchronous_body_dispatch=bool(not a.serial_playback and pipeline.receive_async is not None),
            inter_character_collisions=a.inter_character_collisions,random_only=a.random_only,
            initial_controlled_character=a.controlled_character if not a.random_only else None,
            final_controlled_character=int(state['controlled_actor'])+1 if not a.random_only else None,
            physics_stage_id=body_stage_id,collision_groups=env_groups,
            secondary_physics=bool(streams),secondary_mode='mixed' if a.mixed_secondary else a.secondary_mode,
            mixed_secondary=a.mixed_secondary,initial_secondary_modes=initial_modes if streams else [],
            controlled_secondary_mode=controlled_mode if a.mixed_secondary else None,
            final_secondary_modes=last_secondary_modes if streams else [],
            secondary_physics_hz=[s.metadata['physics_hz'] for s in streams],
            appearance='mmd-toon (cel-studio)',actor_paths=actor_paths,
            video=str(a.record_video.resolve()) if a.record_video else None,video_frames=video_frames,
            video_fps=30 if a.record_video else None,video_time_basis='simulation' if a.record_video else None,
            checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            secondary_source_sha256=hashlib.sha256(resolve_source('physx_secondary.py').read_bytes()).hexdigest(),
            appearance_source_sha256=hashlib.sha256(resolve_source('character_appearance.py').read_bytes()).hexdigest(),
            actors=[dict(index=i+1,falls=int(falls[i]),resets=int(resets[i]),unique_commands=len(distinct[i]),
                         max_distance_from_spawn_m=float(max_distance[i])) for i in range(a.characters)],
            limitations=['Random exploration is not the formal gait acceptance suite',
                'The policy has no learned crowd avoidance; actor body collisions are '+('enabled' if a.inter_character_collisions else 'disabled'),
                'Each actor retains independent one-way hair/skirt physics; no inter-character secondary contacts'])
        (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('ANON_CROWD_RESULT '+json.dumps(report),flush=True)
    except KeyboardInterrupt:
        journal.mark('cancelled','Crowd viewer closed')
    except Exception:
        import traceback
        exit_code=1;journal.mark('failed','See terminal and actor worker logs')
        traceback.print_exc()
        raise
    finally:
        if recorder is not None:recorder.abort()
        if trace is not None:trace.close()
        if pipeline is not None:pipeline.close()
        for stream in streams:stream.close()
        if worker is not None and worker.poll() is None:
            try:
                if channel.connection is not None:channel.send(dict(stop=True))
                worker.wait(timeout=15)
            except (OSError,subprocess.TimeoutExpired):worker.terminate();worker.wait(timeout=10)
        channel.close()
        if body_log is not None:body_log.close()
        if dock_task is not None and not dock_task.done():dock_task.cancel()
        if inspector is not None:inspector.close()
        # All owned workers and output files are closed above. Use Kit's
        # documented immediate shutdown for this multi-SkelRoot viewer; native
        # USD teardown otherwise hit a Windows heap error during the first run.
        if app is not None:app.close(skip_cleanup=True,exit_code=exit_code)
