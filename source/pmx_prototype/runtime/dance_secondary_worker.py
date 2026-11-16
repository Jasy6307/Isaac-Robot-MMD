"""Bounded live secondary-physics producer; no rendered or cached trajectory."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT,config_path
import argparse,json,time,gc,os
startup_started=time.perf_counter();startup_phases={}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--usd',required=True,type=Path);p.add_argument('--pmx',required=True,type=Path)
p.add_argument('--rig-data',type=Path);p.add_argument('--secondary-profile',type=config_path)
p.add_argument('--physics-threads',type=int,default=16)
p.add_argument('--skirt-edge-stride',type=int,choices=[1,2,3],default=1)
p.add_argument('--render-fps',type=int,default=30);p.add_argument('--quality',default='cloth')
p.add_argument('--address');p.add_argument('--authkey');p.add_argument('--output',required=True,type=Path)
p.add_argument('--steps',type=int,default=240);p.add_argument('--runtime-audit',action='store_true')
p.add_argument('--startup-settle-seconds',type=float,default=0.)
p.add_argument('--bake-usd',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.startup_settle_seconds<0 or not __import__('math').isfinite(a.startup_settle_seconds):p.error('settle seconds must be finite and nonnegative')
task_threads=int(os.environ.get('DANCE_WORKER_TASK_THREADS','4'))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True,'disable_viewport_updates':True,
    'renderer':'MinimalRendering',
    'limit_cpu_threads':task_threads,'extra_args':[f'--/persistent/physics/numThreads={a.physics_threads}',f'--/log/file={a.output.resolve()/"kit_worker.log"}']})
startup_phases['app_ready']=time.perf_counter()-startup_started
import numpy as np
import omni.usd
from pxr import Usd,UsdGeom,UsdSkel,Vt
from physx_secondary import BodyMotion,PhysxSecondary
physics=connection=None
gc_probe=None;gc_events=[];gc_ticks={}
try:
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.tensors',True)
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    motion=BodyMotion(a.usd,a.pmx,rig_data=a.rig_data,secondary_profile=a.secondary_profile)
    motion.rig_settings['preview_skirt_edge_stride']=a.skirt_edge_stride
    UsdSkel.Root.Define(stage,'/World/Character').GetPrim().GetReferences().AddReference(str(a.usd.resolve()))
    anim=UsdSkel.Animation.Get(stage,'/World/Character/Animation');anim.CreateJointsAttr(motion.joints)
    lp,lq,_,_=motion.pose(0)
    anim.CreateScalesAttr([(1,1,1)]*len(motion.joints))
    anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(lp.astype(np.float32)))
    anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(lq.astype(np.float32)))
    physics=PhysxSecondary(stage,motion,obstacles=False,quality=a.quality)
    if a.runtime_audit:
        (a.output/'motion_schema.json').write_text(json.dumps(dict(dynamic_mask=motion.dynamic.tolist(),joint_names=motion.names),ensure_ascii=False),encoding='utf-8')
    for _ in range(15):app.update()
    physics.start(context.get_stage_id())
    startup_phases['physics_ready']=time.perf_counter()-startup_started
    settle_steps=round(a.startup_settle_seconds*physics.physics_hz)
    def settle():
        p0,q0=lp,lq
        for sub in range(settle_steps):
            p0,q0=physics.step(0.,produce_pose=sub==settle_steps-1,diagnostics=False,compiled_hair=True,optimized_contacts=True)
        return p0,q0
    initial_p,initial_q=settle()
    physics.runtime_costs[:]=0
    physics.runtime_calls=0
    physics.projection_ms=0.
    startup_phases['settled_ready']=time.perf_counter()-startup_started
    def gc_probe(phase,info):
        gen=info['generation']
        if phase=='start':gc_ticks[gen]=time.perf_counter()
        elif gen in gc_ticks:gc_events.append((time.perf_counter()-gc_ticks.pop(gen))*1000)
    gc.callbacks.append(gc_probe)
    metadata=dict(physics_hz=physics.physics_hz,stream_pose_fps=a.render_fps,joints=physics.joint_count,skirt_edge_stride=a.skirt_edge_stride,skirt_edge_proxies=len(physics.edge_nodes),
        dynamic_bodies=physics.dynamic_view.count,quality_profile=a.quality,
        secondary_profile=str(a.secondary_profile),physics_threads=a.physics_threads,worker_task_threads=task_threads,
        secondary_physics_enabled=True,backend='native omni.physx CPU worker',
        worker_renderer='MinimalRendering (no viewport updates)',
        startup_settle_steps=settle_steps,startup_settle_seconds=settle_steps/physics.physics_hz,startup_phases=startup_phases,
        solver_position_iterations=255 if motion.rig_settings.get('skirt_max_stretch') else 96 if a.quality=='cloth' else 40 if a.quality=='hair' else 24,
        solver_velocity_iterations=16 if a.quality=='cloth' else 8 if a.quality=='hair' else 4,worker_prefetch_frames=1)
    if a.address:
        from anon_policy_transport import connect_worker
        connection=connect_worker(a.address,a.authkey)
        connection.send(dict(seq=-1,metadata=metadata))
    costs=[];audit_p=[];audit_q=[];audit_particles=[];seq=loops=total_steps=playback_steps=0
    baked=None
    if a.bake_usd:
        a.bake_usd.parent.mkdir(parents=True,exist_ok=True)
        baked=Usd.Stage.CreateNew(str(a.bake_usd.resolve()));baked.GetRootLayer().subLayerPaths=[str(a.usd.resolve()).replace('\\','/')]
        baked.SetDefaultPrim(baked.GetPrimAtPath('/Character'));baked.SetTimeCodesPerSecond(a.render_fps);baked.SetFramesPerSecond(a.render_fps)
        baked_anim=UsdSkel.Animation.Get(baked,'/Character/Animation');baked_anim.GetJointsAttr().Set(motion.joints)
        baked_anim.CreateScalesAttr([(1,1,1)]*len(motion.joints))
    first=True;started=time.perf_counter()
    while True:
        request=connection.recv() if connection else dict(seq=seq,stop=seq>=a.steps)
        if request.get('stop'):break
        restart=False
        if request.get('reset') or playback_steps>=(motion.end_frame+1)*(physics.physics_hz//30):
            physics.reset();initial_p,initial_q=settle();playback_steps=0;first=True;loops+=1;restart=True
        tick=time.perf_counter()
        p0,q0=initial_p,initial_q
        if not first:
            substeps=min(physics.physics_hz//a.render_fps,(motion.end_frame+1)*(physics.physics_hz//30)-playback_steps)
            for sub in range(substeps):
                frame=min((playback_steps+1)/(physics.physics_hz//30),motion.end_frame)
                p0,q0=physics.step(frame,produce_pose=sub==substeps-1,diagnostics=False,compiled_hair=True,optimized_contacts=True)
                playback_steps+=1
            total_steps+=substeps
        frame=min(playback_steps/(physics.physics_hz//30),motion.end_frame);first=False
        cost=(time.perf_counter()-tick)*1000;costs.append(cost)
        state=dict(seq=request['seq'],frame=frame,p=p0,q=q0,physics_steps=total_steps,
            restart=restart,loops=loops,work_ms=cost)
        if a.runtime_audit:
            state['particles']=physics.last_positions.copy()
            audit_p.append(p0.copy());audit_q.append(q0.copy());audit_particles.append(physics.last_positions.copy())
        if connection:connection.send(state)
        if baked:
            baked_anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p0.astype(np.float32)),seq)
            baked_anim.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q0.astype(np.float32)),seq)
        seq+=1
    if a.runtime_audit:np.savez_compressed(a.output/'runtime_audit.npz',p=audit_p,q=audit_q,particles=audit_particles)
    if baked:baked.SetStartTimeCode(0);baked.SetEndTimeCode(seq-1);baked.GetRootLayer().Save()
    report=dict(metadata,status='complete',frames=seq,render_frames=seq,physics_steps=total_steps,warmup_physics_steps=settle_steps*(loops+1),loops=loops,
        mean_work_ms=float(np.mean(costs[30:] or costs)),p95_work_ms=float(np.percentile(costs[30:] or costs,95)),
        p99_work_ms=float(np.percentile(costs[30:] or costs,99)),maximum_work_ms=float(max(costs[30:] or costs)),
        gc_pause_ms=dict(count=len(gc_events),maximum=max(gc_events,default=0.),total=sum(gc_events)),
        wall_seconds=time.perf_counter()-started,
        projection_ms_per_substep=physics.projection_ms/max(physics.runtime_calls,1),
        substep_phase_ms=dict(zip(('prepare','simulate_submit','fetch','fabric','postprocess_and_pose','contacts'),
                                  (physics.runtime_costs/max(physics.runtime_calls,1)).tolist())))
    (a.output/'worker_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DANCE_SECONDARY_WORKER_PASS',json.dumps(report),flush=True)
finally:
    if gc_probe in gc.callbacks:gc.callbacks.remove(gc_probe)
    if connection:connection.close()
    if physics:physics.close()
    app.close()
