"""Isolated live Anon secondary simulation, driven by ordered policy poses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT, config_path
import argparse, json, time

parser=argparse.ArgumentParser()
parser.add_argument('--pmx',type=Path,required=True)
parser.add_argument('--profile',type=config_path)
parser.add_argument('--quality',required=True)
parser.add_argument('--physics-threads',type=int,default=16)
parser.add_argument('--task-threads',type=int,default=16)
parser.add_argument('--skirt-edge-stride',type=int,choices=[1,2,3],default=1)
parser.add_argument('--address',required=True)
parser.add_argument('--authkey',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reference-hair-compute',action='store_true')
parser.add_argument('--legacy-compute',action='store_true')
args=parser.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True,'create_new_stage':False,
    'renderer':'MinimalRendering','limit_cpu_threads':args.task_threads,
    # This worker creates its own stage after connecting and never renders a viewport.
    'extra_args':['--/exts/omni.kit.viewport.window/startup/disableWindowOnLoad=true',
                  f'--/persistent/physics/numThreads={args.physics_threads}',
                  f'--/log/file={args.output.resolve()/"kit_worker.log"}']})
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import Gf,UsdGeom
from physx_secondary import BodyMotion,PhysxSecondary
from anon_policy_transport import connect_worker

class LiveBodyMotion(BodyMotion):
    live=None
    def pose(self,frame):
        return super().pose(frame) if self.live is None else self.live

physics=connection=None
try:
    connection=connect_worker(args.address,args.authkey)
    init=connection.recv()
    manifest=init['manifest'];links=init['links']
    motion=LiveBodyMotion(PROJECT_ROOT/'output/pmx_prototype/character.usdc',args.pmx,secondary_profile=args.profile)
    motion.rig_settings['max_node_distance_from_target_m']=2.
    motion.rig_settings['preview_skirt_edge_stride']=args.skirt_edge_stride
    if list(manifest['bone_paths'])!=motion.joints:raise ValueError('Skeleton order mismatch')
    owners=np.array([links.index(n) for n in manifest['bone_owners']])
    rest=np.array(manifest['rest_positions']);origins=np.array([manifest['links'][n]['origin'] for n in manifest['bone_owners']])
    parents=np.array(manifest['parents']);children=np.flatnonzero(parents>=0)
    def body_pose(transforms):
        t=transforms[owners];rotations=Rotation.from_quat(t[:,3:])
        wp=t[:,:3]+rotations.apply(rest-origins);wq=t[:,3:].copy()
        p,q=wp.copy(),wq.copy();parent=parents[children]
        inverse=Rotation.from_quat(wq[parent]).inv()
        p[children]=inverse.apply(wp[children]-wp[parent])
        q[children]=(inverse*Rotation.from_quat(wq[children])).as_quat()
        return p,q,wp,wq
    last=init['poses'];motion.live=body_pose(last)
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.DefinePrim('/World/Character').GetReferences().AddReference(str(PROJECT_ROOT/'output/pmx_prototype/character.usdc'))
    physics=PhysxSecondary(stage,motion,obstacles=False,quality=args.quality)
    floor=UsdGeom.Xformable(stage.GetPrimAtPath('/World/PhysxGround'))
    next(op for op in floor.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(1000,1000,.05))
    for _ in range(30):app.update()
    physics.start(context.get_stage_id())
    accelerated=not (args.legacy_compute or args.reference_hair_compute)
    for _ in range(physics.physics_hz):
        physics.step(0,produce_pose=args.legacy_compute,diagnostics=args.legacy_compute,compiled_hair=accelerated,optimized_contacts=accelerated)
    physics.runtime_costs[:]=0;physics.runtime_calls=0
    connection.send(dict(seq=-1,physics_hz=physics.physics_hz,dynamic_nodes=physics.dynamic_view.count,
        edge_nodes=len(physics.edge_nodes),settings=motion.rig_settings,task_threads=args.task_threads,physics_threads=args.physics_threads,hand_body_indices=motion.hand_body_indices,
        palm_proxy_dimensions_m={str(i):(2*shape[0]).tolist() for i,shape in motion.hand_proxy_shapes.items()}))
    while True:
        request=connection.recv()
        if request.get('stop'):break
        started=time.perf_counter();current=request['poses']
        if request.get('reset'):
            motion.live=body_pose(current);physics.initial_nodes=motion.live[2][motion.nodes].copy();physics.reset();last=current.copy()
        qa,qb=last[:,3:],current[:,3:].copy();qb[(qa*qb).sum(-1)<0]*=-1
        substeps=physics.physics_hz//30
        for substep in range(substeps):
            alpha=(substep+1)/substeps
            interpolated=last*(1-alpha)+current*alpha
            interpolated[:,3:]=qa*(1-alpha)+qb*alpha
            interpolated[:,3:]/=np.linalg.norm(interpolated[:,3:],axis=-1,keepdims=True)
            motion.live=body_pose(interpolated)
            p,q=physics.step(0,produce_pose=args.legacy_compute or substep==substeps-1,diagnostics=args.legacy_compute,
                compiled_hair=accelerated,optimized_contacts=accelerated)
        target=motion.live[2][motion.nodes][physics.dynamic_order]
        distance=np.linalg.norm(physics.last_positions-target,axis=-1)
        groups=motion.system['groups'][physics.dynamic_order]
        deviations=[float(np.sqrt(np.mean(distance[groups==group]**2))) for group in (14,8)]
        connection.send(dict(seq=request['seq'],p=p,q=q,deviations=deviations,steps=physics.steps,work_ms=(time.perf_counter()-started)*1000,
            runtime_costs=(physics.runtime_costs/max(1,physics.runtime_calls)).tolist(),
            hand_contact_steps=physics.hand_contact_steps,edge_hand_contact_steps=physics.edge_hand_contact_steps,
            hand_contact_peak=physics.hand_contact_peak,body_contact_peak=physics.body_contact_peak,body_contact_steps=physics.body_contact_steps,
            particles=physics.last_positions.copy() if request.get('audit') else None))
        last=current.copy()
finally:
    if physics:physics.close()
    if connection:connection.close()
    app.close()
