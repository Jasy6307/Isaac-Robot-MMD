"""Replay identical body states to check secondary physics and shadow equivalence."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=PROJECT_ROOT
sys.path.insert(0,str(ROOT))
from anon_policy_catalog import PRESETS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/realtime_optimization/equivalence')
p.add_argument('--rollout',type=Path,default=ROOT/'output/anon_locomotion/compensation_cleanup/before/rollout.npz')
p.add_argument('--checkpoint',type=Path,default=PRESETS['walk'],help='Physical model configuration associated with the body states')
p.add_argument('--hair-acceleration',action='store_true',help='Isolate the dance hair acceleration; retain existing reconstruction/diagnostic optimizations in both paths')
p.add_argument('--frames',type=int,default=0,help='0 replays the entire input')
p.add_argument('--start-frame',type=int,default=0)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'extra_args':['--/persistent/physics/numThreads=0']})
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import UsdGeom,UsdSkel,UsdLux,Vt,Usd
from physx_secondary import BodyMotion,PhysxSecondary
from secondary_profiles import DEFAULT_SECONDARY_PROFILE
from bangs_shadow import BangsShadow,cast_shadow,cast_shadow_reference
from character_appearance import DefaultCharacterLook

class Motion(BodyMotion):
    live=None
    def pose(self,frame):return super().pose(frame) if self.live is None else self.live

physics=None
result=0
try:
    cfg=json.loads((a.checkpoint.parent/'env_config.json').read_text(encoding='utf-8'))
    manifest=json.loads((Path(cfg['asset']).parent/'manifest.json').read_text(encoding='utf-8'))
    with np.load(a.rollout) as data:
        states=data['body'].copy() if 'body' in data else data['poses'][:,0].copy()
        links=list(data['link_names']) if 'link_names' in data else None
    # Live audit files predate link-name metadata. Their order is the trained
    # articulation's order, saved in the matching full acceptance rollout.
    if links is None:
        companion=a.checkpoint.parent/'evaluation/cpu/rollout.npz'
        with np.load(companion) as data:links=list(data['link_names'])
    end=a.start_frame+a.frames if a.frames else len(states)
    states=states[a.start_frame:end]
    if not len(states):raise ValueError('Body replay is empty')
    profile_hash=hashlib.sha256(DEFAULT_SECONDARY_PROFILE.read_bytes()).hexdigest()
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
    outputs={};shadow_frames=0;shadows={}
    reference='reference' if a.hair_acceleration else 'legacy'
    for optimized in [False,True]:
        legacy=not optimized if not a.hair_acceleration else False
        accelerated=optimized and a.hair_acceleration
        context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900)
        stage.DefinePrim('/World/Character').GetReferences().AddReference(str(ROOT/'output/pmx_prototype/character.usdc'))
        animation=UsdSkel.Animation.Get(stage,'/World/Character/Animation')
        motion=Motion(ROOT/'output/pmx_prototype/character.usdc',ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx',secondary_profile=DEFAULT_SECONDARY_PROFILE)
        motion.rig_settings['max_node_distance_from_target_m']=2.
        motion.live=body_pose(states[0])
        physics=PhysxSecondary(stage,motion,obstacles=False,quality='cloth')
        floor=UsdGeom.Xformable(stage.GetPrimAtPath('/World/PhysxGround'))
        next(op for op in floor.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale).Set((1000.,1000.,.05))
        physics.start(context.get_stage_id())
        for _ in range(physics.physics_hz):
            physics.step(0,produce_pose=legacy,diagnostics=legacy,compiled_hair=accelerated,optimized_contacts=accelerated)
        last=states[0];translations=[];rotations=[];particles=[];substep_positions=[];substep_velocities=[];masks=[]
        look=DefaultCharacterLook(stage,states[0,links.index('pelvis'),:3])
        shadow=look.bangs
        assert shadow is not None,'Approved bangs shadow must be enabled'
        for frame,current in enumerate(states):
            qa,qb=last[:,3:],current[:,3:].copy();qb[(qa*qb).sum(-1)<0]*=-1
            for substep in range(4):
                alpha=(substep+1)/4
                interpolated=last*(1-alpha)+current*alpha
                interpolated[:,3:]=qa*(1-alpha)+qb*alpha
                interpolated[:,3:]/=np.linalg.norm(interpolated[:,3:],axis=-1,keepdims=True)
                motion.live=body_pose(interpolated)
                lp,lq=physics.step(0,produce_pose=legacy or substep==3,diagnostics=legacy,
                                   compiled_hair=accelerated,optimized_contacts=accelerated)
                if a.hair_acceleration:
                    substep_positions.append(physics.last_positions.copy())
                    substep_velocities.append(physics.dynamic_view.get_velocities().copy())
            translations.append(lp.copy());rotations.append(lq.copy());particles.append(physics.last_positions.copy())
            if frame%10==0:
                animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(lp.astype(np.float32)))
                animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(lq.astype(np.float32)))
                points=Vt.Vec3fArray(shadow.rest)
                transforms=shadow.query.ComputeSkinningTransforms(Usd.TimeCode.Default())
                assert shadow.skin.ComputeSkinnedPoints(transforms,points)
                xyz=np.asarray(points,dtype=np.float64)
                old=cast_shadow_reference(xyz[shadow.face_vertices],xyz[shadow.bangs],shadow.direction)
                new=cast_shadow(xyz[shadow.face_vertices],xyz[shadow.bangs],shadow.direction)
                assert np.array_equal(old,new),f'Shadow mismatch frame {frame}'
                masks.append(new.copy())
                shadow_frames+=1
            last=current
            if (frame+1)%150==0:print('REPLAY_PROGRESS',optimized,frame+1,len(states),flush=True)
        name='optimized' if optimized else reference
        outputs[name]={k:np.array(v) for k,v in dict(translations=translations,rotations=rotations,particles=particles).items()}
        np.savez_compressed(a.output/f'{name}.npz',**outputs[name])
        if a.hair_acceleration:
            outputs[name].update(substep_positions=np.array(substep_positions),substep_velocities=np.array(substep_velocities))
        shadows[name]=np.asarray(masks)
        physics.close();physics=None
    errors={k:float(np.max(np.abs(outputs[reference][k]-outputs['optimized'][k]))) for k in outputs[reference]}
    exact={k:bool(np.array_equal(outputs[reference][k],outputs['optimized'][k])) for k in outputs[reference]}
    tolerances={k:(1e-6 if k=='substep_velocities' else 1e-7) for k in errors}
    masks_equal=bool(np.array_equal(shadows[reference],shadows['optimized']))
    unchanged=profile_hash==hashlib.sha256(DEFAULT_SECONDARY_PROFILE.read_bytes()).hexdigest()
    passed=all(errors[k]<=tolerances[k] for k in errors) and masks_equal and unchanged
    report=dict(passed=passed,frames=len(states),physics_hz=120,max_abs_error=errors,bitwise_equal=exact,
                tolerances=tolerances,shadow_frames_identical=shadow_frames,shadow_masks_equal=masks_equal,
                face_samples_per_frame=len(shadow.face_vertices),hair_acceleration=a.hair_acceleration,
                body_input=str(a.rollout.resolve()),body_input_sha256=hashlib.sha256(a.rollout.read_bytes()).hexdigest(),
                secondary_profile=str(DEFAULT_SECONDARY_PROFILE),secondary_profile_sha256=profile_hash,physics_parameters_unchanged=unchanged)
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RUNTIME_EQUIVALENCE',json.dumps(report),flush=True)
    assert passed,errors
except Exception:
    import traceback
    traceback.print_exc();result=1
finally:
    if physics:physics.close()
    app.close(exit_code=result)
