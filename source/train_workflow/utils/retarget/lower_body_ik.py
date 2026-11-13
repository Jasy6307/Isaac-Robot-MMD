"""USD-derived, contact-aware lower-limb retargeting; no clip-specific rules."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from source.train_workflow.utils.retarget.upper_body_ik import (
    RobotArm, SolverWeights, SourceSkeleton, read_joints, unit, skew,
)


@dataclass
class LegTarget:
    elbow: np.ndarray  # knee, shared limb optimizer interface
    wrist: np.ndarray  # ankle
    palm: np.ndarray   # sole orientation
    world_rotation: np.ndarray
    root_position: np.ndarray
    contact: bool = False


class RobotLeg(RobotArm):
    def __init__(self, usd_path, side):
        from pxr import Usd, UsdGeom
        joints = read_joints(usd_path)
        by_child = {j.child:j for j in joints}
        child = side+'_ankle_roll_link'
        self.chain = []
        while child != 'pelvis':
            if child not in by_child or len(self.chain)>10:
                raise ValueError('Invalid leg hierarchy')
            j=by_child[child];self.chain.insert(0,j);child=j.parent
        self.moving=[j for j in self.chain if j.axis is not None]
        self.names=[j.name for j in self.moving]
        if len(self.names)!=6:raise ValueError('Expected six leg joints')
        self.lower=np.array([j.lower for j in self.moving]);self.upper=np.array([j.upper for j in self.moving])
        self.velocity=np.array([j.velocity for j in self.moving])
        if not np.isfinite([self.lower,self.upper,self.velocity]).all():
            raise ValueError('Missing authored leg limits')
        self.knee_name=side+'_knee_joint'
        # Read the visible foot bounds in the ankle body frame, including instanced meshes.
        stage=Usd.Stage.Open(str(usd_path));cache=UsdGeom.XformCache()
        link=next(p for p in stage.Traverse() if p.GetName()==side+'_ankle_roll_link')
        inv=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(link)).T)
        points=[]
        for prim in Usd.PrimRange(link,Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                raw=UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if raw is None:continue
                xyz=np.asarray(raw);m=inv@np.asarray(cache.GetLocalToWorldTransform(prim)).T
                points.append(xyz@m[:3,:3].T+m[:3,3])
        if not points:raise ValueError('Foot mesh is missing')
        points=np.concatenate(points);lo,hi=points.min(0),points.max(0)
        self.sole=np.array([[x,y,lo[2]] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
        self.palm_local=np.eye(3)
        neutral=self.fk(np.zeros(6));self.palm_local=neutral['palm'].T
        # The hip-roll pivot is the lateral hip centre. The hip-pitch pivot
        # precedes a sizeable pelvis offset; treating it as the thigh origin
        # incorrectly narrows the stance on a non-coincident robot hip chain.
        self.shoulder=neutral['origins'][1]
        self.upper_length=np.linalg.norm(neutral['elbow']-self.shoulder)
        self.fore_length=np.linalg.norm(neutral['wrist']-neutral['elbow'])
        self.reach=self.upper_length+self.fore_length

    def fk(self,q):
        r,p=np.eye(3),np.zeros(3);origins=[];axes=[];knee=None;k=0
        for j in self.chain:
            origin=p+r@j.p0;pre=r@j.r0
            if j.name==self.knee_name:knee=origin.copy()
            if j.axis is not None:
                origins.append(origin);axes.append(pre@j.axis)
                motion=Rotation.from_rotvec(j.axis*q[k]).as_matrix();k+=1
            else:motion=np.eye(3)
            r=pre@motion@j.r1.T;p=origin-r@j.p1
        return dict(elbow=knee,wrist=p,palm=r@self.palm_local,
                    body_rotation=r,origins=np.asarray(origins),axes=np.asarray(axes))

    def residual_jacobian(self,q,target,previous,weights=SolverWeights()):
        s=self.fk(q);res=[];jac=[]
        for key,count,weight in [('wrist',6,weights.wrist*(2 if target.contact else 1)),('elbow',3,weights.elbow)]:
            res.append(weight*(s[key]-getattr(target,key))/self.reach)
            j=np.zeros((3,6));j[:,:count]=np.cross(s['axes'][:count],s[key]-s['origins'][:count]).T
            jac.append(weight*j/self.reach)
        res.append(weights.palm*(s['palm']-target.palm).ravel()/np.sqrt(2))
        jac.append(weights.palm*np.column_stack([(skew(a)@s['palm']).ravel() for a in s['axes']])/np.sqrt(2))
        corners=self.sole@s['body_rotation'].T+s['wrist']
        world=corners@target.world_rotation.T+target.root_position
        res.append(40.*np.minimum(world[:,2],0)/self.reach)
        j=np.zeros((4,6))
        for k,point in enumerate(corners):
            j[k]=40.*(target.world_rotation[2]@np.cross(s['axes'],point-s['origins']).T)/self.reach*(world[k,2]<0)
        jac.append(j)
        res.append(weights.continuity*(q-previous));jac.append(weights.continuity*np.eye(6))
        return np.concatenate(res),np.vstack(jac)


def contact_mask(heights,positions,fps,leg_length):
    """Kinematic contact estimate: low AND slow, with minimum dwell time.

    Deliberate sliding/pivots are left free rather than labelled planted feet.
    This is an animation estimate, not a measured force contact.
    """
    velocity=np.linalg.norm(np.gradient(positions,1/fps,axis=0),axis=1) if len(positions)>1 else np.zeros(1)
    mask=(heights<.025*leg_length)&(velocity<.20*leg_length)
    minimum=max(2,round(.08*fps));start=0
    while start<len(mask):
        end=start+1
        while end<len(mask) and mask[end]==mask[start]:end+=1
        if mask[start] and end-start<minimum:mask[start:end]=False
        start=end
    return mask


def build_leg_targets(source,matrices,leg,side,root_positions,root_rotations,fps):
    prefix='左' if side=='left' else '右'
    required=['下半身',prefix+'足',prefix+'ひざ',prefix+'足首',prefix+'つま先']
    if any(n not in source.names for n in required):raise ValueError('Missing MMD leg landmarks')
    ix=[source.names[n] for n in required]
    rest=source.rest
    ankle=rest[ix[3],:3,3];toe=rest[ix[4],:3,3]
    forward=toe-ankle;forward[2]=0;forward=unit(forward)
    up=np.array([0.,0.,1.]);frame=np.column_stack((forward,np.cross(up,forward),up))
    local_frame=rest[ix[3],:3,:3].T@frame
    length=np.linalg.norm(rest[ix[2],:3,3]-rest[ix[1],:3,3])+np.linalg.norm(ankle-rest[ix[2],:3,3])
    ankle_world=np.array([m[ix[3],:3,3] for m in matrices])
    toe_world=np.array([m[ix[4],:3,3] for m in matrices])
    # Height above this rig's rest toe plane; source scaling cancels in thresholds.
    floor=rest[ix[4],2,3]
    heights=np.minimum(ankle_world[:,2]-(ankle[2]-floor),toe_world[:,2])-floor
    contacts=contact_mask(heights,ankle_world,fps,length)
    targets=[]
    for i,m in enumerate(matrices):
        convert=source.to_robot@m[ix[0],:3,:3].T
        hip,knee,ank=[m[j,:3,3] for j in ix[1:4]]
        k=leg.shoulder+leg.upper_length*(convert@unit(knee-hip))
        a=k+leg.fore_length*(convert@unit(ank-knee))
        orient=convert@m[ix[3],:3,:3]@local_frame
        body_r=orient@leg.palm_local.T
        corners=(leg.sole@body_r.T+a)@root_rotations[i].T+root_positions[i]
        # Ground the target geometry without flattening the source foot pitch.
        min_z=corners[:,2].min()
        shift=-min_z if contacts[i] else max(0.,-min_z)
        a=a+root_rotations[i].T@np.array([0.,0.,shift])
        targets.append(LegTarget(k,a,orient,root_rotations[i],root_positions[i],bool(contacts[i])))
    # Lock estimated stance XY over each contiguous segment, retaining foot orientation.
    i=0
    while i<len(targets):
        if not contacts[i]:i+=1;continue
        j=i+1
        while j<len(targets) and contacts[j]:j+=1
        xy=np.median([root_rotations[k]@targets[k].wrist+root_positions[k] for k in range(i,j)],axis=0)[:2]
        for k in range(i,j):
            world=root_rotations[k]@targets[k].wrist+root_positions[k]
            world[:2]=xy;targets[k].wrist=root_rotations[k].T@(world-root_positions[k])
        i=j
    return targets,contacts


def solve_lower_body_arrays(delta,names,baseline,fps,source_animation,robot_usd,root_positions,root_quats_wxyz,*,start_frame=0):
    from pathlib import Path
    import json
    from source.train_workflow.utils.retarget.upper_body_export import summarize
    from source.train_workflow.utils.retarget.upper_body_ik import target_errors
    delta=np.asarray(delta);root_positions=np.asarray(root_positions)
    if len(delta)<1 or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid lower-body timing')
    if root_positions.shape!=(len(delta),3) or np.shape(root_quats_wxyz)!=(len(delta),4):
        raise ValueError('Root trajectory must cover every frame')
    if not np.isfinite(delta).all() or not np.isfinite(root_positions).all():raise ValueError('Nonfinite motion')
    source=SourceSkeleton(source_animation)
    if (start_frame+len(delta)-1)/fps>(source.last-source.first)/source.fps+1e-5:raise ValueError('Source timeline too short')
    matrices=[source.frame((i+start_frame)/fps) for i in range(len(delta))]
    rotations=Rotation.from_quat(np.asarray(root_quats_wxyz)[:,[1,2,3,0]]).as_matrix()
    root_positions=np.asarray(root_positions)
    sidecar=Path(source_animation).with_suffix('.json')
    mode=json.loads(sidecar.read_text(encoding='utf-8')).get('leg_mode','unknown') if sidecar.is_file() else 'unknown'
    result=np.asarray(delta).copy();report=dict(method='skeleton-contact-lower-body-ik-v1',
        source_leg_mode=mode,source_animation=str(source_animation),root_trajectory_changed=False,
        contact_estimation='low-height and low-speed FK/posed-foot landmarks; not force contact',legs={})
    for side in ('left','right'):
        print(f'LOWER_IK {side} {len(delta)} frames',flush=True)
        leg=RobotLeg(robot_usd,side);cols=[names.index(n) for n in leg.names]
        targets,contacts=build_leg_targets(source,matrices,leg,side,root_positions,rotations,fps)
        seeds=delta[:,cols]+np.asarray(baseline)[cols]
        solved=leg.solve_sequence(targets,seeds,fps,weights=SolverWeights(4.,1.,.5,.2))
        result[:,cols]=solved-np.asarray(baseline)[cols]
        def evaluate(qs):
            errors=[];min_z=[];world=[]
            for i,(q,t) in enumerate(zip(qs,targets)):
                e=target_errors(leg,q,t);errors.append(e);s=leg.fk(q)
                min_z.append(((leg.sole@s['body_rotation'].T+s['wrist'])@rotations[i].T+root_positions[i])[:,2].min())
                world.append(rotations[i]@s['wrist']+root_positions[i])
            world=np.asarray(world);pair=contacts[1:]&contacts[:-1]
            slip=np.linalg.norm(np.diff(world[:,:2],axis=0),axis=1)[pair]*fps
            return dict(pose=summarize(errors),penetrating_frames=int(np.sum(np.array(min_z)<-.001)),
                        worst_penetration_m=float(max(0.,-min(min_z))),
                        stance_speed_mean_m_s=float(slip.mean()) if len(slip) else None,
                        stance_speed_p95_m_s=float(np.percentile(slip,95)) if len(slip) else None)
        violations=int(np.sum((solved<leg.lower-1e-7)|(solved>leg.upper+1e-7)))
        speed=int(np.sum(abs(np.diff(solved,axis=0))*fps>leg.velocity+1e-5))
        if violations or speed:raise ValueError('Leg joint limits violated')
        report['legs'][side]=dict(before=evaluate(seeds),after=evaluate(solved),contact_frames=int(contacts.sum()),
            limit_violations=violations,speed_limit_violations=speed,joint_names=leg.names)
    return result,report
