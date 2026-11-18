"""Explicit G1 O6 geometry profile from the accepted 2026-11-17 batch.

Preserves articulated finger capsules and adds real waist obstacles plus the
FCL-identified right pinky/shoulder supporting hull. Opt-in; legacy geometry
remains available for the original Girls regression.
"""
import torch

def apply_batch_geometry(model, robot_usd, *, profile='batch'):
    if profile not in ('batch', 'batch-hips', 'batch-hips-palm'):
        raise ValueError('Unknown batch geometry revision')
    if getattr(model, 'ground_geometry_profile', None):
        raise ValueError('Geometry profile already applied')
    model.ground_geometry_profile = profile
    if profile == 'batch':
        model = add_actual_waist_obstacles(model, robot_usd)
    indices={shape.name:i for i,shape in enumerate(model.shapes)}
    existing={tuple(sorted(pair)) for pair in model.pairs};caps=model.capsule_pairs.cpu().tolist();boxes=model.box_pairs.cpu().tolist();boxbox=model.box_box_pairs.cpu().tolist();added=[]
    for arm in sum(model.arm_names.values(),[]):
        for obstacle in model.hand_leg_targets:
            if "shoulder_" in obstacle and (arm,obstacle)!=("rh_pinky_distal","right_shoulder_pitch_link"):continue
            a,b=indices[arm],indices[obstacle];key=tuple(sorted((a,b)))
            if key in existing:continue
            existing.add(key);i=len(model.pairs);model.pairs.append((a,b));label=f'{arm} / {obstacle}';model.pair_labels.append(label);added.append(label)
            if model.shapes[a].kind==model.shapes[b].kind=='capsule':caps.append((i,a,b))
            elif model.shapes[a].kind==model.shapes[b].kind=='box':boxbox.append((i,a,b))
            else:
                cap,box=(a,b) if model.shapes[a].kind=='capsule' else (b,a);boxes.append((i,cap,box))
    model.capsule_pairs=torch.tensor(caps,device=model.device,dtype=torch.long).reshape(-1,3)
    model.box_pairs=torch.tensor(boxes,device=model.device,dtype=torch.long).reshape(-1,3)
    model.box_box_pairs=torch.tensor(boxbox,device=model.device,dtype=torch.long).reshape(-1,3)
    model.batch_extra_pairs=added
    if profile == 'batch-hips':
        return model
    model = tighten_palm_boxes(model, robot_usd)
    return refine_shoulder_hull(model, robot_usd) if profile == 'batch' else model


def tighten_palm_boxes(model, robot_usd):
    # The historical neutral box included every finger in its rest pose, filling
    # empty space. Articulated finger capsules already cover those meshes.
    # Keep only the palm body in this box, retaining the full finger union.
    from pxr import Usd,UsdGeom
    import numpy as np
    stage=Usd.Stage.Open(str(robot_usd));cache=UsdGeom.XformCache()
    for name in ['lh_hand_base_link','rh_hand_base_link']:
        body=next(p for p in stage.Traverse() if p.GetName()==name)
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T);pieces=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or '/visuals/' not in str(prim.GetPath()):continue
            v=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float);mat=inv@np.array(cache.GetLocalToWorldTransform(prim)).T;pieces.append(v@mat[:3,:3].T+mat[:3,3])
        points=np.concatenate(pieces);lo=points.min(0);hi=points.max(0);i=next(i for i,s in enumerate(model.shapes) if s.name==name)
        model.shapes[i].a=(lo+hi)/2;model.shapes[i].b=(hi-lo)/2
        model.local_a[i]=model._tensor((lo+hi)/2);model.local_b[i]=model._tensor((hi-lo)/2)
    model.hand_geometry_mode='palm-body-box-plus-articulated-finger-capsules-v2'
    return model


def add_actual_waist_obstacles(model, robot_usd):
    from pxr import Usd,UsdGeom
    import numpy as np
    from source.train_workflow.utils.motion.self_collision import Shape
    stage=Usd.Stage.Open(str(robot_usd));cache=UsdGeom.XformCache()
    for name in ['pelvis','waist_yaw_link','left_shoulder_pitch_link','left_shoulder_roll_link','right_shoulder_pitch_link','right_shoulder_roll_link']:
        if any(s.name==name for s in model.shapes):continue
        body=next(p for p in stage.Traverse() if p.GetName()==name)
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T);pieces=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or '/visuals/' not in str(prim.GetPath()):continue
            if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
            v=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float);mat=inv@np.array(cache.GetLocalToWorldTransform(prim)).T;pieces.append(v@mat[:3,:3].T+mat[:3,3])
        points=np.concatenate(pieces);lo=points.min(0);hi=points.max(0)
        model.shapes.append(Shape(name,'box',(lo+hi)/2,(hi-lo)/2));model.hand_leg_targets.append(name)
    model.local_a=model._tensor(np.stack([s.a for s in model.shapes]));model.local_b=model._tensor(np.stack([s.b for s in model.shapes]));model.radii=model._tensor([s.radius for s in model.shapes]);lengths=np.array([np.linalg.norm(s.b-s.a) for s in model.shapes]);model.sample_radii=model._tensor([np.sqrt(s.radius**2+(lengths[i]/32)**2) for i,s in enumerate(model.shapes)])
    return model


def refine_shoulder_hull(model, robot_usd):
    # Tight convex hull for the exact shoulder contact found by FCL. The palm
    # and other finger/torso constraints retain their established envelopes.
    from pxr import Usd,UsdGeom
    from scipy.spatial import ConvexHull
    import numpy as np
    stage=Usd.Stage.Open(str(robot_usd));cache=UsdGeom.XformCache();name='right_shoulder_pitch_link'
    body=next(p for p in stage.Traverse() if p.GetName()==name);inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T);pieces=[]
    for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or '/visuals/' not in str(prim.GetPath()):continue
        points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float);mat=inv@np.array(cache.GetLocalToWorldTransform(prim)).T;pieces.append(points@mat[:3,:3].T+mat[:3,3])
    vertices=np.concatenate(pieces);directions=np.unique(np.round(ConvexHull(vertices).equations[:,:3],6),axis=0);directions/=np.linalg.norm(directions,axis=1)[:,None];planes=np.c_[directions,-(vertices@directions.T).max(0)-1e-7]
    normals=model._tensor(planes[:,:3]);offset=model._tensor(planes[:,3]);cap=next(i for i,s in enumerate(model.shapes) if s.name=='rh_pinky_distal');obstacle=next(i for i,s in enumerate(model.shapes) if s.name==name);pair=model.pairs.index((cap,obstacle));original=model.distances
    def distances(rot,pos,pair_indices=None):
        result=original(rot,pos,pair_indices)
        if pair_indices is not None:
            if pair not in pair_indices:return result
            column=list(pair_indices).index(pair)
        else:column=pair
        ra=rot['rh_pinky_distal'];pa=pos['rh_pinky_distal'];rb=rot[name];pb=pos[name]
        a=(ra@model.local_a[cap,:,None]).squeeze(-1)+pa;b=(ra@model.local_b[cap,:,None]).squeeze(-1)+pa
        points=a[:,None]+model.sample_t[None,:,None]*(b-a)[:,None];local=(points-pb[:,None])@rb
        signed=(local[:,:,None,:]*normals[None,None]).sum(-1)+offset[None,None]
        result[:,column]=signed.amax(-1).amin(-1)-model.sample_radii[cap]
        return result
    model.distances=distances;return model
