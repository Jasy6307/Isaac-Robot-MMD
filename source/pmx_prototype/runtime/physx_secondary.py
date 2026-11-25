"""Native PhysX rigid-node fabric/hair network driving the PMX skin.

Shares mesh topology and bone reconstruction with the XPBD prototype, but never
calls its particle integrator. All integration, distance joints and contacts are
handled by omni.physx, including contacts with ordinary scene colliders.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source, config_path

from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel, UsdShade, PhysxSchema

from convert_pmx import read_pmx
from secondary_motion import prepare_system
from realtime_xpbd import fk, reconstruct, collider_poses, njit, swing, multiply, rotate
from load_vmd import align_vectors
from secondary_profiles import resolve_skirt_grid


@njit(cache=True)
def reconstruct_smooth(lp, lq, wp, wq, parents, dynamic, positions, tangent, left, right, left2, right2, cloth_left=None, cloth_right=None, cloth_translation_mask=None):
    p, q, world_p, world_q = reconstruct(lp, lq, wp, wq, parents, dynamic, positions, tangent)
    # Estimate a material frame from a neighbourhood instead of one short,
    # noisy edge. Particle positions remain the unmodified PhysX output.
    for i in range(len(parents)):
        if not dynamic[i]:
            continue
        baseline = .5*(wp[right[i]]-wp[left[i]]) + .25*(wp[right2[i]]-wp[left2[i]])
        direction = .5*(world_p[right[i]]-world_p[left[i]]) + .25*(world_p[right2[i]]-world_p[left2[i]])
        world_q[i] = multiply(swing(baseline, direction), wq[i])
        if cloth_left is not None and (cloth_left[i] != i or cloth_right[i] != i):
            # Use both material directions for skirt roll; preserve native positions.
            lateral = wp[cloth_right[i]]-wp[cloth_left[i]]
            rotated = rotate(swing(baseline,direction), lateral)
            actual = world_p[cloth_right[i]]-world_p[cloth_left[i]]
            length = np.linalg.norm(direction)
            if length > 1e-8:
                axis = direction/length
                rotated -= axis*np.dot(rotated,axis)
                actual -= axis*np.dot(actual,axis)
                if np.linalg.norm(rotated) > 1e-8 and np.linalg.norm(actual) > 1e-8:
                    world_q[i] = multiply(swing(rotated,actual),world_q[i])
    if cloth_translation_mask is not None:
        for i in range(len(parents)):
            if cloth_translation_mask[i]: world_q[i] = wq[i]
    for i in range(len(parents)):
        parent = parents[i]
        if parent >= 0 and not dynamic[i]:
            world_p[i] = world_p[parent]+rotate(world_q[parent], lp[i])
            world_q[i] = multiply(world_q[parent], lq[i])
    for i in range(len(parents)):
        parent = parents[i]
        if parent >= 0:
            inverse = world_q[parent].copy(); inverse[:3] *= -1
            p[i] = rotate(inverse, world_p[i]-world_p[parent])
            q[i] = multiply(inverse, world_q[i])
    return p, q, world_p, world_q


@njit(cache=True)
def curvature_forces(positions, target, previous, previous_target, triplets, stiffness, damping, dt):
    result = np.zeros_like(positions)
    for e in range(len(triplets)):
        a, b, c = triplets[e]
        for axis in range(3):
            error = (positions[a, axis]-2*positions[b, axis]+positions[c, axis]
                     - target[a, axis]+2*target[b, axis]-target[c, axis])
            old = (previous[a, axis]-2*previous[b, axis]+previous[c, axis]
                   - previous_target[a, axis]+2*previous_target[b, axis]-previous_target[c, axis])
            f = -stiffness[e]*error-damping[e]*(error-old)/dt
            result[a, axis] += f
            result[b, axis] -= 2*f
            result[c, axis] += f
    return result


@njit(cache=True)
def capsule_approach_damping(points, velocities, point_radii, ca, cb, old_ca, old_cb, radii, dt, band, damping, max_force, stiffness=0.):
    """Smooth normal support plus approach damping; native PhysX resolves contact."""
    result=np.zeros_like(points)
    for i in range(len(points)):
        for j in range(len(ca)):
            ab=cb[j]-ca[j]
            t=min(1.,max(0.,np.dot(points[i]-ca[j],ab)/max(np.dot(ab,ab),1e-12)))
            surface=ca[j]+t*ab
            delta=points[i]-surface;distance=np.linalg.norm(delta)
            if distance<1e-10:continue
            gap=distance-radii[j]-point_radii[i]
            if gap>=band:continue
            normal=delta/distance
            surface_velocity=(surface-(old_ca[j]+t*(old_cb[j]-old_ca[j])))/dt
            speed=np.dot(velocities[i]-surface_velocity,normal)
            if speed>=0 and stiffness==0:continue
            u=min(1.,max(0.,(band-gap)/band));weight=u*u*(3.-2.*u)
            magnitude=max_force*np.tanh((damping*weight*max(-speed,0.)+stiffness*band*weight)/max_force)
            result[i]+=magnitude*normal
    return result


class BodyMotion:
    def __init__(self, usd, pmx, rig_data=None, secondary_profile=None):
        self.source = Usd.Stage.Open(str(Path(usd).resolve()))
        if self.source.GetDefaultPrim().GetCustomDataByKey('secondaryMotionSolver'):
            raise ValueError('Use the body animation, not a baked secondary-motion asset')
        self.model = read_pmx(Path(pmx).resolve(), physics=True)
        self.rig_settings = {}
        if rig_data:
            import json
            data = json.loads(Path(rig_data).read_text(encoding='utf-8'))
            self.rig_settings = data
            self.model['bones'].extend(data['bones'])
            for body in self.model['rigid_bodies']:
                if body['bone'] in data.get('locked_hair_bones',[]):
                    body['bone'] = -1
                elif 12 <= body['bone'] <= 80:
                    if body['mode']: body['group'] = 14
                    else: body['bone'] = -1
                else: body['mode'] = 0
                body['mask'] = 65535
            self.model['rigid_bodies'].extend(data['bodies'])
        if secondary_profile:
            import json
            self.rig_settings.update(json.loads(config_path(secondary_profile).read_text(encoding='utf-8')))
        self.skeleton = UsdSkel.Skeleton.Get(self.source, '/Character/Skeleton')
        self.animation = UsdSkel.Animation.Get(self.source, '/Character/Animation')
        self.joints = list(self.skeleton.GetJointsAttr().Get())
        self.names = list(self.skeleton.GetJointNamesAttr().Get())
        resolve_skirt_grid(self.rig_settings,self.names)
        indices = {p: i for i, p in enumerate(self.joints)}
        self.parents = np.array([indices.get(p.rpartition('/')[0], -1) for p in self.joints])
        self.columns = np.array([indices[p] for p in self.animation.GetJointsAttr().Get()])
        self.rest_p = np.array([m.ExtractTranslation() for m in self.skeleton.GetRestTransformsAttr().Get()])
        self.end_frame = int(self.source.GetEndTimeCode())
        lp, lq, wp, wq = self.pose(0)
        self.system = s = prepare_system(self.model, self.source, self.joints, self.names,
                                         self.parents, wp[None], wq[None])
        local_hair=self.rig_settings.get('appearance',{}).get('local_hair_support',{})
        prefixes=tuple(local_hair.get('bone_prefixes',[]))
        if local_hair.get('pin_roots') and prefixes:
            for node,bone in enumerate(s['nodes']):
                name=self.names[bone]
                if name.startswith(prefixes) and name.split('_')[1]=='0':s['invmass'][node]=0
        self.nodes = np.asarray(s['nodes'])
        self.dynamic = np.array([i in s['dynamic'] for i in range(len(self.joints))])
        self.cloth_translation_mask = np.array([n.startswith('qun_') for n in self.names]) if self.rig_settings.get('cloth_translation_only') else None
        self.tangent = np.full(len(self.joints), -1, dtype=np.int64)
        for i in s['dynamic']:
            children = [j for j, p in enumerate(self.parents) if p == i and self.dynamic[j]]
            self.tangent[i] = children[0] if children else (self.parents[i] if self.dynamic[self.parents[i]] else -1)
        cb, offsets, axes = [], [], []
        base_y = min(v[0][1] for v in self.model['vertices'])
        for body in self.model['rigid_bodies']:
            if body['mode'] or body['bone'] < 0:
                continue
            i = s['pmx_to_usd'][body['bone']]
            center = (np.array(body['position'])[[0, 2, 1]]-[0, 0, base_y])*s['scale']
            axis = Rotation.from_euler('xyz', body['rotation']).apply([0, 1, 0])[[0, 2, 1]]
            half = body['size'][1]*s['scale']/2 if body['shape'] == 2 else 0
            cb.append(i); offsets.append(center-s['bone_rest'][i]); axes.append(axis*half)
        self.collider_bones = np.array(cb)
        self.collider_offsets, self.collider_axes = np.array(offsets), np.array(axes)

    def pose(self, frame):
        p = self.rest_p.copy()
        q = np.zeros((len(self.joints), 4)); q[:, 3] = 1
        p[self.columns] = np.asarray(self.animation.GetTranslationsAttr().Get(frame))
        q[self.columns] = np.asarray(self.animation.GetRotationsAttr().Get(frame))
        wp, wq = fk(p, q, self.parents)
        return p, q, wp, wq

    def capsules(self, wp, wq):
        return collider_poses(wp, wq, self.collider_bones, self.collider_offsets, self.collider_axes)

    def configure_hand_contacts(self):
        self.hand_body_indices=[];self.hand_proxy_shapes={}
        if not self.rig_settings.get('appearance',{}).get('hand_skirt_contacts'):return
        proxies=[b for b in self.model['rigid_bodies'] if not b['mode'] and b['bone']>=0]
        for j,b in enumerate(proxies):
            name=self.model['bones'][b['bone']]['name']
            if name in ['左手捩1','右手捩1','左手首','右手首']:self.hand_body_indices.append(j)
            if name not in ['左手首','右手首']:continue
            descendants={b['bone']}
            if self.rig_settings.get('appearance',{}).get('palm_proxy_include_fingers',True):
                for _ in range(8):
                    descendants.update(i for i,bone in enumerate(self.model['bones']) if bone['parent'] in descendants)
            points=[v[0] for v in self.model['vertices'] if sum(w for i,w in zip(v[3],v[4]) if i in descendants)>.5]
            points=(np.asarray(points)[:,[0,2,1]]-np.array([0,0,min(v[0][1] for v in self.model['vertices'])]))*self.system['scale']
            _,_,axes=np.linalg.svd(points-points.mean(0),full_matrices=False)
            axes=axes.T
            if np.linalg.det(axes)<0:axes[:,-1]*=-1
            low,high=np.quantile(points@axes,[.005,.995],axis=0)
            center=axes@((low+high)/2);half=np.maximum((high-low)/2+.002,.007)
            self.collider_offsets[j]=center-self.system['bone_rest'][self.collider_bones[j]]
            self.collider_axes[j]=0
            self.hand_proxy_shapes[j]=(half,Rotation.from_matrix(axes).as_quat())

    def collider_orientations(self,wq,ca,cb):
        q=align_vectors(np.tile([0.,0.,1.],(len(ca),1)),cb-ca).as_quat()
        for i,(_,local) in getattr(self,'hand_proxy_shapes',{}).items():
            q[i]=(Rotation.from_quat(wq[self.collider_bones[i]])*Rotation.from_quat(local)).as_quat()
        return q


class PhysxSecondary:
    def __init__(self, stage, motion, obstacles=True, quality='hair'):
        self.stage, self.motion = stage, motion
        self.runtime_costs = np.zeros(6)
        self.runtime_calls = 0
        self.projection_ms = 0.
        self.quality = quality
        self.hair_quality = quality in ('hair', 'cloth')
        self.track_obstacles = obstacles
        self.physics_hz = int(motion.rig_settings.get('physics_hz',240 if motion.rig_settings.get('skirt_max_stretch') else 120))
        if self.physics_hz not in (120,240,480):raise ValueError('physics_hz must be 120, 240 or 480')
        self.dt, self.time, self.steps = 1/self.physics_hz, 0., 0
        self.physics_scene = scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        api.CreateEnableGPUDynamicsAttr(False)
        api.CreateBroadphaseTypeAttr('MBP')
        api.CreateSolverTypeAttr('PGS')
        api.CreateEnableCCDAttr(True)
        self.paths, self.pinned = [], []
        s = motion.system
        motion.configure_hand_contacts()
        lp, lq, wp, wq = motion.pose(0)
        self.initial_nodes = wp[motion.nodes].copy()
        self.pin_nodes = np.flatnonzero(s['invmass'] == 0)
        self.dynamic_nodes = np.flatnonzero(s['invmass'] > 0)
        self.build_quality_topology()
        self.materials = {}
        if quality != 'legacy':
            for name, static, dynamic in [('Hair', .06 if quality in ('hair', 'cloth') else .12,
                                         .035 if quality in ('hair', 'cloth') else .08), ('Cloth', .08 if quality == 'cloth' else .22, .045 if quality == 'cloth' else .16)]:
                material = UsdShade.Material.Define(stage, '/World/Secondary/Materials/'+name)
                mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                mat.CreateStaticFrictionAttr(static); mat.CreateDynamicFrictionAttr(dynamic)
                mat.CreateRestitutionAttr(0.)
                PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('min')
                self.materials[name] = material
        authored_masks = {s['pmx_to_usd'][b['bone']]: b['mask'] for b in motion.model['rigid_bodies']
                          if b['mode'] and b['bone'] >= 0}
        body_proxies = [b for b in motion.model['rigid_bodies'] if not b['mode'] and b['bone'] >= 0]
        self.node_radii = np.asarray(s['thickness']).copy()
        for i, position in enumerate(self.initial_nodes):
            pin = s['invmass'][i] == 0
            path = f'/World/Secondary/{"Pinned" if pin else "Dynamic"}/n{i:04d}'
            shape = UsdGeom.Sphere.Define(stage, path)
            radius = float(s['thickness'][i])
            if quality in ('hair', 'cloth') and s['groups'][i] == 14:
                name = motion.names[motion.nodes[i]]
                row = int(name.split('_')[1]) if name.startswith('bh_') else 0
                radius = .010-.002*min(row/15., 1.)
            if quality == 'cloth' and s['groups'][i] == 8:
                radius = .024
            profile=motion.rig_settings.get('appearance',{})
            radius=float(profile.get('skirt_radius_m',radius)) if s['groups'][i]==8 else float(profile.get('hair_radius_m',radius)) if s['groups'][i]==14 else radius
            waist=profile.get('skirt_waist_support',{})
            row=int(motion.names[motion.nodes[i]].split('_')[1]) if motion.names[motion.nodes[i]].startswith('qun_') else -1
            radii=waist.get('node_radii_by_row_m',[])
            if 0<=row<len(radii):radius=float(radii[row])
            self.node_radii[i] = radius
            shape.CreateRadiusAttr(radius)
            shape.CreateVisibilityAttr('invisible')
            self.body(shape, position, pin, collision=not pin, mass=.012 if s['groups'][i] == 8 else .004)
            offsets=waist.get('contact_offsets_by_row_m',[])
            if not pin and 0<=row<len(offsets):
                PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim()).CreateContactOffsetAttr(float(offsets[row]))
            if quality != 'legacy' and not pin:
                material = self.materials['Cloth' if s['groups'][i] == 8 else 'Hair']
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose='physics')
            if not pin:
                bone = int(motion.nodes[i])
                while bone not in authored_masks and motion.parents[bone] >= 0:
                    bone = int(motion.parents[bone])
                mask = authored_masks.get(bone, 0xffff)
                group_index = int(s['groups'][i])
                excluded = [Sdf.Path(f'/World/BodyColliders/c{j:02d}') for j, body in enumerate(body_proxies)
                            if not (group_index==8 and j in motion.hand_body_indices) and
                            (not (mask & (1 << body['group'])) or not (body['mask'] & (1 << group_index)))]
                if excluded:
                    UsdPhysics.FilteredPairsAPI.Apply(shape.GetPrim()).CreateFilteredPairsRel().SetTargets(excluded)
            self.paths.append(path)
        # Internal contacts are filtered, while all outside scene colliders
        # remain eligible for native PhysX collision detection.
        group = UsdPhysics.CollisionGroup.Define(stage, '/World/Secondary/NoSelfCollision')
        self.hair_cloth_contacts = bool(motion.rig_settings.get('appearance',{}).get('hair_skirt_contacts',False)) and quality=='cloth'
        if quality in ('hair', 'cloth'):
            hair_nodes = [i for i in self.dynamic_nodes if s['groups'][i] == 14]
            other_nodes = [i for i in self.dynamic_nodes if s['groups'][i] != 14 and not (quality == 'cloth' and s['groups'][i] == 8)]
            group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([Sdf.Path(self.paths[i]) for i in other_nodes])
            hair_group = UsdPhysics.CollisionGroup.Define(stage, '/World/Secondary/HairSelfCollision')
            hair_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([Sdf.Path(self.paths[i]) for i in hair_nodes])
            hair_group.GetCollidersCollectionAPI().CreateExpansionRuleAttr('expandPrims')
            hair_group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
            group.CreateFilteredGroupsRel().SetTargets([group.GetPath(), hair_group.GetPath()])
            if quality == 'cloth':
                cloth_nodes = [i for i in self.dynamic_nodes if s['groups'][i] == 8]
                cloth_group = UsdPhysics.CollisionGroup.Define(stage, '/World/Secondary/ClothSelfCollision')
                cloth_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([Sdf.Path(self.paths[i]) for i in cloth_nodes])
                cloth_group.GetCollidersCollectionAPI().CreateExpansionRuleAttr('expandPrims')
                cloth_group.CreateFilteredGroupsRel().SetTargets([group.GetPath()]+([] if self.hair_cloth_contacts else [hair_group.GetPath()]))
                group.GetFilteredGroupsRel().AddTarget(cloth_group.GetPath())
                if not self.hair_cloth_contacts:hair_group.GetFilteredGroupsRel().AddTarget(cloth_group.GetPath())
                for offset, a in enumerate(cloth_nodes):
                    pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(self.paths[a])).CreateFilteredPairsRel()
                    for b in cloth_nodes[offset+1:]:
                        # Rest neighbours share a fabric patch; only nonlocal patches collide.
                        if np.linalg.norm(self.initial_nodes[a]-self.initial_nodes[b]) < .075:
                            pairs.AddTarget(Sdf.Path(self.paths[b]))
            # Nonlocal hair nodes collide natively. Suppress rest neighbours
            # (and joint-connected pairs) so initial layers cannot fight joints.
            self.hair_excluded_rest_pairs = 0
            for offset, a in enumerate(hair_nodes):
                prim = stage.GetPrimAtPath(self.paths[a])
                pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                for b in hair_nodes[offset+1:]:
                    if np.linalg.norm(self.initial_nodes[a]-self.initial_nodes[b]) < .029:
                        pairs.AddTarget(Sdf.Path(self.paths[b]))
                        self.hair_excluded_rest_pairs += 1
        else:
            group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([Sdf.Path('/World/Secondary/Dynamic')])
            group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
        group.GetCollidersCollectionAPI().CreateExpansionRuleAttr('expandPrims')
        self.joint_count = 0
        for e, (a, b) in enumerate(s['edge_array']):
            if s['invmass'][a]+s['invmass'][b] == 0:
                continue
            na,nb=motion.names[motion.nodes[a]],motion.names[motion.nodes[b]]
            appearance=motion.rig_settings.get('appearance',{})
            drape=appearance.get('skirt_free_drape',appearance.get('free_drape',False)) if na.startswith('qun_') else appearance.get('free_drape',False)
            if appearance.get('skirt_reduced_constraints',False) and na.startswith('qun_') and nb.startswith('qun_'):
                _,ra,ca=na.split('_');_,rb,cb=nb.split('_')
                dr,dc=abs(int(ra)-int(rb)),abs(int(ca)-int(cb));dc=min(dc,21-dc)
                if dr>=2 or dc>=2 or (dc==0 and dr==1):continue
            grid_pair=drape and na.startswith(('qun_','bh_')) and nb.split('_')[0]==na.split('_')[0]
            if grid_pair:
                _,ra,ca=na.split('_');_,rb,cb=nb.split('_')
                dr=abs(int(ra)-int(rb));dc=abs(int(ca)-int(cb))
                if na.startswith('qun_'):dc=min(dc,21-dc)
                # Two-hop chord lengths lock the authored flare/curvature.
                # Local bending forces replace them in the draping profile.
                if dr>=2 or dc>=2:continue
            joint = UsdPhysics.DistanceJoint.Define(stage, f'/World/Secondary/Joints/e{e:04d}')
            joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[a])])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths[b])])
            length = float(s['lengths'][e])
            joint.CreateMinDistanceAttr(length)
            joint.CreateMaxDistanceAttr(length)
            joint.CreateCollisionEnabledAttr(False)
            hair_spring = None
            if quality in ('hair', 'cloth') and s['groups'][a] == 14 and s['groups'][b] == 14:
                na, nb = motion.names[motion.nodes[a]], motion.names[motion.nodes[b]]
                if na.startswith('bh_') and nb.startswith('bh_'):
                    _, ra, ca = na.split('_'); _, rb, cb = nb.split('_')
                    dr, dc = abs(int(ra)-int(rb)), abs(int(ca)-int(cb))
                    if dc:
                        hair_spring = (8. if dr == 0 and dc == 1 else 4., .045)
                if hair_spring is None and s['compliance'][e] >= 1e-5:
                    hair_spring = (8., .07)
            if s['compliance'][e] >= 1e-5 or hair_spring is not None:
                spring = PhysxSchema.PhysxPhysicsDistanceJointAPI.Apply(joint.GetPrim())
                spring.CreateSpringEnabledAttr(True)
                spring.CreateSpringStiffnessAttr(hair_spring[0] if hair_spring else 30.)
                spring.CreateSpringDampingAttr(hair_spring[1] if hair_spring else .3)
            if motion.rig_settings.get('skirt_fabric_relaxation') and s['groups'][a] == 8 and s['groups'][b] == 8:
                ra,ca = map(int,motion.names[motion.nodes[a]].split('_')[1:])
                rb,cb = map(int,motion.names[motion.nodes[b]].split('_')[1:])
                if ca != cb:
                    # Permit sheet shear and modest circumferential stretch;
                    # fixed diagonals otherwise make the lattice a rigid shell.
                    relax=motion.rig_settings.get('appearance',{})
                    joint.GetMinDistanceAttr().Set(length*relax.get('skirt_shear_min',.90))
                    joint.GetMaxDistanceAttr().Set(length*relax.get('skirt_shear_max',1.18))
                    spring=PhysxSchema.PhysxPhysicsDistanceJointAPI.Apply(joint.GetPrim())
                    spring.CreateSpringEnabledAttr(True)
                    spring.CreateSpringStiffnessAttr(relax.get('skirt_lateral_stiffness',18.) if ra==rb else relax.get('skirt_diagonal_stiffness',8.))
                    spring.CreateSpringDampingAttr(.12)
            if grid_pair and dc:
                # Fabric can fold/gather, while its material edges cannot stretch.
                joint.GetMinDistanceAttr().Set(length*.65 if na.startswith('bh_') else -1.)
                joint.GetMaxDistanceAttr().Set(length*1.025)
                PhysxSchema.PhysxPhysicsDistanceJointAPI.Apply(joint.GetPrim()).CreateSpringEnabledAttr(False)
            self.joint_count += 1
        self.tip_limit_count = 0
        if quality in ('hair', 'cloth'):
            by_name = {n:i for i,n in enumerate(motion.names)}
            rest = s['bone_rest']
            for row in range(10,15):
                for col in range(8):
                    if any(f'bh_{r}_{col}' not in by_name for r in (row-1,row,row+1)): continue
                    a,b,c = [by_name[f'bh_{r}_{col}'] for r in (row-1,row,row+1)]
                    l1, l2 = np.linalg.norm(rest[b]-rest[a]), np.linalg.norm(rest[c]-rest[b])
                    v1, v2 = rest[b]-rest[a], rest[c]-rest[b]
                    authored = np.arccos(np.clip(np.dot(v1,v2)/max(l1*l2,1e-12),-1,1))
                    limit = max(np.deg2rad(55), authored+np.deg2rad(15))
                    chord = np.sqrt(max(0.,l1*l1+l2*l2+2*l1*l2*np.cos(limit)))
                    joint = UsdPhysics.DistanceJoint.Define(stage, f'/World/Secondary/TipLimits/t{row}_{col}')
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][a]])])
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][c]])])
                    joint.CreateMinDistanceAttr(float(chord)); joint.CreateMaxDistanceAttr(-1.)
                    joint.CreateCollisionEnabledAttr(False)
                    self.tip_limit_count += 1
                    self.joint_count += 1
        self.cloth_limit_count = 0
        if motion.rig_settings.get('skirt_fabric_relaxation') and not motion.rig_settings.get('appearance',{}).get('skirt_reduced_constraints',False):
            by_name={n:i for i,n in enumerate(motion.names)}
            skirt_indices=[tuple(map(int,n.split('_')[1:])) for n in motion.names if n.startswith('qun_')]
            rows=motion.rig_settings.get('rows',1+max(r for r,c in skirt_indices))
            cols=motion.rig_settings.get('columns',1+max(c for r,c in skirt_indices))
            for row in range(1,rows-1):
                for col in range(cols):
                    a,b,c=[by_name[f'qun_{r}_{col}'] for r in (row-1,row,row+1)]
                    v1=s['bone_rest'][b]-s['bone_rest'][a];v2=s['bone_rest'][c]-s['bone_rest'][b]
                    l1,l2=np.linalg.norm(v1),np.linalg.norm(v2)
                    chord=np.sqrt(l1*l1+l2*l2+2*l1*l2*np.cos(np.deg2rad(55)))
                    joint=UsdPhysics.DistanceJoint.Define(stage,f'/World/Secondary/ClothLimits/r{row}_{col}')
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][a]])])
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][c]])])
                    joint.CreateMinDistanceAttr(float(chord));joint.CreateMaxDistanceAttr(-1.)
                    joint.CreateCollisionEnabledAttr(False)
                    self.cloth_limit_count+=1;self.joint_count+=1
        self.skirt_length_cap_count = 0
        if motion.rig_settings.get('skirt_max_stretch'):
            # D6 radial limits enforce the actual length bound. DistanceJoint
            # dead zones on centimetre-sized links otherwise accumulate visibly.
            ratio=float(motion.rig_settings['skirt_max_stretch'])
            by_name={n:i for i,n in enumerate(motion.names)}
            rows=motion.rig_settings['rows'];cols=motion.rig_settings['columns']
            def length_cap(name, a, b, length):
                joint=UsdPhysics.Joint.Define(stage,'/World/Secondary/LengthCaps/'+name)
                joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][a]])])
                joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][b]])])
                joint.CreateCollisionEnabledAttr(False)
                for axis in ('transX','transY','transZ'):
                    limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                    limit.CreateLowAttr(-float(length));limit.CreateHighAttr(float(length))
                radial=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),'distance')
                radial.CreateLowAttr(0.);radial.CreateHighAttr(float(length))
                limit=PhysxSchema.PhysxLimitAPI.Apply(joint.GetPrim(),'distance')
                limit.CreateStiffnessAttr(0.);limit.CreateDampingAttr(0.)
                self.skirt_length_cap_count+=1;self.joint_count+=1
            for col in range(cols):
                anchor=by_name[f'qun_0_{col}'];total=0.
                for row in range(1,rows):
                    a,b=[by_name[f'qun_{r}_{col}'] for r in (row-1,row)]
                    length=float(np.linalg.norm(s['bone_rest'][a]-s['bone_rest'][b]))
                    total+=length
                    length_cap(f'edge_{row}_{col}',a,b,length*ratio)
                    if row>1: length_cap(f'waist_{row}_{col}',anchor,b,total*ratio)
        self.hair_length_cap_count=0;self.hair_length_links=[]
        hair_ratio=motion.rig_settings.get('hair_max_stretch')
        if hair_ratio:
            ratio=float(hair_ratio)
            if not 1.<=ratio<=1.1:raise ValueError('hair_max_stretch must be between1 and1.1')
            def hair_cap(name,a,b,length):
                joint=UsdPhysics.Joint.Define(stage,'/World/Secondary/HairLengthCaps/'+name)
                joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][a]])])
                joint.CreateBody1Rel().SetTargets([Sdf.Path(self.paths[s['node_index'][b]])])
                joint.CreateCollisionEnabledAttr(False)
                for axis in ('transX','transY','transZ'):
                    limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                    limit.CreateLowAttr(-float(length));limit.CreateHighAttr(float(length))
                radial=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),'distance')
                radial.CreateLowAttr(0.);radial.CreateHighAttr(float(length))
                limit=PhysxSchema.PhysxLimitAPI.Apply(joint.GetPrim(),'distance')
                limit.CreateStiffnessAttr(0.);limit.CreateDampingAttr(0.)
                self.hair_length_cap_count+=1;self.joint_count+=1
            for node in self.dynamic_nodes:
                if s['groups'][node]!=14:continue
                bone=int(motion.nodes[node]);parent=int(motion.parents[bone])
                if parent not in s['node_index']:continue
                length=float(np.linalg.norm(s['bone_rest'][bone]-s['bone_rest'][parent]))
                if length<1e-6:continue
                hair_cap(f'edge_{bone}',parent,bone,length*ratio)
                self.hair_length_links.append((s['node_index'][parent],node,length*ratio))
                anchor=parent;total=length
                while s['invmass'][s['node_index'][anchor]]>0:
                    ancestor=int(motion.parents[anchor])
                    if ancestor not in s['node_index']:break
                    total+=float(np.linalg.norm(s['bone_rest'][anchor]-s['bone_rest'][ancestor]));anchor=ancestor
                if anchor!=parent:hair_cap(f'root_{bone}',anchor,bone,total*ratio)
        ca, cb = motion.capsules(wp, wq)
        body_orientations=motion.collider_orientations(wq,ca,cb)
        for i, (a, b, radius) in enumerate(zip(ca, cb, s['radii'])):
            length = np.linalg.norm(b-a)
            path = f'/World/BodyColliders/c{i:02d}'
            palm=motion.hand_proxy_shapes.get(i)
            if palm:
                shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(2.)
            elif length > 1e-6:
                shape = UsdGeom.Capsule.Define(stage, path)
                shape.CreateAxisAttr('Z')
                shape.CreateHeightAttr(float(length))
            else:
                shape = UsdGeom.Sphere.Define(stage, path)
            radius*=motion.rig_settings.get('appearance',{}).get('body_radius_scale',1.)
            if not palm:shape.CreateRadiusAttr(float(radius))
            shape.CreateVisibilityAttr('invisible')
            orientation = body_orientations[i]
            self.body(shape, (a+b)/2, True, orientation=orientation)
            if palm:UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*palm[0]))
        self.edge_nodes=[];self.edge_paths=[];self.skirt_bend_joint_count=0
        edge_group=None
        if self.hair_cloth_contacts:
            edge_group=UsdPhysics.CollisionGroup.Define(stage,'/World/Secondary/ClothHairEdges')
            edge_group.GetCollidersCollectionAPI().CreateExpansionRuleAttr('expandPrims')
            edge_group.CreateFilteredGroupsRel().SetTargets([group.GetPath(),cloth_group.GetPath(),edge_group.GetPath()])
            group.GetFilteredGroupsRel().AddTarget(edge_group.GetPath())
            cloth_group.GetFilteredGroupsRel().AddTarget(edge_group.GetPath())
        if motion.rig_settings.get('appearance',{}).get('skirt_edge_contacts'):
            by_name={n:i for i,n in enumerate(motion.names)}
            for a,b in s['edge_array']:
                na,nb=motion.names[motion.nodes[a]],motion.names[motion.nodes[b]]
                if not (na.startswith('qun_') and nb.startswith('qun_')):continue
                _,ra,ca=na.split('_');_,rb,cb=nb.split('_')
                if ca!=cb or abs(int(ra)-int(rb))!=1 or s['invmass'][a]+s['invmass'][b]==0:continue
                # Optional preview-only collision-proxy decimation.
                if int(ca)%int(motion.rig_settings.get('preview_skirt_edge_stride',1)):continue
                e=len(self.edge_nodes);self.edge_nodes.append((a,b))
                pa,pb=self.initial_nodes[[a,b]];length=float(np.linalg.norm(pb-pa))
                path=f'/World/SecondaryEdges/Dynamic/e{e:04d}';self.edge_paths.append(path)
                shape=UsdGeom.Capsule.Define(stage,path);shape.CreateAxisAttr('Z');shape.CreateHeightAttr(length)
                shape.CreateRadiusAttr(float(motion.rig_settings.get('appearance',{}).get('skirt_edge_radius_m',.003)));shape.CreateVisibilityAttr('invisible')
                orientation=align_vectors(np.array([[0.,0.,1.]]),np.array([pb-pa])).as_quat()[0]
                appearance=motion.rig_settings.get('appearance',{})
                self.body(shape,(pa+pb)/2,False,mass=float(appearance.get('skirt_edge_mass_kg',.0005)),orientation=orientation)
                edge_px=PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim())
                if 'skirt_edge_linear_damping' in appearance:edge_px.CreateLinearDampingAttr(float(appearance['skirt_edge_linear_damping']))
                if 'skirt_edge_angular_damping' in appearance:edge_px.CreateAngularDampingAttr(float(appearance['skirt_edge_angular_damping']))
                if 'skirt_edge_max_pushout_m_s' in appearance:edge_px.CreateMaxDepenetrationVelocityAttr(float(appearance['skirt_edge_max_pushout_m_s']))
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials['Cloth'],materialPurpose='physics')
                appearance=motion.rig_settings.get('appearance',{})
                allowed=set(motion.hand_body_indices)
                if appearance.get('skirt_edge_body_contacts',False) and min(int(ra),int(rb))>=appearance.get('skirt_edge_body_min_row',0):
                    names=appearance.get('skirt_edge_body_bones')
                    allowed.update(j for j,body in enumerate(body_proxies) if names is None or motion.model['bones'][body['bone']]['name'] in names)
                excluded=[Sdf.Path(f'/World/BodyColliders/c{j:02d}') for j in range(len(body_proxies)) if j not in allowed]
                UsdPhysics.FilteredPairsAPI.Apply(shape.GetPrim()).CreateFilteredPairsRel().SetTargets(excluded)
                (edge_group or group).GetCollidersCollectionAPI().GetIncludesRel().AddTarget(Sdf.Path(path))
                for endpoint,sign in [(a,-1),(b,1)]:
                    joint=UsdPhysics.SphericalJoint.Define(stage,f'/World/SecondaryEdges/Joints/e{e:04d}_{endpoint}')
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(self.paths[endpoint])]);joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
                    joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,sign*length/2));joint.CreateCollisionEnabledAttr(False)
                    self.joint_count+=1
            cone=motion.rig_settings.get('appearance',{}).get('skirt_bend_cone_deg')
            if cone is not None:
                incident={}
                for e,(a,b) in enumerate(self.edge_nodes):
                    incident.setdefault(a,[]).append(e);incident.setdefault(b,[]).append(e)
                for node,segments in incident.items():
                    if len(segments)!=2:continue
                    e0,e1=segments
                    joint=UsdPhysics.SphericalJoint.Define(stage,f'/World/SecondaryEdges/BendLimits/n{node:04d}')
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(self.edge_paths[e0])]);joint.CreateBody1Rel().SetTargets([Sdf.Path(self.edge_paths[e1])])
                    for slot,e in enumerate((e0,e1)):
                        a,b=self.edge_nodes[e];length=float(np.linalg.norm(self.initial_nodes[b]-self.initial_nodes[a]))
                        value=Gf.Vec3f(0,0,(-1 if node==a else 1)*length/2)
                        (joint.CreateLocalPos0Attr if slot==0 else joint.CreateLocalPos1Attr)(value)
                    joint.CreateAxisAttr('Z');joint.CreateConeAngle0LimitAttr(float(cone));joint.CreateConeAngle1LimitAttr(float(cone))
                    soft=motion.rig_settings.get('appearance',{}).get('skirt_bend_soft_limit')
                    if soft:
                        limit=PhysxSchema.PhysxLimitAPI.Apply(joint.GetPrim(),'cone')
                        limit.CreateStiffnessAttr(float(soft['stiffness_per_degree']))
                        limit.CreateDampingAttr(float(soft['damping_per_degree']))
                        limit.CreateRestitutionAttr(0.)
                    joint.CreateCollisionEnabledAttr(False)
                    self.skirt_bend_joint_count+=1;self.joint_count+=1
            self.edge_nodes=np.asarray(self.edge_nodes,dtype=np.int64)
        # These are ordinary USD scene colliders, not custom contact geometry.
        self.obstacles = []
        for name, center, radius, color in [
            ('SkirtSphere', (0, -.24, .65), .13, (.2, .55, .9)),
            ('HairSphere', (0, .27, 1.22), .12, (.85, .4, .15))]:
            shape = UsdGeom.Sphere.Define(stage, '/World/Obstacles/'+name)
            shape.CreateRadiusAttr(radius)
            shape.CreateDisplayColorAttr([color])
            UsdGeom.Xformable(shape).AddTranslateOp().Set(Gf.Vec3d(*center))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(obstacles)
            if not obstacles:
                shape.CreateVisibilityAttr('invisible')
            self.obstacles.append(shape.GetPrim())
        floor = UsdGeom.Cube.Define(stage, '/World/PhysxGround')
        floor.CreateSizeAttr(1)
        xf = UsdGeom.Xformable(floor)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.04))
        xf.AddScaleOp().Set(Gf.Vec3f(10, 10, .05))
        floor.CreateDisplayColorAttr([(.18, .21, .26)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        self.body_contact_peak = 0.
        self.hand_contact_peak=0.;self.hand_contact_steps=0;self.edge_hand_contact_steps=0
        self.hair_cloth_contact_peak=0.;self.hair_cloth_contact_steps=0
        self.body_contact_steps = 0
        self.contact_peak = 0.
        self.contact_steps = 0

    def build_quality_topology(self):
        motion, s = self.motion, self.motion.system
        count = len(motion.joints)
        by_name = {n:j for j,n in enumerate(motion.names)}
        children = {i: [] for i in range(count)}
        for i, p in enumerate(motion.parents):
            if p >= 0 and motion.dynamic[i]:
                children[p].append(i)
        self.left = np.arange(count); self.right = np.arange(count)
        self.left2 = np.arange(count); self.right2 = np.arange(count)
        for i in s['dynamic']:
            parent = int(motion.parents[i])
            self.left[i] = parent if parent >= 0 and motion.dynamic[parent] else i
            self.right[i] = children[i][0] if children[i] else i
        for i in s['dynamic']:
            self.left2[i] = self.left[self.left[i]]
            self.right2[i] = self.right[self.right[i]]
        triplets, stiffness, damping, groups = [], [], [], []
        seen = set()
        def add(a, b, c, group, k, d):
            if any(i not in s['node_index'] for i in (a, b, c)) or len({a,b,c}) != 3:
                return
            key = (min(a,c), b, max(a,c))
            if key in seen: return
            seen.add(key)
            triplets.append([s['node_index'][i] for i in (a,b,c)])
            stiffness.append(k); damping.append(d); groups.append(group)
        for i in s['dynamic']:
            group = int(s['groups'][s['node_index'][i]])
            if group not in (8, 14): continue
            p = int(motion.parents[i])
            for child in children[i]:
                if group == 14:
                    # Hair root keeps its authored curve; tips remain softer.
                    parts = motion.names[i].split('_')
                    row = int(parts[1]) if len(parts) == 3 and parts[0] == 'bh' else 4
                    k = (8.-2.*min(row/15., 1.)) if self.hair_quality else (6.-3.5*min(row/15., 1.))
                    add(p, i, child, group, k, .008 if self.hair_quality else .006)
                else:
                    add(p, i, child, group, 8. if self.quality == 'cloth' else 2.5, .025 if self.quality == 'cloth' else .01)
            name = motion.names[i]
            if name.startswith(('qun_', 'bh_')):
                prefix, row, col = name.split('_'); row, col = int(row), int(col)
                columns = 21 if prefix == 'qun' else 8
                if prefix == 'qun' or 0 < col < columns-1:
                    a = by_name.get(f'{prefix}_{row}_{(col-1)%columns}', -1)
                    c = by_name.get(f'{prefix}_{row}_{(col+1)%columns}', -1)
                    add(a, i, c, group, (6. if self.quality == 'cloth' else 4.) if group == 8 else (.8 if self.hair_quality else 2.),
                        .008 if group == 8 else .004)
        self.skin_left = self.skin_right = None
        if self.quality == 'cloth':
            self.skin_left = np.arange(count); self.skin_right = np.arange(count)
            for i,name in enumerate(motion.names):
                if name.startswith('qun_'):
                    _,row,col = name.split('_'); col = int(col)
                    self.skin_left[i] = by_name[f'qun_{row}_{(col-1)%21}']
                    self.skin_right[i] = by_name[f'qun_{row}_{(col+1)%21}']
        self.curvature_triplets = np.array(triplets, dtype=np.int64)
        self.curvature_stiffness = np.array(stiffness)
        self.curvature_damping = np.array(damping)
        self.curvature_groups = np.array(groups)
        profile=motion.rig_settings.get('appearance',{})
        self.curvature_stiffness*=np.where(self.curvature_groups==8,profile.get('skirt_bend_scale',1.),profile.get('hair_bend_scale',1.))
        if 'front_hair_bend_scale' in profile:
            front=np.array([g==14 and not motion.names[motion.nodes[b]].startswith('bh_') for (_,b,_),g in zip(self.curvature_triplets,self.curvature_groups)])
            self.curvature_stiffness[front]*=profile['front_hair_bend_scale']/max(profile.get('hair_bend_scale',1.),1e-8)
        local_hair=profile.get('local_hair_support',{});prefixes=tuple(local_hair.get('bone_prefixes',[]))
        if prefixes:
            selected=np.array([motion.names[motion.nodes[b]].startswith(prefixes) for _,b,_ in self.curvature_triplets])
            self.curvature_stiffness[selected]*=float(local_hair.get('bend_multiplier',1.))
        nodes = self.motion.nodes
        a,b,c = self.curvature_triplets.T
        rest = s['bone_rest'][nodes]
        self.curvature_lengths = np.maximum((np.linalg.norm(rest[a]-rest[b],axis=1)+
                                            np.linalg.norm(rest[c]-rest[b],axis=1))/2, .005)
        self.cloth_triplets = self.curvature_triplets[self.curvature_groups == 8]
        cloth_nodes = np.flatnonzero(s['groups'] == 8)
        sa,sb = np.triu_indices(len(cloth_nodes),1)
        far = np.linalg.norm(rest[cloth_nodes[sa]]-rest[cloth_nodes[sb]],axis=1) > .10
        self.cloth_pair_a, self.cloth_pair_b = cloth_nodes[sa[far]], cloth_nodes[sb[far]]
        self.quality_samples = []
        self.tip_triplets = []
        for row in range(10,15):
            for col in range(8):
                if all(f'bh_{r}_{col}' in by_name for r in (row-1,row,row+1)):
                    self.tip_triplets.append([s['node_index'][by_name[f'bh_{r}_{col}']] for r in (row-1,row,row+1)])
        if not self.tip_triplets:
            for i in s['dynamic']:
                if s['groups'][s['node_index'][i]] != 14: continue
                parent = motion.parents[i]
                grandparent = motion.parents[parent] if parent >= 0 else -1
                if parent in s['node_index'] and grandparent in s['node_index']:
                    self.tip_triplets.append([s['node_index'][j] for j in (grandparent,parent,i)])
        self.tip_triplets = np.array(self.tip_triplets)
        self.hair_metric_nodes = np.array([i for i in range(len(motion.nodes)) if s['groups'][i] == 14])
        ha,hb = np.triu_indices(len(self.hair_metric_nodes), 1)
        distances = np.linalg.norm(rest[self.hair_metric_nodes[ha]]-rest[self.hair_metric_nodes[hb]],axis=1)
        self.nonlocal_hair_a = self.hair_metric_nodes[ha[distances > .05]]
        self.nonlocal_hair_b = self.hair_metric_nodes[hb[distances > .05]]

    def body(self, shape, position, kinematic, collision=True, mass=.01, orientation=None):
        prim = shape.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        q = [0, 0, 0, 1] if orientation is None else orientation
        xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(bool(kinematic))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            if not kinematic:
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr(.002)
            col.CreateRestOffsetAttr(0.)
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        px.CreateSolverPositionIterationCountAttr(int(self.motion.rig_settings.get('native_solver_position_iterations',
            255 if self.motion.rig_settings.get('skirt_max_stretch') else 96 if self.quality == 'cloth' else 40 if self.hair_quality else 24)))
        px.CreateSolverVelocityIterationCountAttr(int(self.motion.rig_settings.get('native_solver_velocity_iterations',
            16 if self.quality == 'cloth' else 8 if self.hair_quality else 4)))
        px.CreateLinearDampingAttr((4.0 if mass == .012 else 4.5) if self.hair_quality else
                                 (4.0 if mass == .012 else 4.2) if self.quality == 'smooth' else
                                 (3.3 if mass == .012 else 2.2))
        px.CreateMaxLinearVelocityAttr(4.)
        px.CreateMaxDepenetrationVelocityAttr(3. if self.quality == 'cloth' and mass == .012 else 1.)
        px.CreateMaxAngularVelocityAttr(100.)
        profile=self.motion.rig_settings.get('appearance',{})
        damping_key='skirt_linear_damping' if mass==.012 else 'hair_linear_damping' if mass==.004 else None
        if not kinematic and damping_key in profile:px.CreateLinearDampingAttr(float(profile[damping_key]))
        px.CreateEnableCCDAttr(not bool(kinematic))
        contact = self.motion.rig_settings.get('skirt_contact_smoothing')
        if contact and not kinematic and mass == .012:
            col.CreateContactOffsetAttr(float(contact['contact_offset_m']))
            px.CreateMaxDepenetrationVelocityAttr(float(contact['max_pushout_m_s']))
            px.CreateLinearDampingAttr(float(contact['linear_damping']))


    def set_secondary_mode(self, mode, base_settings):
        """Update native iteration counts in place between simulation batches."""
        from copy import deepcopy
        from secondary_profiles import apply_secondary_mode
        settings=deepcopy(apply_secondary_mode(base_settings,mode))
        position=int(settings.get('native_solver_position_iterations',
            255 if settings.get('skirt_max_stretch') else 96 if self.quality=='cloth' else 40 if self.hair_quality else 24))
        velocity=int(settings.get('native_solver_velocity_iterations',16 if self.quality=='cloth' else 8 if self.hair_quality else 4))
        if not hasattr(self,'_solver_budget_attrs'):
            self._solver_budget_attrs=[]
            for view in (self.dynamic_view,self.pin_view,self.body_view,self.edge_view):
                if view is None:continue
                for path in view.prim_paths:
                    api=PhysxSchema.PhysxRigidBodyAPI(self.stage.GetPrimAtPath(path))
                    self._solver_budget_attrs.append((api.GetSolverPositionIterationCountAttr(),api.GetSolverVelocityIterationCountAttr()))
        with Sdf.ChangeBlock():
            for pos,vel in self._solver_budget_attrs:pos.Set(position);vel.Set(velocity)
        self.interface.flush_changes()
        self.motion.rig_settings=settings
        self.hair_constraint_solver=settings.get('hair_constraint_solver')

    def start(self, stage_id):
        import omni.physx
        import omni.physics.tensors as tensors
        import carb
        import omni.kit.app
        from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
        self.settings = carb.settings.get_settings()
        self.write_setting = SETTING_UPDATE_TO_USD
        self.old_write_to_usd = self.settings.get(SETTING_UPDATE_TO_USD)
        self.settings.set(SETTING_UPDATE_TO_USD, False)
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.fabric', True)
        from omni.physxfabric import get_physx_fabric_interface
        self.fabric = get_physx_fabric_interface()
        self.interface = omni.physx.get_physx_simulation_interface()
        self.interface.attach_stage(stage_id)
        self.fabric.attach_stage(stage_id)
        self.interface.simulate(self.dt, 0.)
        self.interface.fetch_results()
        self.fabric.update(self.dt, self.time)
        self.view = tensors.create_simulation_view('numpy', stage_id, backend='physx')
        self.dynamic_view = self.view.create_rigid_body_view('/World/Secondary/Dynamic/*')
        self.pin_view = self.view.create_rigid_body_view('/World/Secondary/Pinned/*')
        self.body_view = self.view.create_rigid_body_view('/World/BodyColliders/*')
        self.dynamic_order = np.array([int(p.rsplit('n', 1)[1]) for p in self.dynamic_view.prim_paths])
        self.pin_order = np.array([int(p.rsplit('n', 1)[1]) for p in self.pin_view.prim_paths])
        self.body_order = np.array([int(p.rsplit('c', 1)[1]) for p in self.body_view.prim_paths])
        self.edge_view=self.view.create_rigid_body_view('/World/SecondaryEdges/Dynamic/*') if len(self.edge_nodes) else None
        if self.edge_view:
            self.edge_order=np.array([int(p.rsplit('e',1)[1]) for p in self.edge_view.prim_paths])
            self.edge_ids=np.arange(self.edge_view.count,dtype=np.uint32)
        self.contact_view = (self.view.create_rigid_contact_view('/World/Secondary/Dynamic/*',
                            ['/World/Obstacles/SkirtSphere', '/World/Obstacles/HairSphere'], max_contact_data_count=65536)
                             if self.track_obstacles else None)
        # One sensor view owns the filter set; reusing its glob in another view
        # with different filters is unsupported by the native tensor backend.
        contact_filters=list(self.body_view.prim_paths)
        if self.hair_cloth_contacts and self.edge_view:contact_filters+=list(self.edge_paths)
        self.body_contact_view = self.view.create_rigid_contact_view('/World/Secondary/Dynamic/*', contact_filters, max_contact_data_count=65536)
        self.edge_hand_contact_view=(self.view.create_rigid_contact_view('/World/SecondaryEdges/Dynamic/*',
            [f'/World/BodyColliders/c{i:02d}' for i in self.motion.hand_body_indices],max_contact_data_count=65536) if self.edge_view else None)
        self.dynamic_ids = np.arange(self.dynamic_view.count, dtype=np.uint32)
        groups = self.motion.system['groups'][self.dynamic_order]
        self.shape_force_gain = np.where(groups == 8, .012*16., .004*7.).astype(np.float32)
        if self.quality != 'legacy':
            self.shape_force_gain = np.where(groups == 8, .012*35., .004*20.).astype(np.float32)
        if self.hair_quality:
            self.shape_force_gain[groups == 14] = .004*45.
        if self.quality == 'cloth':
            self.shape_force_gain[groups == 8] = .012*45.
        profile=self.motion.rig_settings.get('appearance',{})
        self.shape_force_gain[groups==8]*=profile.get('skirt_shape_scale',1.)
        waist_gains=profile.get('skirt_waist_support',{}).get('shape_gain_by_row',[])
        for slot,node in enumerate(self.dynamic_order):
            name=self.motion.names[self.motion.nodes[node]]
            if name.startswith('qun_'):
                row=int(name.split('_')[1])
                if row<len(waist_gains):self.shape_force_gain[slot]*=float(waist_gains[row])
        self.shape_force_gain[groups==14]*=profile.get('hair_shape_scale',1.)
        cushion=self.motion.rig_settings.get('skirt_approach_cushion')
        self.cushion_bodies=np.array([i for i,bone in enumerate(self.motion.collider_bones) if cushion and self.motion.names[bone] in cushion['body_bones']],dtype=np.int64)
        if 'front_hair_shape_scale' in profile:
            front=np.array([g==14 and not self.motion.names[self.motion.nodes[i]].startswith('bh_') for i,g in zip(self.dynamic_order,groups)])
            self.shape_force_gain[front]*=profile['front_hair_shape_scale']/max(profile.get('hair_shape_scale',1.),1e-8)
        local_hair=profile.get('local_hair_support',{});prefixes=tuple(local_hair.get('bone_prefixes',[]))
        gains=local_hair.get('shape_gain_by_row',[])
        if prefixes:
            for slot,node in enumerate(self.dynamic_order):
                name=self.motion.names[self.motion.nodes[node]]
                if name.startswith(prefixes):
                    row=int(name.split('_')[1])
                    if row<len(gains):self.shape_force_gain[slot]*=float(gains[row])
        self.pin_ids = np.arange(self.pin_view.count, dtype=np.uint32)
        self.body_ids = np.arange(self.body_view.count, dtype=np.uint32)
        self.reset()
        motion=self.motion;stage=self.stage
        self.hair_constraint_solver=motion.rig_settings.get('hair_constraint_solver')
        if self.hair_constraint_solver:
            self.constraint_hair=np.array([i for i in self.dynamic_nodes if motion.system['groups'][i]==14],dtype=np.int64)
            self.constraint_cloth=np.array([i for i in self.dynamic_nodes if motion.system['groups'][i]==8],dtype=np.int64) if self.hair_cloth_contacts else np.empty(0,dtype=np.int64)
            allowed=[]
            for node in self.constraint_hair:
                prim=stage.GetPrimAtPath(self.paths[node])
                excluded=set(UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()) if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else set()
                allowed.append([Sdf.Path(f'/World/BodyColliders/c{j:02d}') not in excluded for j in range(len(self.body_order))])
            self.constraint_allowed=np.array(allowed,dtype=bool)
            connected={tuple(sorted((int(a),int(b)))) for a,b in motion.system['edge_array']}
            pairs=[]
            for offset,a in enumerate(self.constraint_hair):
                for b in self.constraint_hair[offset+1:]:
                    if (int(a),int(b)) not in connected and np.linalg.norm(self.initial_nodes[a]-self.initial_nodes[b])>=.029:pairs.append((a,b))
            self.constraint_pairs=np.array(pairs,dtype=np.int64).reshape(-1,2)
            self.constraint_links=np.asarray(self.hair_length_links)
            self.constraint_triplets=self.curvature_triplets[self.curvature_groups==14]
            self.constraint_edges=np.asarray(self.edge_nodes,dtype=np.int64).reshape(-1,2) if self.hair_cloth_contacts else np.empty((0,2),dtype=np.int64)
            self.constraint_body_radii=self.motion.system['radii']*self.motion.rig_settings.get('appearance',{}).get('body_radius_scale',1.)
        if self.quality != 'legacy':
            # Compile/load CPU force and skin-frame kernels before the viewer
            # starts its clock or audio; do not advance or alter PhysX state.
            lp,lq,wp,wq = self.motion.pose(0)
            curvature_forces(self.initial_nodes, self.initial_nodes, self.initial_nodes, self.initial_nodes,
                             self.curvature_triplets, self.curvature_stiffness, self.curvature_damping, self.dt)
            reconstruct_smooth(lp,lq,wp,wq,self.motion.parents,self.motion.dynamic,wp,
                self.motion.tangent,self.left,self.right,self.left2,self.right2,self.skin_left,self.skin_right,self.motion.cloth_translation_mask)

    def reset(self):
        transforms = np.zeros((self.dynamic_view.count, 7), dtype=np.float32); transforms[:, 6] = 1
        transforms[:, :3] = self.initial_nodes[self.dynamic_order]
        self.dynamic_view.set_transforms(transforms, self.dynamic_ids)
        self.dynamic_view.set_velocities(np.zeros((len(transforms), 6), dtype=np.float32), self.dynamic_ids)
        self.last_positions = transforms[:, :3].copy()
        self.previous_particles = self.initial_nodes.copy()
        self.previous_shape_target = self.initial_nodes.copy()
        if self.edge_view:
            a,b=self.initial_nodes[self.edge_nodes[:,0]],self.initial_nodes[self.edge_nodes[:,1]]
            state=np.empty((len(a),7),dtype=np.float32);state[:,:3]=(a+b)/2
            state[:,3:]=align_vectors(np.tile([0.,0.,1.],(len(a),1)),b-a).as_quat()
            self.edge_view.set_transforms(state[self.edge_order],self.edge_ids)
            self.edge_view.set_velocities(np.zeros((len(a),6),dtype=np.float32),self.edge_ids)
        lp, lq, wp, wq = self.motion.pose(0)
        pin = np.zeros((len(self.pin_order), 7), dtype=np.float32); pin[:, 6] = 1
        pin[:, :3] = wp[self.motion.nodes[self.pin_order]]
        self.pin_view.set_transforms(pin, self.pin_ids)
        self.pin_view.set_kinematic_targets(pin, self.pin_ids)
        ca, cb = self.motion.capsules(wp, wq)
        body = np.empty((len(ca), 7), dtype=np.float32)
        body[:, :3] = (ca+cb)/2
        body[:, 3:] = self.motion.collider_orientations(wq,ca,cb)
        self.body_view.set_transforms(body[self.body_order], self.body_ids)
        self.body_view.set_kinematic_targets(body[self.body_order], self.body_ids)
        self.contact_release_weight=np.zeros(self.dynamic_view.count)
        self.contact_release_normal=np.zeros((self.dynamic_view.count,3))
        self.contact_release_body=np.zeros(self.dynamic_view.count,dtype=np.int64)
        self.previous_capsules=(ca.copy(),cb.copy())
        self.steps = 0
        self.time = 0.

    def step(self, motion_frame, produce_pose=True, diagnostics=True, compiled_hair=False, optimized_contacts=False):
        tick=perf_counter()
        lp, lq, wp, wq = self.motion.pose(motion_frame)
        target = wp[self.motion.nodes]
        pin = np.zeros((len(self.pin_order), 7), dtype=np.float32); pin[:, 6] = 1
        pin[:, :3] = target[self.pin_order]
        self.pin_view.set_kinematic_targets(pin, self.pin_ids)
        ca, cb = self.motion.capsules(wp, wq)
        body = np.empty((len(ca), 7), dtype=np.float32)
        body[:, :3] = (ca+cb)/2
        body[:, 3:] = self.motion.collider_orientations(wq,ca,cb)
        self.body_view.set_kinematic_targets(body[self.body_order], self.body_ids)
        forces = (target[self.dynamic_order]-self.last_positions)*self.shape_force_gain[:, None]
        cushion=self.motion.rig_settings.get('skirt_approach_cushion')
        if cushion and len(self.cushion_bodies):
            selected=self.motion.system['groups'][self.dynamic_order]==8
            indices=self.cushion_bodies
            old_ca,old_cb=self.previous_capsules
            radii=self.motion.system['radii'][indices]*self.motion.rig_settings.get('appearance',{}).get('body_radius_scale',1.)
            args=(ca[indices],cb[indices],old_ca[indices],old_cb[indices],radii,self.dt,cushion['band_m'])
            forces[selected]+=capsule_approach_damping(self.last_positions[selected],self.dynamic_view.get_velocities()[selected,:3],self.node_radii[self.dynamic_order][selected],*args,cushion['damping_n_s_m'],cushion['max_force_n'],cushion.get('stiffness_n_m',0.))
            if self.edge_view:
                edge_positions=self.edge_view.get_transforms()[:,:3]
                edges=self.edge_nodes[self.edge_order]
                envelope=np.linalg.norm(self.initial_nodes[edges[:,1]]-self.initial_nodes[edges[:,0]],axis=1)*.5+self.motion.rig_settings['appearance'].get('skirt_edge_radius_m',.003)
                mass_ratio=self.motion.rig_settings['appearance'].get('skirt_edge_mass_kg',.0005)/.012
                edge_forces=capsule_approach_damping(edge_positions,self.edge_view.get_velocities()[:,:3],envelope,*args,cushion['damping_n_s_m']*mass_ratio,cushion['max_force_n']*mass_ratio,cushion.get('stiffness_n_m',0.)*mass_ratio)
                self.edge_view.apply_forces(edge_forces.astype(np.float32),self.edge_ids,True)
        release_damping=np.zeros_like(forces)
        release=self.motion.rig_settings.get('skirt_contact_release')
        if release:
            groups=self.motion.system['groups'][self.dynamic_order]
            selected=(groups==8) & (self.contact_release_weight>1e-4)
            if selected.any():
                collider=self.contact_release_body[selected]
                a,b=ca[collider],cb[collider]
                olda,oldb=self.previous_capsules[0][collider],self.previous_capsules[1][collider]
                axis=b-a
                t=np.clip(np.sum((self.last_positions[selected]-a)*axis,axis=1)/np.maximum(np.sum(axis*axis,axis=1),1e-12),0,1)
                surface_velocity=((a+t[:,None]*axis)-(olda+t[:,None]*(oldb-olda)))/self.dt
                velocity=self.dynamic_view.get_velocities()[selected,:3]-surface_velocity
                normal=self.contact_release_normal[selected]
                speed=np.sum(velocity*normal,axis=1)
                # Suppress rebound velocity without adding position attraction.
                # Tangential damping is kept weaker.
                c=2*np.sqrt(.012*self.shape_force_gain[selected])
                damping=(release['normal_ratio']*np.maximum(speed,0)[:,None]*normal
                         + release['tangent_ratio']*(velocity-speed[:,None]*normal))
                release_damping[selected]=c[:,None]*self.contact_release_weight[selected,None]*damping
        self.previous_capsules=(ca.copy(),cb.copy())
        if self.quality != 'legacy':
            current = target.copy(); current[self.dynamic_order] = self.last_positions
            bend = curvature_forces(current, target, self.previous_particles, self.previous_shape_target,
                                    self.curvature_triplets, self.curvature_stiffness,
                                    self.curvature_damping, self.dt)
            forces += bend[self.dynamic_order]
            self.previous_particles = current.copy()
            self.previous_shape_target = target.copy()
        if release:
            forces-=release_damping
        self.dynamic_view.apply_forces(forces.astype(np.float32), self.dynamic_ids, True)
        prepared=perf_counter()
        self.interface.simulate(self.dt, self.time)
        submitted=perf_counter()
        self.interface.fetch_results()
        fetched=perf_counter()
        # Tensor reads below access PhysX directly. Only the visible endpoint
        # needs Fabric publication; intermediate substeps still solve contacts.
        if produce_pose:self.fabric.update(self.dt, self.time)
        simulated=perf_counter()
        self.steps += 1
        self.time += self.dt
        positions = self.dynamic_view.get_transforms()[:, :3].copy()
        if self.hair_length_links:
            particles=target.copy();particles[self.dynamic_order]=positions
            changed=np.zeros(len(particles),dtype=bool)
            if self.hair_constraint_solver:
                from hair_constraints import project_hair
                if optimized_contacts:
                    from hair_constraints import project_hair_fast as project_hair
                config=self.hair_constraint_solver
                projection_started=perf_counter()
                arguments=(target,self.motion.system['invmass'],self.constraint_links,self.constraint_triplets,self.constraint_hair,self.constraint_cloth,self.constraint_edges,self.node_radii,ca,cb,self.constraint_body_radii,self.constraint_allowed,self.constraint_pairs,float(self.motion.rig_settings['hair_max_stretch']),)
                particles=project_hair(particles,*arguments,int(config.get('iterations',24)),float(config.get('bend_relaxation',.12)))
                refinement=int(config.get('contact_refine_iterations',0))
                if refinement:particles=project_hair(particles,*arguments,refinement,0.)
                self.projection_ms+=(perf_counter()-projection_started)*1000
                changed[self.constraint_hair]=True
            # Parent-before-child traversal bounds every material segment, so
            # cumulative strand length cannot grow. Feed it back into PhysX.
            if optimized_contacts:
                from hair_constraints import bound_hair_lengths
                particles,changed=bound_hair_lengths(particles,changed,self.constraint_links if self.hair_constraint_solver else np.asarray(self.hair_length_links))
            else:
                for parent,child,length in self.hair_length_links:
                    delta=particles[child]-particles[parent];distance=np.linalg.norm(delta)
                    if distance>length:
                        particles[child]=particles[parent]+delta*(length/distance)
                        changed[child]=True
            selected=changed[self.dynamic_order]
            if selected.any():
                state=self.dynamic_view.get_transforms().copy()
                state[selected,:3]=particles[self.dynamic_order[selected]]
                velocity=self.dynamic_view.get_velocities().copy()
                if self.hair_constraint_solver and self.hair_constraint_solver.get('velocity_mode')=='native_tangent':
                    # Contact position corrections must not become artificial
                    # outward impulses. Keep native velocities, removing only
                    # material-separating relative velocity near the length cap.
                    node_velocity=np.zeros_like(particles)
                    node_velocity[self.dynamic_order]=velocity[:,:3]
                    node_velocity[self.pin_order]=self.pin_view.get_velocities()[:,:3]
                    weights=self.motion.system['invmass']
                    if compiled_hair:
                        from hair_contact_velocity import limit_chain_velocity
                        node_velocity=limit_chain_velocity(node_velocity,particles,weights,self.constraint_links if self.hair_constraint_solver else np.asarray(self.hair_length_links))
                    else:
                        for _ in range(3):
                            for parent,child,length in self.hair_length_links:
                                delta=particles[child]-particles[parent];distance=np.linalg.norm(delta)
                                total=weights[parent]+weights[child]
                                if distance<length*.98 or distance<1e-10 or total==0:continue
                                normal=delta/distance
                                separation=np.dot(node_velocity[child]-node_velocity[parent],normal)
                                if separation>0:
                                    impulse=normal*(separation/total)
                                    node_velocity[parent]+=weights[parent]*impulse
                                    node_velocity[child]-=weights[child]*impulse
                    if self.hair_constraint_solver.get('contact_velocity'):
                        from hair_contact_velocity import correct_velocity
                        body_velocity=np.empty_like(self.body_view.get_velocities())
                        body_velocity[self.body_order]=self.body_view.get_velocities()
                        node_velocity=correct_velocity(node_velocity,particles,self.constraint_hair,self.node_radii,ca,cb,self.constraint_body_radii,self.constraint_allowed,body_velocity,self.dt)
                    velocity[selected,:3]=node_velocity[self.dynamic_order[selected]]
                    blend=float(self.hair_constraint_solver.get('projection_velocity_blend',0.))
                    if blend>0:
                        projected=(state[selected,:3]-self.last_positions[selected])/self.dt
                        speed=np.linalg.norm(projected,axis=1,keepdims=True)
                        projected*=np.minimum(1.,4./np.maximum(speed,1e-8))
                        velocity[selected,:3]=(1.-blend)*velocity[selected,:3]+blend*projected
                else:
                    linear=(state[selected,:3]-self.last_positions[selected])/self.dt
                    speed=np.linalg.norm(linear,axis=1,keepdims=True)
                    velocity[selected,:3]=linear*np.minimum(1.,4./np.maximum(speed,1e-8))
                # Tensor setters require full-view data even with sparse IDs.
                self.dynamic_view.set_transforms(state,self.dynamic_ids)
                self.dynamic_view.set_velocities(velocity,self.dynamic_ids)
                positions=state[:,:3].copy()
                if produce_pose:self.fabric.update(self.dt,self.time)
        self.last_positions = positions.copy()
        limit=self.motion.rig_settings.get('max_node_distance_from_target_m')
        out_of_bounds=(np.max(np.linalg.norm(positions-target[self.dynamic_order],axis=-1))>limit
                       if limit is not None else np.max(np.abs(positions))>10)
        if not np.isfinite(positions).all() or out_of_bounds:
            raise RuntimeError(f'PhysX secondary motion diverged at substep {self.steps}; max_abs={np.nanmax(np.abs(positions))}')
        particles = target.copy()
        particles[self.dynamic_order] = positions
        p=q=None
        if produce_pose:
            world_nodes = wp.copy(); world_nodes[self.motion.nodes] = particles
            if self.quality != 'legacy':
                p, q, _, _ = reconstruct_smooth(lp, lq, wp, wq, self.motion.parents,
                    self.motion.dynamic, world_nodes, self.motion.tangent,
                    self.left, self.right, self.left2, self.right2, self.skin_left, self.skin_right, self.motion.cloth_translation_mask)
            else:
                p, q, _, _ = reconstruct(lp, lq, wp, wq, self.motion.parents,
                                         self.motion.dynamic, world_nodes, self.motion.tangent)
        reconstructed=perf_counter()
        if diagnostics and self.steps % (self.physics_hz//4) == 0:
            a,b,c = self.curvature_triplets.T
            error = ((particles[a]-2*particles[b]+particles[c])-
                     (target[a]-2*target[b]+target[c]))/self.curvature_lengths[:, None]
            values = {}
            for group, name in [(14, 'hair'), (8, 'skirt')]:
                selected = error[self.curvature_groups == group]
                values[name+'_curvature_rms'] = float(np.sqrt(np.mean(np.sum(selected*selected,axis=1))))
            ta,tb,tc = self.tip_triplets.T
            u,v = particles[tb]-particles[ta], particles[tc]-particles[tb]
            angles = np.rad2deg(np.arccos(np.clip(np.sum(u*v,axis=1)/
                np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-10),-1,1)))
            values['hair_tip_angle_p95_deg'] = float(np.percentile(angles,95))
            values['hair_tip_fold_fraction_over_75deg'] = float(np.mean(angles > 75))
            separation = np.linalg.norm(particles[self.nonlocal_hair_a]-particles[self.nonlocal_hair_b],axis=1)
            values['hair_nonlocal_pairs_under_12mm'] = int(np.sum(separation < .012))
            a,b,c = self.cloth_triplets.T
            u,v = particles[b]-particles[a],particles[c]-particles[b]
            angles = np.rad2deg(np.arccos(np.clip(np.sum(u*v,axis=1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-10),-1,1)))
            values['skirt_fold_fraction_over_90deg'] = float(np.mean(angles > 90))
            values['skirt_angle_p95_deg'] = float(np.percentile(angles,95))
            values['skirt_nonlocal_pairs_under_30mm'] = int(np.sum(np.linalg.norm(particles[self.cloth_pair_a]-particles[self.cloth_pair_b],axis=1) < .030))
            skirt = np.flatnonzero(self.motion.system['groups'] == 8)
            depth = .13+self.node_radii[skirt]-np.linalg.norm(particles[skirt]-np.array([0,-.24,.65]),axis=1)
            values['skirt_scene_sphere_max_proxy_penetration_mm'] = float(max(0.,np.max(depth))*1000) if self.track_obstacles else 0.
            self.quality_samples.append(values)
        contact_forces = self.body_contact_view.get_contact_force_matrix(self.dt)
        body_forces = contact_forces[:,:len(self.body_order)]
        if diagnostics and self.hair_cloth_contacts and self.edge_view:
            hair=self.motion.system['groups'][self.dynamic_order]==14
            peak=float(np.linalg.norm(contact_forces[hair,len(self.body_order):],axis=-1).max())
            self.hair_cloth_contact_peak=max(self.hair_cloth_contact_peak,peak)
            self.hair_cloth_contact_steps+=peak>1e-6
        if diagnostics and self.motion.hand_body_indices:
            skirt=self.motion.system['groups'][self.dynamic_order]==8
            hands=np.isin(self.body_order,self.motion.hand_body_indices)
            hand_peak=float(np.linalg.norm(body_forces[skirt][:,hands],axis=-1).max())
            edge_peak=float(np.linalg.norm(self.edge_hand_contact_view.get_contact_force_matrix(self.dt),axis=-1).max()) if self.edge_hand_contact_view else 0.
            self.hand_contact_peak=max(self.hand_contact_peak,hand_peak,edge_peak)
            self.hand_contact_steps+=max(hand_peak,edge_peak)>1e-6
            self.edge_hand_contact_steps+=edge_peak>1e-6
        if self.motion.rig_settings.get('skirt_contact_release'):
            magnitude=np.linalg.norm(body_forces,axis=-1)
            strongest=np.argmax(magnitude,axis=1)
            peak=magnitude[np.arange(len(magnitude)),strongest]
            active=peak>.005
            release=self.motion.rig_settings['skirt_contact_release']
            decay=np.exp(-self.dt/release['release_seconds'])
            self.contact_release_weight*=decay
            if release.get('rise_seconds',0)>0:
                old_weight=self.contact_release_weight[active]/decay
                self.contact_release_weight[active]=old_weight+(1.-old_weight)*(1.-np.exp(-self.dt/release['rise_seconds']))
            else:self.contact_release_weight[active]=1.
            normals=body_forces[np.arange(len(magnitude))[active],strongest[active]]/peak[active,None]
            if release.get('normal_blend_seconds',0)>0:
                old=self.contact_release_normal[active]
                same=(self.contact_release_body[active]==self.body_order[strongest[active]]) & (np.linalg.norm(old,axis=1)>.5) & (np.sum(old*normals,axis=1)>0)
                blend=1.-np.exp(-self.dt/release['normal_blend_seconds'])
                normals[same]=old[same]*(1.-blend)+normals[same]*blend
                normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-10)
            self.contact_release_normal[active]=normals
            self.contact_release_body[active]=self.body_order[strongest[active]]
        if diagnostics:
            body_peak = float(np.max(np.linalg.norm(body_forces,axis=-1)))
            self.body_contact_peak = max(self.body_contact_peak, body_peak)
            self.body_contact_steps += body_peak > 1e-6
            forces = self.contact_view.get_contact_force_matrix(self.dt) if self.contact_view else np.zeros((1, 3))
            peak = float(np.max(np.linalg.norm(forces, axis=-1)))
            self.contact_peak = max(self.contact_peak, peak)
            self.contact_steps += peak > 1e-6
        self.particles, self.ca, self.cb = particles, ca, cb
        self.runtime_costs += np.array([prepared-tick,submitted-prepared,fetched-submitted,simulated-fetched,reconstructed-simulated,perf_counter()-reconstructed])*1000
        self.runtime_calls += 1
        return p, q

    def close(self):
        if hasattr(self, 'interface'):
            if hasattr(self, 'fabric'):
                self.fabric.detach_stage()
            self.interface.detach_stage()
            self.settings.set(self.write_setting, self.old_write_to_usd)
