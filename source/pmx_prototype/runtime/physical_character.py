"""Build a separate, floating-base humanoid from the PMX rest skeleton.

Z up, character forward -Y, left +X. Masses/limits are engineering starting
values, not measurements of the fictional character. Source assets stay read-only.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel, PhysxSchema, Vt


def build(stage, character_usd, fixed=False, lift=0.003, articulated_neck=False, articulated_wrists=False):
    source = Usd.Stage.Open(str(character_usd.resolve()))
    skel = UsdSkel.Skeleton.Get(source, '/Character/Skeleton')
    names = list(skel.GetJointNamesAttr().Get())
    paths = list(skel.GetJointsAttr().Get())
    rest = np.array([list(m.ExtractTranslation()) for m in skel.GetBindTransformsAttr().Get()])
    parents = [paths.index(p.rsplit('/', 1)[0]) if '/' in p else -1 for p in paths]
    points = dict(zip(names, rest))
    required = ['下半身', '上半身', '上半身2', '首', '頭']
    required += [s+n for s in ('左', '右') for n in ('足', 'ひざ', '足首', 'つま先', '腕', 'ひじ', '手首')]
    missing = set(required)-points.keys()
    if missing:
        raise ValueError(f'Missing anatomical bones: {missing}')
    root = '/World/Anon'
    UsdGeom.Xform.Define(stage, root)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root))
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSolverPositionIterationCountAttr(16)
    art.CreateSolverVelocityIterationCountAttr(4)
    visual = UsdSkel.Root.Define(stage, root+'/Visual').GetPrim()
    visual.GetReferences().AddReference(str(character_usd.resolve()))
    # An independent live animation wins over every referenced demo time sample.
    anim = UsdSkel.Animation.Define(stage, root+'/LivePose')
    anim.CreateJointsAttr(paths)
    anim.CreateScalesAttr([Gf.Vec3h(1)]*len(names))
    live_skel = UsdSkel.Skeleton.Get(stage, root+'/Visual/Skeleton')
    UsdSkel.BindingAPI.Apply(live_skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    material = UsdShade.Material.Define(stage, root+'/FootMaterial')
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(0.9); mat.CreateDynamicFrictionAttr(0.8)
    mat.CreateRestitutionAttr(0.0)
    links, joints, mapping = {}, [], {}

    def body(name, origin, mass, a=None, b=None, radius=.04, box=None):
        origin = np.asarray(origin, dtype=float)
        path = root+'/Bodies/'+name
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*(origin+[0, 0, lift])))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr(0.0); rb.CreateAngularDampingAttr(0.05)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        ma = UsdPhysics.MassAPI.Apply(prim); ma.CreateMassAttr(mass)
        if box:
            center, size = map(np.asarray, box)
            geom = UsdGeom.Cube.Define(stage, path+'/Collider')
            geom.CreateSizeAttr(1.)
            geom.AddTranslateOp().Set(Gf.Vec3d(*(center-origin)))
            geom.AddScaleOp().Set(Gf.Vec3f(*size))
            inertia = mass/12*np.array([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
            orientation = Rotation.identity()
        elif a is not None:
            a, b = np.asarray(a), np.asarray(b)
            center = (a+b)/2
            length = float(np.linalg.norm(b-a))
            orientation = Rotation.align_vectors([(b-a)/length], [[0, 0, 1]])[0]
            geom = UsdGeom.Capsule.Define(stage, path+'/Collider')
            geom.CreateRadiusAttr(radius); geom.CreateHeightAttr(max(.001, length-2*radius))
            geom.AddTranslateOp().Set(Gf.Vec3d(*(center-origin)))
            q = orientation.as_quat()
            geom.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            # Conservative cylinder approximation including capsule end caps.
            inertia = np.array([mass*(3*radius**2+length**2)/12]*2+[mass*radius**2/2])
        else:
            center = origin
            inertia = np.full(3, mass*.025**2*0.4)
            orientation = Rotation.identity()
            geom = None  # Small positive-inertia intermediate joint frame.
        ma.CreateCenterOfMassAttr(Gf.Vec3f(*(center-origin)))
        ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        q = orientation.as_quat()
        ma.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        if geom:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            geom.CreateDisplayColorAttr([(0.12, .5, .8)])
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose='physics')
        links[name] = dict(path=path, origin=origin.tolist(), mass=mass)
        return name

    def joint(parent, child, anchor, axis=None, limits=None, effort=100., stiffness=180., damping=15.):
        name = child+'_joint'
        path = root+'/Joints/'+name
        j = (UsdPhysics.RevoluteJoint if axis else UsdPhysics.FixedJoint).Define(stage, path)
        j.CreateBody0Rel().SetTargets([links[parent]['path']])
        j.CreateBody1Rel().SetTargets([links[child]['path']])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(anchor)-links[parent]['origin'])))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(anchor)-links[child]['origin'])))
        j.CreateCollisionEnabledAttr(False)
        if axis:
            axis_vector = None if isinstance(axis, str) else np.asarray(axis, dtype=float)
            if axis_vector is not None:
                axis_vector /= np.linalg.norm(axis_vector)
                orientation = Rotation.align_vectors([axis_vector], [[1., 0, 0]])[0].as_quat()
                local = Gf.Quatf(float(orientation[3]), Gf.Vec3f(*orientation[:3]))
                j.CreateLocalRot0Attr(local); j.CreateLocalRot1Attr(local)
            j.CreateAxisAttr(axis if axis_vector is None else 'X'); j.CreateLowerLimitAttr(limits[0]); j.CreateUpperLimitAttr(limits[1])
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'angular')
            # USD angular drive gains are per degree; the tensor/PD interface
            # uses radians. Keep our manifest gains in Nm/rad and Nm s/rad.
            drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(stiffness*np.pi/180.)
            drive.CreateDampingAttr(damping*np.pi/180.); drive.CreateMaxForceAttr(effort)
            drive.CreateTargetPositionAttr(0.)
            PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(float(np.degrees(12.)))
            joints.append(dict(name=name, parent=parent, child=child, axis=axis if axis_vector is None else 'X',
                               limits_deg=limits, effort_nm=effort, stiffness=stiffness, damping=damping))
            if axis_vector is not None: joints[-1]['axis_vector'] = axis_vector.tolist()

    def chain(parent, label, anchor, specs, mass, a=None, b=None, radius=.04, box=None):
        for k, (suffix, axis, limits) in enumerate(specs):
            last = k == len(specs)-1
            child = body(label+'_'+suffix, anchor, mass if last else .06,
                         a=a if last else None, b=b if last else None, radius=radius,
                         box=box if last else None)
            leg = any(part in label for part in ('hip', 'knee', 'ankle'))
            joint(parent, child, anchor, axis, limits, effort=120 if leg else 70,
                  stiffness=1000. if 'ankle' in label else (600. if leg or label == 'waist' else 180.),
                  damping=60. if 'ankle' in label else (35. if leg or label == 'waist' else 15.))
            parent = child
        return parent

    pelvis = body('pelvis', points['下半身'], 8., box=(points['下半身'], [.22, .14, .16]))
    mapping['下半身'] = pelvis
    torso = chain(pelvis, 'waist', points['上半身'], [('yaw', 'Z', [-45, 45]), ('roll', 'Y', [-25, 25]), ('pitch', 'X', [-30, 35])],
                  14., a=points['上半身'], b=points['首'], radius=.095)
    mapping.update({'上半身': torso, '上半身2': torso})
    head = body('head', points['首'], 4., a=points['首'], b=points['頭']+[0, 0, .12], radius=.075)
    neck_parent = torso
    if articulated_neck:
        for suffix, axis, limits in [('yaw', 'Z', [-45, 45]), ('pitch', 'X', [-25, 25])]:
            child = body('neck_'+suffix, points['首'], .06)
            joint(neck_parent, child, points['首'], axis, limits, effort=15., stiffness=60., damping=5.)
            neck_parent = child
    joint(neck_parent, head, points['首']); mapping.update({'首': head, '頭': head})
    for side, jp in [('left', '左'), ('right', '右')]:
        hip, knee, ankle, toe = [points[jp+n] for n in ('足', 'ひざ', '足首', 'つま先')]
        thigh = chain(pelvis, side+'_hip', hip, [('yaw', 'Z', [-45, 45]), ('roll', 'Y', [-40, 40]), ('pitch', 'X', [-100, 45])],
                      5., a=hip, b=knee, radius=.045)
        shin = chain(thigh, side+'_knee', knee, [('pitch', 'X', [-5, 145])], 3., a=knee, b=ankle, radius=.035)
        foot_center = np.array([ankle[0], (toe[1]+ankle[1])/2, .035])
        foot = chain(shin, side+'_ankle', ankle, [('pitch', 'X', [-45, 45]), ('roll', 'Y', [-25, 25])],
                     1., box=(foot_center, [.095, .23, .07]))
        for n, link in [('足', thigh), ('ひざ', shin), ('足首', foot), ('つま先', foot)]:
            mapping[jp+n] = link
            mapping[jp+n+'D'] = link
        shoulder, elbow, wrist = [points[jp+n] for n in ('腕', 'ひじ', '手首')]
        arm = chain(torso, side+'_shoulder', shoulder, [('pitch', 'X', [-100, 100]), ('roll', 'Y', [-100, 100]), ('yaw', 'Z', [-90, 90])],
                    1.5, a=shoulder, b=elbow, radius=.027)
        if articulated_wrists:
            flex = body(side+'_elbow_flex', elbow, .04)
            joint(arm, flex, elbow, 'Z', [-145,145], effort=70., stiffness=60., damping=6.)
            forearm = body(side+'_forearm_twist', elbow, .96, a=elbow, b=wrist, radius=.023)
            along = (wrist-elbow)/np.linalg.norm(wrist-elbow)
            joint(flex, forearm, elbow, along.tolist(), [-90,90], effort=15., stiffness=35., damping=3.)
            bend = np.cross(along, [0.,0,1.]); bend /= np.linalg.norm(bend)
            deviation = np.cross(along, bend)
            parent = forearm
            for suffix, axis in [('pitch',bend),('deviation',deviation)]:
                child = body(side+'_wrist_'+suffix, wrist, .03)
                joint(parent, child, wrist, axis.tolist(), [-45,45], effort=8., stiffness=20., damping=2.)
                parent=child
            hand=body(side+'_hand',wrist,.34,a=wrist,b=wrist+(wrist-elbow)*.22,radius=.025)
            joint(parent,hand,wrist)
            for suffix in ['手捩','手捩1','手捩2','手捩3']:mapping[jp+suffix]=forearm
        else:
            forearm = chain(arm, side+'_elbow', elbow, [('flex', 'Z', [-145, 145])], 1., a=elbow, b=wrist, radius=.023)
            hand = body(side+'_hand', wrist, .4, a=wrist, b=wrist+(wrist-elbow)*.22, radius=.025)
            joint(forearm, hand, wrist)
        mapping.update({jp+'腕': arm, jp+'ひじ': forearm, jp+'手首': hand})
    if fixed:
        j = UsdPhysics.FixedJoint.Define(stage, root+'/SupportForJointTest')
        j.CreateBody1Rel().SetTargets([links[pelvis]['path']])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(points['下半身']+[0, 0, lift])))
    # Exclude all zero-geometry frames and near-neighbour bodies in each limb.
    # Distal bodies from different limbs still collide.
    adjacency = {n: set() for n in links}
    for j in stage.Traverse():
        if j.IsA(UsdPhysics.Joint):
            a = UsdPhysics.Joint(j).GetBody0Rel().GetTargets()
            b = UsdPhysics.Joint(j).GetBody1Rel().GetTargets()
            if a and b:
                na, nb = a[0].name, b[0].name
                adjacency[na].add(nb); adjacency[nb].add(na)
    for name, link in links.items():
        nearby = {name}; frontier = {name}
        for _ in range(4):
            frontier = set().union(*(adjacency[n] for n in frontier))-nearby
            nearby |= frontier
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(link['path'])).CreateFilteredPairsRel().SetTargets(
            [links[n]['path'] for n in sorted(nearby-{name})])
    owners = []
    for i, name in enumerate(names):
        current, owner = i, None
        while current >= 0:
            if names[current] in mapping:
                owner = mapping[names[current]]; break
            current = parents[current]
        owners.append(owner or pelvis)
    manifest = dict(schema='anon_physical_v1', source=str(character_usd.resolve()), root=root,
                    forward_axis='-Y', up_axis='Z', fixed_base=fixed, lift=lift,
                    links=links, joints=joints, total_mass_kg=sum(x['mass'] for x in links.values()),
                    bone_names=names, bone_paths=paths, parents=parents, rest_positions=rest.tolist(),
                    bone_owners=owners, anatomical_mapping=mapping,
                    assumptions='Estimated masses, inertia, torque and limits; not anthropometric measurements.')
    return manifest


class SkinBridge:
    """Rigid skin segment motion from actual PhysX link poses, never animation targets."""
    def __init__(self, stage, manifest, link_names):
        self.m = manifest
        self.rest = np.asarray(manifest['rest_positions'])
        self.parents = manifest['parents']
        self.indices = np.array([list(link_names).index(n) for n in manifest['bone_owners']])
        self.origins = np.array([manifest['links'][n]['origin'] for n in manifest['bone_owners']])
        self.anim = UsdSkel.Animation.Get(stage, manifest['root']+'/LivePose')

    def update(self, transforms):
        t = transforms[self.indices]
        rot = Rotation.from_quat(t[:, 3:])
        wp = t[:, :3]+rot.apply(self.rest-self.origins)
        wq = rot.as_quat()
        lp, lq = wp.copy(), wq.copy()
        for i, p in enumerate(self.parents):
            if p >= 0:
                inverse = Rotation.from_quat(wq[p]).inv()
                lp[i] = inverse.apply(wp[i]-wp[p])
                lq[i] = (inverse*Rotation.from_quat(wq[i])).as_quat()
        self.anim.CreateTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(lp.astype(np.float32)))
        self.anim.CreateRotationsAttr().Set([Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])) for q in lq])
        return wp, wq
