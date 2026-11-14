"""Minimal PMX -> animated UsdSkel prototype, using Isaac Sim's bundled USD.

Mesh/material/bone sections are read; physics=True also reads rigid bodies and
spring joints after skipping morph/display records. SDEF/QDEF are approximated by
linear skinning; morphs, append translation, IK and MMD physics are not evaluated.
Append rotations are evaluated in common rest axes for the demonstration.
The source PMX and textures are never modified.
"""
from __future__ import annotations

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import struct


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, size):
        if size < 0 or self.offset + size > len(self.data):
            raise ValueError(f"Invalid/truncated PMX at byte {self.offset}")
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def unpack(self, fmt):
        values = struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))
        return values[0] if len(values) == 1 else values

    def text(self):
        return self.take(self.unpack('i')).decode(self.encoding)

    def index(self, size, unsigned=False):
        return self.unpack(({1: 'B', 2: 'H', 4: 'I'} if unsigned else
                            {1: 'b', 2: 'h', 4: 'i'})[size])

    def count(self):
        n = self.unpack('i')
        if n < 0 or n > len(self.data):
            raise ValueError(f"Invalid PMX count {n}")
        return n


def read_pmx(path, physics=False):
    r = Reader(path.read_bytes())
    if r.take(4) != b'PMX ':
        raise ValueError('Expected PMX signature')
    version = r.unpack('f')
    if not (abs(version - 2.0) < 0.001 or abs(version - 2.1) < 0.001):
        raise ValueError(f'Unsupported PMX version {version}')
    header = r.take(r.unpack('B'))
    encoding, extra_uv, vertex_size, texture_size, material_size, bone_size, morph_size, rigid_size = header[:8]
    r.encoding = {0: 'utf-16-le', 1: 'utf-8'}[encoding]
    name, english, comment, english_comment = [r.text() for _ in range(4)]
    vertices, kinds = [], Counter()
    for _ in range(r.count()):
        position, normal, uv = r.unpack('3f'), r.unpack('3f'), r.unpack('2f')
        r.take(extra_uv * 16)
        kind = r.unpack('B')
        kinds[kind] += 1
        n = {0: 1, 1: 2, 2: 4, 3: 2, 4: 4}[kind]
        indices = [r.index(bone_size) for _ in range(n)]
        if kind == 0:
            weights = [1.0]
        elif kind in (1, 3):
            w = r.unpack('f')
            weights = [w, 1 - w]
            if kind == 3:
                r.take(36)
        else:
            weights = list(r.unpack('4f'))
        r.take(4)  # edge scale
        vertices.append((position, normal, uv, indices, weights))
    indices = [r.index(vertex_size, True) for _ in range(r.count())]
    if len(indices) % 3 or any(i >= len(vertices) for i in indices):
        raise ValueError('Invalid triangle indices')
    textures = [r.text() for _ in range(r.count())]
    materials = []
    for _ in range(r.count()):
        material_name, material_english = r.text(), r.text()
        diffuse = r.unpack('4f')
        r.take(12 + 4 + 12)
        flags = r.unpack('B')
        r.take(16 + 4)
        texture = r.index(texture_size)
        r.index(texture_size)  # sphere texture
        r.take(1)  # sphere mode
        if r.unpack('B') == 0:
            r.index(texture_size)
        else:
            r.take(1)
        r.text()  # material memo
        materials.append(dict(name=material_name, diffuse=diffuse, flags=flags,
                              texture=texture, count=r.count()))
    bones = []
    for _ in range(r.count()):
        bone_name, bone_english = r.text(), r.text()
        position, parent = r.unpack('3f'), r.index(bone_size)
        layer = r.unpack('i')
        flags = r.unpack('H')
        if flags & 1:
            r.index(bone_size)
        else:
            r.take(12)
        append = None
        if flags & 0x300:
            append = (r.index(bone_size), r.unpack('f'))
        if flags & 0x400:
            r.take(12)
        if flags & 0x800:
            r.take(24)
        if flags & 0x2000:
            r.take(4)
        ik = None
        if flags & 0x20:
            target = r.index(bone_size)
            loops, angle = r.unpack('i'), r.unpack('f')
            links = []
            for _ in range(r.count()):
                link = r.index(bone_size)
                limits = None
                if r.unpack('B'):
                    limits = (r.unpack('3f'), r.unpack('3f'))
                links.append(dict(bone=link, limits=limits))
            ik = dict(target=target, loops=loops, angle=angle, links=links)
        bones.append(dict(name=bone_name, english=bone_english, position=position,
                          parent=parent, flags=flags, append=append, layer=layer, ik=ik))
    if sum(m['count'] for m in materials) != len(indices):
        raise ValueError('Material triangle ranges do not cover the mesh')
    if not bones:
        raise ValueError('PMX has no bones')
    for bone in bones:
        if bone['parent'] < -1 or bone['parent'] >= len(bones):
            raise ValueError('Invalid bone parent')
    result = dict(name=name, english=english, comment=comment,
                vertices=vertices, indices=indices, textures=textures,
                materials=materials, bones=bones, weight_types=dict(kinds))
    if physics:
        for _ in range(r.count()):
            r.text(); r.text()
            r.take(1)
            kind = r.unpack('B')
            sizes = {0: morph_size + 4, 1: vertex_size + 12, 2: bone_size + 28,
                     **{i: vertex_size + 16 for i in range(3, 8)},
                     8: material_size + 113, 9: morph_size + 4, 10: rigid_size + 25}
            r.take(r.count() * sizes[kind])
        for _ in range(r.count()):
            r.text(); r.text(); r.take(1)
            for _ in range(r.count()):
                kind = r.unpack('B')
                r.index(bone_size if kind == 0 else morph_size)
        bodies = []
        for _ in range(r.count()):
            body_name, body_english = r.text(), r.text()
            bone, group, mask = r.index(bone_size), r.unpack('B'), r.unpack('H')
            shape, size = r.unpack('B'), r.unpack('3f')
            position, rotation = r.unpack('3f'), r.unpack('3f')
            mass, linear, angular, restitution, friction = r.unpack('5f')
            mode = r.unpack('B')
            bodies.append(dict(name=body_name, bone=bone, group=group, mask=mask,
                               shape=shape, size=size, position=position, rotation=rotation,
                               mass=mass, linear_damping=linear, angular_damping=angular,
                               restitution=restitution, friction=friction, mode=mode))
        joints = []
        for _ in range(r.count()):
            joint_name, joint_english, kind = r.text(), r.text(), r.unpack('B')
            a, b = r.index(rigid_size), r.index(rigid_size)
            if kind != 0:
                raise ValueError(f'Unsupported PMX physics joint type {kind}')
            values = [r.unpack('3f') for _ in range(8)]
            joints.append(dict(name=joint_name, a=a, b=b, position=values[0], rotation=values[1],
                               lower=values[2], upper=values[3], angular_lower=values[4],
                               angular_upper=values[5], spring=values[6], angular_spring=values[7]))
        result.update(rigid_bodies=bodies, physics_joints=joints)
    return result


def convert(source, output, height=1.6, credits='Bushiroad Games/BungleScrungle/海新_/小麦弥望'):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, Vt

    model = read_pmx(source)
    output.mkdir(parents=True, exist_ok=True)
    vertices, bones = model['vertices'], model['bones']
    min_y = min(v[0][1] for v in vertices)
    model_height = max(v[0][1] for v in vertices) - min_y
    if model_height <= 0 or height <= 0:
        raise ValueError('Model height must be positive')
    scale = height / model_height

    def point(p):
        # PMX: Y up, character front -Z. USD: Z up, character front -Y.
        return Gf.Vec3f(p[0] * scale, p[2] * scale, (p[1] - min_y) * scale)

    order, visiting = [], set()
    def visit(i):
        if i in visiting:
            raise ValueError('Cyclic PMX bone hierarchy')
        if i in order:
            return
        visiting.add(i)
        if bones[i]['parent'] >= 0:
            visit(bones[i]['parent'])
        visiting.remove(i)
        order.append(i)
    for i in range(len(bones)):
        visit(i)
    remap = {old: new for new, old in enumerate(order)}
    paths = {}
    for i in order:
        prefix = paths[bones[i]['parent']] + '/' if bones[i]['parent'] >= 0 else ''
        paths[i] = prefix + f'j{i:03d}'
    joints = [paths[i] for i in order]
    positions = [point(b['position']) for b in bones]
    translations = [positions[i] - positions[bones[i]['parent']] if bones[i]['parent'] >= 0
                    else positions[i] for i in order]
    destination = output / 'character.usdc'
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdSkel.Root.Define(stage, '/Character')
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey('credits', credits)
    skeleton = UsdSkel.Skeleton.Define(stage, '/Character/Skeleton')
    skeleton.CreateJointsAttr(joints)
    skeleton.CreateJointNamesAttr([b['name'] for b in (bones[i] for i in order)])
    skeleton.CreateBindTransformsAttr([Gf.Matrix4d().SetTranslate(Gf.Vec3d(positions[i])) for i in order])
    skeleton.CreateRestTransformsAttr([Gf.Matrix4d().SetTranslate(Gf.Vec3d(t)) for t in translations])
    animation = UsdSkel.Animation.Define(stage, '/Character/Animation')
    animation.CreateJointsAttr(joints)
    animation.CreateTranslationsAttr(translations)
    animation.CreateScalesAttr([Gf.Vec3h(1)] * len(bones))
    rotation_attr = animation.CreateRotationsAttr([Gf.Quatf(1)] * len(bones))
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
    mesh = UsdGeom.Mesh.Define(stage, '/Character/Body')
    points = [point(v[0]) for v in vertices]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(model['indices']) // 3))
    # Axis swap changes handedness; reverse triangle winding.
    faces = [model['indices'][i + k] for i in range(0, len(model['indices']), 3) for k in (2, 1, 0)]
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr([Gf.Vec3f(v[1][0], v[1][2], v[1][1]) for v in vertices])
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateExtentAttr([Gf.Vec3f(-height, -height, -height), Gf.Vec3f(height, height, height * 2)])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
        [Gf.Vec2f(v[2][0], 1 - v[2][1]) for v in vertices])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1))
    binding.CreateSkinningMethodAttr('classicLinear')
    ji, jw = [], []
    for vertex in vertices:
        if any(i not in remap for i, w in zip(vertex[3], vertex[4]) if w > 0):
            raise ValueError('Invalid weighted bone index')
        influences = [(remap[i], max(0.0, w)) for i, w in zip(vertex[3], vertex[4]) if w > 0 and i >= 0]
        if not influences:
            raise ValueError('Invalid weighted bone index')
        total = sum(w for _, w in influences)
        ji.extend([i for i, _ in influences] + [0] * (4 - len(influences)))
        jw.extend([w / total for _, w in influences] + [0.0] * (4 - len(influences)))
    binding.CreateJointIndicesPrimvar(False, 4).Set(ji)
    binding.CreateJointWeightsPrimvar(False, 4).Set(jw)
    texture_paths, missing = {}, []
    for i, texture in enumerate(model['textures']):
        original = (source.parent / texture.replace('\\', '/')).resolve()
        if not original.is_file():
            missing.append(str(original))
            continue
        target = output / 'textures' / f'texture_{i:02d}{original.suffix.lower()}'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        texture_paths[i] = './textures/' + target.name
    face_offset = 0
    for i, m in enumerate(model['materials']):
        material = UsdShade.Material.Define(stage, f'/Character/Looks/Material_{i:02d}')
        material.GetPrim().SetDisplayName(m['name'])
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Surface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m['diffuse'][:3]))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.85)
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(m['diffuse'][3])
        if m['texture'] in texture_paths:
            reader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('UV'))
            reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            tex = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Texture'))
            tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_paths[m['texture']]))
            tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            tex.CreateOutput('a', Sdf.ValueTypeNames.Float)
            shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
            shader.GetInput('opacity').ConnectToSource(tex.ConnectableAPI(), 'a')
            shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(0.1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        n = m['count'] // 3
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, f'material_{i:02d}', 'face',
                                               list(range(face_offset, face_offset + n)), 'materialBind')
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
        face_offset += n
    UsdGeom.Subset.SetFamilyType(mesh, 'materialBind', 'partition')
    # Separate phases make it easy to see which bone is controlled.
    names = {b['name']: remap[i] for i, b in enumerate(bones)}
    controls = [('左腕', (0, 1, 0), -55), ('左ひじ', (0, 0, 1), -85),
                ('左足', (1, 0, 0), -30), ('左ひざ', (1, 0, 0), 65)]
    if any(name not in names for name, _, _ in controls):
        raise ValueError('This prototype expects standard Japanese MMD arm/elbow/leg/knee bone names')
    fps, duration = 30, 8
    for frame in range(fps * duration + 1):
        t = frame / fps
        rotations = [Gf.Quatf(1)] * len(bones)
        for k, (name, axis, degrees) in enumerate(controls):
            phase = t - k * 2
            angle = degrees * math.sin(math.pi * phase / 2) ** 2 if 0 <= phase <= 2 else 0
            rotations[names[name]] = Gf.Quatf(Gf.Rotation(Gf.Vec3d(*axis), angle).GetQuat())
        # This model skins the legs to D bones, which inherit rotation from
        # the visible MMD control bones. Evaluate that dependency explicitly.
        resolved, resolving = {}, set()
        def rotation(i):
            if i in resolved:
                return resolved[i]
            if i in resolving:
                raise ValueError('Cyclic append-rotation dependency')
            resolving.add(i)
            q = rotations[remap[i]]
            append = bones[i]['append']
            if bones[i]['flags'] & 0x100 and append:
                source_index, influence = append
                inherited = Gf.Slerp(influence, Gf.Quatf(1), rotation(source_index))
                q = q * inherited
            resolving.remove(i)
            resolved[i] = q
            return q
        rotations = [rotation(i) for i in order]
        rotation_attr.Set(rotations, frame)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(fps * duration)
    stage.GetRootLayer().Save()
    # Verify actual skin deformation with USD's CPU evaluator, independent of rendering.
    cache = UsdSkel.Cache()
    cache.Populate(root, Usd.PrimDefaultPredicate)
    query = cache.GetSkelQuery(skeleton)
    skin = cache.GetSkinningQuery(mesh.GetPrim())
    displacements = {}
    for frame in (0, 30, 90, 150, 210, 240):
        deformed = Vt.Vec3fArray(points)
        if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(frame), deformed, frame):
            raise RuntimeError('USD skinning evaluation failed')
        displacements[str(frame)] = max((a - b).GetLength() for a, b in zip(points, deformed))
    if displacements['0'] > 1e-4 or displacements['240'] > 1e-4:
        raise RuntimeError('Bind pose does not reproduce source mesh')
    if any(displacements[str(f)] < 0.005 for f in (30, 90, 150, 210)):
        raise RuntimeError('One of the demo bones did not deform the mesh')
    report = dict(source=str(source), usd=str(destination), name=model['name'],
                  vertices=len(vertices), triangles=len(faces)//3, bones=len(bones),
                  materials=len(model['materials']), height_m=height,
                  weight_types=model['weight_types'], missing_textures=missing,
                  controls=[c[0] for c in controls], max_displacement_m=displacements,
                  limitations=['animation only; no physics articulation', 'linear skinning approximation for SDEF/QDEF',
                               'no morphs, IK, append translation, toon/sphere shaders or hair/skirt physics',
                               'append rotation evaluated in common rest axes for this demo'],
                  bone_map=[dict(index=remap[i], pmx_index=i, name=bones[i]['name'], joint=paths[i]) for i in order])
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'bone_map'}, ensure_ascii=False, indent=2))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pmx', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/pmx_prototype'))
    parser.add_argument('--height', type=float, default=1.6)
    parser.add_argument('--credits', default='Bushiroad Games/BungleScrungle/海新_/小麦弥望')
    args = parser.parse_args()
    convert(args.pmx.resolve(), args.output.resolve(), args.height, args.credits)
