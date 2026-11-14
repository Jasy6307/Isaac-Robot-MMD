"""Fixed-step XPBD secondary motion, baked onto a dancing UsdSkel.

PMX hair/skirt bone networks become particles with stretch/bend constraints.
Animated PMX body proxies provide capsule/sphere collisions. This is an offline
particle simulation, not a one-to-one PhysX import of MMD's Bullet constraints.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source


import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
# The bundled Numba and coverage versions have incompatible optional tracing
# APIs. Disable that optional hook only during import in this standalone tool;
# do not change installed packages or the project's training environment.
try:
    import coverage
except ImportError:
    coverage = None
if coverage is not None and not hasattr(getattr(coverage, 'types', None), 'Tracer'):
    sys.modules['coverage'] = None
try:
    from numba import njit
finally:
    if coverage is not None:
        sys.modules['coverage'] = coverage
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from convert_pmx import read_pmx
from load_vmd import align_vectors


@njit(cache=True)
def simulate(targets, capsule_a, capsule_b, capsule_r, inverse_mass, edges,
             lengths, compliance, groups, thickness, substeps, iterations,
             initial_position=None, initial_velocity=None, skip_first=False, gravity=9.81):
    frames, nodes, _ = targets.shape
    result = np.empty_like(targets)
    current = targets[0].copy() if initial_position is None else initial_position.copy()
    velocity = np.zeros_like(current) if initial_velocity is None else initial_velocity.copy()
    dt = 1.0 / (30 * substeps)
    max_speed = 0.0
    result[0] = current
    for frame in range(1 if skip_first else 0, frames):
        previous_frame = max(0, frame - 1)
        for step in range(substeps):
            mix = (step + 1) / substeps
            target = targets[previous_frame] * (1 - mix) + targets[frame] * mix
            ca = capsule_a[previous_frame] * (1 - mix) + capsule_a[frame] * mix
            cb = capsule_b[previous_frame] * (1 - mix) + capsule_b[frame] * mix
            before = current.copy()
            for i in range(nodes):
                if inverse_mass[i] == 0:
                    current[i] = target[i]
                    continue
                stiffness = 16.0 if groups[i] == 8 else 7.0
                damping = 0.973 if groups[i] == 8 else 0.982
                for axis in range(3):
                    force = stiffness * (target[i, axis] - current[i, axis])
                    if axis == 2:
                        force -= gravity
                    velocity[i, axis] = velocity[i, axis] * damping + force * dt
                    current[i, axis] += velocity[i, axis] * dt
            multipliers = np.zeros(len(edges))
            for iteration in range(iterations):
                for e in range(len(edges)):
                    a, b = edges[e]
                    wa, wb = inverse_mass[a], inverse_mass[b]
                    if wa + wb == 0:
                        continue
                    dx = current[b, 0] - current[a, 0]
                    dy = current[b, 1] - current[a, 1]
                    dz = current[b, 2] - current[a, 2]
                    distance = (dx*dx + dy*dy + dz*dz)**0.5
                    if distance < 1e-10:
                        continue
                    alpha = compliance[e] / (dt * dt)
                    change = (-(distance - lengths[e]) - alpha*multipliers[e]) / (wa + wb + alpha)
                    multipliers[e] += change
                    for axis, delta in enumerate((dx, dy, dz)):
                        correction = change * delta / distance
                        current[a, axis] -= wa * correction
                        current[b, axis] += wb * correction
                # Alternate constraints and collision projections; final iteration
                # also projects contacts, so stretch cannot put nodes inside limbs.
                if iteration % 2 == 0 or iteration == iterations - 1:
                    for i in range(nodes):
                        if inverse_mass[i] == 0:
                            continue
                        for c in range(len(capsule_r)):
                            # Earrings and short bangs need their authored scalp
                            # embedding; large skirt/back-hair grids collide fully.
                            if groups[i] == 13:
                                continue
                            vx = cb[c, 0] - ca[c, 0]
                            vy = cb[c, 1] - ca[c, 1]
                            vz = cb[c, 2] - ca[c, 2]
                            dx = current[i, 0] - ca[c, 0]
                            dy = current[i, 1] - ca[c, 1]
                            dz = current[i, 2] - ca[c, 2]
                            denom = vx*vx + vy*vy + vz*vz
                            amount = min(1.0, max(0.0, (dx*vx + dy*vy + dz*vz) / max(denom, 1e-15)))
                            dx -= vx*amount
                            dy -= vy*amount
                            dz -= vz*amount
                            distance = (dx*dx + dy*dy + dz*dz)**0.5
                            radius = capsule_r[c] + thickness[i]
                            if distance < radius:
                                if distance < 1e-10:
                                    dx, dy, dz, distance = 1.0, 0.0, 0.0, 1.0
                                push = radius / distance - 1.0
                                current[i, 0] += dx * push
                                current[i, 1] += dy * push
                                current[i, 2] += dz * push
                        current[i, 2] = max(current[i, 2], thickness[i] + 0.003)
            for i in range(nodes):
                if inverse_mass[i] == 0:
                    velocity[i] = 0
                    continue
                for axis in range(3):
                    velocity[i, axis] = (current[i, axis] - before[i, axis]) / dt
                speed = np.sqrt(np.sum(velocity[i]**2))
                # A collider can cross a very short cloth segment in a single
                # frame. Bound the carried velocity, not the contact position.
                if speed > 6:
                    velocity[i] *= 6 / speed
                max_speed = max(max_speed, min(speed, 6))
        result[frame] = current
    return result, max_speed, velocity


def read_animation(stage, max_frames=0):
    skel = UsdSkel.Skeleton.Get(stage, '/Character/Skeleton')
    anim = UsdSkel.Animation.Get(stage, '/Character/Animation')
    joints = list(skel.GetJointsAttr().Get())
    names = list(skel.GetJointNamesAttr().Get())
    index = {p: i for i, p in enumerate(joints)}
    parent = np.array([index.get(p.rpartition('/')[0], -1) for p in joints])
    count = int(stage.GetEndTimeCode()) + 1
    if max_frames:
        count = min(count, max_frames)
    rest = skel.GetRestTransformsAttr().Get()
    q = np.zeros((count, len(joints), 4))
    q[:, :, 3] = 1
    p = np.tile(np.array([m.ExtractTranslation() for m in rest]), (count, 1, 1))
    columns = [index[path] for path in anim.GetJointsAttr().Get()]
    for frame in range(count):
        rotations = anim.GetRotationsAttr().Get(frame)
        q[frame, columns] = [(r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2], r.GetReal()) for r in rotations]
        p[frame, columns] = np.asarray(anim.GetTranslationsAttr().Get(frame))
    wp, wq = p.copy(), q.copy()
    for i, ancestor in enumerate(parent):
        if ancestor >= 0:
            rotation = Rotation.from_quat(wq[:, ancestor])
            wp[:, i] = wp[:, ancestor] + rotation.apply(p[:, i])
            wq[:, i] = (rotation * Rotation.from_quat(q[:, i])).as_quat()
    return joints, names, parent, p, q, wp, wq


def penetration(points, ca, cb, radii, margin=0.0):
    """Capsule-proxy metric, not an exact triangle/triangle intersection test."""
    worst = np.zeros(len(points))
    for a, b, radius in zip(ca, cb, radii):
        direction = b-a
        t = np.clip(((points-a) @ direction) / max(direction @ direction, 1e-12), 0, 1)
        distance = np.linalg.norm(points - (a + t[:, None]*direction), axis=1)
        worst = np.maximum(worst, radius + margin - distance)
    return dict(count=int(np.sum(worst > .001)), max_m=float(worst.max(initial=0)),
                sum_m=float(worst.sum()))


def prepare_system(model, source, joints, names, parents, wp, wq):
    """Build static topology/proxies from one or many body poses; no simulation."""
    bodies, pmx_bones = model['rigid_bodies'], model['bones']
    frames = len(wp)
    by_name = {name: i for i, name in enumerate(names)}
    pmx_to_usd = {i: by_name[b['name']] for i, b in enumerate(pmx_bones)}
    rest_points = np.asarray(UsdGeom.Mesh.Get(source, '/Character/Body').GetPointsAttr().Get())
    raw_points = np.array([v[0] for v in model['vertices']])
    scale = np.ptp(rest_points[:, 2]) / np.ptp(raw_points[:, 1])
    base_z = raw_points[:, 1].min()
    def convert_point(p):
        return (np.asarray(p)[[0, 2, 1]] - np.array([0, 0, base_z])) * scale
    pmx_by_name = {b['name']: b for b in pmx_bones}
    bone_rest = np.array([convert_point(pmx_by_name[name]['position']) for name in names])
    groups_by_bone = {pmx_to_usd[b['bone']]: b['group'] for b in bodies if b['mode'] and b['bone'] >= 0}
    dynamic = set(groups_by_bone)
    # Include the leaf/tip bones. PMX rigid bodies stop one row before the hem.
    for i, parent in enumerate(parents):
        if parent in dynamic:
            dynamic.add(i)
            groups_by_bone.setdefault(i, groups_by_bone[parent])
    anchors = {parents[i] for i in dynamic if parents[i] >= 0 and parents[i] not in dynamic}
    nodes = sorted(dynamic | anchors)
    node_index = {bone: i for i, bone in enumerate(nodes)}
    groups = np.array([groups_by_bone.get(i, -1) for i in nodes])
    invmass = np.ones(len(nodes))
    invmass[[node_index[i] for i in anchors]] = 0
    # First grid rows are pivots fixed to pelvis/scalp; rotations remain free
    # through the next row of particles.
    for i in dynamic:
        if names[i].startswith(('qun_0_', 'bh_0_')):
            invmass[node_index[i]] = 0
    edges = {}
    def edge(a, b, compliance):
        if a in node_index and b in node_index and a != b:
            key = tuple(sorted((node_index[a], node_index[b])))
            edges[key] = min(edges.get(key, 1), compliance)
    for i in dynamic:
        edge(i, int(parents[i]), 1e-9)
        grandparent = parents[parents[i]] if parents[i] >= 0 else -1
        edge(i, int(grandparent), 3e-5)
    for joint in model['physics_joints']:
        a, b = bodies[joint['a']]['bone'], bodies[joint['b']]['bone']
        if a >= 0 and b >= 0 and pmx_to_usd[a] in dynamic and pmx_to_usd[b] in dynamic:
            edge(pmx_to_usd[a], pmx_to_usd[b], 2e-8)
    # Preserve the cloth sheet's shear and bending resistance as well as its
    # longitudinal chains. This avoids isolated strands opening large gaps.
    for prefix, columns in [('qun', 21), ('bh', 8)]:
        for i in dynamic:
            if not names[i].startswith(prefix + '_'):
                continue
            _, row, col = names[i].split('_')
            row, col = int(row), int(col)
            for dr, dc, compliance in [(0, 1, 2e-8), (1, 1, 8e-8), (0, 2, 4e-5), (2, 0, 4e-5)]:
                next_col = (col+dc) % columns if prefix == 'qun' else col+dc
                other = by_name.get(f'{prefix}_{row+dr}_{next_col}')
                if other is not None:
                    edge(i, other, compliance)
    edge_array = np.array(list(edges), dtype=np.int64)
    lengths = np.linalg.norm(bone_rest[np.array(nodes)[edge_array[:, 1]]] -
                             bone_rest[np.array(nodes)[edge_array[:, 0]]], axis=1)
    compliance = np.array(list(edges.values()))
    targets = wp[:, nodes].copy()
    thickness = np.where(groups == 8, .018, .012)
    # PMX body proxies are animated kinematically by the baked body skeleton.
    ca, cb, radii, collider_names = [], [], [], []
    for body in bodies:
        if body['mode'] or body['bone'] < 0:
            continue
        i = pmx_to_usd[body['bone']]
        rotation = Rotation.from_quat(wq[:, i])
        offset = convert_point(body['position']) - bone_rest[i]
        center = wp[:, i] + rotation.apply(np.tile(offset, (frames, 1)))
        body_rotation = Rotation.from_euler('xyz', body['rotation'])
        axis = body_rotation.apply([0, 1, 0])[[0, 2, 1]]
        half = body['size'][1] * scale / 2 if body['shape'] == 2 else 0
        displacement = rotation.apply(np.tile(axis * half, (frames, 1)))
        ca.append(center-displacement)
        cb.append(center+displacement)
        radii.append(body['size'][0]*scale)
        collider_names.append(body['name'])
    ca, cb = np.stack(ca, axis=1), np.stack(cb, axis=1)
    radii = np.asarray(radii)
    return {key: value for key, value in locals().copy().items() if key in ['nodes', 'dynamic', 'node_index', 'groups', 'invmass', 'edge_array', 'lengths', 'compliance', 'thickness', 'ca', 'cb', 'radii', 'collider_names', 'targets', 'pmx_to_usd', 'rest_points', 'bone_rest', 'scale']}


def bake(args):
    started = time.perf_counter()
    model = read_pmx(args.pmx.resolve(), physics=True)
    bodies, pmx_bones = model['rigid_bodies'], model['bones']
    source = Usd.Stage.Open(str(args.usd.resolve()))
    joints, names, parents, lp, lq, wp, wq = read_animation(source, args.frames)
    frames = len(wp)
    system = prepare_system(model, source, joints, names, parents, wp, wq)
    nodes = system['nodes']
    dynamic = system['dynamic']
    node_index = system['node_index']
    groups = system['groups']
    invmass = system['invmass']
    edge_array = system['edge_array']
    lengths = system['lengths']
    compliance = system['compliance']
    thickness = system['thickness']
    ca = system['ca']
    cb = system['cb']
    radii = system['radii']
    collider_names = system['collider_names']
    targets = system['targets']
    pmx_to_usd = system['pmx_to_usd']
    rest_points = system['rest_points']
    bone_rest = system['bone_rest']
    scale = system['scale']
    print(f'XPBD_START frames={frames} nodes={len(nodes)} edges={len(edge_array)} colliders={len(radii)}', flush=True)
    simulated, max_speed, _ = simulate(targets, ca, cb, radii, invmass, edge_array, lengths,
                                    compliance, groups, thickness, args.substeps, args.iterations)
    if not np.isfinite(simulated).all() or np.max(np.abs(simulated)) > 20:
        raise RuntimeError('Secondary motion diverged')
    # Reconstruct world-space bone rotations from the simulated chain tangents.
    new_wp, new_wq = wp.copy(), wq.copy()
    children = {i: [] for i in range(len(joints))}
    for i, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(i)
    for i in dynamic:
        new_wp[:, i] = simulated[:, node_index[i]]
    for i in sorted(dynamic):
        candidates = [child for child in children[i] if child in dynamic]
        if candidates:
            child = candidates[0]
            baseline = wp[:, child] - wp[:, i]
            direction = new_wp[:, child] - new_wp[:, i]
        elif parents[i] in dynamic:
            parent = parents[i]
            baseline = wp[:, i] - wp[:, parent]
            direction = new_wp[:, i] - new_wp[:, parent]
        else:
            continue
        new_wq[:, i] = (align_vectors(baseline, direction) * Rotation.from_quat(wq[:, i])).as_quat()
    # Non-simulated descendants retain their animation relative to their now
    # simulated parent, rather than being left behind in world space.
    for i, parent in enumerate(parents):
        if parent >= 0 and i not in dynamic:
            rotation = Rotation.from_quat(new_wq[:, parent])
            new_wp[:, i] = new_wp[:, parent] + rotation.apply(lp[:, i])
            new_wq[:, i] = (rotation * Rotation.from_quat(lq[:, i])).as_quat()
    body_bones = sorted({pmx_to_usd[b['bone']] for b in bodies if not b['mode'] and b['bone'] >= 0})
    body_drift = float(np.max(np.abs(new_wp[:, body_bones] - wp[:, body_bones])))
    if body_drift > 1e-6:
        raise RuntimeError('Secondary simulation altered the main body dance')
    for i, parent in enumerate(parents):
        if parent >= 0:
            inverse = Rotation.from_quat(new_wq[:, parent]).inv()
            lp[:, i] = inverse.apply(new_wp[:, i] - new_wp[:, parent])
            lq[:, i] = (inverse * Rotation.from_quat(new_wq[:, i])).as_quat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = Usd.Stage.CreateNew(str(args.output.resolve()))
    out.GetRootLayer().subLayerPaths = [str(args.usd.resolve()).replace('\\', '/')]
    out.SetDefaultPrim(out.GetPrimAtPath('/Character'))
    out.GetDefaultPrim().SetCustomDataByKey('secondaryMotionSolver', 'offline XPBD particles + capsule contacts')
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1)
    animation = UsdSkel.Animation.Define(out, '/Character/Animation')
    animation.CreateJointsAttr(joints)
    qr, tr = animation.CreateRotationsAttr(), animation.CreateTranslationsAttr()
    animation.CreateScalesAttr([Gf.Vec3h(1)] * len(joints))
    for frame in range(frames):
        values = Vt.QuatfArray([Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])) for q in lq[frame]])
        translations = Vt.Vec3fArray.FromNumpy(lp[frame].astype(np.float32))
        qr.Set(values, frame)
        tr.Set(translations, frame)
        if frame == 0:
            qr.Set(values)
            tr.Set(translations)
    out.SetStartTimeCode(0)
    out.SetEndTimeCode(frames-1)
    out.SetTimeCodesPerSecond(30)
    out.SetFramesPerSecond(30)
    out.GetRootLayer().Save()
    # Evaluate final skinned vertices, not just physics particles. Quantify
    # penetration against the SAME body proxy capsules before and after.
    check = Usd.Stage.Open(str(args.output.resolve()))
    caches = []
    for stage in (source, check):
        cache = UsdSkel.Cache()
        cache.Populate(UsdSkel.Root.Get(stage, '/Character'), Usd.PrimDefaultPredicate)
        caches.append((cache, cache.GetSkelQuery(UsdSkel.Skeleton.Get(stage, '/Character/Skeleton')),
                       cache.GetSkinningQuery(stage.GetPrimAtPath('/Character/Body'))))
    weighted = np.array([any(pmx_to_usd[b] in dynamic and w > .2 for b, w in zip(v[3], v[4]) if b >= 0)
                         for v in model['vertices']])
    samples = sorted({f for f in (0, 150, 300, 600, 900, 1500, 2400, 3000, 4500, 6000, frames-1) if f < frames})
    metrics = {}
    for frame in samples:
        values = []
        for cache, query, skin in caches:
            points = Vt.Vec3fArray.FromNumpy(rest_points.astype(np.float32))
            if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(frame), points, frame):
                raise RuntimeError('Physics bake skinning failed')
            array = np.asarray(points)
            if not np.isfinite(array).all():
                raise RuntimeError('Non-finite physics mesh')
            values.append(penetration(array[weighted], ca[frame], cb[frame], radii))
        metrics[frame] = dict(before=values[0], after=values[1])
    before_sum = sum(v['before']['sum_m'] for v in metrics.values())
    after_sum = sum(v['after']['sum_m'] for v in metrics.values())
    reduction = 1-after_sum/max(before_sum, 1e-12)
    if before_sum > 0.1 and reduction < .05:
        raise RuntimeError('Sampled mesh/capsule penetration did not improve')
    edge_lengths = np.linalg.norm(simulated[:, edge_array[:, 1]] - simulated[:, edge_array[:, 0]], axis=2)
    stretch = np.abs(edge_lengths - lengths) / np.maximum(lengths, 1e-6)
    summary = dict(source=str(args.usd), output=str(args.output), solver='offline XPBD particles + capsule contacts',
                   frames=frames, substeps=args.substeps, iterations=args.iterations,
                   pmx_bodies=len(bodies), pmx_joints=len(model['physics_joints']), nodes=len(nodes),
                   dynamic_nodes=len(dynamic), constraints=len(edge_array), body_colliders=collider_names,
                   groups=dict(Counter(groups.tolist())), max_carried_speed_m_s=float(max_speed),
                   main_body_position_drift_m=body_drift,
                   sampled_penetration_sum_reduction_fraction=reduction,
                   constraint_relative_stretch_p95=float(np.percentile(stretch, 95)),
                   vertex_capsule_metrics=metrics, elapsed_seconds=time.perf_counter()-started,
                   limitations=['Not a direct MMD Bullet/PhysX rigid-body import',
                                'Capsule contact approximation; no triangle self-collision',
                                'Fixed 30fps bake with substeps; rerun simulation after motion changes'])
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('XPBD_BAKE_PASS ' + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pmx', type=Path, default=Path('media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'))
    parser.add_argument('--usd', type=Path, default=Path('output/pmx_prototype/ji_le_jing_tu.usdc'))
    parser.add_argument('--output', type=Path, default=Path('output/pmx_prototype/ji_le_jing_tu_physics.usdc'))
    parser.add_argument('--frames', type=int, default=0, help='0: entire dance; positive: bounded trial')
    parser.add_argument('--substeps', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=12)
    bake(parser.parse_args())
