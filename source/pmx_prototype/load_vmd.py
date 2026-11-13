"""Bake a native MMD VMD onto the converted PMX UsdSkel, including leg IK.

Uses PMX rest positions rather than G1 retargeting. Feet use an analytical
two-link solve with PMX knee limits, followed by the one-link toe IK solve.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import unicodedata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from source.paths import resolve_dance_asset

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from convert_pmx import Reader, read_pmx


def name_key(name):
    return unicodedata.normalize('NFKC', name)


def read_motion(path):
    path = resolve_dance_asset(path)
    r = Reader(path.read_bytes())
    header = r.take(30)
    if not header.startswith(b'Vocaloid Motion Data 0002'):
        raise ValueError('Expected VMD 0002 header')
    model_name = r.take(20).split(b'\0')[0].decode('cp932', errors='replace')
    tracks = defaultdict(dict)
    for _ in range(r.count()):
        # VMD's 15-byte field can truncate the final Shift-JIS character.
        name = r.take(15).split(b'\0')[0].decode('cp932', errors='ignore')
        frame = r.unpack('I')
        position, quat, interpolation = r.unpack('3f'), r.unpack('4f'), r.take(64)
        tracks[name_key(name)][frame] = (position, quat, interpolation)
    if not tracks:
        raise ValueError('VMD contains no bone animation')
    morph_count = r.count()
    r.take(morph_count * 23)
    switches = defaultdict(list)
    # Camera/light/shadow blocks precede optional model visibility / IK switches.
    for record_size in (61, 28, 9):
        if r.offset == len(r.data):
            break
        r.take(r.count() * record_size)
    if r.offset < len(r.data):
        for _ in range(r.count()):
            frame, visible, count = r.unpack('I'), r.unpack('B'), r.count()
            for _ in range(count):
                name = r.take(20).split(b'\0')[0].decode('cp932', errors='strict')
                switches[name_key(name)].append((frame, bool(r.unpack('B'))))
    return tracks, switches, dict(model_name=model_name, morph_keyframes=morph_count)


def bezier(progress, parameters):
    """Invert Bezier X, then evaluate Y (VMD destination-key interpolation)."""
    x1, y1, x2, y2 = np.asarray(parameters, dtype=float) / 127.0
    lo, hi = np.zeros_like(progress), np.ones_like(progress)
    for _ in range(22):
        t = (lo + hi) * 0.5
        x = 3*(1-t)**2*t*x1 + 3*(1-t)*t*t*x2 + t**3
        lo = np.where(x < progress, t, lo)
        hi = np.where(x >= progress, t, hi)
    t = (lo + hi) * 0.5
    return 3*(1-t)**2*t*y1 + 3*(1-t)*t*t*y2 + t**3


def sample_track(keys, count):
    frames = sorted(keys)
    pos = np.empty((count, 3))
    quat = np.empty((count, 4))
    pos[:frames[0]+1] = keys[frames[0]][0]
    quat[:frames[0]+1] = keys[frames[0]][1]
    pos[frames[-1]:] = keys[frames[-1]][0]
    quat[frames[-1]:] = keys[frames[-1]][1]
    for a, b in zip(frames, frames[1:]):
        pa, qa, _ = keys[a]
        pb, qb, curve = keys[b]
        if b == a+1:
            # Dense VMD tracks have no interior integer sample to interpolate.
            pos[a], pos[b] = pa, pb
            quat[a], quat[b] = qa, qb
            continue
        progress = np.arange(b-a+1) / (b-a)
        for axis in range(3):
            parameters = [curve[axis + k] for k in (0, 4, 8, 12)]
            pos[a:b+1, axis] = pa[axis] + (pb[axis] - pa[axis]) * bezier(progress, parameters)
        parameters = [curve[3 + k] for k in (0, 4, 8, 12)]
        amount = bezier(progress, parameters)
        quat[a:b+1] = Slerp([0, 1], Rotation.from_quat([qa, qb]))(amount).as_quat()
    return pos, Rotation.from_quat(quat).as_quat()


def align_vectors(source, destination):
    a = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    b = destination / np.maximum(np.linalg.norm(destination, axis=1, keepdims=True), 1e-12)
    q = np.concatenate([np.cross(a, b), (1 + np.sum(a*b, axis=1))[:, None]], axis=1)
    opposite = np.linalg.norm(q, axis=1) < 1e-8
    if opposite.any():
        auxiliary = np.zeros_like(a[opposite])
        auxiliary[:, 0] = 1
        auxiliary[np.abs(a[opposite, 0]) > .9] = [0, 1, 0]
        q[opposite, :3] = np.cross(a[opposite], auxiliary)
        q[opposite, 3] = 0
    return Rotation.from_quat(q)


def bake(pmx, vmd, asset, output, leg_mode='ik', translation_scale=1.0, stance_reference=None, leg_clearance=False):
    vmd = resolve_dance_asset(vmd)
    if leg_mode not in ('fk', 'ik'):
        raise ValueError('leg_mode must be fk or ik')
    model = read_pmx(pmx)
    bones = model['bones']
    tracks, switches, metadata = read_motion(vmd)
    count = max(max(keys) for keys in tracks.values()) + 1
    stage = Usd.Stage.Open(str(asset))
    skeleton = UsdSkel.Skeleton.Get(stage, '/Character/Skeleton')
    names = list(skeleton.GetJointNamesAttr().Get())
    paths = list(skeleton.GetJointsAttr().Get())
    rest = skeleton.GetRestTransformsAttr().Get()
    if len(names) != len(bones) or set(names) != {b['name'] for b in bones}:
        raise ValueError('USD skeleton does not match this PMX')
    by_name = {name_key(b['name']): i for i, b in enumerate(bones)}
    # Recover the exact mesh scale from its vertical bounding range.
    mesh_points = UsdGeom.Mesh.Get(stage, '/Character/Body').GetPointsAttr().Get()
    scale = (max(p[2] for p in mesh_points) - min(p[2] for p in mesh_points)) / (
        max(v[0][1] for v in model['vertices']) - min(v[0][1] for v in model['vertices']))
    identity = np.tile([0., 0., 0., 1.], (count, 1))
    zero = np.zeros((count, 3))
    positions = np.asarray([b['position'] for b in bones])
    parents = [b['parent'] for b in bones]
    local_pos = [positions[i] - positions[p] if p >= 0 else positions[i] for i, p in enumerate(parents)]
    base_q, offsets = {}, {}
    matched = []
    for name, keys in tracks.items():
        if name in by_name:
            i = by_name[name]
            offsets[i], base_q[i] = sample_track(keys, count)
            offsets[i] *= translation_scale
            matched.append(name)
    stance_adjustments = {}
    if stance_reference is not None:
        reference = read_pmx(stance_reference)
        reference_by_name = {name_key(b['name']): b for b in reference['bones']}
        model_height = np.ptp([v[0][1] for v in model['vertices']])
        reference_height = np.ptp([v[0][1] for v in reference['vertices']])
        for i, bone in enumerate(bones):
            # Shift the two leg controllers only; child toe controllers inherit
            # the shift. VMD displacement and rotation curves remain unchanged.
            if (name_key(bone['name']) not in ('左足IK', '右足IK')
                    or not bone['ik'] or len(bone['ik']['links']) != 2):
                continue
            other = reference_by_name.get(name_key(bone['name']))
            if other is None:
                raise ValueError(f"Reference has no leg controller {bone['name']}")
            shift = other['position'][0] * model_height / reference_height - bone['position'][0]
            offsets[i] = offsets.get(i, zero).copy()
            offsets[i][:, 0] += shift
            stance_adjustments[bone['name']] = float(shift * scale)
    q_cache, world_cache = {}, {}
    def local_rotation(i):
        if i not in q_cache:
            q = Rotation.from_quat(base_q.get(i, identity))
            append = bones[i]['append']
            if append and bones[i]['flags'] & 0x100:
                source, influence = append
                q = q * Rotation.from_rotvec(local_rotation(source).as_rotvec() * influence)
            q_cache[i] = q
        return q_cache[i]

    def world(i):
        if i < 0:
            return zero, Rotation.from_quat(identity)
        if i not in world_cache:
            pp, pq = world(parents[i])
            world_cache[i] = (pp + pq.apply(local_pos[i] + offsets.get(i, zero)), pq * local_rotation(i))
        return world_cache[i]

    def enabled(i):
        result = np.ones(count, dtype=bool)
        for frame, value in sorted(switches.get(name_key(bones[i]['name']), [])):
            result[frame:] = value
        return result

    source_base_q = {i:q.copy() for i,q in base_q.items()}
    def solve_ik():
        errors = {}
        # Solve the native PMX chains, not robot joint targets.
        for controller in sorted((i for i, b in enumerate(bones) if b['ik']), key=lambda i: (bones[i]['layer'], i)):
            if leg_mode == 'fk':
                break  # Explicit joint-driven mode: do not evaluate VMD foot IK controllers.
            ik = bones[controller]['ik']
            target = ik['target']
            links = ik['links']
            active = enabled(controller)
            target_position = world(controller)[0].copy()
            if len(links) == 2:
                knee, hip = links[0]['bone'], links[1]['bone']
                if parents[target] != knee or parents[knee] != hip or not links[0]['limits']:
                    raise ValueError('Unsupported two-link IK topology')
                hpos, hq = world(hip)
                u, v = local_pos[knee], local_pos[target]
                length = np.linalg.norm(target_position - hpos, axis=1)
                a, b = u[1]*v[1] + u[2]*v[2], u[2]*v[1] - u[1]*v[2]
                desired = (length**2 - np.dot(u, u) - np.dot(v, v))/2 - u[0]*v[0]
                theta = np.arctan2(b, a) - np.arccos(np.clip(desired / np.hypot(a, b), -1, 1))
                theta = np.clip(theta, links[0]['limits'][0][0], links[0]['limits'][1][0])
                knee_q = Rotation.from_rotvec(np.column_stack([theta, np.zeros((count, 2))]))
                bent = u + knee_q.apply(np.tile(v, (count, 1)))
                hip_world = align_vectors(hq.apply(bent), target_position - hpos) * hq
                hip_local = (world(parents[hip])[1].inv() * hip_world).as_quat()
                base_q[hip] = np.where(active[:, None], hip_local, base_q.get(hip, identity))
                base_q[knee] = np.where(active[:, None], knee_q.as_quat(), base_q.get(knee, identity))
            elif len(links) == 1:
                link = links[0]['bone']
                p, q = world(link)
                correction = align_vectors(world(target)[0] - p, target_position - p)
                local = (world(parents[link])[1].inv() * correction * q).as_quat()
                base_q[link] = np.where(active[:, None], local, base_q.get(link, identity))
            else:
                raise ValueError(f'Unsupported IK chain: {bones[controller]["name"]}')
            q_cache.clear()
            world_cache.clear()
            distance = np.linalg.norm(world(target)[0] - target_position, axis=1) * scale
            errors[bones[controller]['name']] = dict(mean_m=float(distance.mean()), max_m=float(distance.max()),
                                                  enabled_frames=int(active.sum()),
                                                  active_mean_m=float(distance[active].mean()) if active.any() else None,
                                                  active_max_m=float(distance[active].max()) if active.any() else None,
                                                  error_scope='mean_m/max_m include inactive frames; active_* measure enabled IK only')
        return errors

    errors = solve_ik()
    clearance_report = None
    if leg_clearance:
        from leg_clearance import fit_leg_proxies, proxy_gaps, smooth_upper_envelope
        proxies = fit_leg_proxies(model)
        controller_ids = [i for i,b in enumerate(bones) if name_key(b['name']) in ('左足IK','右足IK')]
        if leg_mode != 'ik' or len(controller_ids) != 2:
            raise ValueError('leg-clearance requires two active leg IK controllers')
        original_q = source_base_q
        original_offsets = {i:p.copy() for i,p in offsets.items()}
        def apply_padding(padding_m):
            base_q.clear(); base_q.update({i:q.copy() for i,q in original_q.items()})
            offsets.clear(); offsets.update({i:p.copy() for i,p in original_offsets.items()})
            for i in controller_ids:
                offsets.setdefault(i,zero.copy())
                offsets[i][:,0] += np.sign(positions[i,0])*padding_m/scale
            q_cache.clear(); world_cache.clear()
            solve_ik()
            return proxy_gaps(proxies,{i:world(i)[0] for p in proxies for i in (p['a'],p['b'])})*scale
        before = proxy_gaps(proxies,{i:world(i)[0] for p in proxies for i in (p['a'],p['b'])})*scale
        required = np.full(count,np.nan)
        margin_m = .005
        # Search the minimum outward displacement per frame, at 2 mm resolution.
        for padding in np.arange(0,.202,.002):
            gaps = apply_padding(np.full(count,padding))
            accepted = np.isnan(required) & (gaps >= margin_m)
            required[accepted] = padding
            if np.isfinite(required).all(): break
        unresolved = int(np.isnan(required).sum())
        required[np.isnan(required)] = .2
        padding = smooth_upper_envelope(required)
        after = apply_padding(padding)
        # Interacting pose constraints need a final pass after temporal smoothing.
        for _ in range(25):
            failed = after < margin_m-.0001
            if not failed.any(): break
            padding[failed] += .002
            padding = smooth_upper_envelope(padding)
            after = apply_padding(padding)
        errors = solve_ik()
        clearance_report = dict(method='95th percentile bare-leg tapered sphere proxies; minimum per-frame outward IK padding',
            margin_m=margin_m, search_resolution_m=.002, temporal_envelope_frames=12,
            before_min_gap_m=float(before.min()),after_min_gap_m=float(after.min()),
            before_penetration_frames=int((before<0).sum()),after_penetration_frames=int((after<0).sum()),
            search_unresolved_frames=unresolved,margin_failed_frames=int((after<margin_m-.0001).sum()),
            extra_padding_mean_m=float(padding.mean()),extra_padding_max_m=float(padding.max()),
            padding_m=padding.tolist(),proxy_radii_m=[(p['radii']*scale).tolist() for p in proxies])
        print('LEG_CLEARANCE', {k:v for k,v in clearance_report.items() if k not in ('padding_m','proxy_radii_m')},flush=True)
    # Resolve all append rotations after the FK control bones have been IK-solved.
    animated = set(base_q) | set(offsets)
    for _ in bones:
        previous = len(animated)
        animated.update(i for i, b in enumerate(bones) if b['append'] and b['append'][0] in animated)
        if len(animated) == previous:
            break
    ordered = [(j, by_name[name_key(name)]) for j, name in enumerate(names)
               if by_name[name_key(name)] in animated]
    baked_q, baked_t = [], []
    for j, i in ordered:
        q = local_rotation(i).as_quat()
        # Reflection C(x,y,z)=(x,z,y): axial quaternion vector becomes -C*v.
        baked_q.append(np.column_stack([q[:, 3], -q[:, 0], -q[:, 2], -q[:, 1]]))
        delta = offsets.get(i, zero)[:, [0, 2, 1]] * scale
        baked_t.append(delta + np.asarray(rest[j].ExtractTranslation()))
    baked_q, baked_t = np.stack(baked_q, axis=1), np.stack(baked_t, axis=1)
    if not np.isfinite(baked_q).all() or not np.isfinite(baked_t).all():
        raise RuntimeError('Non-finite baked animation')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A separate animation override references the original asset and its textures.
    dance = Usd.Stage.CreateNew(str(output))
    dance.GetRootLayer().subLayerPaths = [str(asset.resolve()).replace('\\', '/')]
    dance.SetDefaultPrim(dance.GetPrimAtPath('/Character'))
    UsdGeom.SetStageUpAxis(dance, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(dance, 1.0)
    animation = UsdSkel.Animation.Define(dance, '/Character/Animation')
    animation.CreateJointsAttr([paths[j] for j, _ in ordered])
    qr, tr = animation.CreateRotationsAttr(), animation.CreateTranslationsAttr()
    qr.Clear()
    tr.Clear()
    qr.Set(Vt.QuatfArray([Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])) for q in baked_q[0]]))
    tr.Set(Vt.Vec3fArray.FromNumpy(baked_t[0].astype(np.float32)))
    animation.CreateScalesAttr([Gf.Vec3h(1)] * len(ordered))
    for frame in range(count):
        qr.Set(Vt.QuatfArray([Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])) for q in baked_q[frame]]), frame)
        tr.Set(Vt.Vec3fArray.FromNumpy(baked_t[frame].astype(np.float32)), frame)
    dance.SetStartTimeCode(0)
    dance.SetEndTimeCode(count - 1)
    dance.SetTimeCodesPerSecond(30)
    dance.SetFramesPerSecond(30)
    dance.GetRootLayer().Save()
    # Reload from disk and exercise UsdSkel's actual composed skinning evaluator.
    check = Usd.Stage.Open(str(output))
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root.Get(check, '/Character'), Usd.PrimDefaultPredicate)
    query = cache.GetSkelQuery(UsdSkel.Skeleton.Get(check, '/Character/Skeleton'))
    skin = cache.GetSkinningQuery(check.GetPrimAtPath('/Character/Body'))
    bounds = {}
    for frame in sorted({0, 300, 600, 900, 1500, count-1}):
        if frame >= count:
            continue
        points = Vt.Vec3fArray(mesh_points)
        if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(frame), points, frame):
            raise RuntimeError('Dance skinning failed')
        array = np.asarray(points)
        if not np.isfinite(array).all() or np.max(np.abs(array)) > 20:
            raise RuntimeError('Invalid/degenerated dance mesh')
        bounds[frame] = [array.min(axis=0).tolist(), array.max(axis=0).tolist()]
    report = dict(vmd=str(vmd), pmx=str(pmx), usd=str(output), frames=count, seconds=(count-1)/30, leg_mode=leg_mode, translation_scale=translation_scale,
                  stance_reference=str(stance_reference) if stance_reference else None, stance_shift_m=stance_adjustments, leg_clearance=clearance_report,
                  matched_tracks=len(matched), animated_joints=len(ordered),
                  unmatched_tracks=sorted(set(tracks)-set(by_name)), ik=errors, sample_bounds=bounds,
                  limitations=['No facial morphs, MMD hair/skirt physics or audio',
                               ('Joint-driven FK; foot IK controllers ignored' if leg_mode == 'fk' else
                                'Analytical PMX leg IK and one-link toe IK; not a full MMD runtime')], **metadata)
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('VMD_BAKE_PASS ' + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pmx', type=Path, default=Path('media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'))
    parser.add_argument('--vmd', type=Path, default=Path('media/dance/ji_le_jing_tu/ji_le_jing_tu.vmd'))
    parser.add_argument('--asset', type=Path, default=Path('output/pmx_prototype/character.usdc'))
    parser.add_argument('--output', type=Path, default=Path('output/pmx_prototype/ji_le_jing_tu.usdc'))
    parser.add_argument('--leg-mode', choices=('fk', 'ik'), default='ik',
                        help='fk: use VMD joint rotations only; ik: evaluate native PMX foot IK (legacy default)')
    parser.add_argument('--translation-scale', type=float, default=1.0,
                        help='Scale VMD translations into PMX model units before IK; rotations are unchanged')
    parser.add_argument('--leg-clearance', action='store_true', help='Fit bare-leg collision proxies and solve minimum smooth outward IK padding')
    parser.add_argument('--stance-reference', type=Path,
                        help='Optional PMX reference for normalized left/right foot IK rest spacing')
    args = parser.parse_args()
    if not np.isfinite(args.translation_scale) or args.translation_scale <= 0:
        parser.error('translation-scale must be finite and positive')
    bake(args.pmx.resolve(), args.vmd.resolve(), args.asset.resolve(), args.output.resolve(), leg_mode=args.leg_mode,
         translation_scale=args.translation_scale,
         stance_reference=args.stance_reference.resolve() if args.stance_reference else None,
         leg_clearance=args.leg_clearance)
