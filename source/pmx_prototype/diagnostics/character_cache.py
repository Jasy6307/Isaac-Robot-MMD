"""Mutsumi USD cache diagnostics; separate spatial, temporal and reference metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT


def open_skeleton(path):
    from pxr import Usd, UsdSkel
    stage = Usd.Stage.Open(str(Path(path).resolve()))
    if stage is None:
        raise ValueError(f'Unable to open USD cache: {path}')
    skeleton = UsdSkel.Skeleton.Get(stage, '/Character/Skeleton')
    if not skeleton:
        raise ValueError(f'Missing /Character/Skeleton: {path}')
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root.Get(stage, '/Character'), Usd.PrimDefaultPredicate)
    return stage, skeleton, cache, cache.GetSkelQuery(skeleton), list(skeleton.GetJointNamesAttr().Get())


def audit_cloth(args):
    from pxr import UsdGeom, Vt, Gf
    from scipy.spatial.transform import Rotation
    from convert_pmx import read_pmx
    s, sk, c, q, names = open_skeleton(args.cache)
    bind = sk.GetBindTransformsAttr().Get()
    rest = np.array([x.ExtractTranslation() for x in bind])
    by = {n: i for i, n in enumerate(names)}
    mesh = UsdGeom.Mesh.Get(s, '/Character/Part_09')
    skin = c.GetSkinningQuery(mesh.GetPrim())
    points = mesh.GetPointsAttr().Get()
    used = np.unique(mesh.GetFaceVertexIndicesAttr().Get())
    m = read_pmx(PROJECT_ROOT / 'media/mutsmi/mutsmi v1.pmx', physics=True)
    raw = np.array([v[0] for v in m['vertices']])
    scale = np.ptp(np.array(points)[:, 2]) / np.ptp(raw[:, 1])
    zbase = raw[:, 1].min()
    proxies = []
    for b in m['rigid_bodies']:
        if b['bone'] < 0:
            continue
        name = m['bones'][b['bone']]['name']
        if name not in ['左足', '右足', '左ひざ', '右ひざ']:
            continue
        i = by[name]
        center = (np.array(b['position'])[[0, 2, 1]] - [0, 0, zbase]) * scale
        axis = Rotation.from_euler('xyz', b['rotation']).apply([0, 1, 0])[[0, 2, 1]] * b['size'][1] * scale / 2
        proxies.append((i, center - rest[i] - axis, center - rest[i] + axis, b['size'][0] * scale))
    rows = 1 + max((int(n.split('_')[1]) for n in names if n.startswith('qun_')))
    grid = np.array([[by[f'qun_{r}_{col}'] for col in range(21)] for r in range(rows)])
    records = []
    for f in range(0, min(args.frames, int(s.GetEndTimeCode()) + 1), args.stride):
        mats = q.ComputeJointSkelTransforms(f)
        pos = np.array([x.ExtractTranslation() for x in mats])
        v = pos[grid]
        u, w = (v[1:-1] - v[:-2], v[2:] - v[1:-1])
        angles = np.degrees(np.arccos(np.clip(np.sum(u * w, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1), 1e-10), -1, 1)))
        pp = Vt.Vec3fArray(points)
        assert skin.ComputeSkinnedPoints(q.ComputeSkinningTransforms(f), pp, f)
        vertices = np.array(pp)[used]
        depth = np.full(len(vertices), -np.inf)
        for i, aa, bb, radius in proxies:
            pa = np.array(mats[i].Transform(Gf.Vec3d(*aa)))
            pb = np.array(mats[i].Transform(Gf.Vec3d(*bb)))
            axis = pb - pa
            t = np.clip((vertices - pa) @ axis / max(np.dot(axis, axis), 1e-12), 0, 1)
            dist = np.linalg.norm(vertices - (pa + t[:, None] * axis), axis=-1)
            depth = np.maximum(depth, radius - dist)
        records.append(dict(frame=f, fold_fraction=float(np.mean(angles > 90)), leg_proxy_vertex_fraction_over_2mm=float(np.mean(depth > 0.002)), leg_proxy_vertex_penetration_p95_mm=float(np.percentile(np.maximum(depth, 0), 95) * 1000), leg_proxy_vertex_penetration_max_mm=float(max(0, depth.max()) * 1000)))
    r = dict(cache=str(args.cache.resolve()), sample_stride_frames=args.stride, sample_count=len(records), common_leg_proxy_radius_scale=1.0, metrics={k: float(np.mean([v[k] for v in records])) for k in records[0] if k != 'frame'}, samples=records, limitation='Skinned vertices versus common PMX leg capsule proxies; not mesh-mesh ground truth')
    return r


def audit_temporal(args):
    from scipy.signal import savgol_filter, welch
    s, sk, cache, q, names = open_skeleton(args.cache)
    ids = [i for i, n in enumerate(names) if n.startswith('qun_') and int(n.split('_')[1]) > 0]
    hip = names.index('下半身')
    x = []
    for f in range(min(args.frames, int(s.GetEndTimeCode()) + 1)):
        mats = q.ComputeJointSkelTransforms(f)
        points = np.array([mats[i].ExtractTranslation() for i in ids])
        inv = np.linalg.inv(np.array(mats[hip]))
        x.append(np.c_[points, np.ones(len(points))] @ inv)
    x = np.asarray(x)[:, :, :3]
    smooth = savgol_filter(x, 9, 2, axis=0)
    hp = x - smooth
    acc = np.diff(x, n=2, axis=0) * 900.0
    slice_ = slice(30, -4)
    h = np.linalg.norm(hp, axis=-1)[slice_]
    am = np.linalg.norm(acc, axis=-1)[slice_]
    perframe = np.sqrt(np.mean(h * h, axis=1))
    freq, psd = welch(x[30:], fs=30, axis=0, nperseg=min(256, len(x) - 30))
    band = (freq >= 6) & (freq <= 14)
    r = dict(frames=len(x), diagnostic='Pelvis-local node high-pass residual: Savitzky-Golay 9 frames, order 2; no output filtering', metrics=dict(highpass_rms_mm=float(np.sqrt(np.mean(h * h)) * 1000), highpass_p99_mm=float(np.percentile(h, 99) * 1000), accel_rms_m_s2=float(np.sqrt(np.mean(am * am))), accel_p99_m_s2=float(np.percentile(am, 99)), accel_p999_m_s2=float(np.percentile(am, 99.9)), accel_max_m_s2=float(am.max()), highpass_max_mm=float(h.max() * 1000), band_6_14_hz_power=float(np.mean(psd[band]))), worst_frames=(np.argsort(perframe)[-12:] + 30).tolist())
    return r


def audit_reference(args):
    from pxr import UsdGeom
    import unicodedata
    names = ['左足IK', '右足IK', '左足首', '右足首', '上半身', '下半身', '頭', '左腕', '右腕']

    def read(path):
        s, sk, cache, query, joint_names = open_skeleton(path)
        joints = [unicodedata.normalize('NFKC', str(x)) for x in joint_names]
        ids = [joints.index(x) for x in names]
        xf = UsdGeom.XformCache().GetLocalToWorldTransform(sk.GetPrim())
        values = []
        for frame in range(int(s.GetEndTimeCode()) + 1):
            mats = query.ComputeJointSkelTransforms(frame)
            values.append([tuple(xf.Transform(mats[i].ExtractTranslation())) for i in ids])
        return np.asarray(values)
    old, new = (read(args.previous), read(args.current))
    assert old.shape == new.shape, (old.shape, new.shape)
    assert np.isfinite(new).all(), 'Nonfinite skeleton position'
    d = np.linalg.norm(old - new, axis=-1)
    body = json.loads(args.current.with_suffix('.json').read_text(encoding='utf-8'))
    assert body['leg_clearance'] is None, 'Extra clearance still active'
    assert np.max(d[:, 4:]) < 1e-06, 'Nonleg joint changed'
    r = dict(frames=len(new), extra_leg_clearance=False, max_world_change_m={name: float(d[:, i].max()) for i, name in enumerate(names)}, native_ik_errors=body.get('ik'), upper_body_unchanged=True, finite=True, previous=str(args.previous.resolve()), current=str(args.current.resolve()))
    return r



def main(argv=None, *, action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if action is None:
        children = parser.add_subparsers(dest='action', required=True)
        parsers = [(name, children.add_parser(name)) for name in ('cloth', 'temporal', 'reference')]
    else:
        parser.set_defaults(action=action)
        parsers = [(action, parser)]
    for name, child in parsers:
        if name == 'reference':
            child.add_argument('--previous', type=Path, required=True)
            child.add_argument('--current', type=Path, required=True)
        else:
            child.add_argument('--cache', type=Path, required=True)
            child.add_argument('--frames', type=int, default=7011)
            if name == 'cloth': child.add_argument('--stride', type=int, default=30)
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if getattr(args, 'frames', 1) < 1 or getattr(args, 'stride', 1) < 1:
        parser.error('Frame count and stride must be positive')
    if args.dry_run:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError(f'Result exists; choose a new output: {args.output}')
    result = {'cloth': audit_cloth, 'temporal': audit_temporal, 'reference': audit_reference}[args.action](args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    if args.action == 'cloth':
        print('CLOTH_CACHE_AUDIT', args.cache, result['metrics'], flush=True)
    elif args.action == 'temporal':
        print('SKIRT_TEMPORAL_AUDIT', result, flush=True)
    else:
        first_ik = list(result['max_world_change_m'].values())[:2]
        print('MOTION_AUDIT_PASS', args.current.parent.name, result['frames'], max(first_ik), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
