"""Measure the skinned palm, fingers and skirt on a full-secondary replay."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import fcl
from numba import njit
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_pmx import read_pmx
from diagnostics.replay_anon_secondary import sha256


SCENES = ('stand', 'walk_before_stop', 'stop_after_walk', 'walk_backward', 'walk')
SIDES = ('left', 'right')
PARTS = ('palm', 'fingers')
SIGNS = np.array([1., -1.])
MARGIN = .005


class SkinGeometry:
    def __init__(self, pmx, manifest):
        model = read_pmx(pmx)
        bones = model['bones']
        order = np.array([int(path.rsplit('/', 1)[-1][1:]) for path in manifest['bone_paths']])
        remap = np.argsort(order)
        rest = np.array([v[0] for v in model['vertices']])[:, [0, 2, 1]]
        floor = rest[:, 2].min()
        scale = 1.6 / (rest[:, 2].max() - floor)
        rest[:, 2] -= floor
        rest *= scale
        bone_rest = np.array([bones[i]['position'] for i in order])[:, [0, 2, 1]]
        bone_rest[:, 2] -= floor
        bone_rest *= scale
        np.testing.assert_allclose(bone_rest, manifest['rest_positions'], atol=1e-6, rtol=0)
        ids = np.zeros((len(rest), 4), dtype=int)
        weights = np.zeros(ids.shape)
        for i, vertex in enumerate(model['vertices']):
            for j, (bone, weight) in enumerate(zip(vertex[3], vertex[4])):
                if weight > 0:
                    ids[i, j], weights[i, j] = remap[bone], weight
        weights /= weights.sum(axis=1, keepdims=True)
        faces = np.asarray(model['indices']).reshape(-1, 3)
        skirt = []
        cursor = 0
        for material in model['materials']:
            end = cursor + material['count'] // 3
            if material['name'] in ('裙子', '裙纱'):
                skirt.append(faces[cursor:end])
            cursor = end

        def submesh(triangles):
            vertices, tri = np.unique(triangles, return_inverse=True)
            joints = ids[vertices]
            return joints, weights[vertices], rest[vertices, None] - bone_rest[joints], tri.reshape(-1, 3)

        self.skirt = submesh(np.concatenate(skirt))
        self.hands = []
        self.samples = []
        directions = np.array([v for v in product((-1, 0, 1), repeat=3) if any(v)])
        for side, name in zip(SIDES, ('左手首', '右手首')):
            wrist = next(i for i, bone in enumerate(bones) if bone['name'] == name)
            hand = {wrist}
            while True:
                more = {i for i, bone in enumerate(bones) if bone['parent'] in hand} - hand
                if not more:
                    break
                hand.update(more)
            hand_weight = (weights * np.isin(ids, remap[sorted(hand)])).sum(axis=1)
            finger_weight = (weights * np.isin(ids, remap[sorted(hand - {wrist})])).sum(axis=1)
            triangles = faces[(hand_weight[faces] > .5).all(axis=1)]
            finger_faces = finger_weight[triangles].mean(axis=1) >= .5
            regions = (triangles[~finger_faces], triangles[finger_faces])
            self.hands.append([submesh(tri) for tri in regions])
            sample_regions = []
            for tri in regions:
                vertices = np.unique(tri)
                skin_weights = weights[vertices] * (np.asarray(manifest['bone_owners'])[ids[vertices]] == side + '_hand')
                # Both palm and fingers are rigidly attached to the last policy link.
                vertices = vertices[skin_weights.sum(axis=1) > .999]
                points = rest[vertices] - manifest['links'][side + '_hand']['origin']
                extreme = np.unique((points @ directions.T).argmax(axis=0))
                sample_regions.append(points[extreme])
            self.samples.append(sample_regions)
        self.faces = dict(skirt=len(self.skirt[-1]), hand_parts=[[len(sub[-1]) for sub in hand] for hand in self.hands])

    @staticmethod
    def skin(mesh, positions, matrices):
        joints, weights, rest, tri = mesh
        transformed = positions[joints] + np.einsum('...ij,...j->...i', matrices[joints], rest)
        return (transformed * weights[..., None]).sum(axis=1), tri


def world_skinning(local_p, local_q, parents):
    positions = local_p.astype(np.float64)
    rotations = Rotation.from_quat(local_q.reshape(-1, 4)).as_matrix().reshape(*local_q.shape[:2], 3, 3)
    parents = np.asarray(parents)
    done = parents < 0
    while not done.all():
        children = np.flatnonzero(~done & done[parents])
        if not len(children):
            raise ValueError('Disconnected PMX skeleton')
        r = rotations[:, parents[children]]
        positions[:, children] = positions[:, parents[children]] + np.einsum('...ij,...j->...i', r, positions[:, children])
        rotations[:, children] = r @ rotations[:, children]
        done[children] = True
    return positions, rotations


def collision_mesh(vertices, faces):
    mesh = fcl.BVHModel()
    mesh.beginModel()
    mesh.addSubModel(vertices, faces.astype(np.int32, copy=False))
    mesh.endModel()
    return fcl.CollisionObject(mesh)


@njit(cache=True)
def skirt_projection(vertices, faces, y, z):
    """Rasterize the outer skirt surface along the pelvis lateral axis."""
    surface = np.zeros((2, len(z), len(y)), dtype=np.float32)
    for face in faces:
        a, b, c = vertices[face]
        ab, ac = b - a, c - a
        determinant = ab[1] * ac[2] - ab[2] * ac[1]
        if abs(determinant) < 1e-12:
            continue
        y0 = np.searchsorted(y, min(a[1], b[1], c[1]))
        y1 = np.searchsorted(y, max(a[1], b[1], c[1]), side='right')
        z0 = np.searchsorted(z, min(a[2], b[2], c[2]))
        z1 = np.searchsorted(z, max(a[2], b[2], c[2]), side='right')
        for i in range(z0, z1):
            for j in range(y0, y1):
                dy, dz = y[j] - a[1], z[i] - a[2]
                u = (dy * ac[2] - dz * ac[1]) / determinant
                v = (ab[1] * dz - ab[2] * dy) / determinant
                if u >= -1e-7 and v >= -1e-7 and u + v <= 1.0000001:
                    x = a[0] + u * ab[0] + v * ac[0]
                    if x >= 0:
                        surface[0, i, j] = max(surface[0, i, j], x)
                    else:
                        surface[1, i, j] = max(surface[1, i, j], -x)
    return surface


def longest_run(flags):
    changes = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    return int(np.max(changes[1::2] - changes[::2], initial=0))


def summarize(labels, frames, hits, gap, wrists):
    summary = {}
    for label in SCENES:
        mask = labels == label
        if not mask.any():
            continue
        local_frames = frames - frames[mask][0]
        windows = {'whole': mask}
        if label in ('stand', 'walk_before_stop', 'stop_after_walk', 'walk'):
            windows['settled'] = mask & (local_frames >= 60)
        if label == 'stop_after_walk':
            windows['braking'] = mask & (local_frames < 60)
        summary[label] = {}
        for name, picked in windows.items():
            if not picked.any():
                continue
            touching = hits[picked].any(axis=(1, 2))
            summary[label][name] = dict(
                frames=int(picked.sum()), intersect_frames=int(touching.sum()),
                longest_intersection_s=longest_run(touching) / 30,
                intersect_frames_by_hand_and_region=hits[picked].sum(axis=0).tolist(),
                surface_distance_mm_p05=np.round(np.quantile(gap[picked] * 1000, .05, axis=0), 2).tolist(),
                surface_distance_mm_median=np.round(np.median(gap[picked] * 1000, axis=0), 2).tolist(),
                wrist_swing_fore_aft_mm=np.round(1000 * np.ptp(wrists[picked, :, 1], axis=0), 2).tolist(),
                wrist_lateral_mm_median=np.round(1000 * np.median(wrists[picked, :, 0] * SIGNS, axis=0), 2).tolist(),
            )
    return summary


def analyze(replay, *, probe):
    began = perf_counter()
    metadata = json.loads((replay / 'secondary.json').read_text(encoding='utf-8'))
    rollout = Path(metadata['rollout'])
    pmx = Path(metadata['pmx'])
    profile = Path(metadata['secondary_profile'])
    for path, key in ((rollout, 'rollout_sha256'), (pmx, 'pmx_sha256'), (profile, 'secondary_profile_sha256')):
        if sha256(path) != metadata[key]:
            raise ValueError(f'Input has changed since the secondary replay: {path}')
    report = json.loads(rollout.with_name('report.json').read_text(encoding='utf-8'))
    manifest = json.loads(Path(report['env_config']['asset']).with_name('manifest.json').read_text(encoding='utf-8'))
    geometry = SkinGeometry(pmx, manifest)
    with np.load(rollout) as trace, np.load(replay / 'secondary.npz') as cache:
        labels = trace['labels'][:metadata['frames']]
        frames = np.flatnonzero(np.isin(labels, SCENES))
        body = trace['poses'][frames, 0]
        commands = trace['planned_commands'][frames, 0]
        phases = trace['phases'].reshape(len(trace['labels']), -1)[frames, 0]
        links = trace['link_names'].tolist()
        positions, matrices = world_skinning(cache['positions'][frames], cache['rotations'][frames], manifest['parents'])
    labels = labels[frames]
    pelvis = body[:, links.index('pelvis')]
    local = Rotation.from_quat(pelvis[:, 3:]).inv().as_matrix()
    hands = body[:, [links.index(side + '_hand') for side in SIDES]]
    wrist = np.einsum('tij,tsj->tsi', local, hands[..., :3] - pelvis[:, None, :3])
    y, z = np.linspace(-.4, .4, 81), np.linspace(-.42, .12, 55)
    shapes = []
    hits = np.zeros((len(frames), 2, 2), dtype=bool)
    gaps = np.zeros(hits.shape, dtype=np.float64)
    request = fcl.CollisionRequest()
    distance = fcl.DistanceRequest()
    for frame in range(len(frames)):
        p = (positions[frame] - pelvis[frame, :3]) @ local[frame].T
        r = local[frame] @ matrices[frame]
        skirt, triangles = geometry.skin(geometry.skirt, p, r)
        cloth_mesh = collision_mesh(skirt, triangles)
        if probe:
            shapes.append(skirt_projection(skirt, triangles, y, z))
        for side, hand in enumerate(geometry.hands):
            for region, submesh in enumerate(hand):
                mesh = collision_mesh(*geometry.skin(submesh, p, r))
                hits[frame, side, region] = bool(fcl.collide(mesh, cloth_mesh, request, fcl.CollisionResult()))
                if not hits[frame, side, region]:
                    gaps[frame, side, region] = fcl.distance(mesh, cloth_mesh, distance, fcl.DistanceResult())
    if not np.isfinite(gaps).all() or np.min(gaps) < -1e-7:
        raise ValueError('Invalid mesh distances')
    np.savez_compressed(replay / 'hand_skirt_measurements.npz', frames=frames, labels=labels, hits=hits, gaps_m=gaps,
                        wrist_local_m=wrist)
    result = dict(source_rollout_sha256=metadata['rollout_sha256'], secondary_profile_sha256=metadata['secondary_profile_sha256'],
                  regions=list(PARTS), sides=list(SIDES), mesh_triangles=geometry.faces,
                  frames_checked=len(frames), measures=summarize(labels, frames, hits, gaps, wrist),
                  duration_seconds=perf_counter() - began)
    (replay / 'hand_skirt_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('HAND_SKIRT_MEASURED ' + json.dumps({k: v.get('settled', v['whole'])['intersect_frames']
                                              for k, v in result['measures'].items()}), flush=True)
    if probe:
        return prepare_reward(replay, geometry, y, z, np.asarray(shapes), frames, labels, pelvis, hands, hits,
                              commands, phases, report['env_config'], probe, result['measures'])


def prepare_reward(replay, geometry, y, z, shapes, frames, labels, pelvis, hands, hits, commands, phases, config,
                   probe, baseline_measures):
    """Fit only on alternating half-second blocks; hold out the second walk and reverse walk."""
    fit = np.isin(labels, ('stand', 'walk_before_stop', 'stop_after_walk')) & ((frames // 15) % 2 == 0)
    holdout = ~fit
    envelope = np.zeros((len(z), len(y)), dtype=np.float32)
    for scene in ('stand', 'walk_before_stop', 'stop_after_walk'):
        measured = shapes[fit & (labels == scene)]
        envelope = np.maximum(envelope, np.quantile(measured, .9, axis=0).max(axis=0))
    envelope = gaussian_filter(maximum_filter(envelope, size=3), sigma=.75).astype(np.float32)
    points = [np.concatenate(parts) for parts in geometry.samples]
    point_count = max(map(len, points))
    points = np.stack([np.pad(p, ((0, point_count - len(p)), (0, 0)), mode='edge') for p in points])
    probe_measurement = json.loads((probe / 'hand_skirt_metrics.json').read_text(encoding='utf-8'))
    with np.load(probe / 'hand_skirt_measurements.npz') as diagnostic:
        probe_frames = diagnostic['frames']
        anchor_idxs = np.searchsorted(frames, probe_frames)
        np.testing.assert_array_equal(frames[anchor_idxs], probe_frames)
        actual_wrist = Rotation.from_quat(pelvis[anchor_idxs, 3:]).inv().as_matrix()
        actual_wrist = np.einsum('fij,fsj->fsi', actual_wrist, hands[anchor_idxs, :, :3] - pelvis[anchor_idxs, None, :3])
        outward = (diagnostic['wrist_local_m'] - actual_wrist)[..., 0] * SIGNS
        allowance = round(float(np.median(outward)), 3)
    settings = dict(weight=1.2, margin_m=MARGIN, reference_allowance_m=allowance,
                    y_bounds_m=y[[0, -1]].tolist(), z_bounds_m=z[[0, -1]].tolist(),
                    envelope_m=np.round(envelope, 5).tolist(), hand_points_m=points.tolist())
    validation = validate_reward(settings, pelvis, commands, phases, hands, hits, holdout, labels, config)
    probe_changes = {}
    for scene in ('stand', 'walk_before_stop', 'stop_after_walk'):
        prior = baseline_measures[scene]['settled']
        current = probe_measurement['measures'][scene]['settled']
        probe_changes[scene] = dict(
            intersect_frames=[prior['intersect_frames'], current['intersect_frames']],
            regional_intersections=[int(np.sum(prior['intersect_frames_by_hand_and_region'])),
                                    int(np.sum(current['intersect_frames_by_hand_and_region']))],
        )
    physical_probe_supports_outward = (probe_changes['walk_before_stop']['intersect_frames'][1]
                                       < probe_changes['walk_before_stop']['intersect_frames'][0])
    validation.update(fitted_frames=int(fit.sum()), holdout_frames=int(holdout.sum()), point_count_per_hand=point_count,
                      profile_sha256=json.loads((replay / 'secondary.json').read_text(encoding='utf-8'))['secondary_profile_sha256'],
                      reference_allowance_m=allowance, offline_shoulder_probe_changes=probe_changes)
    validation['passed'] &= physical_probe_supports_outward
    (replay / 'offline_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    if not validation['passed']:
        raise RuntimeError(f'Reward did not pass the offline checks: {validation}')
    (replay / 'training_override.json').write_text(json.dumps({'train_hand_skirt_clearance': settings}, indent=2), encoding='utf-8')
    print('HAND_SKIRT_CALIBRATION_PASSED ' + json.dumps(validation), flush=True)


def validate_reward(settings, body, commands, phases, hands, hits, holdout, labels, config):
    import torch
    from full_motion_reference import FullMotionReference
    from training.hand_skirt_clearance import HandSkirtClearance
    from unified_walk_run_reference import prepare, sample

    reward = HandSkirtClearance(settings, 'cpu')
    pelvis = torch.from_numpy(body)
    hand = torch.from_numpy(hands)
    command = torch.from_numpy(commands)
    with torch.inference_mode():
        motion = FullMotionReference(config['full_motion_path'], 'cpu')
        prepare(motion)
        reference = sample(motion, torch.from_numpy(phases), command, config)
        links = [motion.links.index(link) for link in ('pelvis', 'left_hand', 'right_hand')]
        ref = torch.cat((reference['positions'][:, links], reference['rotations'][:, links]), dim=-1)
        rate, gap, deficit, _ = reward(pelvis, hand, ref[:, 0], ref[:, 1:], command)
        axis = Rotation.from_quat(body[:, 3:]).apply([1., 0., 0.])[:, None, :] * SIGNS[None, :, None]
        outward = hand.clone()
        outward[:, :, :3] += torch.from_numpy(axis * .01)
        shifted_rate, _, _, _ = reward(pelvis, outward, ref[:, 0], ref[:, 1:], command)
        clear = hand.clone()
        clear[:, :, :3] += torch.from_numpy(axis * .25)
        clear_rate, _, _, _ = reward(pelvis, clear, ref[:, 0], ref[:, 1:], command)
        running = command.clone()
        running[:, 0] = 1.2
        running_rate, _, _, _ = reward(pelvis, hand, ref[:, 0], ref[:, 1:], running)
    colliding = hits.any(axis=-1)
    seen = (gap.numpy() < settings['margin_m'])
    counts = {}
    for label in SCENES:
        pick = holdout & (labels == label)
        actual = colliding[pick]
        counts[label] = dict(colliding_hands=int(actual.sum()), detected_hands=int((actual & seen[pick]).sum()),
                             clear_hands=int((~actual).sum()), clear_within_proxy=int(((~actual) & seen[pick]).sum()))
    total = sum(row['colliding_hands'] for row in counts.values())
    detected = sum(row['detected_hands'] for row in counts.values())
    feasible_shift = np.quantile(deficit.numpy()[holdout], .9, axis=0)
    responsive = bool((shifted_rate >= rate - 1e-4).all() and (shifted_rate > rate + 1e-4).any())
    zero_when_clear = bool((clear_rate == 0).all())
    running_zero = bool((running_rate == 0).all())
    recall = detected / max(total, 1)
    return dict(passed=bool(total and recall >= .85 and responsive and zero_when_clear and running_zero
                            and np.max(feasible_shift) < .095),
                heldout_collision_recall=recall, heldout_scenes=counts, responsive_to_10mm_outward=responsive,
                zero_at_clearance=zero_when_clear, zero_at_running_speed=running_zero,
                proxy_deficit_p90_mm=np.round(1000 * feasible_shift, 1).tolist(),
                mean_penalty=float(-rate.mean()), max_penalty=float(-rate.min()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--calibration-probe', type=Path)
    args = parser.parse_args()
    analyze(args.replay, probe=args.calibration_probe)


if __name__ == '__main__':
    main()
