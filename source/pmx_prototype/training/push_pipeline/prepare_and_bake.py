"""Validate matching trials and solve Anon's separate 120 Hz hair and skirt tracks."""

import hashlib
import json
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / 'source/pmx_prototype'))
import source_layout
from trials import read_batch
from anon_secondary_stream import AnonSecondaryStream
from secondary_profiles import DEFAULT_SECONDARY_PROFILE, DEFAULT_SECONDARY_QUALITY
from render import WIDTH, HEIGHT, HAPERTURE, FOCAL, camera_pose


FPS = 30
LEAD = 45
AFTER = 150
LANES = ('base', 'trained')
def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fit_camera(chapter, centers, bodies):
    points = np.stack(bodies)
    world_centers = np.column_stack((centers, np.full(len(centers), .82)))
    for scale in (1., 1.12, 1.25, 1.4, 1.6, 1.9, 2.3, 2.8, 3.4, 4.2):
        chapter['camera_scale'] = scale
        center, eye = camera_pose(chapter, centers[0])
        offset = eye - center
        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = points - world_centers[None, :, None] - offset
        zoom = WIDTH*FOCAL/HAPERTURE / (relative @ forward)
        x = WIDTH/2 + zoom*(relative @ right)
        y = HEIGHT/2 - zoom*(relative @ up)
        if x.min() >= 55 and x.max() <= WIDTH-55 and y.min() >= 160 and y.max() <= 920:
            return
    raise ValueError(f'Cannot keep both actors within the matched view: {chapter["trial_id"]}')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=OUT)
    OUT = parser.parse_args().work_dir.resolve()
    plan = json.loads((OUT / 'video_plan.json').read_text(encoding='utf-8'))
    shots = {shot['trial_id']:shot for shot in plan['shots']}
    metadata = [json.loads((OUT / lane / 'protocol.json').read_text(encoding='utf-8')) for lane in LANES]
    records = []
    for protocol in metadata:
        _, rows = read_batch(Path(protocol['acceptance_reference']))
        records.append(rows)
    traces = []
    for lane in LANES:
        with np.load(OUT / lane / 'poses.npz') as handle:
            traces.append(dict(handle))
    a, b = traces
    for field in ('trial_ids', 'link_names', 'dt', 'onset', 'commands'):
        np.testing.assert_array_equal(a[field], b[field])
    if metadata[0]['asset'] != metadata[1]['asset'] or not np.isclose(float(a['dt']), 1/FPS):
        raise ValueError('Expected the same body asset and 30 Hz timing on both lanes')
    links = a['link_names'].tolist()
    pelvis = links.index('pelvis')
    manifest = json.loads(Path(metadata[0]['asset']).with_name('manifest.json').read_text(encoding='utf-8'))
    chapters = []
    scenes = {lane: dict(poses=[], velocity=[], camera_xy=[], arrow_xy=[]) for lane in LANES}
    full_pose_paths = []
    for clip_idx, trial in enumerate(a['trial_ids'].tolist()):
        onset = int(a['onset'][clip_idx])
        begin, end = onset - LEAD + 1, onset + AFTER + 1
        origin = a['poses'][onset, clip_idx, pelvis, :2]
        base_row, trial_row = (rows[trial] for rows in records)
        shot = shots[trial]
        chapter = dict(trial_id=trial, title=shot['title'], direction_cn=shot['direction_cn'],
                       phase_bin=int(base_row['phase_bin']), case=base_row['case'],
                       direction=base_row['direction'], command=a['commands'][clip_idx].tolist(),
                       impulse_m_s=base_row['impulse_m_s'], impulse_Ns=float(base_row['impulse_Ns']),
                       peak_force_N=float(base_row['peak_force_N']),
                       fps=FPS, start_frame=clip_idx*(LEAD+AFTER), impact_frame=LEAD,
                       duration_frames=LEAD+AFTER, local_clip_start=begin,
                       lanes={lane: {key: row[key] for key in ('fell','fall_seconds','recovered','recovery_seconds')}
                              for lane, row in zip(LANES, (base_row,trial_row))})
        if any(base_row[key] != trial_row[key] for key in ('case','direction','phase_bin','impulse_m_s')):
            raise ValueError(f'Unpaired trial ID {trial}')
        if [row['recovered'] for row in (base_row,trial_row)] != shot['expected_recovery']:
            raise ValueError(f'Trial selection does not match the verified recovery outcome: {trial}')
        heading_left = a['left_xy'][clip_idx]
        chapters.append(chapter)
        pair_full = []
        for lane, trace in zip(LANES, traces):
            body = trace['poses'][:end, clip_idx].copy()
            body[..., :2] -= origin
            pair_full.append(body)
            scenes[lane]['poses'].append(body[begin:])
            scenes[lane]['velocity'].append(trace['frames'][begin:end,clip_idx,:3])
            scenes[lane]['arrow_xy'].append(trace['force_direction_xy'][clip_idx])
        full_pose_paths.append(pair_full)
        centers = (pair_full[0][begin:,pelvis,:2] + pair_full[1][begin:,pelvis,:2]) * .5
        # Match camera translation across both panels and avoid 30 Hz sway.
        width = 13
        camera_xy = np.stack([np.convolve(np.pad(values, width//2, mode='edge'),
                                         np.ones(width)/width, mode='valid')
                              for values in centers.T], axis=1)
        for lane in LANES:
            scenes[lane]['camera_xy'].append(camera_xy)
        chapter['camera_left_xy'] = heading_left.tolist()
        fit_camera(chapter, camera_xy, [body[begin:,:,:3] for body in pair_full])

    (OUT/'chapters.json').write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding='utf-8')
    bake_metadata = {}
    for lane_index, lane in enumerate(LANES):
        data = scenes[lane]
        files = OUT / lane
        settings = SimpleNamespace(output=files, quality=DEFAULT_SECONDARY_QUALITY, physics_threads=8,
            skirt_edge_stride=1, secondary_mode='full', secondary_profile=DEFAULT_SECONDARY_PROFILE,
            legacy_secondary_physics=False, reference_hair_compute=False, legacy_compute=False,
            pmx=ROOT / 'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
        stream = AnonSecondaryStream(settings, full_pose_paths[0][lane_index][0], manifest, links,
                                     SimpleNamespace(update=lambda: None))
        frames = 0
        data['positions'], data['rotations'] = [], []
        try:
            details = stream.metadata
            if details['physics_hz'] != 120 or details['edge_nodes'] <= 0:
                raise ValueError('Expected the standard 120 Hz hair and skirt profile')
            for chapter, pair in zip(chapters, full_pose_paths):
                positions, rotations = [], []
                for t, pose in enumerate(pair[lane_index]):
                    stream.send(frames, dict(poses=pose, restart=(t == 0 and frames != 0)))
                    current = stream.receive(frames)
                    frames += 1
                    if t >= chapter['local_clip_start']:
                        positions.append(current['p'])
                        rotations.append(current['q'])
                data['positions'].append(np.stack(positions))
                data['rotations'].append(np.stack(rotations))
                print(f'BAKE {lane} trial={chapter["trial_id"]} body_frames={frames}', flush=True)
        finally:
            stream.close()
        payload = {key: np.stack(values) for key, values in data.items()}
        if payload['positions'].shape[:2] != (len(chapters), LEAD+AFTER):
            raise ValueError('Incomplete secondary replay')
        if not all(np.isfinite(value).all() for value in payload.values()):
            raise ValueError('Nonfinite recorded transforms')
        np.savez_compressed(files/'baked.npz', **payload, link_names=links)
        bake_metadata[lane] = dict(physics_hz=details['physics_hz'], edge_nodes=details['edge_nodes'],
                                   processed_body_frames=frames, displayed_frames=len(chapters)*(LEAD+AFTER),
                                   source=digest(files/'poses.npz'), baked=digest(files/'baked.npz'))
    (OUT/'bake_validation.json').write_text(json.dumps(bake_metadata, indent=2), encoding='utf-8')
    print('BAKE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
