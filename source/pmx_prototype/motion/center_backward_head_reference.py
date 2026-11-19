"""Build a versioned reference override for forward gaze during backward travel."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import config_path
from build_unified_walk_run import FIELDS
from retarget_100style import Retarget


HEAD_CLIPS = ('backward_040_final', 'run_backward')
HEAD_FORWARD = [0., -1., 0.]
NECK_JOINTS = ('neck_yaw_joint', 'neck_pitch_joint')
HEAD_LINKS = ('neck_yaw', 'neck_pitch', 'head')
GAZE_LIMIT = np.deg2rad(3.)


def facing_angles(direction):
    horizontal = np.hypot(direction[:, 0], direction[:, 1])
    return np.stack((np.arctan2(direction[:, 0], -direction[:, 1]),
                     np.arctan2(direction[:, 2], horizontal)), axis=-1)


def gaze_direction(angles):
    yaw, pitch = angles.T
    return np.stack((np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)), axis=-1)


def center_clip(clip, model, joint_names, link_names, period, command):
    if command[0] >= 0 or not np.allclose(command[1:], 0., atol=1e-6):
        raise ValueError('Head reference correction requires a straight backward clip')
    dt = period / len(clip['q'])
    joint_ids = [joint_names.index(name) for name in NECK_JOINTS]
    link_ids = [link_names.index(name) for name in HEAD_LINKS]
    joints = [model.joints[model.names.index(name)] for name in NECK_JOINTS]
    axes = [joint.get('axis_vector', np.eye(3)['XYZ'.index(joint['axis'])]) for joint in joints]
    if not np.allclose(axes, [[0, 0, 1], [1, 0, 0]]) or tuple(
        model.parents[link] for link in HEAD_LINKS
    ) != ('waist_pitch', 'neck_yaw', 'neck_pitch'):
        raise ValueError('Expected a Z-yaw and X-pitch neck below waist_pitch')

    original = facing_angles(Rotation.from_quat(clip['rotations'][:, link_ids[-1]]).apply(HEAD_FORWARD))
    smooth = gaussian_filter1d(np.unwrap(original, axis=0), .10 / dt, axis=0, mode='wrap')
    # Keep a small periodic trace of the captured gaze motion around level forward.
    swing = GAZE_LIMIT * np.tanh((smooth - smooth.mean(axis=0)) * .25 / GAZE_LIMIT)
    swing -= swing.mean(axis=0)
    scale = np.maximum(1., np.max(np.abs(swing), axis=0) / GAZE_LIMIT)
    desired = gaze_direction(swing / scale)

    torso_id = link_names.index('waist_pitch')
    torso = Rotation.from_quat(clip['rotations'][:, torso_id])
    local = facing_angles(torso.inv().apply(desired))
    local[:, 1] *= -1
    limits = model.limits[[model.names.index(name) for name in NECK_JOINTS]]
    angles = np.clip(local, limits[:, 0] + .005, limits[:, 1] - .005)

    positions, matrices = [], []
    parent_position = clip['positions'][:, torso_id].astype(float)
    parent_rotation = torso
    parent_link = 'waist_pitch'
    for k, link in enumerate(HEAD_LINKS):
        offset = model.origins[link] - model.origins[parent_link]
        position = parent_position + parent_rotation.apply(offset)
        rotation = parent_rotation
        if k < len(joints):
            rotation = parent_rotation * Rotation.from_rotvec(angles[:, k, None] * axes[k])
        positions.append(position)
        matrices.append(rotation.as_matrix())
        parent_position, parent_rotation, parent_link = position, rotation, link

    positions = np.stack(positions, axis=1)
    matrices = np.stack(matrices, axis=1)
    rotation_delta = np.roll(matrices, -1, axis=0) @ np.roll(matrices, 1, axis=0).swapaxes(-1, -2)
    omega = Rotation.from_matrix(rotation_delta.reshape(-1, 3, 3)).as_rotvec().reshape(positions.shape) / (2 * dt)

    def derivative(values):
        return (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2 * dt)

    clip['q'][:, joint_ids] = angles
    clip['qd'][:, joint_ids] = derivative(angles)
    clip['positions'][:, link_ids] = positions
    clip['rotations'][:, link_ids] = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(-1, len(link_ids), 4)
    linear = derivative(positions) + np.array([command[1], -command[0], 0.])
    clip['velocities'][:, link_ids] = np.concatenate((linear, omega), axis=-1)
    corrected = facing_angles(parent_rotation.apply(HEAD_FORWARD))
    return dict(source_median_deg=np.rad2deg(np.median(original, axis=0)).tolist(),
                corrected_median_deg=np.rad2deg(np.median(corrected, axis=0)).tolist(),
                corrected_max_abs_deg=np.rad2deg(np.abs(corrected).max(axis=0)).tolist(),
                limit_clamped_frames=int(np.count_nonzero((local != angles).any(axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=config_path, help='Checkpoint environment configuration')
    parser.add_argument('--output', required=True, type=Path, help='New reference directory')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    motion_path = Path(config['full_motion_path'])
    with np.load(motion_path) as archive:
        bank = dict(archive)
    model = Retarget(Path(config['asset']).parent / 'manifest.json')
    out = args.output.resolve()
    expert_paths = [Path(path).resolve() for path in config.get('amp_expert_paths', [])]
    if out in {motion_path.resolve().parent, *(path.parent for path in expert_paths)}:
        raise ValueError('Write corrected references in a separate directory')
    out.mkdir(parents=True, exist_ok=True)
    names = bank['clip_names'].tolist()
    changed = [names.index(name) for name in HEAD_CLIPS]
    results = {}
    for i in changed:
        clip = {key: bank[key][i] for key in FIELDS}
        results[names[i]] = center_clip(clip, model, bank['joint_names'].tolist(),
                                       bank['link_names'].tolist(), bank['periods'][i], bank['commands'][i])
    corrected_bank = out / 'full_motion.npz'
    np.savez_compressed(corrected_bank, **bank)

    experts = []
    for path in expert_paths:
        with np.load(path) as archive:
            command = archive['command']
            matching = [i for i in changed if np.allclose(command, bank['commands'][i])]
            if matching:
                data = dict(archive)
                i, = matching
                if not (np.array_equal(data['contacts'], bank['contacts'][i])
                        and np.isclose(data['period'], bank['periods'][i])):
                    raise ValueError(f'AMP expert timing differs from the reference bank: {path}')
                joint_names = bank['joint_names'].tolist()
                link_names = bank['link_names'].tolist()
                if data['joint_names'].tolist() != joint_names or data['link_names'].tolist() != link_names:
                    raise ValueError(f'AMP expert rig differs from the reference bank: {path}')
                joint_ids = [joint_names.index(name) for name in NECK_JOINTS]
                link_ids = [link_names.index(name) for name in HEAD_LINKS]
                for key in ('q', 'qd'):
                    data[key][:, joint_ids] = bank[key][i][:, joint_ids]
                for key in ('positions', 'rotations', 'velocities'):
                    data[key][:, link_ids] = bank[key][i][:, link_ids]
                path = out / path.name
                np.savez_compressed(path, **data)
        experts.append(str(path.resolve()))
    override = dict(full_motion_path=str(corrected_bank), amp_expert_paths=experts)
    (out / 'reference_override.json').write_text(json.dumps(override, indent=2), encoding='utf-8')
    report = dict(source=str(motion_path), targeted_clips=results, head_angles_order=['yaw', 'pitch'])
    (out / 'reference_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
