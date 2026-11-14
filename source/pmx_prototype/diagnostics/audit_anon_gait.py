"""Measure gait/arm motion from recorded PhysX poses; no renderer dependencies."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

ROOT = PROJECT_ROOT


def audit(path):
    d = np.load(path)
    dt = float(d['dt'])
    poses, joints, commands = d['poses'][:, 0], d['joints'][:, 0], d['commands'][:, 0]
    links, names = list(d['link_names']), list(d['dof_names'])
    moving = np.linalg.norm(commands[:, :2], axis=-1) > .05
    manifest = json.loads((ROOT/'output/anon_physical/stand/manifest.json').read_text(encoding='utf-8'))
    rest = np.asarray(manifest['rest_positions']); bone_names = manifest['bone_names']
    result = dict(rollout=str(Path(path).resolve()), frames=len(poses), seconds=len(poses)*dt,
                  resets=int(d['dones'].sum()), feet={}, slip_by_segment={})
    for side, jp in [('left', '左'), ('right', '右')]:
        foot = poses[:, links.index(side+'_ankle_roll')]
        rotation = Rotation.from_quat(foot[:, 3:])
        ankle = np.array(manifest['links'][side+'_ankle_roll']['origin'])
        toe = rest[bone_names.index(jp+'つま先')]
        center = np.array([ankle[0], (ankle[1]+toe[1])/2, .035])-ankle
        corners = np.array([[x, y, z] for x in [-.0475, .0475] for y in [-.115, .115] for z in [-.035, .035]])+center
        world = np.stack([foot[:, :3]+rotation.apply(c) for c in corners], axis=1)
        bottom = world[:, :, 2].min(-1)
        contact = bottom < .008
        sustained = contact & np.roll(contact, 1) & np.roll(contact, 2) & moving
        sustained[:2] = False
        speed = np.linalg.norm(np.diff(world[:, :, :2], axis=0)/dt, axis=-1)
        near_floor = world[1:, :, 2] < bottom[1:, None]+.002
        slip = (speed*near_floor).sum(-1)/np.maximum(near_floor.sum(-1), 1)
        for label in dict.fromkeys(d['labels']):
            valid = sustained[1:] & (d['labels'][1:] == label)
            if valid.any():
                result['slip_by_segment'].setdefault(str(label), {})[side] = float(np.mean(slip[valid]))
        tilt = np.degrees(np.arccos(np.clip(rotation.apply(np.tile([0., 0., 1.], (len(foot), 1)))[:, 2], -1, 1)))
        peaks = find_peaks(bottom, height=.012, distance=round(.25/dt))[0]
        peaks = peaks[moving[peaks]]
        result['feet'][side] = dict(sustained_contact_slip_m_s=float(np.mean(slip[sustained[1:]])),
                                   moving_contact_fraction=float(contact[moving].mean()),
                                   stance_tilt_deg=float(np.mean(tilt[contact & moving])),
                                   median_clearance_peak_m=float(np.median(bottom[peaks])) if len(peaks) else 0.,
                                   swing_peaks=int(len(peaks)))
    left = joints[:, names.index('left_shoulder_pitch_joint')]
    right = joints[:, names.index('right_shoulder_pitch_joint')]
    forward = commands[:, 0] > .1
    result['arms'] = dict(left_pitch_range_deg=float(np.degrees(np.ptp(left[forward]))),
                          right_pitch_range_deg=float(np.degrees(np.ptp(right[forward]))),
                          contralateral_correlation=float(np.corrcoef(left[forward], right[forward])[0, 1]))
    result['upper_joint_ranges_deg'] = {
        n: float(np.degrees(np.ptp(joints[moving, names.index(n)])))
        for n in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'neck_yaw_joint', 'neck_pitch_joint') if n in names}
    head_rotation = Rotation.from_quat(poses[:, links.index('head'), 3:])
    head_up = head_rotation.apply(np.tile([0., 0., 1.], (len(poses), 1)))
    head_tilt = np.degrees(np.arccos(np.clip(head_up[:, 2], -1, 1)))
    result['head_tilt_p95_deg'] = float(np.percentile(head_tilt[moving], 95))
    head_omega = (head_rotation[:-1].inv()*head_rotation[1:]).as_rotvec()/dt
    result['head_roll_pitch_angular_speed_rms_rad_s'] = float(np.sqrt(np.mean(head_omega[moving[1:], :2]**2)))
    leg_ids = [i for i, n in enumerate(names) if any(k in n for k in ('hip_', 'knee_', 'ankle_'))]
    acc = np.diff(joints[:, leg_ids], n=2, axis=0)/dt**2
    result['leg_acceleration_rms_rad_s2'] = float(np.sqrt(np.mean(acc[moving[2:]]**2)))
    result['leg_joint_acceleration_rms_rad_s2'] = {
        names[j]:float(np.sqrt(np.mean(acc[moving[2:],k]**2))) for k,j in enumerate(leg_ids)}
    pelvis = poses[:, links.index('pelvis'), 2]
    oscillations = []
    labels = d['labels']
    for label in dict.fromkeys(labels):
        ids = np.flatnonzero((labels == label) & moving)
        if len(ids) > 60:
            values = pelvis[ids[30:]]
            oscillations.append(float(np.percentile(values, 95)-np.percentile(values, 5)))
    result['mean_segment_pelvis_vertical_range_m'] = float(np.mean(oscillations))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = {'candidate': audit(args.rollout)}
    if args.baseline: report['baseline'] = audit(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
