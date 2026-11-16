"""Compare fixed-scope locomotion reports without relaxing their existing gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DIRECTIONS = ('run_backward', 'run_left', 'run_right')


def joint_smoothness(path):
    """Physical joint acceleration after the same two-second segment warmup."""
    with np.load(path) as data:
        q = data['joints'].reshape(len(data['joints']), -1)
        dt = float(data['dt'])
        labels = data['labels'].astype(str)
        names = list(data['dof_names'].astype(str))
        acceleration = np.diff(q, n=2, axis=0) / dt**2
        result = {}
        for label in DIRECTIONS:
            indices = np.flatnonzero(labels == label)
            indices = indices[round(2/dt)+2:]
            indices = indices[(indices >= 2) & (indices < len(q))]
            leg_ids = [i for i, name in enumerate(names)
                       if any(part in name.lower() for part in ('hip', 'knee', 'ankle'))]
            result[label] = dict(
                leg_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration[indices-2][:, leg_ids]**2))),
                leg_joint_names=[names[i] for i in leg_ids])
        return result


def compare(baseline_path, candidate_path, reduction):
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    assert baseline['evaluator_sha256'] == candidate['evaluator_sha256'], 'Evaluator changed'
    assert baseline['env_config'].get('simulation_device', 'cuda:0') == candidate['env_config'].get('simulation_device', 'cuda:0'), 'Backend mismatch'
    assert baseline['total_seconds'] == candidate['total_seconds'] == 325, 'Full suite required'
    assert [(s['label'], s['seconds'], s['command']) for s in baseline['segments']] == [(s['label'], s['seconds'], s['command']) for s in candidate['segments']], 'Commands changed'
    rows = []
    for label in DIRECTIONS:
        old = next(s for s in baseline['segments'] if s['label'] == label)
        new = next(s for s in candidate['segments'] if s['label'] == label)
        ratio = new['stance_slip_geometry_m_s'] / old['stance_slip_geometry_m_s']
        retained_speed = new['mean_speed'] >= .95*old['mean_speed']
        rows.append(dict(label=label, baseline_slip_m_s=old['stance_slip_geometry_m_s'],
                         candidate_slip_m_s=new['stance_slip_geometry_m_s'], reduction_fraction=1-ratio,
                         baseline_speed_m_s=old['mean_speed'], candidate_speed_m_s=new['mean_speed'],
                         retained_baseline_speed=bool(retained_speed), minimum_speed_retention_fraction=.95,
                         passed=bool(new['passed'] and retained_speed and ratio <= 1-reduction)))
    smoothness = {}
    for name, report_path in [('baseline', baseline_path), ('candidate', candidate_path)]:
        rollout = report_path.parent/'rollout.npz'
        if rollout.is_file():
            smoothness[name] = joint_smoothness(rollout)
    return dict(baseline=str(baseline_path.resolve()), candidate=str(candidate_path.resolve()),
                baseline_report_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                candidate_report_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                required_reduction_fraction=reduction, directions=rows,
                joint_smoothness=smoothness,
                passed=bool(candidate['passed_unified_gate'] and all(row['passed'] for row in rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--minimum-reduction', type=float, default=.2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.candidate, args.minimum_reduction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
