"""Validate and summarize the push grids using the walk/run tracking tolerances."""

import csv
import json

import numpy as np


def moving_average(values, width):
    sums = np.cumsum(values, axis=0, dtype=np.float64)
    cumulative = np.concatenate((np.zeros_like(sums[:1]), sums), axis=0)
    return (cumulative[width:] - cumulative[:-width]) / width


def recovery_confirmation(velocities, target, dt):
    linear_width, yaw_width, hold = (round(duration / dt) for duration in (1., 2., .5))
    hold += 1
    linear_limit = max(.06, .22 * np.linalg.norm(target[:2]))
    yaw_limit = .1 if not np.any(target) else .2
    linear_average = moving_average(velocities[:, :2], linear_width)[yaw_width-linear_width:]
    yaw_average = moving_average(velocities[:, 2], yaw_width)
    good = np.linalg.norm(linear_average - target[:2], axis=1) < linear_limit
    good &= np.abs(yaw_average - target[2]) < yaw_limit
    if not np.any(target):
        speed_average = moving_average(np.linalg.norm(velocities[:, :2], axis=1), linear_width)
        good &= speed_average[yaw_width-linear_width:] < .08
    confirmed = np.flatnonzero(np.convolve(good, np.ones(hold, dtype=int), mode='valid') == hold)
    return float((confirmed[0]+yaw_width+hold-1)*dt) if len(confirmed) else None


def read_batch(folder):
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    with (folder / 'trials.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    expected = len(protocol['cases']) * protocol['gait_phase_samples']
    expected *= 1 + len(protocol['direction_names']) * len(protocol['levels_m_s'])
    if len(rows) != protocol['trials'] or len(rows) != expected:
        raise ValueError(f'Incomplete factorial design in {folder}')
    keys = {(row['case'], row['phase_bin'], row['direction'], row['impulse_m_s']) for row in rows}
    if len(keys) != expected:
        raise ValueError(f'Duplicate trial in {folder}')
    if protocol['calibration']['max_error_m_s'] > .02:
        raise ValueError(f'Failed applied impulse calibration in {folder}')
    dt = protocol['control_dt']
    frames = round(protocol['observation_after_push_seconds'] / dt)
    with np.load(folder / 'telemetry.npz') as trace:
        telemetry = trace['frames']
        onset = trace['onset']
        first_fall = trace['first_fall']
        target_commands = trace['commands']
        if not np.isfinite(telemetry).all():
            raise ValueError(f'Nonfinite rollout measurements in {folder}')
        for row in rows:
            i = int(row['trial_id'])
            paired = rows[int(row['control_id'])]
            if (paired['direction'] != 'none' or paired['case'] != row['case']
                    or paired['phase_bin'] != row['phase_bin']):
                raise ValueError('Invalid unforced control pairing')
            velocities = telemetry[onset[i]+1:onset[i]+frames+1, i, :3]
            if len(velocities) != frames or float(row['planned_at_impact_error']) >= .025:
                raise ValueError(f'Incomplete frame history or unsettled command in trial {i}')
            pre_fall = 0 <= first_fall[i] < onset[i]
            fell = onset[i] <= first_fall[i] < onset[i]+frames
            if pre_fall != (row['precondition_fell'] == 'True') or fell != (row['fell'] == 'True'):
                raise ValueError(f'Fall log mismatch in trial {i}')
            seconds = None if pre_fall or fell else recovery_confirmation(velocities, target_commands[i], dt)
            row.update(batch=folder.name, impulse_m_s=float(row['impulse_m_s']),
                       precondition_fell=bool(pre_fall), fell=bool(fell), recovered=seconds is not None,
                       recovery_seconds=seconds)
        for row in rows:
            row['paired_control_recovered'] = rows[int(row['control_id'])]['recovered']
    return protocol, rows

