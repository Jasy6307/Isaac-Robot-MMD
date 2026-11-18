"""Response diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_audit_response_windows(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    from pathlib import Path
    import numpy as np
    ROOT = Path(REPO_ROOT)
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117')

    def first_dwell(flags, dt, dwell=1.0):
        length = round(dwell / dt)
        times = []
        for e in range(flags.shape[1]):
            hits = np.flatnonzero(np.convolve(flags[:, e].astype(int), np.ones(length, dtype=int), mode='valid') == length)
            times.append(float(hits[0] * dt) if len(hits) else None)
        finite = [t for t in times if t is not None]
        return dict(by_env_seconds=times, reached=len(finite), total=len(times), median_seconds=float(np.median(finite)) if finite else None, maximum_seconds=max(finite) if finite else None)

    def main():
        schedule = json.loads(workspace.read_text(OUT / 'transition_schedule.json'))
        result = []
        for case, segments in schedule.items():
            data = {name: np.load(workspace.input_path(OUT / 'transitions' / f'{name}_{case}.npz')) for name in ['old', 'new']}
            dt = float(np.diff(data['old']['time']).mean())
            offset = 0
            logical = {}
            for name, z in data.items():
                h = z['headings']
                v = z['root_velocity']
                logical[name] = np.stack([np.sin(h) * v[:, :, 0] - np.cos(h) * v[:, :, 1], np.cos(h) * v[:, :, 0] + np.sin(h) * v[:, :, 1], v[:, :, 5]], -1)
            for index, (seconds, cmd) in enumerate(segments):
                n = round(seconds / dt)
                end = offset + n
                valid = ~(data['old']['done'][:end].any(0) | data['new']['done'][:end].any(0))
                row = dict(case=case, segment=index, command=cmd, seconds=seconds, valid_envs=int(valid.sum()), total_envs=len(valid), models={})
                target = np.array(cmd)
                speed = np.linalg.norm(target[:2])
                stationary = np.abs(target).sum() < 0.001
                linear_limit = max(0.06, speed * 0.22)
                yaw_limit = 0.1 if stationary else 0.2
                for name, z in data.items():
                    physical = logical[name][offset:end, valid]
                    planned = z['ref_commands'][offset:end, valid]
                    if not valid.any():
                        row['models'][name] = dict(falls=int(z['done'][offset:end].sum()), valid_metrics=False)
                        continue
                    windows = {}
                    for label, begin, finish in [('prefix_5_seconds', 2.0, min(5.0, seconds)), ('whole_original_exclusion', 2.0, seconds), ('last_3_seconds', max(0.0, seconds - 3), seconds), ('later_start_3_seconds', 3.0, seconds)]:
                        if finish <= begin:
                            continue
                        chosen = physical[round(begin / dt):round(finish / dt)]
                        plan_window = planned[round(begin / dt):round(finish / dt)]
                        planning_bias = (plan_window[:, :, :2] - target[:2]).mean(0)
                        tracking_bias = (chosen[:, :, :2] - plan_window[:, :, :2]).mean(0)
                        bias = np.linalg.norm((chosen[:, :, :2] - target[:2]).mean(0), axis=-1)
                        yaw = np.abs((chosen[:, :, 2] - target[2]).mean(0))
                        within = (bias < linear_limit) & (yaw < yaw_limit)
                        if stationary:
                            within &= np.linalg.norm(chosen[:, :, :2], axis=-1).mean(0) < 0.08
                        windows[label] = dict(start=begin, end=finish, velocity_error_mean=float(bias.mean()), velocity_error_max=float(bias.max()), yaw_error_mean=float(yaw.mean()), yaw_error_max=float(yaw.max()), planning_bias_vector_mean=planning_bias.mean(0).tolist(), physical_vs_planned_bias_vector_mean=tracking_bias.mean(0).tolist(), original_tolerance_fraction=float(within.mean()))
                    plan_flag = (np.linalg.norm(planned[:, :, :2] - target[:2], axis=-1) < linear_limit) & (abs(planned[:, :, 2] - target[2]) < yaw_limit)
                    plan_close = (np.linalg.norm(planned[:, :, :2] - target[:2], axis=-1) < 0.02) & (abs(planned[:, :, 2] - target[2]) < 0.02)
                    width = round(1 / dt)
                    mean = np.stack([physical[i - width + 1:i + 1].mean(0) for i in range(width - 1, n)])
                    physical_flag = (np.linalg.norm(mean[:, :, :2] - target[:2], axis=-1) < linear_limit) & (abs(mean[:, :, 2] - target[2]) < yaw_limit)
                    linear_flag = np.linalg.norm(mean[:, :, :2] - target[:2], axis=-1) < linear_limit
                    if stationary:
                        mean_speed = np.stack([np.linalg.norm(physical[i - width + 1:i + 1, :, :2], axis=-1).mean(0) for i in range(width - 1, n)])
                        physical_flag &= mean_speed < 0.08
                        linear_flag &= mean_speed < 0.08
                    padded = np.concatenate([np.zeros((width - 1, int(valid.sum())), dtype=bool), physical_flag])
                    linear_padded = np.concatenate([np.zeros((width - 1, int(valid.sum())), dtype=bool), linear_flag])
                    row['models'][name] = dict(falls=int(z['done'][offset:end].sum()), valid_metrics=True, windows=windows, plan_within_original_tolerance=first_dwell(plan_flag, dt), plan_within_002=first_dwell(plan_close, dt), linear_response_within_original_tolerance=first_dwell(linear_padded, dt), physical_within_original_tolerance=first_dwell(padded, dt))
                result.append(row)
                offset = end
        summary = dict(training=False, rows=result, method='Physical linear response: one-second trailing mean, first entry into original tracking tolerances sustained for one second. Yaw steady bias is reported separately: gait oscillations can prevent a short combined velocity/yaw dwell despite a passing long-window yaw bias. Planning-near-target uses0.02 only as a timing diagnostic, never as an acceptance threshold.', official_gates_changed=False, limitation='Response metrics use the matched subset with no prior fall before segment end; all falls are separately retained, not excused. Alternative windows are sensitivity diagnostics only.')
        workspace.write_text(OUT / 'response_window_audit.json', json.dumps(summary, indent=2))
        print(json.dumps([dict(case=r['case'], segment=r['segment'], command=r['command'], valid=r['valid_envs'], metrics={name: dict(falls=v['falls'], response=v.get('physical_within_original_tolerance', {}).get('median_seconds'), plan=v.get('plan_within_002', {}).get('median_seconds'), windows=v.get('windows')) for name, v in r['models'].items()}) for r in result], indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_replay_acceptance_v2(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import argparse
    import copy
    import hashlib
    import json
    from pathlib import Path
    import sys
    import numpy as np
    import torch
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'diagnostics')]
    import source_layout
    from evaluate_unified_walk_run import command, linear_tracking_error
    from locomotion_acceptance import score_segment, summarize, specification
    from command_plan import command_increment

    def replay(folder):
        old = json.loads(workspace.read_text(folder / 'report.json'))
        z = np.load(workspace.input_path(folder / 'rollout.npz'))
        dt = float(z['dt'])
        cfg = old['env_config']
        vel = z['velocities'].mean(1)
        root = list(z['link_names']).index(cfg['reference_root_link_name'])
        pos = z['poses'][:, 0, root, :3]
        delta = np.vstack([np.zeros((1, 3)), np.diff(pos, axis=0)]) / dt
        heading = z['headings'][:, 0]
        geo = np.stack([np.sin(heading) * delta[:, 0] - np.cos(heading) * delta[:, 1], np.cos(heading) * delta[:, 0] + np.sin(heading) * delta[:, 1]], -1)
        current = torch.zeros((1, 3))
        items = []
        start = 0
        for segment in old['segments']:
            n = round(segment['seconds'] / dt)
            initial = current[0].numpy().copy()
            planned = []
            requested = []
            for t in range(n):
                cmd = command(segment['label'], t * dt, segment['seconds'], segment['command'])
                requested.append(cmd)
                current += command_increment(torch.tensor([cmd], dtype=torch.float32), current, dt, cfg)
                if z['dones'][start + t].any():
                    current.zero_()
                planned.append(current[0].numpy().copy())
            raw = np.asarray(requested, dtype=np.float32)
            item = copy.deepcopy(segment)
            if not z['dones'][start:start + n].any():
                error = linear_tracking_error(geo[start + 60:start + n] - raw[60:, :2], segment['label'].startswith('ramp_') or segment['label'] == 'direction_sweep', dt)
                if abs(error - segment['geometric_velocity_error']) > 0.0002:
                    raise ValueError('Root-position replay mismatch')
            item['legacy_passed'] = segment.get('legacy_passed', segment['passed'])
            item['response_steady'] = score_segment(item, vel[start:start + n], np.asarray(planned), raw, geo[start:start + n], initial, cfg, dt)
            item['passed'] = item['response_steady']['passed']
            items.append(item)
            start += n
        return dict(source_report=str(folder / 'report.json'), replay_only=True, training=False, acceptance_protocol='response_steady_v2', acceptance_protocol_sha256=hashlib.sha256((SOURCE / 'training/locomotion_acceptance.py').read_bytes()).hexdigest(), source_complete_scope=old['complete_scope'], source_diagnostic=old.get('diagnostic', False), specification=specification(), summary=summarize(items), segments=items, caveat='Archived short stops may have insufficient steady observation. Such segments remain inconclusive; this replay cannot promote a model or manufacture missing samples.')
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

