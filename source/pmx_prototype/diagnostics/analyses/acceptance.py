"""Acceptance diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_audit_scratch_acceptance_limits(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import argparse
    import json
    from pathlib import Path
    import sys
    import numpy as np
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'diagnostics')]
    from evaluate_unified_walk_run import command, linear_tracking_error, yaw_tracking_error

    def failures(s, report):
        result = []

        def upper(name, value, limit):
            if value >= limit:
                result.append(dict(metric=name, value=value, limit=limit, excess=value - limit, excess_fraction=(value - limit) / limit))
        if s['falls']:
            result.append(dict(metric='falls', value=s['falls'], limit=0))
        upper('velocity_error', s['velocity_error'], s['velocity_limit'])
        upper('geometric_velocity_error', s['geometric_velocity_error'], s['velocity_limit'])
        stationary = s['label'] in ['stand', 'stand_end', 'stop']
        upper('yaw_error', s['yaw_error'], 0.1 if stationary else 0.2)
        upper('keypoint_rmse_m', s['metrics']['Imitation/keypoint_rmse_m'], 0.075)
        if stationary:
            upper('mean_speed', s['mean_speed'], 0.08)
        if s['label'] in ['stand', 'stand_end']:
            upper('displacement_m', s['displacement_m'], 0.25)
            upper('pelvis_yaw_drift_rad', s['pelvis_yaw_drift_rad'], 0.35)
        if s['label'] in ['run_forward', 'run_backward', 'run_left', 'run_right']:
            force = report['env_config'].get('unified_eval_force_flight')
            value = s['force_flight_fraction'] if force else s['flight_fraction']
            if value <= 0.01:
                result.append(dict(metric='flight_fraction', value=value, limit=0.01, shortfall=0.01 - value))
        upper('clearance_deficit_m', s['metrics'].get('Imitation/near_hip_clearance_deficit_m', 1), 0.015)
        if report['env_config'].get('unified_command_frame_metrics'):
            upper('pelvis_orientation_rmse_rad', s['metrics']['Imitation/pelvis_orientation_rmse_rad'], 0.25)
        assert bool(result) == (not s['passed']), (s['label'], result, s['passed'])
        return result

    def analyze(folder):
        report = read(folder / 'report.json')
        data = np.load(workspace.input_path(folder / 'rollout.npz'))
        dt = float(data['dt'])
        start = 0
        rows = []
        for index, s in enumerate(report['segments']):
            n = round(s['seconds'] / dt)
            vel = data['velocities'][start:start + n, 0]
            target = np.asarray([command(s['label'], i * dt, s['seconds'], s['command']) for i in range(n)], dtype=np.float32)
            variable = s['label'].startswith('ramp_') or s['label'] == 'direction_sweep'
            error = vel[:, :2] - target[:, :2]
            score = linear_tracking_error(error[60:], variable, dt)
            yaw = yaw_tracking_error((vel[:, 2] - target[:, 2])[60:], target[60:, 2])
            assert abs(score - s['velocity_error']) < 0.0001, (index, score, s['velocity_error'])
            assert abs(yaw - s['yaw_error']) < 0.0001, (index, yaw, s['yaw_error'])
            reasons = failures(s, report)
            windows = {}
            for label, begin in [('official', 2.0), ('last_two_seconds', max(2.0, s['seconds'] - 2)), ('last_second', max(2.0, s['seconds'] - 1))]:
                ix = round(begin / dt)
                windows[label] = dict(start_seconds=begin, linear_error=linear_tracking_error(error[ix:], variable, dt), yaw_error=yaw_tracking_error((vel[:, 2] - target[:, 2])[ix:], target[ix:, 2]), mean_velocity=vel[ix:].mean(0).tolist(), mean_command=target[ix:].mean(0).tolist())
            rows.append(dict(index=index, label=s['label'], previous_label=report['segments'][index - 1]['label'] if index else None, command=s['command'], seconds=s['seconds'], passed=s['passed'], failures=reasons, windows=windows))
            start += n
        assert start == len(data['velocities'])
        return dict(report=str(folder / 'report.json'), evaluator_sha256=report['evaluator_sha256'], passed=report['passed_unified_gate'], falls=report['total_falls'], passed_segments=sum((s['passed'] for s in report['segments'])), failure_rows=[s for s in rows if not s['passed']], walk_rows=[s for s in rows if s['label'] == 'walk'], note='Later windows measure sensitivity only; the official two-second exclusion and every threshold remain unchanged.')

    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--report-folder', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(legacy_argv)
        result = analyze(args.report_folder)
        tests = ROOT / 'output/anon_locomotion/split_command_tests_20261117'
        current = ROOT / 'output/anon_locomotion/scratch_split_20261117/acceptance/checkpoint_006000/lite'
        result['matched_lite_comparison'] = {}
        for name, folder in [('previous_policy_same_planner', tests / 'split_120_lite'), ('scratch_6000_same_planner', current)]:
            if not (folder / 'report.json').exists():
                continue
            report = read(folder / 'report.json')
            assert report['evaluator_sha256'] == result['evaluator_sha256']
            result['matched_lite_comparison'][name] = [dict(label=s['label'], seconds=s['seconds'], passed=s['passed'], velocity_error=s['velocity_error'], velocity_limit=s['velocity_limit'], mean_speed=s['mean_speed'], yaw_error=s['yaw_error']) for s in report['segments'] if s['label'] in ['walk', 'run_diagonal_135', 'reverse_turn']]
        baseline = read(ROOT / 'output/anon_locomotion/command_plan_tests_20261116/baseline_full/report.json')
        assert baseline['evaluator_sha256'] == result['evaluator_sha256']
        result['previous_default_full'] = dict(passed=baseline['passed_unified_gate'], falls=baseline['total_falls'], caveat='This full baseline uses the previous0.6 planner. The matched lite comparison above uses the same split planner.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        workspace.write_text(args.output, json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k not in ['walk_rows']}, indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_smoke_acceptance_v2(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    ROOT = Path(REPO_ROOT)
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_fix_20261117')

    def main():
        if (OUT / 'smoke_result.json').exists():
            raise RuntimeError('Integration output already exists')
        models_path = workspace.input_dir / 'models.json'
        if workspace.input_dir == workspace.original and not models_path.exists():
            models_path = ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117/models.json'
        models = json.loads(workspace.read_text(models_path))
        result = {}
        for name, path in models.items():
            before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            start = time.perf_counter()
            execute(name + '_smoke', [ROOT / 'source/pmx_prototype/train_anon_locomotion.py',
                '--mode', 'eval', '--headless', '--num-envs', '1', '--checkpoint', path,
                '--output', OUT / (name + '_smoke'), '--config-override', OUT / 'smoke_config.json'])
            report = json.loads(workspace.read_text(OUT / (name + '_smoke') / 'report.json'))
            assert report['acceptance_protocol'] == 'response_steady_v2' and report['diagnostic'] and (not report['passed_unified_gate'])
            assert report['acceptance_protocol_sha256']
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == before
            result[name] = dict(checkpoint_unchanged=True, seconds=time.perf_counter() - start, falls=report['total_falls'], legacy_passed=sum((s['legacy_passed'] for s in report['segments'])), v2_summary=report['response_steady_summary'], segments=[dict(label=s['label'], legacy_passed=s['legacy_passed'], v2=s['response_steady']) for s in report['segments']])
            workspace.write_text(OUT / 'smoke_result.partial.json', json.dumps(result, indent=2))
        assert next((s for s in result['new']['segments'] if s['label'] == 'walk'))['v2']['status'] == 'failed'
        workspace.write_text(OUT / 'smoke_result.json', json.dumps(dict(training=False, models=result), indent=2))
        workspace.write_text(OUT / 'smoke_status.json', json.dumps(dict(stage='complete', training=False)))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

