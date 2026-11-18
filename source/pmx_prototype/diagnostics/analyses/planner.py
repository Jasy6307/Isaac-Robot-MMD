"""Planner diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_compare_command_plans(workspace, legacy_argv=None):
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
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/command_plan_tests_20261116')
    CHECKPOINT = ROOT / 'logs/rsl_rl/anon_locomotion/retention_strength_20261116_original/policy_final.pt'

    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        if (OUT / 'plan.json').exists():
            raise RuntimeError('Duplicate experiment')
        start = time.perf_counter()
        before_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        variants = {'baseline': {}, 'tau_020': {'command_smoothing_seconds': 0.2}, 'accel_090': {'command_linear_acceleration_limit': 0.9}, 'accel_120': {'command_linear_acceleration_limit': 1.2}}
        plan = dict(training=False, checkpoint=str(CHECKPOINT), checkpoint_sha256=before_hash, variants=variants, unchanged='Reference blend formulas, weights, physics, rewards, yaw acceleration and original acceptance thresholds.', baseline=dict(command_smoothing_seconds=0.35, command_linear_acceleration_limit=0.6, command_yaw_acceleration_limit=1.2), promotion='No production/default update; any winning plan is only a validated diagnostic candidate.')
        write(OUT / 'plan.json', plan)
        reports = {}
        rows = {}
        durations = {}
        sys.path.insert(0, str(SOURCE / 'training'))
        from continue_sampling_pivot import regression
        for name, delta in variants.items():
            config = dict(unified_eval_lite=True, unified_record_controller=True, **delta)
            write(OUT / (name + '_config.json'), config)
            durations[name] = execute(name, [SOURCE / 'train_anon_locomotion.py', '--mode', 'eval', '--headless', '--num-envs', '1', '--checkpoint', CHECKPOINT, '--output', OUT / name, '--config-override', OUT / (name + '_config.json')])
            report = read(OUT / name / 'report.json')
            reports[name] = report
            targets = {s['label']: s for s in report['segments'] if s['label'] in ('run_diagonal_135', 'reverse_turn')}
            checks = regression(reports['baseline'], report)
            rows[name] = dict(delta=delta, total_falls=report['total_falls'], screen_passed=report['passed_lite_gate'], passed_segments=sum((s['passed'] for s in report['segments'])), targets=targets, regression=checks, no_regression=all((s['passed'] for s in checks)))
            write(OUT / 'result.json', dict(plan=plan, rows=rows, durations=durations, outcome='running'))
        candidates = [name for name in variants if name != 'baseline' and rows[name]['no_regression'] and (rows[name]['total_falls'] == 0) and all((s['passed'] for s in rows[name]['targets'].values()))]
        winner = None
        if candidates:
            winner = min(candidates, key=lambda n: sum((s['velocity_error'] for s in rows[n]['targets'].values())))
            write(OUT / 'pivot_models.json', dict(candidate=str(CHECKPOINT)))
            durations['candidate_pivots'] = execute('candidate_pivots', [SOURCE / 'diagnostics/diagnose_pivot_failure.py', '--models', OUT / 'pivot_models.json', '--output', OUT / 'candidate_pivots', '--config-override', OUT / (winner + '_config.json')])
            probe = read(OUT / 'candidate_pivots/report.json')
            rows[winner]['pivot_probe'] = probe
            if sum((s['total_falls'] for s in probe['models']['candidate'])) == 0:
                write(OUT / 'full_config.json', dict(unified_record_controller=True, **variants[winner]))
                durations['candidate_full'] = execute('candidate_full', [SOURCE / 'train_anon_locomotion.py', '--mode', 'eval', '--headless', '--num-envs', '1', '--checkpoint', CHECKPOINT, '--output', OUT / 'candidate_full', '--config-override', OUT / 'full_config.json'])
                full = read(OUT / 'candidate_full/report.json')
                old = {s['label']: s for s in read(ROOT / 'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
                slip = all((s['mean_speed'] >= old[s['label']]['mean_speed'] * 0.95 and s['stance_slip_geometry_m_s'] <= old[s['label']]['stance_slip_geometry_m_s'] * 0.8 for s in full['segments'] if s['label'] in ['run_backward', 'run_left', 'run_right']))
                rows[winner]['full_passed'] = bool(full['passed_unified_gate'] and full['total_falls'] == 0 and slip)
                rows[winner]['full_report'] = str(OUT / 'candidate_full/report.json')
        assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == before_hash
        write(OUT / 'result.json', dict(plan=plan, rows=rows, durations=durations, winner=winner, training=False, checkpoint_unchanged=True, wall_seconds=time.perf_counter() - start, outcome='complete'))
        write(OUT / 'status.json', dict(stage='complete', training=False))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_compare_split_command_plans(workspace, legacy_argv=None):
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
    import numpy as np
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/split_command_tests_20261117')
    CHECKPOINT = ROOT / 'logs/rsl_rl/anon_locomotion/retention_strength_20261116_original/policy_final.pt'
    PREVIOUS = ROOT / 'output/anon_locomotion/command_plan_tests_20261116'

    def evaluate(name, config):
        write(OUT / (name + '_config.json'), config)
        seconds = execute(name, [SOURCE / 'train_anon_locomotion.py', '--mode', 'eval', '--headless', '--num-envs', '1', '--checkpoint', CHECKPOINT, '--output', OUT / name, '--config-override', OUT / (name + '_config.json')])
        return (read(OUT / name / 'report.json'), seconds)

    def timing(name):
        data = np.load(workspace.input_path(OUT / name / 'rollout.npz'))
        dt = float(data['dt'])
        result = {}
        for label in ['run_diagonal_135', 'reverse_turn']:
            ix = np.flatnonzero(data['labels'] == label)
            velocity = data['velocities'][ix, 0, 0]
            target = float(data['commands'][ix[-1], 0, 0])
            window = round(0.5 / dt)
            filtered = np.convolve(velocity, np.ones(window) / window, mode='valid')
            reached = np.flatnonzero(filtered <= target * 0.9)
            result[label] = dict(actual_forward_90_percent_seconds=float((reached[0] + window - 1) * dt) if len(reached) else None, filter='Trailing .5-second mean')
        return result

    def full_checks(report, baseline):
        old = {s['label']: s for s in read(ROOT / 'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
        checks = []
        for segment in report['segments']:
            if segment['label'] not in ['run_backward', 'run_left', 'run_right']:
                continue
            a = old[segment['label']]
            speed = segment['mean_speed'] / a['mean_speed']
            slip = segment['stance_slip_geometry_m_s'] / a['stance_slip_geometry_m_s']
            checks.append(dict(label=segment['label'], speed_ratio=speed, slip_ratio=slip, passed=speed >= 0.95 and slip <= 0.8))
        regressions = regression(baseline, report)
        protected = [s for s in regressions if s['label'] in ['stop', 'stand', 'stand_end', 'run_backward', 'reverse_turn']]
        return dict(original_full_passed=report['passed_unified_gate'], total_falls=report['total_falls'], passed_segments=sum((s['passed'] for s in report['segments'])), slip_speed_checks=checks, stop_backward_checks=protected, accepted=bool(report['passed_unified_gate'] and report['total_falls'] == 0 and all((s['passed'] for s in checks)) and all((s['passed'] for s in protected))))
    sys.path.insert(0, str(SOURCE / 'training'))
    from continue_sampling_pivot import regression

    def main():
        if (OUT / 'plan.json').exists():
            raise RuntimeError('Duplicate experiment; choose a new output folder')
        start = time.perf_counter()
        sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        variants = {'baseline': {}, 'split_120': dict(command_plan_mode='split', command_speedup_limit=1.2, command_direction_change_limit=1.2, command_braking_limit=0.6), 'split_090': dict(command_plan_mode='split', command_speedup_limit=0.9, command_direction_change_limit=0.9, command_braking_limit=0.6)}
        brakes = [('stand', 4, [0, 0, 0]), ('run_forward', 8, [1.2, 0, 0]), ('stop', 5, [0, 0, 0]), ('run_backward', 8, [-1.2, 0, 0]), ('stop', 5, [0, 0, 0]), ('run_turn_left', 8, [1.1, 0, 0.6]), ('stop', 5, [0, 0, 0]), ('run_turn_right', 8, [1.1, 0, -0.6]), ('stop', 5, [0, 0, 0]), ('reverse_turn', 8, [-0.9, 0, 0.6]), ('stop', 5, [0, 0, 0]), ('stand_end', 4, [0, 0, 0])]
        plan = dict(training=False, checkpoint=str(CHECKPOINT), checkpoint_sha256=sha, variants=variants, brake_cases=brakes, unchanged='Actor, normalizers, reference blending, physics, rewards, yaw planner and all original acceptance thresholds.', promotion='No default/eval UI changes; experimental opt-in configuration only.', selection='Both target switches pass; zero falls; no regression versus fixed-actor baseline in all non-pivot lite cases and all brake cases. Then zero falls in 64 pivot probes, original full gate and slip/speed gate, plus stop/backward regression checks.', cumulative_training_seconds=51378.49378980009)
        write(OUT / 'plan.json', plan)
        reports = {}
        brake_reports = {}
        rows = {}
        durations = {}
        for name, delta in variants.items():
            report, seconds = evaluate(name + '_lite', dict(unified_eval_lite=True, unified_record_controller=True, **delta))
            reports[name] = report
            durations[name + '_lite'] = seconds
            brake, seconds = evaluate(name + '_brakes', dict(unified_eval_cases=brakes, unified_record_controller=True, **delta))
            brake_reports[name] = brake
            durations[name + '_brakes'] = seconds
            targets = {s['label']: s for s in report['segments'] if s['label'] in ['run_diagonal_135', 'reverse_turn']}
            checks = regression(reports['baseline'], report)
            brake_checks = regression(brake_reports['baseline'], brake)
            rows[name] = dict(targets=targets, response=timing(name + '_lite'), lite_passed=report['passed_lite_gate'], lite_passed_segments=sum((s['passed'] for s in report['segments'])), lite_falls=report['total_falls'], lite_regression=checks, brakes_passed=all((s['passed'] for s in brake['segments'])), brakes_falls=brake['total_falls'], brake_regression=brake_checks)
            rows[name]['eligible'] = bool(name != 'baseline' and report['total_falls'] == 0 and (brake['total_falls'] == 0) and all((s['passed'] for s in targets.values())) and all((s['passed'] for s in checks)) and all((s['passed'] for s in brake_checks)) and rows[name]['brakes_passed'])
            write(OUT / 'result.json', dict(plan=plan, rows=rows, durations=durations, outcome='screening', training=False))
        candidates = sorted([n for n in variants if rows[n]['eligible']], key=lambda n: sum((s['velocity_error'] for s in rows[n]['targets'].values())))
        winner = None
        baseline_full = read(PREVIOUS / 'baseline_full/report.json')
        for name in candidates:
            write(OUT / 'pivot_models.json', dict(candidate=str(CHECKPOINT)))
            durations[name + '_pivots'] = execute(name + '_pivots', [SOURCE / 'diagnostics/diagnose_pivot_failure.py', '--models', OUT / 'pivot_models.json', '--output', OUT / (name + '_pivots'), '--config-override', OUT / (name + '_lite_config.json')])
            pivots = read(OUT / (name + '_pivots/report.json'))
            rows[name]['pivot_probe'] = pivots
            if sum((s['total_falls'] for s in pivots['models']['candidate'])) != 0:
                continue
            full, seconds = evaluate(name + '_full', dict(unified_record_controller=True, **variants[name]))
            durations[name + '_full'] = seconds
            assert full['evaluator_sha256'] == baseline_full['evaluator_sha256']
            rows[name]['full_checks'] = full_checks(full, baseline_full)
            write(OUT / 'result.json', dict(plan=plan, rows=rows, durations=durations, outcome='full_validation', training=False))
            if rows[name]['full_checks']['accepted']:
                winner = name
                break
        assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == sha
        write(OUT / 'result.json', dict(plan=plan, rows=rows, durations=durations, winner=winner, checkpoint_unchanged=True, training=False, training_seconds=0, cumulative_training_seconds=plan['cumulative_training_seconds'], wall_seconds=time.perf_counter() - start, outcome='complete'))
        write(OUT / 'status.json', dict(stage='complete', training=False, winner=winner))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

