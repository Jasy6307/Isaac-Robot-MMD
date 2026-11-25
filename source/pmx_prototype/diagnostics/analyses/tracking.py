"""Tracking diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_diagnose_tracking_margin(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/tracking_margin_diagnosis_20261116')

    def execute(stage, checkpoint, config):
        return workspace.execute(stage, [SOURCE / 'train_anon_locomotion.py', '--mode', 'eval',
            '--headless', '--num-envs', '1', '--checkpoint', checkpoint,
            '--output', OUT / stage, '--config-override', config], started=False)

    def main():
        started = time.perf_counter()
        sys.path.insert(0, str(SOURCE))
        import source_layout
        from evaluate_unified_walk_run import suite
        models = read(OUT / 'models.json')
        rows = []
        write(OUT / 'repeat_config.json', dict(unified_record_controller=True))
        execute('repeat_after', models['after'], OUT / 'repeat_config.json')
        report = read(OUT / 'repeat_after/report.json')
        rows.append(dict(stage='repeat_after', model='after', report=report))
        cfg = report['env_config']
        schedule = suite(protocol=report['acceptance_protocol'], idle_head=cfg.get('unified_idle_head_acceptance', False),
                         backward_walk=cfg.get('unified_eval_backward_walk', False),
                         stand_seconds=cfg.get('unified_eval_stand_hold_seconds'),
                         stop_timing=report.get('full_stop_timing', 'original'))
        for label, seconds in [('run_diagonal_135', 8), ('reverse_turn', 10)]:
            cases = []
            for item in schedule:
                cases.append([item[0], seconds if item[0] == label else item[1], item[2]])
                if item[0] == label:
                    break
            config = OUT / (label + '_config.json')
            write(config, dict(unified_eval_cases=cases, unified_record_controller=True))
            for name, checkpoint in models.items():
                stage = label + '_' + name
                execute(stage, checkpoint, config)
                rows.append(dict(stage=stage, model=name, target=label, report=read(OUT / stage / 'report.json')))
                write(OUT / 'partial_results.json', rows)
        write(OUT / 'result.json', dict(training=False, wall_seconds=time.perf_counter() - started, results=rows))
        write(OUT / 'status.json', dict(stage='complete', training=False))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_repeat_reverse_turn(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/reverse_turn_diagnosis_20261116')

    def main():
        models = read(OUT / 'models.json')
        rows = []
        started = time.perf_counter()
        for repeat, order in [(1, ['before', 'after']), (2, ['after', 'before'])]:
            for name in order:
                stage = f'repeat{repeat}_{name}'
                execute(stage, [SOURCE / 'train_anon_locomotion.py', '--mode', 'eval', '--headless', '--num-envs', '1', '--checkpoint', models[name], '--output', OUT / stage, '--config-override', OUT / 'prefix_config.json'])
                report = read(OUT / stage / 'report.json')
                segment = next((s for s in report['segments'] if s['label'] == 'reverse_turn'))
                rows.append(dict(repeat=repeat, model=name, segment=segment, total_falls=report['total_falls'], report=str(OUT / stage / 'report.json')))
                write(OUT / 'repeat_results.json', rows)
        schedule = {'stand_reverse_left': [[2, [0, 0, 0]], [8, [-0.9, 0, 0.6]]], 'stand_reverse_right': [[2, [0, 0, 0]], [8, [-0.9, 0, -0.6]]], 'right_run_to_reverse': [[2, [0, 0, 0]], [5, [1.1, 0, -0.6]], [5, [-0.9, 0, 0.6]]], 'backward_to_reverse': [[2, [0, 0, 0]], [5, [-0.9, 0, 0]], [5, [-0.9, 0, 0.6]]]}
        write(OUT / 'schedule.json', schedule)
        execute('phase_probes', [SOURCE / 'diagnostics/diagnose_pivot_failure.py', '--models', OUT / 'models.json', '--schedule', OUT / 'schedule.json', '--output', OUT / 'phase_probes'])
        result = dict(training=False, repeats=rows, phase_probes=read(OUT / 'phase_probes/report.json'), wall_seconds=time.perf_counter() - started)
        write(OUT / 'result.json', result)
        write(OUT / 'status.json', dict(stage='complete', training=False))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

