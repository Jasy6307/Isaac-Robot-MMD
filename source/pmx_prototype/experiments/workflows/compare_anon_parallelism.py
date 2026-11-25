"""Finish the active scratch lineage, then compare equal-sample fresh PPO runs.

Run with Isaac's Python outside the restricted socket sandbox. No GUI is used.
The controller serializes all Isaac jobs and writes a resumable state/report.
"""
from source.paths import REPO_ROOT
from source.pmx_prototype.experiments import jobs
from source.pmx_prototype.experiments.jobs import read, write
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(REPO_ROOT)
PYTHON = Path('I:/isaac61/envs/isaac61/python.exe')
ENTRY = ROOT / 'source/pmx_prototype/train_anon_locomotion.py'
CURRENT = ROOT / 'output/anon_locomotion/scratch_v8_unified_20261116_12288'
CURRENT_RUN = ROOT / 'logs/rsl_rl/anon_locomotion/v8_scratch_unified_20261116_12288'
OUT = ROOT / 'output/anon_locomotion/equal_samples_20261116'






def launch(run, output, config, envs, iterations, checkpoint=None, keep_budget=False):
    command = [str(PYTHON), '-u', str(ENTRY), '--mode', 'train', '--headless', '--no-acceptance-video',
               '--num-envs', str(envs), '--iterations', str(iterations), '--seed', '42',
               '--run-dir', str(run), '--experiment-output', str(output),
               '--scratch-target', 'latest_unified', '--acceptance-interval', str(3750 if envs == 8192 else 2500),
               '--config-override', str(config)]
    command += ['--scratch-continuation', '--checkpoint', str(checkpoint)] if checkpoint else ['--from-scratch']
    if keep_budget:
        command += ['--scratch-continue-after-pass']
    output.mkdir(parents=True, exist_ok=True)
    write(OUT / 'controller_status.json', dict(phase='running', run_dir=str(run), num_envs=envs,
          iterations_to_run=iterations, command=command, updated_unix=time.time()))
    log_path = output / (run.name + '.log')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        write(OUT / 'active_process.json', dict(pid=process.pid, run_dir=str(run), log=str(log_path)))
        code = process.wait()
    if code or not (run / 'training_time.json').exists():
        raise RuntimeError(f'Training failed ({code}); inspect {log_path}')
    return read(run / 'training_time.json')


def summarize(envs):
    output = OUT / str(envs)
    history = read(output / 'history.json')
    first = next((item for item in history if item['passed']), None)
    latest = history[-1]
    return dict(num_envs=envs, completed_iterations=latest['completed_iterations'],
                total_environment_steps=latest['completed_iterations'] * envs * 48,
                active_training_seconds=latest['active_training_seconds'],
                evaluation_seconds=latest['evaluation_seconds_cumulative'],
                train_plus_evaluation_seconds=latest['active_training_seconds'] + latest['evaluation_seconds_cumulative'],
                first_pass=None if first is None else dict(
                    iterations=first['completed_iterations'],
                    total_environment_steps=first['completed_iterations'] * envs * 48,
                    active_training_seconds=first['active_training_seconds'],
                    evaluation_seconds=first['evaluation_seconds_cumulative'],
                    train_plus_evaluation_seconds=first['active_training_seconds'] + first['evaluation_seconds_cumulative'],
                    checkpoint=first['checkpoint'], report=first['report']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=int, default=18000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # The existing continuation was launched by the assistant; never overlap it.
    waiting = Path(str(CURRENT_RUN) + '_cont1')
    deadline = time.monotonic() + args.wait_seconds
    while not (waiting / 'training_time.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f'Existing continuation has not finished: {waiting}')
        write(OUT / 'controller_status.json', dict(phase='waiting_for_current_continuation', run_dir=str(waiting)))
        time.sleep(20)
    # Allow Kit to release CUDA after its final timing write.
    time.sleep(30)
    timing = read(waiting / 'training_time.json')
    for continuation in range(2, 10):
        if read(CURRENT / 'experiment_status.json')['passed']:
            break
        next_run = Path(str(CURRENT_RUN) + f'_cont{continuation}')
        if (next_run / 'training_time.json').exists():
            timing = read(next_run / 'training_time.json')
        else:
            timing = launch(next_run, CURRENT, CURRENT / 'config_override.json', 12288, 1000,
                            checkpoint=waiting / 'policy_final.pt')
        waiting = next_run
    current = read(CURRENT / 'experiment_status.json')
    write(OUT / 'current_lineage_result.json', dict(status=current, timing=timing))
    if not current['passed']:
        raise RuntimeError('Current lineage did not pass after 8 further continuations; paired comparison has not started')

    baseline = read(CURRENT / 'config_override.json')
    curriculum_keys = ['curriculum_iterations', 'continuous_curriculum_start', 'unified_curriculum_start',
                       'unified_curriculum_iterations', 'unified_backward_curriculum_start',
                       'unified_backward_curriculum_iterations', 'unified_right_curriculum_iterations']
    plan = dict(seed=42, steps_per_env_per_iteration=48, env_counts=[8192, 12288],
                sample_interval=8192*750*48, ppo_epochs=5, ppo_minibatches=4,
                schedule='Sequential runs; equal sample budgets, sample-aligned curricula and validation intervals',
                curriculum_keys_scaled=curriculum_keys,
                acceptance='Identical full unified gate, zero falls and original slip/speed thresholds',
                stop_rule='Keep each arm through the common budget; expand both budgets until both have a passing checkpoint',
                limitation='One seed; first sampled passing checkpoint is an upper bound. No visual acceptance is implied.')
    write(OUT / 'plan.json', plan)
    for envs in (8192, 12288):
        config = dict(baseline)
        factor = 8192 / envs
        for key in curriculum_keys:
            if key in config:
                config[key] = baseline[key] * factor
        config['scratch_anneal_iterations'] = 300 * factor
        write(OUT / str(envs) / 'config_override.json', config)

    # Every budget is an exact multiple of both arms' validation intervals.
    for target_12288 in (4500, 6000, 7500, 9000, 10500, 12000):
        results = []
        for envs in (8192, 12288):
            target = target_12288 * 12288 // envs
            output = OUT / str(envs)
            run = ROOT / f'logs/rsl_rl/anon_locomotion/equal_samples_20261116_{envs}_{target}'
            if not (run / 'training_time.json').exists():
                previous_status = read(output / 'experiment_status.json') if (output / 'experiment_status.json').exists() else None
                completed = previous_status['completed_iterations'] if previous_status else 0
                checkpoint = None
                if previous_status:
                    previous_run = Path(previous_status['checkpoint']).parent
                    checkpoint = previous_run / 'policy_final.pt'
                if target > completed:
                    launch(run, output, output / 'config_override.json', envs, target-completed,
                           checkpoint=checkpoint, keep_budget=True)
            results.append(summarize(envs))
        assert results[0]['total_environment_steps'] == results[1]['total_environment_steps']
        report = dict(plan=plan, results=results, shared_total_environment_steps=results[0]['total_environment_steps'])
        if all(result['first_pass'] for result in results):
            a, b = [result['first_pass'] for result in results]
            report['training_time_ratio_12288_over_8192'] = b['active_training_seconds']/a['active_training_seconds']
            report['total_time_ratio_12288_over_8192'] = b['train_plus_evaluation_seconds']/a['train_plus_evaluation_seconds']
        write(OUT / 'comparison_report.json', report)
        if all(result['first_pass'] for result in results):
            write(OUT / 'controller_status.json', dict(phase='complete', report=str(OUT/'comparison_report.json')))
            return
    raise RuntimeError('Equal-sample budget exhausted without both arms passing; inspect comparison_report.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(OUT / 'controller_status.json', dict(phase='error', error=str(error)))
        raise
