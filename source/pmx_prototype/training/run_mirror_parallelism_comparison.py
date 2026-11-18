"""Train the accepted mirror recipe from scratch at 12,288 environments."""

from pathlib import Path
import json
import os

import run_balanced_scratch as training

BASE = training.ROOT / 'output/anon_locomotion/scratch_mirror_augmentation_v1'
NAME = 'scratch_mirror_augmentation_v1_12288'
OUTPUT = BASE.parent / NAME
RUN = training.ROOT / 'logs/rsl_rl/anon_locomotion' / NAME
NUM_ENVS = 12288


def compare():
    baseline = training.read(BASE / 'result.json')
    candidate = training.read(OUTPUT / 'result.json')
    points = {}
    for folder, result in ((BASE, baseline), (OUTPUT, candidate)):
        plan = training.read(folder / 'experiment_plan.json')
        history = training.read(folder / 'acceptance/history.json')
        steps_per_iteration = result['num_envs'] * plan['steps_per_env_per_iteration']
        points[result['num_envs']] = {
            row['completed_iterations'] * steps_per_iteration: row for row in history
        }

    matched = []
    for steps in sorted(points[8192].keys() & points[NUM_ENVS].keys()):
        old, new = points[8192][steps], points[NUM_ENVS][steps]
        old_time, new_time = (row['active_training_seconds'] for row in (old, new))
        matched.append(dict(
            environment_steps=steps,
            baseline_iterations=old['completed_iterations'],
            candidate_iterations=new['completed_iterations'],
            baseline_training_seconds=old_time,
            candidate_training_seconds=new_time,
            training_time_change_fraction=new_time / old_time - 1,
            throughput_ratio=old_time / new_time,
            baseline_acceptance_kind=old['acceptance_kind'],
            candidate_acceptance_kind=new['acceptance_kind'],
            baseline_report=old['report'],
            candidate_report=new['report'],
        ))

    first_passes = {}
    for result in (baseline, candidate):
        passed = result['first_complete_and_low_speed_pass']
        first_passes[str(result['num_envs'])] = (
            dict(iterations=passed['completed_iterations'],
                 environment_steps=passed['completed_iterations'] * result['num_envs'] * 48,
                 training_seconds=passed['training_seconds'],
                 acceptance_seconds=result['acceptance_seconds'],
                 wall_seconds_including_setup=result['wall_seconds_including_setup'])
            if passed else None
        )
    report = dict(
        baseline_result=str(BASE / 'result.json'),
        candidate_result=str(OUTPUT / 'result.json'),
        matched_environment_steps=matched,
        first_complete_and_low_speed_pass=first_passes,
        candidate_completed_iterations=candidate['completed_iterations'],
        candidate_training_seconds=candidate['training_seconds'],
        cumulative_training_seconds=candidate['cumulative_training_seconds'],
        measurement_note='One run per setting, with actual shared-GPU timings. '
                         'At equal sample counts, 12288 environments require fewer PPO updates.',
    )
    training.write(OUTPUT / 'comparison.json', report)


def main():
    prepared = training.read(BASE / 'prepared_plan.json')
    result = training.read(BASE / 'result.json')
    if not result['accepted'] or result['num_envs'] != 8192:
        raise ValueError('Comparison requires the accepted 8192-environment baseline')
    # The reusable controller now takes an environment count. Train and
    # evaluator source, recipe, references, and physics files must match.
    sources = {
        training.SOURCE / name: sha for name, sha in prepared['source_hashes'].items()
        if name != 'training/run_balanced_scratch.py'
    }
    expected_files = {**sources, **{Path(name): sha for name, sha in prepared['input_hashes'].items()},
                      Path(prepared['recipe']): prepared['recipe_sha256']}
    changed = [str(path) for path, sha in expected_files.items()
               if not path.is_file() or training.hashlib.sha256(path.read_bytes()).hexdigest() != sha]
    if changed:
        raise ValueError('Baseline files changed: ' + ', '.join(changed))

    training.OUT = OUTPUT
    training.RUN = RUN
    training.RECIPE = Path(prepared['recipe'])
    training.NUM_ENVS = NUM_ENVS
    training.MAXIMUM = prepared['maximum_iterations']
    training.BEFORE = result['cumulative_training_seconds']
    training.main()
    compare()


if __name__ == '__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    try:
        main()
    except Exception as error:
        training.write(OUTPUT / 'comparison_error.json', dict(error=str(error)))
        raise
