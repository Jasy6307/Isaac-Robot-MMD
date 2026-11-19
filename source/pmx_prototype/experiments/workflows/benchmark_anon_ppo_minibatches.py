"""Short, serial 12,288-environment PPO timing comparison."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from source.pmx_prototype.experiments import jobs

OUTPUT = ROOT / 'output/anon_locomotion/ppo_minibatch_benchmark_20261118'
ENTRY = ROOT / 'source/pmx_prototype/train_anon_locomotion.py'
STOP_STATE = OUTPUT.parent / 'scratch_mirror_augmentation_v1_12288/stop_state_004018.json'
NUM_ENVS = 12288
WARMUP = 6
MEASURED = 20
ORDER = (4, 6, 6, 4)


def metric(rows, samples_per_iteration):
    collection = sum(row[0] for row in rows)
    update = sum(row[1] for row in rows)
    samples = len(rows) * samples_per_iteration
    return dict(
        iterations=len(rows), collection_seconds_per_iteration=collection / len(rows),
        update_seconds_per_iteration=update / len(rows),
        combined_seconds_per_iteration=(collection + update) / len(rows),
        collection_steps_per_second=samples / collection,
        update_sample_equivalents_per_second=samples / update,
        throughput_steps_per_second=samples / (collection + update),
    )


def read_iteration_timings(log, start_iteration):
    pattern = r'Learning iteration[ \t]+(\d+)/\d+[^\n]*\n'
    fields = re.split(pattern, log.read_text(encoding='utf-8', errors='replace'), flags=re.M)
    iterations = []
    timings = []
    for index, body in zip(fields[1::2], fields[2::2]):
        collection = re.search(r'Collection time:\s*([0-9.]+)s', body)
        update = re.search(r'Learning time:\s*([0-9.]+)s', body)
        if collection is None or update is None:
            raise ValueError(f'Missing PPO timings for iteration {index} in {log.name}')
        iterations.append(int(index))
        timings.append((float(collection[1]), float(update[1])))
    if iterations != list(range(start_iteration, start_iteration + WARMUP + MEASURED)):
        raise ValueError(f'Unexpected iteration indices in {log.name}')
    return timings


def main(resume=False):
    stopped = jobs.read(STOP_STATE)
    checkpoint = Path(stopped['checkpoint'])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checksum != stopped['checkpoint_sha256']:
        raise ValueError('Benchmark checkpoint differs from the recorded stopped run')
    plan_path = OUTPUT / 'plan.json'
    plan = dict(
        num_envs=NUM_ENVS, seed=42, checkpoint=str(checkpoint), checkpoint_sha256=checksum,
        warmup_iterations=WARMUP, measured_iterations=MEASURED, execution_order=list(ORDER),
        acceptance_runs=0, timing='Total environment steps divided by rollout + learning time.',
        cumulative_training_seconds_before=stopped['cumulative_training_seconds'],
    )
    if resume:
        if jobs.read(plan_path) != plan:
            raise ValueError('Saved benchmark plan differs from current parameters')
    else:
        if plan_path.exists():
            raise FileExistsError(f'Benchmark already exists: {OUTPUT}')
        jobs.write(plan_path, plan)
    (OUTPUT / 'controller.pid').write_text(str(os.getpid()), encoding='ascii')
    runs = []
    timings_by_count = {count: [] for count in set(ORDER)}
    baseline_config = None
    for index, count in enumerate(ORDER, 1):
        stage = f'{index:02d}_mb{count}'
        run = OUTPUT / stage
        wall = None
        if run.exists():
            if not resume or not (run / 'training_time.json').exists():
                raise FileExistsError(f'Benchmark arm directory already exists: {run}')
        else:
            wall = jobs.execute(ROOT, OUTPUT, stage, [
                ENTRY, '--mode', 'train', '--headless', '--no-acceptance-video', '--seed', '42', '--num-envs', NUM_ENVS,
                '--iterations', WARMUP + MEASURED, '--ppo-mini-batches', count,
                '--checkpoint', checkpoint, '--run-dir', run,
            ], training=True)
        config = jobs.read(run / 'agent_config.json')
        actual_count = config['algorithm'].pop('num_mini_batches')
        if actual_count != count or (baseline_config is not None and baseline_config != config):
            raise ValueError(f'Unexpected PPO config difference in {stage}')
        baseline_config = config
        batch = NUM_ENVS * config['num_steps_per_env']
        time = jobs.read(run / 'training_time.json')
        status = jobs.read(run / 'status.json')
        audit = jobs.read(run / 'resume_identity_audit.json')
        if status['num_envs'] != NUM_ENVS or any(
            not row['normalizer_identical'] or row['output_max_error'] or row['normalized_input_max_error']
            for row in audit.values()
        ):
            raise ValueError(f'Checkpoint or environment mismatch in {stage}')
        timings = read_iteration_timings(OUTPUT / (stage + '.log'), stopped['completed_iterations'])
        measured = timings[WARMUP:]
        timings_by_count[count].extend(measured)
        runs.append(dict(stage=stage, mini_batches=count, raw_samples_per_mini_batch=batch // count,
                         mirrored_samples_per_mini_batch=2 * batch // count,
                         ppo_updates_per_iteration=count * config['algorithm']['num_learning_epochs'],
                         training_seconds_including_warmup=time['training_seconds'], wall_seconds=wall,
                         gpu_peak_allocated_mib=status['gpu_peak_allocated_mb'], **metric(measured, batch)))
        jobs.write(OUTPUT / 'completed_runs.json', runs)

    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checksum:
        raise ValueError('The starting checkpoint changed during the benchmark')
    aggregate = {count: metric(rows, batch) for count, rows in timings_by_count.items()}
    benchmark_training = sum(row['training_seconds_including_warmup'] for row in runs)
    jobs.write(OUTPUT / 'result.json', dict(
        runs=runs, aggregate=aggregate, environment_steps_per_iteration=batch,
        throughput_change_fraction=aggregate[6]['throughput_steps_per_second'] / aggregate[4]['throughput_steps_per_second'] - 1,
        benchmark_training_seconds=benchmark_training,
        cumulative_training_seconds=stopped['cumulative_training_seconds'] + benchmark_training,
    ))
    jobs.write(OUTPUT / 'status.json', dict(stage='complete', training=False))
    jobs.write(OUTPUT / 'active_process.json', dict(stage='complete', pid=None))


if __name__ == '__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Reuse finished benchmark arms')
    try:
        main(resume=parser.parse_args().resume)
    except Exception as error:
        jobs.write(OUTPUT / 'status.json', dict(stage='error', training=False, error=str(error)))
        raise
