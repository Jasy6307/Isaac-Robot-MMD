"""Shared training, trial validation and recording for Anon push recovery."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
ROOT = SOURCE.parents[1]
CONFIG = SOURCE / 'configs/training/unified'
WORKERS = SOURCE / 'training/push_pipeline'
DEFAULT_BASE = ROOT / ('logs/rsl_rl/anon_locomotion/'
                       'lab_direct_backward_start_continue_02409_to_10000_8192_20261124/scratch_010000.pt')
LEVELS = (.075, .10, .15, .175, .20, .225, .25, .30, .35, .40, .525)
GRIDS = {'fixed': (), 'offset': ('--angle-offset-deg', 22.5, '--phase-offset', .0625, '--lead-seconds', 4.4)}
MEDIUM_FLOOR = 833
COUNTS = 896
STRONG_TARGET = 269
WALK_RUN_COUNT = 38
CURRICULUM = (
    ('light', 'push_recovery_pilot.json', 200, 42),
    ('directions', 'push_recovery_round3.json', 150, 44),
    ('tracking_seed', 'push_recovery_round4.json', 51, 45),
    ('tracking', 'push_recovery_tracking.json', 150, 47),
)
STRONG_PROFILE = 'push_recovery_corrective_steps.json'
ADAPTER_PROFILE = 'push_recovery_frozen_adapter.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checkpoint_inputs(checkpoint):
    return {str(path.resolve()): digest(path) for path in (
        checkpoint, checkpoint.with_name('env_config.json'), checkpoint.with_name('agent_config.json'))}


def inputs(baseline, *, record_video=True):
    config = read(baseline.with_name('env_config.json'))
    if (not config.get('unified_walk_run') or not config.get('continuous_commands')
            or config.get('guarded_recovery_actor') or config.get('train_push_recovery')):
        raise ValueError('Start with the unified continuous walk/run checkpoint before push training')
    if not config.get('full_motion_path') or not config.get('amp_expert_paths'):
        raise ValueError('The walking checkpoint requires its full motion and AMP references')

    sources = [Path(config[key]) for key in ('asset', 'full_motion_path')]
    sources += [Path(path) for path in config['amp_expert_paths']]
    profiles = [stage[1] for stage in CURRICULUM] + [STRONG_PROFILE, ADAPTER_PROFILE, 'lab_direct_eval.json']
    sources += [CONFIG / name for name in profiles]
    sources += [SOURCE/'train_anon_locomotion.py', *sorted(WORKERS.glob('*.py'))]
    sources += [SOURCE/'module_locations.json', SOURCE/'diagnostics/evaluate_unified_walk_run.py']
    sources += [SOURCE/'training'/name for name in ('anon_environment.py', 'anon_lab_direct_env.py',
                'anon_imitation_env.py', 'anon_push_recovery.py', 'anon_recovery_actor.py', 'retention_anchor.py')]
    if record_video:
        from runtime.anon_crowd_recording import find_ffmpeg

        for path in (ROOT/'output/pmx_prototype/character.usdc', ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'):
            if not path.is_file():
                raise FileNotFoundError(f'Missing recording asset: {path}')
            sources.append(path)
        find_ffmpeg()
    return checkpoint_inputs(baseline) | {str(path.resolve()): digest(path) for path in sources}


@contextmanager
def exclusive_run(root):
    with (root/'run.lock').open('a+b') as handle:
        handle.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f'Another process is running {root}') from error
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Campaign:
    def __init__(self, root, signature, resume):
        self.root = root
        self.journal = root/'progress.json'
        if resume:
            self.progress = read(self.journal)
            recorded = self.progress['inputs_sha256']
            if recorded != signature:
                if not matches_source_migration(recorded, signature):
                    raise ValueError('Run inputs or pipeline source changed. Start a fresh output directory.')
                self.progress['inputs_sha256'] = signature
                write(self.journal, self.progress)
        else:
            self.progress = dict(inputs_sha256=signature, stages={}, current=None)
            write(self.journal, self.progress)

    def stage(self, name, action):
        if name in self.progress['stages']:
            return self.root / self.progress['stages'][name]['directory']
        parent = self.root/'stages'/name
        parent.mkdir(parents=True, exist_ok=True)
        attempt = parent/f'attempt_{len(list(parent.glob("attempt_*"))) + 1:02}'
        attempt.mkdir()
        directory = attempt.relative_to(self.root).as_posix()
        self.progress['current'] = dict(name=name, directory=directory)
        write(self.journal, self.progress)
        print(f'\n[{name}] {attempt}', flush=True)
        started = time.perf_counter()
        action(attempt)
        self.progress['stages'][name] = dict(directory=directory, seconds=time.perf_counter()-started)
        self.progress['current'] = None
        write(self.journal, self.progress)
        return attempt


def matches_source_migration(recorded, signature):
    """Permit only the exact source changes that retired the standalone course."""
    migration = read(WORKERS/'resume_source_migration.json')
    before = {str((SOURCE / path).resolve()): sha256 for path, sha256 in migration['before'].items()}
    after = {str((SOURCE / path).resolve()): sha256 for path, sha256 in migration['after'].items()}
    if any(recorded.get(path) != sha256 for path, sha256 in before.items()):
        return False
    if (after.keys() - before.keys()) & recorded.keys():
        return False
    expected = {path: sha256 for path, sha256 in recorded.items() if path not in before}
    return (expected | after) == signature


def run(folder, log_name, program, *arguments):
    log_path = folder/f'{log_name}.log'
    command = [sys.executable, '-u', str(program), *map(str, arguments)]
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        print(subprocess.list2cmdline(command), file=log, flush=True)
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding='utf-8', errors='replace',
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith(('爱音训练 ', 'ANON_TRAIN_TIME ', 'PUSH_TRIALS_DONE ',
                                        'STEP_SEARCH ', 'BAKE ', 'RENDER ', 'VIDEO_VERIFIED ')):
                        print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError(f'{program.name} failed ({process.returncode}). Log: {log_path}')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                raise


def train(campaign, name, source, profile, iterations, seed, num_envs, *, freeze=False):
    def action(work):
        before = checkpoint_inputs(source)
        run(work, 'train', SOURCE/'train_anon_locomotion.py', '--mode', 'train',
            '--checkpoint', source, '--config-override', CONFIG/profile, '--run-dir', work/'train',
            '--num-envs', num_envs, '--iterations', iterations, '--seed', seed,
            '--headless', '--no-acceptance-video')
        import torch
        result = work/'train/policy_final.pt'
        old = torch.load(source, map_location='cpu', weights_only=True)
        new = torch.load(result, map_location='cpu', weights_only=True)
        if new['iter'] - old['iter'] != iterations or checkpoint_inputs(source) != before:
            raise ValueError(f'{name} did not complete the requested continuation')
        frozen = {}
        for role in ('actor_state_dict', 'critic_state_dict'):
            frozen[role] = [key for key in old[role] if key.startswith('obs_normalizer.')
                            or (freeze and role == 'actor_state_dict' and not key.startswith('correction.residual.'))]
            if any(not torch.equal(old[role][key], new[role][key]) for key in frozen[role]):
                raise ValueError(f'{name} changed frozen tensors in {role}')
        write(work/'lineage.json', dict(source=str(source), source_inputs=before, checkpoint=str(result),
              checkpoint_inputs=checkpoint_inputs(result), iterations=iterations, seed=seed, profile=str(CONFIG/profile), frozen=frozen,
              timing=read(work/'train/training_time.json')))
    work = campaign.stage(name, action)
    receipt = read(work/'lineage.json')
    result = work/'train/policy_final.pt'
    if receipt['source_inputs'] != checkpoint_inputs(source) or receipt['checkpoint_inputs'] != checkpoint_inputs(result):
        raise ValueError(f'{name} checkpoint lineage changed')
    return result


def measure(folder, label, policy, *, levels=LEVELS, options=()):
    run(folder, label, WORKERS/'measure.py', '--checkpoint', policy, '--output', folder/label,
        '--levels', *levels, *options)
    batch = folder/label
    if not (batch/'protocol.json').is_file():
        raise RuntimeError(f'Missing measurement protocol for {batch}')
    return batch


def guard(medium, walk_run):
    return (medium['total'] == COUNTS and medium['recovered'] >= MEDIUM_FLOOR
            and walk_run['passed'] and walk_run['segments'] == WALK_RUN_COUNT
            and walk_run['falls'] == 0)


def validate_candidate(campaign, name, policy, *, baseline=False):
    def action(work):
        from trials import read_batch
        checkpoint_signature = checkpoint_inputs(policy)
        clean_count = None
        if not baseline:
            run(work, 'walk_run', SOURCE/'train_anon_locomotion.py', '--mode', 'eval',
                '--checkpoint', policy, '--config-override', CONFIG/'lab_direct_eval.json',
                '--output', work/'walk_run', '--no-acceptance-video', '--headless')
            clean = read(work/'walk_run/report.json')
            summary = clean['response_steady_summary']
            clean_count = dict(passed=clean['passed_unified_gate'] and clean['complete_scope'] and summary['passed']
                               and clean['acceptance_mode'] == 'full'
                               and summary['failed_segments'] == summary['inconclusive_segments'] == 0,
                               segments=summary['passed_segments'], falls=clean['total_falls'],
                               report=str(work/'walk_run/report.json'))
        grids = {}
        for grid, options in GRIDS.items():
            batch = measure(work, grid, policy, options=options)
            protocol, rows = read_batch(batch)
            offsets = (0., 0., 4.) if grid == 'fixed' else (22.5, .0625, 4.4)
            actual_offsets = (protocol['angle_offset_degrees'], protocol['phase_offset_cycles'],
                              protocol['minimum_onset_seconds'])
            if (tuple(protocol['levels_m_s']) != LEVELS or protocol['gait_phase_samples'] != 8
                    or actual_offsets != offsets
                    or protocol['inputs_sha256'] != checkpoint_signature):
                raise ValueError(f'Mismatched acceptance protocol: {batch}')
            grids[grid] = {}
            for level in (0.0, *LEVELS):
                cases = [row for row in rows if row['impulse_m_s'] == level]
                if len(cases) != (56 if level == 0 else 448):
                    raise ValueError(f'Unexpected number of trials at {level}: {batch}')
                grids[grid][str(level)] = dict(total=len(cases), recovered=sum(row['recovered'] for row in cases),
                    falls=sum(row['fell'] or row['precondition_fell'] for row in cases))
        pooled = {level: {key: sum(grid[level][key] for grid in grids.values())
                         for key in ('total', 'recovered', 'falls')} for level in grids['fixed']}
        controls = pooled['0.0']['recovered'] == 112
        passed = guard(pooled['0.2'], clean_count) if clean_count else False
        if checkpoint_inputs(policy) != checkpoint_signature:
            raise ValueError(f'Measured checkpoint changed: {policy}')
        record = dict(checkpoint=str(policy), checkpoint_inputs=checkpoint_signature, directory=str(work),
                      grids=grids, pooled=pooled,
                      walk_run=clean_count, controls=controls, accepted=passed,
                      reached_30_percent=pooled['0.525']['recovered'] >= STRONG_TARGET)
        write(work/'acceptance.json', record)
        print(f'{name}  0.20 m/s {pooled["0.2"]["recovered"]}/{COUNTS}  '
              f'0.525 m/s {pooled["0.525"]["recovered"]}/{COUNTS}  门槛通过={passed}', flush=True)

    result = read(campaign.stage(f'eval_{name}', action)/'acceptance.json')
    if result['checkpoint_inputs'] != checkpoint_inputs(policy):
        raise ValueError(f'Measured checkpoint changed: {policy}')
    return result


def make_video(work, base, candidate, accepted, *,
               base_label='原始走跑 · 无抗扰续训', candidate_label=None,
               levels=LEVELS, shot_selector=None):
    from shots import shot_plan
    from trials import read_batch
    batches = [Path(record['directory'])/'fixed' for record in (base, candidate)]
    clips = (shot_selector or shot_plan)(*(read_batch(batch)[1] for batch in batches))
    if candidate_label is None:
        candidate_label = '抗扰训练 · 通过验收' if accepted else '抗扰候选 · 未通过验收'
    write(work/'video_plan.json', dict(chosen=candidate['checkpoint'], clean_passed=accepted, shots=clips,
        base_label=base_label, candidate_label=candidate_label))
    for lane, model, reference in zip(('base', 'trained'), (base, candidate), batches, strict=True):
        measure(work, lane, model['checkpoint'], levels=levels, options=('--match-reference', reference,
                '--trial-ids', *(clip['trial_id'] for clip in clips)))
    run(work, 'bake', WORKERS/'prepare_and_bake.py', '--work-dir', work)
    # Keep already verified physics and bake when the headless viewport needs a second initialization.
    errors = []
    for attempt in (1, 2):
        try:
            run(work, f'render_{attempt}', WORKERS/'render.py', '--work-dir', work,
                '--destination', work/'acceptance.mp4')
            receipt = read(work/'video_validation.json')
            if receipt['sha256'] != digest(work/'acceptance.mp4'):
                raise ValueError('Rendered video hash mismatch')
            return
        except RuntimeError as error:
            errors.append(str(error))
            for name in ('base_panel.mp4', 'trained_panel.mp4', 'acceptance.mp4'):
                partial = work/name
                if partial.exists():
                    partial.rename(work/f'incomplete_{attempt}_{name}')
    raise RuntimeError('\n'.join(errors))
