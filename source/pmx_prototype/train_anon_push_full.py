"""Train the Anon standing push-recovery lineage from the original walk/run policy."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from training.push_pipeline import campaign as pipeline

sys.path.insert(0, str(pipeline.WORKERS))
from pack_standing import package


RECIPE = pipeline.CONFIG/'push_recovery_full.json'
MIXED_RECIPE = pipeline.CONFIG/'push_recovery_mixed4.json'
FRONTIER_LEVELS = (.55, .60, .65, .70, .75, .80)
WAVE_LEVELS = (.40, .45, .475, .49, .50, .51, .525, .5375, .55, .565, .575, .60,
               .625, .65, .675, .70, .725, .75, .80, .85, .90, .95, 1.0, 1.1)
DIRECTIONS = ('left', 'right', 'front', 'back')


def overlay(base, changes):
    config = base.copy()
    for key, value in changes.items():
        config[key] = overlay(base.get(key, {}), value) if isinstance(value, dict) else value
    return config


def read_recipe():
    recipe = pipeline.read(RECIPE)
    config = pipeline.read(pipeline.CONFIG/pipeline.ADAPTER_PROFILE)
    stages = {}
    seen = set()
    for group in ('higher', 'standing', 'lateral'):
        stages[group] = []
        for step in recipe[group]:
            name = step['name']
            if name in seen or step['iterations'] < 1 or 'train_retention_teacher_checkpoint' in step['settings']:
                raise ValueError(f'Invalid training recipe stage {name}')
            seen.add(name)
            config = overlay(config, step['settings'])
            stages[group].append({**step, 'settings':config})
    if len(stages['standing']) != 2 or len(stages['lateral']) < 3 or not stages['higher']:
        raise ValueError('Incomplete walking, standing or lateral training recipe')
    return stages


def require_checkpoint(folder):
    checkpoint = folder/'policy_final.pt'
    pipeline.checkpoint_inputs(checkpoint)
    return checkpoint


def read_mixed_recipe(replay):
    bases = {
        'walk_run': pipeline.read(pipeline.CONFIG/pipeline.STRONG_PROFILE),
        'waist_legs': replay['higher'][-1]['settings'],
        'standing_support': replay['lateral'][0]['settings'],
        'standing_lateral': replay['lateral'][-1]['settings'],
    }
    phases = pipeline.read(MIXED_RECIPE)
    if [phase['name'] for phase in phases] != list(bases):
        raise ValueError('The mixed course requires four ordered phases')
    return [{**phase, 'settings': overlay(bases[phase['name']], phase['settings'])} for phase in phases]


def moving_policy(campaign, baseline, num_envs, prefix_envs, stages):
    policy = baseline
    for name, config, updates, seed in pipeline.CURRICULUM:
        policy = pipeline.train(campaign, name, policy, config, updates, seed, prefix_envs)
    strong = pipeline.train(campaign, 'corrective_steps', policy, pipeline.STRONG_PROFILE, 180, 49, prefix_envs)
    fitted = fit_moving_adapter(campaign, policy, strong)
    online = pipeline.train(campaign, 'frozen_waist_legs', require_checkpoint(fitted/'strict'),
                            pipeline.ADAPTER_PROFILE, 120, 64, prefix_envs, freeze=True)
    teacher = speed_guard(campaign, online, require_checkpoint(fitted/'wider'))
    return train_steps(campaign, stages, teacher, teacher, num_envs)


def fit_moving_adapter(campaign, policy, strong):
    def rehearse(work):
        medium = pipeline.measure(work, 'medium', policy, levels=(.18, .20, .22),
                                  options=('--phases', 10, '--angle-offset-deg', 11.25, '--phase-offset', .03125,
                                           '--lead-seconds', 3.7, '--rehearsal'))
        heavy = pipeline.measure(work, 'heavy', strong, levels=(.43, .48, .525),
                                 options=('--phases', 12, '--angle-offset-deg', 33.75, '--phase-offset', .0417,
                                          '--lead-seconds', 4.1, '--rehearsal'))
        pipeline.run(work, 'fit', pipeline.WORKERS/'rebalance.py', '--medium', policy, '--strong', strong,
                     '--medium-batch', medium, '--strong-batch', heavy, '--output', work/'model')
        report = pipeline.read(work/'model/training.json')
        if 'result' not in report:
            raise RuntimeError(f'The two teachers could not supply a rebalanced checkpoint: {report}')
        require_checkpoint(work/'model/train')

    frozen = require_checkpoint(campaign.stage('rehearsal', rehearse)/'model/train')
    samples = pipeline.validate_candidate(campaign, 'search_reference', frozen, baseline=True)
    reference = Path(samples['directory'])

    demonstrations = []
    for seed, cells in ((61, 111), (62, 160), (63, 160)):
        def search(work):
            pipeline.run(work, 'search', pipeline.WORKERS/'search_steps.py',
                         '--output', work/'search', '--seed', seed, '--cells', cells,
                         '--base', frozen, '--prior', strong, '--frontier', *(reference/grid for grid in pipeline.GRIDS))
            pipeline.read(work/'search/report.json')

        demonstrations.append(campaign.stage(f'moving_examples_{seed}', search)/'search')

    def fit(work):
        fixture = pipeline.measure(work, 'activation', frozen, levels=(.10, .18, .20, .22, .25),
                                   options=('--phases', 10, '--angle-offset-deg', 17.6, '--phase-offset', .018,
                                            '--lead-seconds', 3.9, '--rehearsal'))
        pipeline.run(work, 'fit', pipeline.WORKERS/'train_residual.py', '--base', frozen, '--search',
                     *demonstrations, '--fixture', fixture, '--output', work/'adapter')
        report = pipeline.read(work/'adapter/training.json')
        if not report['candidates']:
            raise RuntimeError(f'Insufficient successful moving demonstrations: {report["skipped"]}')
        for name in ('strict', 'wider'):
            require_checkpoint(work/'adapter'/name)

    return campaign.stage('moving_adapter', fit)/'adapter'


def speed_guard(campaign, source, wider):
    def guard(work):
        pipeline.run(work, 'package', pipeline.WORKERS/'package_adapter.py', '--source', source,
                     '--wider', wider, '--output', work/'model')
        require_checkpoint(work/'model')

    return require_checkpoint(campaign.stage('speed_guard', guard)/'model')


def train_steps(campaign, stages, source, teacher, num_envs):
    for step in stages:
        source = train_phase(campaign, step, source, teacher, num_envs, freeze=True)
    return source


def train_phase(campaign, phase, source, teacher, num_envs, *, freeze):
    folder = campaign.root/'profiles'
    folder.mkdir(exist_ok=True)
    config = folder/f'{phase["name"]}.json'
    pipeline.write(config, {**phase['settings'], 'train_retention_teacher_checkpoint':str(teacher)})
    return pipeline.train(campaign, phase['name'], source, config, phase['iterations'], phase['seed'],
                          num_envs, freeze=freeze)


def moving_teacher(campaign, checkpoint, iterations):
    def select(work):
        import torch

        end = torch.load(checkpoint, map_location='cpu', weights_only=True)['iter']
        target = end - iterations//4
        snapshots = {int(path.stem.removeprefix('model_')): path for path in checkpoint.parent.glob('model_*.pt')
                     if path.stem.removeprefix('model_').isdecimal()}
        earlier = [step for step in snapshots if end - iterations < step < end]
        if not earlier:
            raise RuntimeError('The walk/run phase did not save an intermediate teacher')
        iteration = min(earlier, key=lambda step: (abs(step - target), step))
        teacher = snapshots[iteration]
        if torch.load(teacher, map_location='cpu', weights_only=True)['iter'] != iteration:
            raise ValueError(f'The saved teacher iteration does not match {teacher}')
        pipeline.write(work/'teacher.json', dict(checkpoint=str(teacher), inputs=pipeline.checkpoint_inputs(teacher),
                                                 source_inputs=pipeline.checkpoint_inputs(checkpoint)))

    record = pipeline.read(campaign.stage('moving_teacher', select)/'teacher.json')
    teacher = Path(record['checkpoint'])
    if (record['inputs'] != pipeline.checkpoint_inputs(teacher)
            or record['source_inputs'] != pipeline.checkpoint_inputs(checkpoint)):
        raise ValueError('The moving rehearsal teachers changed')
    return teacher


def lateral_fixture(campaign, source):
    def collect(work):
        pipeline.measure(work, 'standing', source, levels=(.20, .70),
                         options=('--only', 'stand', '--phases', 16, '--rehearsal'))

    return campaign.stage('standing_fixture', collect)/'standing'


def side_examples(campaign, source, label, seed, onsets, levels):
    def collect(work):
        pipeline.run(work, 'search', pipeline.WORKERS/'search_lateral_steps.py',
                     '--base', source, '--prior', source, '--seed', seed, '--output', work/'samples',
                     '--reference-steps', '--onsets', onsets, '--strengths', *levels)
        pipeline.read(work/'samples/report.json')

    return campaign.stage(label, collect)/'samples'


def usable_frames(samples):
    import numpy as np

    with np.load(samples/'steps.npz') as data:
        return int((data['corrected'] & (data['frames'] <= 25)).sum())


def standing_policy(campaign, moving, num_envs, stages):
    standing_seed = require_checkpoint(campaign.stage(
        'standing_seed', lambda work: package(moving, work/'model'))/'model')
    standing = train_steps(campaign, stages['standing'], standing_seed, standing_seed, num_envs)
    lateral = stages['lateral']
    first = train_steps(campaign, lateral[:1], standing, standing, num_envs)
    fitted, fixture = fit_standing_lateral(campaign, first)
    teacher = require_checkpoint(fitted/'seed')
    onset = train_steps(campaign, lateral[1:2], require_checkpoint(fitted/'fit_1.0'), teacher, num_envs)
    idle_guarded = require_checkpoint(campaign.stage(
        'idle_guard', lambda work: package(onset, work/'model', fixture))/'model')
    return train_steps(campaign, lateral[2:], idle_guarded, teacher, num_envs)


def fit_standing_lateral(campaign, first):
    fixture = lateral_fixture(campaign, first)
    fit_levels = (.33, .37, .41, .47)
    holdout_levels = (.35, .45, .55, .65, .70)
    fit = side_examples(campaign, first, 'lateral_examples', 902, 8, fit_levels)
    holdout = side_examples(campaign, first, 'lateral_holdout', 274, 5, holdout_levels)
    train_data = [fit]
    if usable_frames(fit) < 250:
        train_data.append(side_examples(campaign, first, 'lateral_examples_extra', 903, 12, fit_levels))
    if usable_frames(holdout) < 100:
        holdout = side_examples(campaign, first, 'lateral_holdout_extra', 275, 12, holdout_levels)

    def fit_lateral(work):
        pipeline.run(work, 'fit', pipeline.WORKERS/'fit_lateral_demonstrations.py',
                     '--source', first, '--fit', *train_data, '--holdout', holdout, '--retention', fixture,
                     '--lateral-gate', '--output', work/'model')
        for name in ('seed', 'fit_1.0'):
            require_checkpoint(work/'model'/name)

    return campaign.stage('lateral_adapter', fit_lateral)/'model', fixture


def mixed_policy(campaign, baseline, num_envs, phases):
    walk, waist, support, lateral = phases
    strong = train_phase(campaign, walk, baseline, baseline, num_envs, freeze=False)
    medium = moving_teacher(campaign, strong, walk['iterations'])
    fitted = fit_moving_adapter(campaign, medium, strong)
    teacher = speed_guard(campaign, require_checkpoint(fitted/'strict'), require_checkpoint(fitted/'wider'))
    moving = train_phase(campaign, waist, teacher, teacher, num_envs, freeze=True)
    standing_seed = require_checkpoint(campaign.stage(
        'standing_seed', lambda work: package(moving, work/'model'))/'model')
    standing = train_phase(campaign, support, standing_seed, standing_seed, num_envs, freeze=True)
    fitted, fixture = fit_standing_lateral(campaign, standing)
    source = require_checkpoint(fitted/'fit_1.0')
    guarded = require_checkpoint(campaign.stage(
        'idle_guard', lambda work: package(source, work/'model', fixture))/'model')
    return train_phase(campaign, lateral, guarded, require_checkpoint(fitted/'seed'), num_envs, freeze=True)


def count(rows):
    return dict(total=len(rows), recovered=sum(r['recovered'] for r in rows),
                falls=sum(r['fell'] or r['precondition_fell'] for r in rows))


def measured(work, label, policy, levels, *, grid='fixed', waveform=None):
    from trials import read_batch

    options = pipeline.GRIDS[grid]
    if waveform is not None:
        options += ('--only', 'stand', '--phases', 16, '--force-profile', waveform)
    protocol, rows = read_batch(pipeline.measure(work, label, policy, levels=levels, options=options))
    expected_grid = (0., 0., 4.) if grid == 'fixed' else (22.5, .0625, 4.4)
    actual_grid = tuple(protocol[key] for key in ('angle_offset_degrees', 'phase_offset_cycles', 'minimum_onset_seconds'))
    if (protocol['inputs_sha256'] != pipeline.checkpoint_inputs(policy) or tuple(protocol['levels_m_s']) != levels
            or actual_grid != expected_grid or protocol['gait_phase_samples'] != (16 if waveform else 8)
            or protocol['force_profile'] != (waveform or 'impulse')):
        raise ValueError(f'Unexpected measurement inputs in {work/label}')
    return rows


def frontier(work, policy):
    grids = {}
    for grid in pipeline.GRIDS:
        rows = measured(work, grid, policy, FRONTIER_LEVELS, grid=grid)
        grids[grid] = {}
        for level in FRONTIER_LEVELS:
            group = [r for r in rows if r['impulse_m_s'] == level]
            stand = [r for r in group if r['case'] == 'stand']
            if len(group) != 448 or len(stand) != 64:
                raise ValueError(f'Incomplete standing and moving measurements in {work/grid}')
            grids[grid][str(level)] = dict(all=count(group), standing=count(stand),
                directions={d:count([r for r in stand if r['direction'] == d]) for d in DIRECTIONS})
    pipeline.write(work/'frontier.json', dict(checkpoint=str(policy), checkpoint_inputs=pipeline.checkpoint_inputs(policy),
                                               directory=str(work), grids=grids))


def shape_results(work, policy):
    reports = {}
    for shape in ('impulse', 'sustained', 'ramp', 'double'):
        rows = measured(work, shape, policy, WAVE_LEVELS, waveform=shape)
        reports[shape] = {str(level):{d:count([r for r in rows if r['impulse_m_s'] == level and r['direction'] == d])
                                     for d in DIRECTIONS} for level in WAVE_LEVELS}
    pipeline.write(work/'shapes.json', dict(checkpoint_inputs=pipeline.checkpoint_inputs(policy), reports=reports))


def baseline_video_reference(work, policy):
    measured(work, 'fixed', policy, FRONTIER_LEVELS)
    pipeline.write(work/'reference.json', dict(checkpoint=str(policy), directory=str(work),
                                              checkpoint_inputs=pipeline.checkpoint_inputs(policy)))


def targets(standard, high, waves):
    grids = high['grids']
    directions = grids['fixed']
    medium = standard['pooled']['0.2']['recovered']
    heavy = standard['pooled']['0.525']['recovered']
    fore_aft = all(grid['0.7']['directions'][d]['recovered'] == 8
                   for grid in grids.values() for d in ('front', 'back'))
    accepted = standard['accepted'] and standard['controls'] and fore_aft and heavy >= pipeline.STRONG_TARGET
    standing = sum(grid['0.7']['standing']['recovered'] for grid in grids.values())
    wave = waves['reports']
    checks = dict(
        medium=medium >= 866, strong=heavy >= 531, standing_070=standing >= 103,
        pure_lateral_070=all(directions['0.7']['directions'][d]['recovered'] == 8 for d in DIRECTIONS[:2]),
        pure_lateral_075=all(directions['0.75']['directions'][d]['recovered'] == 8 for d in DIRECTIONS[:2]),
        pure_lateral_080=(directions['0.8']['directions']['left']['recovered'] == 8
                         and directions['0.8']['directions']['right']['recovered'] >= 1),
        left_impulse_100=wave['impulse']['1.0']['left']['recovered'] >= 15,
        right_shaped_080=all(wave[name]['0.8']['right']['recovered'] >= threshold
                            for name, threshold in (('sustained', 15), ('ramp', 13), ('double', 8))))
    return bool(accepted), checks, standing


def video_shots(base, final):
    from shots import DIRECTION_LABEL, TRIAL_KEYS

    pairs = list(zip(base, final, strict=True))
    for left, right in pairs:
        if (left['case'] == 'stand' or right['case'] == 'stand') and any(
                left[key] != right[key] for key in TRIAL_KEYS):
            raise ValueError('Comparison scenes do not have the same applied push and onset')
    picks = []
    for level, direction in ((.70, 'left'), (.70, 'right'), (.75, 'right'), (.80, 'left'), (.80, 'right')):
        options = [(a, b) for a, b in pairs if a['case'] == 'stand' and a['impulse_m_s'] == level
                   and a['direction'] == direction and not a['precondition_fell'] and not b['precondition_fell']]
        if not options:
            raise ValueError(f'No matched video scene for {level} {direction}')
        a, b = max(options, key=lambda pair: (pair[1]['recovered'], not pair[0]['recovered']))
        picks.append(dict(trial_id=int(a['trial_id']), title='站立受推恢复',
                          direction_cn=DIRECTION_LABEL[direction], impulse_m_s=level,
                          expected_recovery=[a['recovered'], b['recovered']]))
    return picks


def execute(campaign, baseline, num_envs, prefix_envs, course, stages, record_video):
    if course == 'mixed4':
        policy = mixed_policy(campaign, baseline, num_envs, stages)
    else:
        moving = moving_policy(campaign, baseline, num_envs, prefix_envs, stages['higher'])
        policy = standing_policy(campaign, moving, num_envs, stages)
    standard = pipeline.validate_candidate(campaign, 'final', policy)
    high = pipeline.read(campaign.stage('frontier_final', lambda work: frontier(work, policy))/'frontier.json')
    waves = pipeline.read(campaign.stage('shapes_final', lambda work: shape_results(work, policy))/'shapes.json')
    identity = pipeline.checkpoint_inputs(policy)
    if any(record['checkpoint_inputs'] != identity for record in (standard, high, waves)):
        raise ValueError('The final checkpoint or acceptance inputs changed')
    accepted, checks, standing = targets(standard, high, waves)
    matched = accepted and all(checks.values())
    report = dict(course=course, baseline=str(baseline), checkpoint=str(policy), accepted=accepted,
                  accepted_checkpoint=str(policy) if accepted else None, matches_round_15=matched,
                  target_checks=checks, standing_070=dict(recovered=standing, total=128),
                  standard=standard, frontier=high, waves=waves, video=None,
                  num_envs=num_envs, prefix_num_envs=prefix_envs)
    path = campaign.root/'result.json'
    pipeline.write(path, report)
    if record_video:
        reference = campaign.stage('video_baseline', lambda work: baseline_video_reference(work, baseline))
        initial = pipeline.read(reference/'reference.json')
        if initial['checkpoint_inputs'] != pipeline.checkpoint_inputs(baseline):
            raise ValueError('The original policy changed after measuring video reference trials')
        label = '四段混合课程' if course == 'mixed4' else '第十五轮课程'
        status = '通过保底验收' if accepted else '保底验收未通过'
        video = campaign.stage('video', lambda work: pipeline.make_video(
            work, initial, high, accepted, levels=FRONTIER_LEVELS, shot_selector=video_shots,
            candidate_label=f'{label} · {status}'))
        receipt = pipeline.read(video/'video_validation.json')
        if receipt['sha256'] != pipeline.digest(video/'acceptance.mp4'):
            raise ValueError('The verified comparison video changed')
        report['video'] = str(video/'acceptance.mp4')
        pipeline.write(path, report)
    print(f'FINAL {path} accepted={accepted} matches_round_15={matched}', flush=True)
    return 0 if matched else 3 if accepted else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course', choices=('replay', 'mixed4'), default='replay',
                        help='Replay the historical stages or run four fixed-mixture training phases')
    parser.add_argument('--baseline', type=Path, default=pipeline.DEFAULT_BASE, help='Unperturbed walk/run checkpoint')
    parser.add_argument('--num-envs', type=int, default=8192, help='Parallel training environments (default: 8192)')
    parser.add_argument('--prefix-num-envs', type=int, help='Override only the six foundation stages (historically 2048)')
    parser.add_argument('--output', type=Path, help='Fresh output directory or the existing directory for --resume')
    parser.add_argument('--resume', action='store_true', help='Reuse validated stages and retry the interrupted stage')
    parser.add_argument('--check', action='store_true', help='Validate inputs and print the run plan without starting Isaac Lab')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=True, help='Record one matched comparison after final acceptance')
    args = parser.parse_args()
    if args.num_envs < 1 or (args.prefix_num_envs is not None and args.prefix_num_envs < 1):
        parser.error('Parallel environment counts must be positive')
    if args.resume and not args.output:
        parser.error('--resume requires --output')
    if args.course == 'mixed4' and args.prefix_num_envs is not None:
        parser.error('--prefix-num-envs applies to the replay course')
    baseline = args.baseline.resolve()
    prefix_envs = args.prefix_num_envs or args.num_envs
    stages = read_recipe()
    if args.course == 'mixed4':
        stages = read_mixed_recipe(stages)
    signature = pipeline.inputs(baseline, record_video=args.video)
    for path in (Path(__file__), RECIPE, *(pipeline.SOURCE/'training'/f'{name}.py' for name in (
            'anon_lateral_actor', 'anon_lateral_support', 'anon_lateral_followup', 'anon_push_waveforms', 'anon_push_bands'))):
        signature[str(path.resolve())] = pipeline.digest(path)
    signature.update(num_envs=args.num_envs, prefix_num_envs=prefix_envs, record_video=args.video)
    if args.course == 'mixed4':
        signature[str(MIXED_RECIPE.resolve())] = pipeline.digest(MIXED_RECIPE)
        signature['course'] = args.course
    if args.check:
        print(f'INPUTS_OK {baseline} course={args.course} prefix_envs={prefix_envs} envs={args.num_envs} video={args.video}')
        if args.course == 'mixed4':
            for phase in stages:
                print(f'{phase["name"]}: {phase["iterations"]} PPO iterations (fixed mixture)')
        else:
            print('foundation: ' + ', '.join([stage[0] for stage in pipeline.CURRICULUM] +
                                             ['corrective_steps', 'frozen_waist_legs']))
            for name, group in stages.items():
                print(f'{name}: {", ".join(step["name"] for step in group)}')
        return 0
    root = (args.output or pipeline.ROOT/'output/anon_push_full'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if args.resume:
        if not (root/'progress.json').is_file():
            parser.error(f'No existing run at {root}')
    else:
        root.mkdir(parents=True, exist_ok=False)
    with pipeline.exclusive_run(root):
        return execute(pipeline.Campaign(root, signature, args.resume), baseline, args.num_envs,
                       prefix_envs, args.course, stages, args.video)


if __name__ == '__main__':
    sys.exit(main())
