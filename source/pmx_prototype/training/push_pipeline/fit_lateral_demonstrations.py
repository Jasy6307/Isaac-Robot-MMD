"""Fit only the standing correction to successful physical stepping examples."""

import argparse
import copy
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from tensordict import TensorDict

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path.insert(0, str(SRC))
import source_layout
from anon_recovery_actor import RecoveryActor
from anon_lateral_actor import LateralRecoveryActor
from training.push_pipeline.campaign import checkpoint_inputs, digest, read, write


def demos(folder, source, joints):
    report = read(folder / 'report.json')
    data = folder / 'steps.npz'
    if report['source_sha256'] != digest(source) or report['dataset_sha256'] != digest(data):
        raise ValueError(f'Demonstration lineage differs in {folder}')
    group = len(report['variants']) + 2
    with np.load(data) as archive:
        if not np.array_equal(archive['lower'], joints):
            raise ValueError('Waist and leg joints differ from the frozen policy')
        take = archive['corrected'] & (archive['frames'] <= 25)
        positive = archive['observations'][take]
        reference = archive['reference'][take]
        action_delta = archive['actions'][take] - reference
        cell = archive['trial_ids'][take] // group
        side_of_cell = np.asarray([entry['direction'] < 180 for entry in report['cells']])
    return (torch.from_numpy(positive.copy()), torch.from_numpy(reference.copy()),
            torch.from_numpy(action_delta.copy()), torch.from_numpy(side_of_cell[cell].copy()))


def retention(folder, source):
    protocol = read(folder / 'protocol.json')
    data = folder / 'teacher.npz'
    if protocol['inputs_sha256'] != checkpoint_inputs(source) or digest(data) != protocol['teacher']['sha256']:
        raise ValueError('Protected cases must come from the same policy')
    with (folder / 'trials.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    safe = np.asarray([row['recovered'] == 'True' and (float(row['impulse_m_s']) <= .20
                       or row['direction'] in {'front', 'back'}) for row in rows])
    with np.load(data) as archive:
        retained = safe[archive['trial_ids']]
        return torch.from_numpy(archive['observations'][retained].copy()), torch.from_numpy(archive['actions'][retained].copy())


def fit():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--fit', nargs='+', required=True, type=Path)
    parser.add_argument('--holdout', required=True, type=Path)
    parser.add_argument('--retention', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lateral-gate', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(9103)
    source = args.source.resolve()
    snapshot = torch.load(source, map_location='cpu', weights_only=True)
    agent = read(source.with_name('agent_config.json'))
    cfg = copy.deepcopy(agent['actor'])
    del cfg['class_name']
    joints = np.sort(cfg['recovery']['joint_indices'])
    train = [demos(path, source, joints) for path in args.fit]
    raw, original_actions, delta, side = (torch.cat(items) for items in zip(*train))
    holdout, holdout_base, holdout_delta, _ = demos(args.holdout, source, joints)
    safe, safe_actions = retention(args.retention, source)
    if len(raw) < 250 or len(holdout) < 100 or len(safe) < 800:
        raise ValueError('Too few physical trajectories or retained action samples')

    def obs(raw):
        return TensorDict({'policy': raw}, batch_size=[len(raw)])

    actor = RecoveryActor(obs(raw[:1]), agent['obs_groups'], 'actor', 31, **copy.deepcopy(cfg)).eval()
    actor.load_state_dict(snapshot['actor_state_dict'])
    original_model = actor
    if args.lateral_gate:
        if not cfg['recovery'].get('standing') or not cfg['recovery'].get('freeze_gate'):
            raise ValueError('The lateral gate requires the previously frozen standing model')
        cfg['recovery']['lateral_standing'] = dict(lateral_onset=.23, lateral_width=.18,
            dominance_margin=.065, dominance_width=.15, maximum_action=1.15)
        agent['actor'].update(recovery=cfg['recovery'], class_name='anon_lateral_actor:LateralRecoveryActor')
        actor = LateralRecoveryActor(obs(raw[:1]), agent['obs_groups'], 'actor', 31, **copy.deepcopy(cfg)).eval()
        remapped = {}
        for key, value in snapshot['actor_state_dict'].items():
            if key.startswith('correction.'):
                remapped[key.replace('correction.', 'correction.reference.', 1)] = value
            if not key.startswith('correction.') or key.startswith('correction.residual.'):
                remapped[key] = value
        actor.load_state_dict(remapped)
        with torch.no_grad():
            for probe in (raw, safe, holdout):
                torch.testing.assert_close(actor(obs(probe)), original_model(obs(probe)), rtol=0., atol=0.)
    actor.requires_grad_(False)
    actor.correction.residual.requires_grad_(True)
    target_scale = .92 if args.lateral_gate else .7
    expected = original_actions[:, joints] + target_scale * delta[:, joints].clamp(-.75, .75)
    expected_holdout = holdout_base[:, joints] + target_scale * holdout_delta[:, joints].clamp(-.75, .75)
    # Teacher actions are logged after the controller ran. Match the protected
    # policy measured on the stored observations before using them as targets.
    with torch.no_grad():
        torch.testing.assert_close(actor(obs(safe))[:, joints], safe_actions[:, joints], rtol=3e-4, atol=3e-5)
        original = actor(obs(holdout))[:, joints]
    loss = torch.nn.functional.smooth_l1_loss
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {key: value.detach().clone() for key, value in actor.state_dict().items()}

    def package(name, variant):
        actor.load_state_dict(variant)
        path = args.output / name
        path.mkdir()
        weights = path / 'policy_final.pt'
        torch.save({**snapshot, 'actor_state_dict': variant}, weights)
        write(path / 'agent_config.json', agent)
        shutil.copyfile(source.with_name('env_config.json'), path / 'env_config.json')
        exported = torch.jit.script(actor.as_jit().eval())
        torch.jit.save(exported, str(path / 'policy_jit.pt'))
        with torch.no_grad():
            actual = actor(obs(holdout))[:, joints]
            torch.testing.assert_close(actor(obs(holdout[:50])), exported(holdout[:50]))
            return dict(checkpoint=str(weights), inputs=checkpoint_inputs(weights),
                        holdout_error=float((actual - expected_holdout).square().mean().sqrt()),
                        retained_action_drift=float((actor(obs(safe))[:, joints] - safe_actions[:, joints]).square().mean().sqrt()))

    seed = package('seed', frozen) if args.lateral_gate else None
    opt = torch.optim.AdamW(actor.correction.residual.parameters(), lr=8e-5, weight_decay=.001)
    weaker = (~side).nonzero().flatten()
    for _ in range(390):
        sample = torch.randint(len(raw), (128,), generator=rng)
        if len(weaker):
            sample = torch.cat((sample, weaker[torch.randint(len(weaker), (64,), generator=rng)]))
        stable = torch.randint(len(safe), (320,), generator=rng)
        imitation = loss(actor(obs(raw[sample]))[:, joints], expected[sample], beta=.09)
        preserved = loss(actor(obs(safe[stable]))[:, joints], safe_actions[stable][:, joints], beta=.07)
        cost = imitation + 1.9 * preserved
        opt.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(actor.correction.residual.parameters(), 1.)
        opt.step()

    learned = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    candidate_records = {}
    for scale in (.5, 1.):
        variant = {key: torch.lerp(frozen[key], value, scale) if key.startswith('correction.residual.') else value
                   for key, value in learned.items()}
        if any(not torch.equal(frozen[key], value) for key, value in variant.items()
               if not key.startswith('correction.residual.')):
            raise ValueError('The frozen controller changed')
        candidate_records[str(scale)] = package(f'fit_{scale:.1f}', variant)
    write(args.output / 'fit.json', dict(inputs=checkpoint_inputs(source), train_frames=len(raw), held_out_frames=len(holdout),
          retained_frames=len(safe), prior_holdout_error=float((original - expected_holdout).square().mean().sqrt()),
          seed=seed, candidates=candidate_records))


if __name__ == '__main__':
    fit()
