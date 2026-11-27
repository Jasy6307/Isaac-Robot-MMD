"""Fit the strong-push descendant to both frozen teachers without changing its observation contract."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from rsl_rl.models import MLPModel
from tensordict import TensorDict
import torch

from measure import sha256 as digest
from trials import read_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('medium', 'strong', 'medium-batch', 'strong-batch', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random = torch.Generator().manual_seed(54)
    medium, strong = args.medium.resolve(), args.strong.resolve()
    medium_meta, _ = read_batch(args.medium_batch)
    strong_meta, rows = read_batch(args.strong_batch)
    inputs = [args.medium_batch/'teacher.npz', args.strong_batch/'teacher.npz']
    for policy, protocol, archive in zip((medium, strong), (medium_meta, strong_meta), inputs, strict=True):
        if protocol['inputs_sha256'].get(str(policy)) != digest(policy) or protocol['teacher']['sha256'] != digest(archive):
            raise ValueError('Rehearsal data or teacher checkpoint changed')
    with np.load(inputs[0]) as a, np.load(inputs[1]) as b:
        levels = np.asarray([row['impulse_m_s'] for row in rows])[b['trial_ids']]
        keep = levels > 0.
        if not all(rows[i]['recovered'] for i in np.unique(b['trial_ids'][keep])):
            raise ValueError('A failed trajectory entered strong-push rehearsal')
        groups = np.concatenate((a['groups'], np.where(levels[keep] == .525, 4, 3)))
        obs, target = (torch.from_numpy(np.concatenate((a[key], b[key][keep])))
                       for key in ('observations', 'actions'))
    pools = [torch.from_numpy(np.flatnonzero(groups == i)) for i in range(5)]
    if min(map(len, pools)) < 500:
        (root/'training.json').write_text(json.dumps(dict(skipped='Insufficient successful rehearsal samples',
            groups=list(map(len, pools)))), encoding='utf-8')
        return
    train = root/'train'
    train.mkdir()
    agent = json.loads((strong.parent/'agent_config.json').read_text(encoding='utf-8'))
    params = copy.deepcopy(agent['actor'])
    del params['class_name']
    actor = MLPModel(TensorDict({'policy':obs[:1]}, batch_size=[1]), agent['obs_groups'], 'actor',
                     target.shape[-1], **params).eval()
    base = torch.load(medium,map_location='cpu',weights_only=True)['actor_state_dict']
    checkpoint = torch.load(strong,map_location='cpu',weights_only=True)
    original = checkpoint['actor_state_dict']
    if any(not torch.equal(original[key],base[key]) for key in original if key.startswith('obs_normalizer.')):
        raise ValueError('Actor normalization differs between teachers')

    def means(indices):
        return actor(TensorDict({'policy':obs[indices]}, batch_size=[len(indices)]))

    with torch.no_grad():
        actor.load_state_dict(base)
        check = torch.cat(pools[:3])[::17]
        base_difference = float((means(check)-target[check]).abs().max())
        actor.load_state_dict(original)
        check = torch.cat(pools[3:])[::17]
        strong_difference = float((means(check)-target[check]).abs().max())
        if max(base_difference,strong_difference) > .0001:
            raise ValueError(f'The reconstruction does not reproduce its teacher labels: {base_difference}, {strong_difference}')

    def residuals():
        with torch.no_grad():
            return [float((means(indices)-target[indices]).square().mean().sqrt()) for indices in pools]

    history = [{'input': 'strong', 'rmse': residuals()}]
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=.00006)
    # Keep the strong recovery behavior present while more directly weighting the 0.20 m/s band.
    for step in range(1, 241):
        indices = torch.cat([pool[torch.randint(len(pool),(draw,),generator=random)]
                             for pool,draw in zip(pools,(384,384,896,512,704))])
        loss = (means(indices)-target[indices]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.)
        optimizer.step()
        if step in (40, 80, 120, 180, 240):
            history.append({'step':step,'rmse':residuals()})

    result = train/'policy_final.pt'
    checkpoint['actor_state_dict'] = actor.state_dict()
    checkpoint['optimizer_state_dict']['state'] = {}
    torch.save(checkpoint,result)
    for name in ('agent_config.json','env_config.json'):
        shutil.copy2(strong.parent/name,train/name)
    manifest = dict(source=str(strong),source_sha256=digest(strong),round6=str(medium),round6_sha256=digest(medium),
                    result=str(result),result_sha256=digest(result),iterations=0,final_iteration=checkpoint['iter'],
                    offline_updates=240,rehearsal_sha256={str(path): digest(path) for path in inputs},
                    teacher_max_label_error=[base_difference,strong_difference],
                    action_rmse=history,cleared_optimizer_moments=True)
    (root/'training.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
