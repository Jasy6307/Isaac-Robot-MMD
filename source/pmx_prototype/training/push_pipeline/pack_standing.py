"""Initialize the standing specialization and derive its idle-reference guard."""

import copy
from pathlib import Path
import shutil

from training.push_pipeline.campaign import checkpoint_inputs, digest, read, write


def package(source, target, fixture=None):
    import numpy as np
    import torch
    from tensordict import TensorDict

    import source_layout
    from anon_lateral_actor import LateralRecoveryActor
    from anon_recovery_actor import RecoveryActor

    torch.set_num_threads(4)
    inputs = checkpoint_inputs(source)
    env = read(source.with_name('env_config.json'))
    original = read(source.with_name('agent_config.json'))
    agent = copy.deepcopy(original)
    settings = agent['actor']['recovery']
    snapshot = torch.load(source, map_location='cpu', weights_only=True)
    weights = snapshot['actor_state_dict']
    variant = weights.copy()
    receipt = dict(source=str(source), source_inputs=inputs)
    if fixture is None:
        if original['actor']['class_name'] != 'anon_recovery_actor:RecoveryActor' or settings.get('standing'):
            raise ValueError('The standing specialization needs a guarded moving policy')
        actor_class = RecoveryActor
        settings['standing'] = True
        protected = {key.replace('correction.residual.', 'correction.protected_residual.', 1):value.clone()
                     for key, value in weights.items() if key.startswith('correction.residual.')}
        if not protected:
            raise ValueError('The moving policy has no learned correction to freeze')
        variant.update(protected)
        raw = torch.randn(768, 248, generator=torch.Generator().manual_seed(19))
        raw[:256, 9:12] = 0.
        raw[256:512, 9:12] = torch.tensor([.4, 0., 0.])
        raw[512:, 9:12] = torch.tensor([1.2, 0., 0.])
    else:
        if original['actor']['class_name'] != 'anon_lateral_actor:LateralRecoveryActor':
            raise ValueError('The idle guard needs the fitted lateral correction')
        actor_class = LateralRecoveryActor
        protocol = read(fixture/'protocol.json')
        trace = fixture/'teacher.npz'
        motion = Path(env['full_motion_path'])
        if (Path(protocol['asset']).resolve() != Path(env['asset']).resolve()
                or digest(Path(protocol['reference'])) != digest(motion)
                or protocol['teacher']['sha256'] != digest(trace)):
            raise ValueError('The idle-guard fixture uses different body/reference inputs')
        order = protocol['dof_names']
        with np.load(motion) as bank:
            clip = bank['clip_names'].tolist().index('idle')
            if len(order) != 31 or len(set(order)) != 31 or np.any(bank['commands'][clip] != 0.):
                raise ValueError('Invalid idle clip or action order')
            indices = [bank['joint_names'].tolist().index(name) for name in order]
            positions, rates = bank['q'][clip][:, indices], bank['qd'][clip][:, indices]*.05
        settings['lateral_standing']['idle_reference'] = dict(
            lower_pose=(positions.min(0)-.0005).tolist(), upper_pose=(positions.max(0)+.0005).tolist(),
            lower_rate=(rates.min(0)-.0005).tolist(), upper_rate=(rates.max(0)+.0005).tolist(),
            pose_width=.020, rate_width=.006)
        with np.load(trace) as data:
            raw = torch.from_numpy(data['observations'].copy())
        if len(raw) < 500 or raw.shape[1] != 248 or not (raw[:, 9:12] == 0.).all():
            raise ValueError('The idle guard needs a zero-command standing fixture')
        receipt.update(fixture_sha256=digest(trace), motion_sha256=digest(motion))

    obs = TensorDict({'policy':raw}, batch_size=[len(raw)])

    def model(config, state):
        params = copy.deepcopy(config['actor'])
        del params['class_name']
        actor = actor_class(obs, config['obs_groups'], 'actor', 31, **params).eval()
        actor.load_state_dict(state)
        return actor

    base, candidate = model(original, weights), model(agent, variant)
    with torch.inference_mode():
        expected = base(obs)
        if fixture is not None:
            torch.testing.assert_close(candidate.correction.idle_reference(raw), torch.ones(len(raw)), rtol=0., atol=0.)
        torch.testing.assert_close(candidate(obs), expected, rtol=0., atol=0.)
        exported = torch.jit.script(candidate.as_jit().eval())
        torch.testing.assert_close(exported(raw), expected, rtol=0., atol=0.)
    if inputs != checkpoint_inputs(source):
        raise ValueError('The source checkpoint changed during packaging')
    target.mkdir()
    checkpoint = target/'policy_final.pt'
    if fixture is None:
        torch.save({**snapshot, 'actor_state_dict':variant}, checkpoint)
    else:
        shutil.copyfile(source, checkpoint)
    shutil.copyfile(source.with_name('env_config.json'), target/'env_config.json')
    write(target/'agent_config.json', agent)
    torch.jit.save(exported, str(target/'policy_jit.pt'))
    write(target/'package.json', {**receipt, 'checkpoint_inputs':checkpoint_inputs(checkpoint),
                                 'unchanged_probe_actions':len(raw)})
