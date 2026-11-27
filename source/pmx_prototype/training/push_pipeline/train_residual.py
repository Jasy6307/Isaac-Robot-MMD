"""Learn the gated waist/leg correction from physically successful counterfactuals."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from tensordict import TensorDict
import torch

HERE=Path(__file__).resolve().parent
from measure import sha256 as digest
sys.path.insert(0, str(HERE.parents[1]))
import source_layout
from anon_recovery_actor import RecoveryActor


def make_actor(agent, indices, raw):
    agent=copy.deepcopy(agent)
    settings=agent['actor']
    del settings['class_name']
    return RecoveryActor(TensorDict({'policy':raw[:1]},batch_size=[1]),agent['obs_groups'],'actor',31,
                         dict(joint_indices=indices,threshold=.8,width=.12,gain=1.),**settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search',nargs='+',required=True,type=Path)
    parser.add_argument('--fixture',required=True,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(902)
    random=torch.Generator().manual_seed(61333)
    base=args.base.resolve()
    original=torch.load(base,map_location='cpu',weights_only=True)
    original_actor=original['actor_state_dict']
    agent=json.loads((base.parent/'agent_config.json').read_text(encoding='utf-8'))
    env=json.loads((base.parent/'env_config.json').read_text(encoding='utf-8'))
    demos, targets, demo_fit, controls = [],[],[],[]
    indices=None
    dataset_audit={}
    total_cells=0
    for folder in args.search:
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if not report['rescued_cells']:
            continue
        if report['source_sha256']!=digest(base) or report['dataset_sha256']!=digest(folder/'steps.npz'):
            raise ValueError(f'The frozen policy or the source demonstrations changed in {folder}')
        total_cells+=report['rescued_cells']
        with np.load(folder/'steps.npz') as d:
            if indices is None:
                indices=d['lower'].tolist()
            elif indices!=d['lower'].tolist():
                raise ValueError('Inconsistent waist/leg action order')
            group=len(report['variants'])+2
            labels=d['actions'][:,indices]-d['reference'][:,indices]
            valid=d['corrected'] & (d['frames']<=39) & np.isfinite(labels).all(axis=1)
            unchanged=(d['trial_ids']%group)==0
            demos.append(torch.from_numpy(d['observations'][valid]))
            targets.append(torch.from_numpy(labels[valid]).clamp(-.82,.82))
            demo_fit.append(torch.from_numpy(((d['trial_ids'][valid]//group+report['seed'])%7)!=0))
            controls.append(torch.from_numpy(d['observations'][unchanged]))
        dataset_audit[str(folder.resolve())] = dict(sha256=report['dataset_sha256'], rescued=report['rescued_cells'])

    fixture=json.loads((args.fixture/'protocol.json').read_text(encoding='utf-8'))
    if fixture['inputs_sha256'].get(str(base.resolve()))!=digest(base) or fixture['teacher']['sha256']!=digest(args.fixture/'teacher.npz'):
        raise ValueError('The protected activation fixture changed')
    with np.load(args.fixture/'teacher.npz') as archive:
        all_safe=torch.from_numpy(archive['observations'].copy())
        fit=torch.from_numpy(archive['trial_ids']%7!=0)
        safe_fit=all_safe[fit]
        safe_validation=all_safe[~fit]
    if not demos:
        (args.output/'training.json').write_text(json.dumps({'candidates': {}, 'skipped': 'No recovered demonstrations'}),
                                                 encoding='utf-8')
        return
    positive=torch.cat(demos)
    expected=torch.cat(targets)
    train=torch.cat(demo_fit)
    if len(positive)<500 or int((~train).sum())<100:
        (args.output/'training.json').write_text(json.dumps({'candidates': {}, 'positive_frames': len(positive),
            'skipped': 'Insufficient demonstrations for fit and holdout'}), encoding='utf-8')
        return
    actor=make_actor(agent,indices,positive)
    mismatch=actor.load_state_dict(original_actor,strict=False)
    if mismatch.unexpected_keys or any(not name.startswith('correction.') for name in mismatch.missing_keys):
        raise ValueError(f'Unexpected actor differences: {mismatch}')
    with torch.no_grad():
        normalized=actor.obs_normalizer(positive)
        protected=actor.obs_normalizer(torch.cat((safe_fit,*controls)))
        validation=actor.obs_normalizer(safe_validation)

    def draw(indices,size):
        return indices[torch.randint(len(indices),(size,),generator=random)]

    pos_ids=train.nonzero().flatten()
    neg_ids=torch.arange(len(protected))
    reg=actor.correction
    opt=torch.optim.AdamW(reg.parameters(),lr=.00075,weight_decay=.002)
    for _ in range(1000):
        ii=draw(pos_ids,256)
        jj=draw(neg_ids,512)
        hard_negative=reg.score(protected[jj])
        effective=reg.score(normalized[ii])
        gate_loss=torch.nn.functional.softplus(-effective).mean() + 1.7*torch.nn.functional.softplus(hard_negative).mean()
        action_loss=torch.nn.functional.smooth_l1_loss(reg.delta(normalized[ii]),expected[ii],beta=.10)
        step_loss=action_loss + .16*gate_loss
        opt.zero_grad()
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(reg.parameters(),2.)
        opt.step()

    with torch.no_grad():
        safe=reg.score(validation).sigmoid()
        hard=reg.score(normalized[~train]).sigmoid()
        pred=reg.delta(normalized[~train]); actual=expected[~train]
        error=float((pred-actual).square().mean().sqrt())
        raw_error=float(actual.square().mean().sqrt())
        thresholds={'strict':min(.985,max(.46,float(torch.quantile(safe,.9995))+.025)),
                    'wider':min(.965,max(.40,float(torch.quantile(safe,.995))+.01))}

    if any(not torch.equal(actor.state_dict()[name],value) for name,value in original_actor.items()):
        raise RuntimeError('The original actor was modified')
    results={}
    env['guarded_recovery_actor']=True
    for name,threshold in thresholds.items():
        path=args.output/name
        path.mkdir(exist_ok=False)
        cfg=copy.deepcopy(agent)
        recovery=dict(joint_indices=indices,threshold=threshold,width=min(.08,.9*(1.-threshold)),
                      gain=1. if name=='wider' else .82)
        cfg['actor'].update(class_name='anon_recovery_actor:RecoveryActor',recovery=recovery)
        actor.correction.threshold=threshold
        actor.correction.gain=recovery['gain']
        actor.correction.width=recovery['width']
        checkpoint=copy.deepcopy(original)
        checkpoint['actor_state_dict']=actor.state_dict()
        checkpoint['infos']=dict(recovery_supervised_updates=1000, frozen_source_sha256=digest(base))
        target=path/'policy_final.pt'
        torch.save(checkpoint,target)
        (path/'agent_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        (path/'env_config.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
        exported=torch.jit.script(actor.as_jit().eval())
        torch.jit.save(exported,str(path/'policy_jit.pt'))
        with torch.no_grad():
            probe=all_safe[::211]
            direct=actor(TensorDict({'policy':probe},batch_size=[len(probe)]))
            torch.testing.assert_close(exported(probe),direct)
            latent=actor.obs_normalizer(probe)
            reference=actor.mlp(latent)
            outside=torch.ones(31,dtype=torch.bool);outside[indices]=False
            torch.testing.assert_close(direct[:,outside],reference[:,outside],atol=0.,rtol=0.)
            inactive=actor.correction.score(latent).sigmoid()<=threshold
            torch.testing.assert_close(direct[inactive],reference[inactive],atol=0.,rtol=0.)
        results[name]=dict(checkpoint=str(target.resolve()),sha256=digest(target),recovery=recovery,
                           protected_active_fraction=float((safe>threshold).float().mean()),
                           held_out_recovery_active_fraction=float((hard>threshold).float().mean()))
    metrics=dict(source=str(base),source_sha256=digest(base),datasets=dataset_audit,
                 fixed_actor_unchanged=True,positive_frames=len(positive),rescued_training_cells=total_cells,
                 response_rmse=error,unmodified_response_rmse=raw_error,candidates=results)
    (args.output/'training.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
