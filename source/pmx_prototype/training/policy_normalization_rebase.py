"""Change observation statistics without changing a model's raw-input function."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import torch


def rebase_state(state,new_statistics,eps=.01):
    result={k:v.clone() for k,v in state.items()}
    old_mean=state['obs_normalizer._mean'].flatten()
    old_scale=state['obs_normalizer._std'].flatten()+eps
    new_mean=new_statistics['obs_normalizer._mean'].flatten()
    new_scale=new_statistics['obs_normalizer._std'].flatten()+eps
    weight=state['mlp.0.weight']
    result['mlp.0.weight']=weight*(new_scale/old_scale)[None]
    result['mlp.0.bias']=state['mlp.0.bias']+weight@((new_mean-old_mean)/old_scale)
    for key,value in new_statistics.items():
        if key.startswith('obs_normalizer.'):result[key]=value.clone()
    return result


def apply(runner,env,checkpoint):
    loaded=torch.load(checkpoint,map_location=env.device,weights_only=False)
    observations=env.get_observations();audit={}
    with torch.no_grad():
        for role,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
            before=model(observations).clone()
            model.load_state_dict(rebase_state(model.state_dict(),loaded[role+'_state_dict'],model.obs_normalizer.eps))
            delta=float((before-model(observations)).abs().max())
            if delta>1e-4:raise RuntimeError(f'{role} normalization rebase parity failed: {delta}')
            audit[role+'_raw_output_max_delta']=delta
    return dict(statistics_checkpoint=str(checkpoint),**audit,normalization_updates_frozen=True)
