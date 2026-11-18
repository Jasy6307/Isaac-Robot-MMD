"""Capture and audit checkpoint input/output identity before learning."""
import torch


def capture(models,obs):
    fixed=obs[:256].clone()
    snapshots={}
    with torch.no_grad():
        for name,model in models.items():
            snapshots[name]=dict(normalizer={k:v.clone() for k,v in model.state_dict().items() if k.startswith('obs_normalizer.')},
                latent=model.get_latent(fixed).clone(),output=model(fixed).clone())
    return fixed,snapshots


def audit(models,captured):
    fixed,before=captured;result={}
    with torch.no_grad():
        for name,model in models.items():
            state=model.state_dict();old=before[name]
            same=all(torch.equal(state[k],v) for k,v in old['normalizer'].items())
            latent_error=float((model.get_latent(fixed)-old['latent']).abs().max())
            output_error=float((model(fixed)-old['output']).abs().max())
            result[name]=dict(normalizer_identical=same,normalized_input_max_error=latent_error,
                              output_max_error=output_error,observations=len(fixed),
                              count=float(state['obs_normalizer.count']))
            if not same or latent_error!=0 or output_error!=0:
                raise RuntimeError(f'Checkpoint continuation changed {name} before learning: {result[name]}')
    return result
