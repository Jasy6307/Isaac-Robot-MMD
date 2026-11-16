"""Training-only rehearsal of physical-teacher labels; unused during inference."""
import numpy as np
import torch
from tensordict import TensorDict

def install(runner,env):
    cfg=env.cfg
    data=np.load(cfg['train_teacher_anchor_path'])
    x=torch.tensor(data['observations'],device=env.device);y=torch.tensor(data['actions'],device=env.device)
    groups=torch.tensor(data['groups'],device=env.device)
    pools=[(groups==group).nonzero().flatten() for group in cfg.get('train_teacher_anchor_groups',[0,1])]
    if any(len(pool)==0 for pool in pools):
        raise ValueError('Teacher rehearsal dataset is missing a requested skill group')
    optimizer=torch.optim.Adam(runner.alg.actor.mlp.parameters(),lr=cfg.get('train_teacher_anchor_lr',2e-5))
    update=runner.alg.update
    def wrapped():
        losses=update();errors=[]
        for _ in range(cfg.get('train_teacher_anchor_steps',4)):
            ids=torch.cat([pool[torch.randint(len(pool),(512,),device=env.device)] for pool in pools])
            predicted=runner.alg.actor(TensorDict({'policy':x[ids]},batch_size=[len(ids)]))
            loss=(predicted-y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(runner.alg.actor.mlp.parameters(),1.);optimizer.step()
            errors.append(loss.detach())
        losses['teacher_rehearsal_rmse']=torch.stack(errors).mean().sqrt().item()
        return losses
    runner.alg.update=wrapped
