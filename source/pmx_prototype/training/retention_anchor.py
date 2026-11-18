"""Training-only teacher constraint on previous skills, outside transition pool."""
import copy
import torch


def install(runner,env):
    teacher=copy.deepcopy(runner.alg.actor).eval()
    if cfg_path:=env.cfg.get('train_retention_teacher_checkpoint'):
        state=torch.load(cfg_path,map_location=env.device,weights_only=False)
        teacher.load_state_dict(state['actor_state_dict'])
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    update=runner.alg.update
    cfg=env.cfg
    first=(round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))+
           round(100*cfg.get('unified_pivot_env_fraction',0.))+round(100*cfg.get('unified_reversal_env_fraction',0.)))
    count=round(100*cfg.get('unified_transition_env_fraction',0.))
    pool=env.ids[(env.ids.remainder(100)<first)|(env.ids.remainder(100)>=first+count)]
    if not len(pool):raise ValueError('Retention constraint needs non-transition environments')
    def wrapped():
        with torch.no_grad():
            ids=pool[torch.randint(len(pool),(min(len(pool),cfg.get('train_retention_batch_size',4096)),),device=env.device)]
            fixed=env.get_observations()[ids].clone()
            target=teacher(fixed).clone()
        losses=update()
        errors=[]
        for _ in range(cfg.get('train_retention_steps',2)):
            predicted=runner.alg.actor(fixed)
            mse=(predicted-target).square().mean()
            loss=float(cfg.get('train_retention_weight',2.))*mse
            runner.alg.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.alg.actor.parameters(),1.)
            runner.alg.optimizer.step();errors.append(mse.detach())
        losses['retention_action_rmse']=float(torch.stack(errors).mean().sqrt())
        return losses
    runner.alg.update=wrapped
