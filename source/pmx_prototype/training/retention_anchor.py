"""Training-only teacher constraint on previous skills, outside transition pool."""
import copy
import math
import torch


def install(runner,env):
    teacher=copy.deepcopy(runner.alg.actor).eval()
    if cfg_path:=env.cfg.get('train_retention_teacher_checkpoint'):
        state=torch.load(cfg_path,map_location=env.device,weights_only=False)
        teacher.load_state_dict(state['actor_state_dict'])
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    update=runner.alg.update
    cfg=env.cfg
    clean_only = cfg.get('train_retention_clean_only', False)
    keep_mild_pushes = cfg.get('train_retention_mild_pushes', False)
    keep_fore_aft = cfg.get('train_retention_fore_aft_pushes', False)
    recovery = getattr(env, 'push_recovery', None)
    lateral_weight = float(cfg.get('train_retention_lateral_weight', 1.))
    if not math.isfinite(lateral_weight) or lateral_weight <= 0.:
        raise ValueError('Lateral retention weight must be finite and positive')
    if clean_only and recovery is None:
        raise ValueError('Clean-only retention requires a push-recovery curriculum')
    if keep_mild_pushes and (not clean_only or recovery.steps is None):
        raise ValueError('Mild-push retention requires a corrective-step curriculum and clean retention')
    if keep_fore_aft and (not clean_only or not recovery.fore_aft_rehearsal):
        raise ValueError('Fore-aft retention requires standing fore-aft rehearsal and clean retention')
    first=(round(100*cfg.get('unified_backward_env_fraction',0.))+round(100*cfg.get('unified_right_env_fraction',0.))+
           round(100*cfg.get('unified_pivot_env_fraction',0.))+round(100*cfg.get('unified_reversal_env_fraction',0.)))
    count=round(100*cfg.get('unified_transition_env_fraction',0.))
    pool=env.ids[(env.ids.remainder(100)<first)|(env.ids.remainder(100)>=first+count)]
    if not len(pool):raise ValueError('Retention constraint needs non-transition environments')
    def wrapped():
        available = pool
        if clean_only:
            retained = ~recovery.enabled[pool]
            if keep_mild_pushes:
                retained |= (recovery.remaining[pool] > 0.) & ~recovery.strong[pool]
            if keep_fore_aft:
                retained |= (recovery.remaining[pool] > 0.) & recovery.protected_fore_aft[pool]
            available = pool[retained]
        if not len(available):
            return update()
        with torch.no_grad():
            ids=available[torch.randint(len(available),(min(len(available),cfg.get('train_retention_batch_size',4096)),),device=env.device)]
            fixed=env.get_observations()[ids].clone()
            target=teacher(fixed).clone()
            weights = None
            if lateral_weight != 1.:
                forward, lateral, yaw = env.commands[ids].abs().unbind(dim=-1)
                side_step = (lateral >= .2) & (forward <= .12) & (yaw <= .12)
                weights = torch.where(side_step, lateral_weight, 1.)
                weights /= weights.mean()
        losses=update()
        errors=[]
        for _ in range(cfg.get('train_retention_steps',2)):
            predicted=runner.alg.actor(fixed)
            squared_error = (predicted-target).square()
            mse = squared_error.mean() if weights is None else (squared_error.mean(dim=-1) * weights).mean()
            loss=float(cfg.get('train_retention_weight',2.))*mse
            runner.alg.optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.alg.actor.parameters(),1.)
            runner.alg.optimizer.step();errors.append(mse.detach())
        losses['retention_action_rmse']=float(torch.stack(errors).mean().sqrt())
        if keep_mild_pushes:
            losses['retention_mild_samples'] = int(recovery.enabled[ids].sum())
        return losses
    runner.alg.update=wrapped
