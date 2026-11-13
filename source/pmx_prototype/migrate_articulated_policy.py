"""Name-based action and observation migration for the imitation observation schema."""
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

def migrate(runner,env,state,old_names):
    old_n=len(old_names);new_n=env.num_dofs
    joint_map=[env.names.index(n) for n in old_names]
    columns=list(range(12))
    for block in range(6):columns += [12+block*new_n+j for j in joint_map]
    columns+=list(range(12+6*new_n,20+6*new_n))
    columns += [20+6*new_n+j for j in joint_map]
    columns+=list(range(20+7*new_n,31+7*new_n))
    assert len(columns)==state['actor_state_dict']['mlp.0.weight'].shape[1]
    for role,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
        target=model.state_dict()
        for key,old in state[role+'_state_dict'].items():
            value=target[key]
            if key=='mlp.0.weight':value.zero_();value[:,columns]=old
            elif role=='actor' and key in ['mlp.6.weight','mlp.6.bias']:
                value.zero_();value[joint_map]=old
            elif role=='actor' and key=='distribution.log_std_param':
                value.fill_(float(torch.log(torch.tensor(.08))));value[joint_map]=old
            elif key.startswith('obs_normalizer.') and value.numel()==31+7*new_n:
                value.fill_(1. if key.endswith(('_var','_std')) else 0.)
                value[...,columns]=old
            elif value.shape==old.shape:value.copy_(old)
            else:raise ValueError(f'Unmapped migration tensor {role}:{key}, {old.shape}->{value.shape}')
        model.load_state_dict(target)
    runner.current_learning_iteration=int(state['iter'])
    x=torch.randn(64,len(columns),device=env.device)
    old_actor=MLPModel(TensorDict({'policy':x},batch_size=[64]),{'actor':['policy']},'actor',old_n,
        hidden_dims=[256,128,128],activation='elu',obs_normalization=True,
        distribution_cfg=dict(class_name='GaussianDistribution',init_std=.5,std_type='log')).to(env.device)
    old_actor.load_state_dict(state['actor_state_dict']);old_actor.eval();runner.alg.actor.eval()
    new_x=torch.zeros(64,31+7*new_n,device=env.device);new_x[:,columns]=x
    with torch.inference_mode():
        expected=old_actor(TensorDict({'policy':x},batch_size=[64]))
        actual=runner.alg.actor(TensorDict({'policy':new_x},batch_size=[64]))
        error=float((actual[:,joint_map]-expected).abs().max())
        new_ids=[i for i in range(new_n) if i not in joint_map]
        added=float(actual[:,new_ids].abs().max())
    runner.alg.actor.train()
    assert error<1e-5 and added==0.,(error,added)
    return dict(kind='named_imitation_joint_extension',old_actions=old_n,new_actions=new_n,
        max_existing_action_error=error,max_new_action=added,optimizer='fresh',amp='reset')
