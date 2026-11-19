"""Training-only anatomical mirrors for the 31-DOF unified policy.

Transforms raw observations and residual actions before normalization.
Lateral running and turn-stop lanes use the phase of their mirrored reference.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch


def opposite(name):
    return name.replace('left_', 'SIDE_').replace('right_', 'left_').replace('SIDE_', 'right_')


class AnatomicalMirror:
    def __init__(self, names, key_names, manifest, offsets, device='cpu'):
        self.names=list(names); self.n=len(names); self.device=device
        if self.n!=31 or len(key_names)!=8:
            raise ValueError('Lateral symmetry requires the audited 31-DOF / eight-keypoint layout')
        joints={j['name']:j for j in manifest['joints']}
        reflection=np.diag([-1.,1.,1.])
        def axis(name):
            joint=joints[name]
            return np.array(joint.get('axis_vector',np.eye(3)['XYZ'.index(joint['axis'])]))
        self.index=torch.tensor([self.names.index(opposite(n)) for n in names],device=device)
        dots=[float(axis(n)@(-reflection@axis(opposite(n)))) for n in names]
        if not np.allclose(np.abs(dots),1.,atol=1e-5):raise ValueError('Non-mirror joint axes')
        signs=np.sign(dots).tolist()
        self.sign=torch.tensor(signs,device=device)
        for i,name in enumerate(names):
            a,b=joints[name],joints[opposite(name)]
            if not np.allclose(a['limits_deg'],np.sort(np.array(b['limits_deg'])*signs[i]),atol=1e-5):
                raise ValueError('Non-mirror limits: '+name)
        self.key_index=torch.tensor([key_names.index(opposite(n)) for n in key_names],device=device)
        self.polar=torch.tensor([-1.,1.,1.],device=device)
        self.axial=torch.tensor([1.,-1.,-1.],device=device)
        self.quaternion=torch.tensor([1.,-1.,-1.,1.],device=device)
        self.lateral_phase_delta=float(offsets[13]-offsets[14])
        self.pivot_phase_delta=0.

    @classmethod
    def from_env(cls,env):
        if not (env.cfg.get('unified_walk_run') and env.cfg.get('imitation_invariant_observation')
                and env.cfg.get('imitation_heading_observation')):
            raise ValueError('Unsupported observation convention for lateral symmetry')
        manifest=json.loads((Path(env.cfg['asset']).parent/'manifest.json').read_text(encoding='utf-8'))
        return cls(env.names,env.key_names,manifest,env.cfg['unified_contact_offsets'],env.device)

    def joints(self,x):return x[...,self.index]*self.sign

    def observations(self,x,pivot_direction=None):
        if x.shape[-1]!=248:raise ValueError('Observation layout changed; re-audit symmetry')
        y=x.clone(); y[...,0:3]=x[...,0:3]*self.polar
        y[...,3:6]=x[...,3:6]*self.axial; y[...,6:9]=x[...,6:9]*self.polar
        y[...,9:12]=x[...,9:12]*self.axial  # forward, left, yaw
        for start in range(12,198,31):y[...,start:start+31]=self.joints(x[...,start:start+31])
        y[...,199:202]=x[...,199:202]*self.polar
        y[...,202:204]=x[...,202:204].flip(-1)
        angle=2*math.pi*self.lateral_phase_delta*torch.sign(x[...,10])
        if pivot_direction is not None:
            turn_angle = (2 * math.pi * self.pivot_phase_delta) * pivot_direction
            angle = torch.where(pivot_direction != 0., turn_angle, angle)
        s,c=x[...,204],x[...,205]
        y[...,204]=s*angle.cos()+c*angle.sin();y[...,205]=c*angle.cos()-s*angle.sin()
        y[...,206:237]=self.joints(x[...,206:237])
        y[...,237:241]=x[...,237:241]*self.quaternion
        y[...,241:245]=x[...,241:245]*self.quaternion
        y[...,245:248]=x[...,245:248]*self.polar
        return y

    def amp(self,x):
        """Mirror unnormalized AMP states or temporal windows, preserving order."""
        if x.shape[-1]!=96:raise ValueError('AMP feature layout changed; re-audit symmetry')
        y=x.clone();y[...,1:4]=x[...,1:4]*self.polar
        y[...,4:7]=x[...,4:7]*self.polar;y[...,7:10]=x[...,7:10]*self.axial
        y[...,10:41]=self.joints(x[...,10:41]);y[...,41:72]=self.joints(x[...,41:72])
        p=x[...,72:].reshape(*x.shape[:-1],8,3)
        y[...,72:]=(p[...,self.key_index,:]*self.polar).flatten(-2)
        return y


def eligibility(env):
    """Store rollout metadata, not a new actor input or guessed clock phase."""
    target=env.commands; current=env.continuous_ref_commands
    future=current+.1*env.continuous_ref_rate
    run_threshold=float(getattr(env,'cfg',{}).get('walk_run_blend_end',.9))
    mask=torch.ones(len(target),device=env.device,dtype=torch.bool)
    for command in (target,current,future):
        mask&=(command[:,0].abs()<1e-4)&(command[:,2].abs()<1e-4)&(command[:,1].abs()>=run_threshold)
        mask&=command[:,1]*target[:,1]>0
    return mask[:,None]


def pivot_reference_phase_delta(motion, cfg):
    """Find the phase shift from the left reference to its right counterpart."""
    from unified_walk_run_reference import sample

    phases = torch.arange(motion.count, device=motion.commands.device, dtype=torch.float32) / motion.count
    command = torch.zeros((motion.count, 3), device=phases.device)
    command[:, 2] = cfg['max_yaw']
    left = sample(motion, phases, command, cfg)['contacts'].flip(-1).cpu().numpy()
    right = sample(motion, phases, -command, cfg)['contacts'].cpu().numpy()
    errors = [np.mean((left - np.roll(right, -shift, axis=0)) ** 2) for shift in range(len(left))]
    return float(np.argmin(errors)) / len(left)


def rollout_metadata(env):
    metadata = {'mirror_eligible': eligibility(env)}
    if env.cfg.get('train_mirror_pivot_stop', False):
        from low_speed_commands import pivot_stop_directions

        if not hasattr(env, 'mirror_pivot_direction'):
            env.mirror_pivot_direction = pivot_stop_directions(env.cfg, env.ids)[:, None]
        # A dedicated lane keeps its turn direction through braking and idle.
        direction = env.mirror_pivot_direction
        metadata['mirror_eligible'] |= direction != 0.
        metadata['mirror_pivot_direction'] = direction
    return metadata


def paired_reference_std(q,mirror,floor):
    """Reward scales from both anatomical copies, without averaging the motion."""
    std=torch.cat([q,mirror.joints(q)],0).std(0)
    # GPU reductions can differ by a few ulps across swapped columns. Share
    # their scalar scale exactly; individual reference poses stay untouched.
    return (.5*(std+std[mirror.index])).clamp_min(floor)


def augmentation(env,obs=None,actions=None):
    """RSL-RL Symmetry API: original samples first, followed by their mirrors."""
    mirror=env.lateral_mirror
    augmented_obs=augmented_actions=None
    if obs is not None:
        from tensordict import TensorDict
        mask=obs['mirror_eligible'].bool()
        transformed={key:value.clone() for key,value in obs.items()}
        direction = None
        if 'mirror_pivot_direction' in obs:
            pivot = obs['mirror_pivot_direction']
            direction = pivot[..., 0]
            transformed['mirror_pivot_direction'] = torch.where(mask, -pivot, pivot)
        transformed['policy']=torch.where(mask,mirror.observations(obs['policy'], direction),obs['policy'])
        augmented_obs=TensorDict({key:torch.cat([value,transformed[key]],0) for key,value in obs.items()},
                                batch_size=[2*len(obs)])
    if actions is not None:
        # compute_loss calls this with means but without observations. The
        # original-slice eligibility from the preceding augmentation is needed.
        mask=obs['mirror_eligible'].bool() if obs is not None else env._lateral_batch_mask
        augmented_actions=torch.cat([actions,torch.where(mask,mirror.joints(actions),actions)],0)
    if obs is not None:env._lateral_batch_mask=obs['mirror_eligible'].bool().detach()
    return augmented_obs,augmented_actions


def configure(agent_config,env):
    if not env.cfg.get('train_lateral_mirror',False):return
    if env.cfg.get('amp_upper_reference_mix',1.)!=1.:
        raise ValueError('Mirror training requires full upper reference mixing')
    coefficient=float(env.cfg.get('train_mirror_loss_coeff',.05))
    if not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('Mirror loss coefficient must be finite and nonnegative')
    env.lateral_mirror=AnatomicalMirror.from_env(env)
    if env.cfg.get('train_mirror_pivot_stop', False):
        env.lateral_mirror.pivot_phase_delta = pivot_reference_phase_delta(env.motion, env.cfg)
    agent_config['algorithm']['symmetry_cfg']=dict(data_augmentation_func='lateral_symmetry:augmentation',
        use_data_augmentation=bool(env.cfg.get('train_mirror_augmentation',True)),
        use_mirror_loss=coefficient>0,mirror_loss_coeff=coefficient)


def install_normalization(runner,env):
    if not env.cfg.get('train_lateral_mirror',False):return
    if not env.cfg.get('train_mirror_augmentation',True):return
    # Raw original+mirror observations use the actor's existing normalizer.
    # Installing this hook never edits inherited statistics or model weights.
    for model in (runner.alg.actor,runner.alg.critic):
        original=model.update_normalization
        def update(obs,_original=original):
            paired,_=augmentation(env,obs=obs)
            return _original(paired)
        model.update_normalization=update
