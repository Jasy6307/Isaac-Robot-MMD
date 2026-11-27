"""Frozen Anon policy with a bounded, observation-gated waist and leg correction."""

import copy
import math

from rsl_rl.models import MLPModel
import torch
from torch import nn


class WaistLegCorrection(nn.Module):
    def __init__(self, obs_size, joint_indices, threshold, width, gain, low_speed=None, straight_run=None,
                 standing=False):
        super().__init__()
        self.register_buffer('joints',torch.tensor(joint_indices,dtype=torch.long))
        self.register_buffer('gate_channels',torch.tensor(list(range(12))+list(range(198,206))+list(range(237,248))))
        self.threshold = float(threshold)
        self.width = float(width)
        self.gain = float(gain)
        if not (0. <= self.threshold < 1. and 0. < self.width <= 1.-self.threshold
                and 0. < self.gain < math.inf):
            raise ValueError('Invalid recovery activation or gain')
        self.split_speed = low_speed is not None
        slower = low_speed or dict(threshold=threshold,width=width,gain=gain)
        self.low_threshold = float(slower['threshold'])
        self.low_width = float(slower['width'])
        self.low_gain = float(slower['gain'])
        if not (0. <= self.low_threshold < 1. and 0. < self.low_width <= 1.-self.low_threshold
                and 0. < self.low_gain < math.inf):
            raise ValueError('Invalid low-speed recovery activation or gain')
        self.split_run = straight_run is not None
        run = straight_run or dict(threshold=threshold, width=width, gain=gain)
        self.run_threshold = float(run['threshold'])
        self.run_width = float(run['width'])
        self.run_gain = float(run['gain'])
        if not (0. <= self.run_threshold < 1. and 0. < self.run_width <= 1.-self.run_threshold
                and 0. < self.run_gain < math.inf):
            raise ValueError('Invalid straight-run recovery activation or gain')
        self.gate = nn.Sequential(nn.Linear(len(self.gate_channels),40),nn.ELU(),nn.Linear(40,24),nn.ELU(),nn.Linear(24,1))
        self.residual = nn.Sequential(nn.Linear(obs_size,96),nn.ELU(),nn.Linear(96,64),nn.ELU(),nn.Linear(64,len(joint_indices)))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.protected_residual = copy.deepcopy(self.residual).requires_grad_(False) if standing else None

    def score(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.gate(normalized.index_select(-1,self.gate_channels).clamp(-10.,10.)).squeeze(-1)

    def delta(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.residual(normalized.clamp(-10.,10.)).clamp(-.9,.9)

    def forward(self, raw: torch.Tensor, normalized: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(self.score(normalized))
        gate = ((probability-self.threshold)/self.width).clamp(0.,1.) * self.gain
        # The recovery demonstrations cover walking/standing and straight forward running.
        command = raw[:,9:12]
        speed_squared = command[:,:2].square().sum(-1)
        walking = speed_squared <= .55**2
        running_forward = (command[:,0] >= .55) & (command[:,0] <= 1.25) & (command[:,1].abs() <= .15)
        if self.split_speed or self.split_run:
            slow_gate = ((probability-self.low_threshold)/self.low_width).clamp(0.,1.) * self.low_gain
            if self.split_speed:
                walking_fraction = ((speed_squared.sqrt()-.12)/.16).clamp(0.,1.)
                gate = torch.lerp(slow_gate,gate,walking_fraction)
            if self.split_run:
                straight_gate = ((probability-self.run_threshold)/self.run_width).clamp(0.,1.) * self.run_gain
                run_gate = torch.where(command[:,2].abs() <= .1, straight_gate, slow_gate)
                gate = torch.where(running_forward & ~walking, run_gate, gate)
        gate = gate * (walking | running_forward)
        if self.protected_residual is None:
            delta = self.delta(normalized)
        else:
            features = normalized.clamp(-10., 10.)
            reference = self.protected_residual(features)
            standing = (1. - speed_squared.sqrt()/.06).clamp(0., 1.)
            standing *= (1. - command[:,2].abs()/.08).clamp(0., 1.)
            change = (self.residual(features) - reference).clamp(-.8, .8)
            delta = reference.clamp(-.9, .9) + standing.unsqueeze(-1)*change
        action = base.index_select(-1,self.joints) + delta*gate.unsqueeze(-1)
        indices = self.joints.unsqueeze(0).expand(action.shape[0],-1)
        return base.scatter(-1,indices,action)


class RecoveryActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, recovery, **kwargs):
        super().__init__(obs,obs_groups,obs_set,output_dim,**kwargs)
        if self.obs_dim != 248 or output_dim != 31 or self.obs_groups != ['policy']:
            raise ValueError('The recovery actor requires Anon’s 248-channel unified walk/run observation')
        joints = recovery['joint_indices']
        if len(joints)!=15 or len(set(joints))!=15 or not all(0 <= index < 31 for index in joints):
            raise ValueError('Expected the three waist and twelve leg action indices')
        self.requires_grad_(False)
        self.correction=WaistLegCorrection(self.obs_dim,joints,recovery['threshold'],recovery['width'],recovery['gain'],
                                           recovery.get('low_speed'), recovery.get('straight_run'),
                                           recovery.get('standing', False))
        if recovery.get('freeze_gate',False):
            self.correction.gate.requires_grad_(False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        normalized = self.get_latent(obs,masks,hidden_state)
        output = self.correction(obs['policy'],normalized,self.mlp(normalized))
        if self.distribution is None:
            return output
        if stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    def update_normalization(self, obs):
        # The original network and its normalization jointly define the frozen policy.
        pass

    def as_jit(self):
        return RecoveryExport(self)

    def as_onnx(self, verbose=False):
        return RecoveryExport(self)


class RecoveryExport(nn.Module):
    is_recurrent = False

    def __init__(self, actor):
        super().__init__()
        self.obs_normalizer=copy.deepcopy(actor.obs_normalizer)
        self.mlp=copy.deepcopy(actor.mlp)
        self.correction=copy.deepcopy(actor.correction)
        self.deterministic_output=actor.distribution.as_deterministic_output_module() if actor.distribution else nn.Identity()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_normalizer(observation)
        return self.deterministic_output(self.correction(observation,normalized,self.mlp(normalized)))

    def get_dummy_inputs(self):
        return (torch.zeros(1,248),)

    @property
    def input_names(self):
        return ['obs']

    @property
    def output_names(self):
        return ['actions']

    @torch.jit.export
    def reset(self) -> None:
        pass
