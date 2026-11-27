"""A frozen-referenced extension for completing the standing corrective step."""

import copy
import math

import torch
from torch import nn

from anon_lateral_actor import LateralRecoveryActor


class LateralContinuation(nn.Module):
    def __init__(self, reference, observation_indices, threshold=.95, width=.04, maximum_action=1.15):
        super().__init__()
        if reference.idle_reference is None:
            raise ValueError('The continuation needs a verified idle reference')
        if (not 0. <= threshold < 1. or not 0. < width <= 1.-threshold
                or not 0. < maximum_action < math.inf or len(set(observation_indices)) != len(observation_indices)
                or not all(0 <= index < 248 for index in observation_indices)):
            raise ValueError('Invalid continuation gate settings')
        self.reference = reference.requires_grad_(False)
        self.residual = copy.deepcopy(reference.residual).requires_grad_(True)
        self.register_buffer('observation_indices', torch.tensor(observation_indices, dtype=torch.long), persistent=False)
        self.gate = nn.Sequential(nn.Linear(len(observation_indices), 48), nn.ELU(), nn.Linear(48, 24),
                                  nn.ELU(), nn.Linear(24, 1)).requires_grad_(False)
        self.threshold = threshold
        self.width = width
        self.maximum_action = maximum_action

    def score(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.gate(normalized.index_select(-1, self.observation_indices).clamp(-10., 10.)).squeeze(-1)

    def activation(self, raw: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        probability = self.score(normalized).sigmoid()
        active = ((probability-self.threshold) / self.width).clamp(0., 1.)
        command = raw[:, 9:12].abs().amax(dim=-1)
        clean_stop = (1.-command/.035).clamp(0., 1.)
        if self.reference.idle_reference is not None:
            clean_stop *= self.reference.idle_reference(raw)
        return active*clean_stop

    def forward(self, raw: torch.Tensor, normalized: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        original = self.reference(raw, normalized, base)
        features = normalized.clamp(-10., 10.)
        change = (self.residual(features) - self.reference.residual(features)).clamp(-self.maximum_action, self.maximum_action)
        joints = self.reference.reference.joints
        adjusted = original.index_select(-1, joints) + self.activation(raw, normalized).unsqueeze(-1)*change
        return original.scatter(-1, joints.unsqueeze(0).expand(adjusted.shape[0], -1), adjusted)


class LateralContinuationActor(LateralRecoveryActor):
    def __init__(self, obs, obs_groups, obs_set, output_dim, recovery, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, recovery, **kwargs)
        self.correction = LateralContinuation(self.correction, **recovery['lateral_continuation'])
