"""Independent lateral specializations on the same frozen standing policy."""

import copy
import math

import torch

from anon_lateral_actor import LateralStandingCorrection
from anon_recovery_actor import RecoveryActor


class DirectionalStandingCorrection(LateralStandingCorrection):
    def __init__(self, reference, *, right_onset, right_width, **lateral):
        super().__init__(reference, **lateral)
        if not all(math.isfinite(value) and value > 0. for value in (right_onset, right_width)):
            raise ValueError('Directional correction thresholds must be finite and positive')
        self.left_residual = copy.deepcopy(self.residual).requires_grad_(False)
        self.right_onset = right_onset
        self.right_width = right_width

    def forward(self, raw: torch.Tensor, normalized: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        original = self.reference(raw, normalized, base)
        features = normalized.clamp(-10., 10.)
        previous = self.reference.residual(features)
        gate = self.activation(raw).unsqueeze(-1)
        joints = self.reference.joints
        protected = original.index_select(-1, joints)
        left = protected + gate * (self.left_residual(features) - previous).clamp(
            -self.maximum_action, self.maximum_action)
        right = protected + gate * (self.residual(features) - previous).clamp(
            -self.maximum_action, self.maximum_action)
        right_speed = -.5 * raw[:, 0]
        share = ((right_speed - self.right_onset)/self.right_width).clamp(0., 1.)
        share = share.square() * (3. - 2.*share)
        indices = joints.unsqueeze(0).expand(left.shape[0], -1)
        return torch.lerp(original.scatter(-1, indices, left), original.scatter(-1, indices, right), share[:, None])


class LateralDirectionalActor(RecoveryActor):
    def __init__(self, obs, obs_groups, obs_set, output_dim, recovery, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, recovery, **kwargs)
        if not recovery.get('standing') or not recovery.get('freeze_gate'):
            raise ValueError('Directional recovery requires the frozen standing controller')
        self.correction = DirectionalStandingCorrection(
            self.correction, **recovery['lateral_standing'], **recovery['directional_standing'])
