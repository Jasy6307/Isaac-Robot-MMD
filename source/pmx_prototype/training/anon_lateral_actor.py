"""A directional standing correction layered over Anon's frozen recovery policy."""

import copy
import math

import torch
from torch import nn

from anon_recovery_actor import RecoveryActor


class IdleReferenceGuard(nn.Module):
    """Distinguish standing from a zero command that is still decelerating."""

    def __init__(self, lower_pose, upper_pose, lower_rate, upper_rate, pose_width=.025, rate_width=.008):
        super().__init__()
        limits = torch.tensor([lower_pose, upper_pose, lower_rate, upper_rate], dtype=torch.float32)
        if (limits.shape != (4, 31) or not torch.isfinite(limits).all()
                or (limits[::2] > limits[1::2]).any()
                or not all(math.isfinite(value) and value > 0. for value in (pose_width, rate_width))):
            raise ValueError('Invalid idle reference envelope')
        # These bounds belong to the motion and observation configuration. Keep
        # the checkpoint tensor layout compatible with the previous controller.
        self.register_buffer('limits', limits, persistent=False)
        self.pose_width = pose_width
        self.rate_width = rate_width

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        q = raw[:, 105:136]
        dq = raw[:, 136:167]
        future = raw[:, 206:237]
        below = self.limits[0] - torch.minimum(q, future)
        above = torch.maximum(q, future) - self.limits[1]
        pose_error = torch.maximum(below, above).amax(dim=-1) / self.pose_width
        rate_error = torch.maximum(self.limits[2] - dq, dq - self.limits[3]).amax(dim=-1) / self.rate_width
        return 1. - torch.maximum(pose_error, rate_error).clamp(0., 1.)


class LateralStandingCorrection(nn.Module):
    def __init__(self, reference, *, lateral_onset=.23, lateral_width=.18,
                 dominance_margin=.065, dominance_width=.15, maximum_action=1.15, idle_reference=None):
        super().__init__()
        if not all(math.isfinite(value) and value > 0. for value in (
                lateral_onset, lateral_width, dominance_margin, dominance_width, maximum_action)):
            raise ValueError('Lateral activation thresholds must be finite and positive')
        self.reference = reference.requires_grad_(False)
        self.residual = copy.deepcopy(reference.residual).requires_grad_(True)
        self.lateral_onset = lateral_onset
        self.lateral_width = lateral_width
        self.dominance_margin = dominance_margin
        self.dominance_width = dominance_width
        self.maximum_action = maximum_action
        self.idle_reference = IdleReferenceGuard(**idle_reference) if idle_reference is not None else None

    def activation(self, raw: torch.Tensor) -> torch.Tensor:
        # Policy observations use twice the body-frame linear velocity. Anon's
        # local X points left and local -Y points forward.
        speed_xy = .5 * raw[:, :2].abs()
        lateral, forward = speed_xy.unbind(-1)
        impact = ((lateral - self.lateral_onset) / self.lateral_width).clamp(0., 1.)
        dominance = ((lateral - 1.2*forward - self.dominance_margin) / self.dominance_width).clamp(0., 1.)
        command = raw[:, 9:12]
        standing = (1. - command[:, :2].square().sum(-1).sqrt() / .06).clamp(0., 1.)
        standing *= (1. - command[:, 2].abs() / .08).clamp(0., 1.)
        if self.idle_reference is not None:
            standing *= self.idle_reference(raw)
        return impact * dominance * standing

    def forward(self, raw: torch.Tensor, normalized: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        original = self.reference(raw, normalized, base)
        features = normalized.clamp(-10., 10.)
        change = self.residual(features) - self.reference.residual(features)
        change = change.clamp(-self.maximum_action, self.maximum_action)
        joints = self.reference.joints
        actions = original.index_select(-1, joints) + self.activation(raw).unsqueeze(-1) * change
        return original.scatter(-1, joints.unsqueeze(0).expand(actions.shape[0], -1), actions)


class LateralRecoveryActor(RecoveryActor):
    def __init__(self, obs, obs_groups, obs_set, output_dim, recovery, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, recovery, **kwargs)
        if not recovery.get('standing') or not recovery.get('freeze_gate'):
            raise ValueError('Standing lateral recovery requires the frozen standing controller')
        self.correction = LateralStandingCorrection(self.correction, **recovery['lateral_standing'])
