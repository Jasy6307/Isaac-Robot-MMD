"""Training-only clearance loss against a surface fitted to simulated skirt meshes."""

import math

import torch
from torch.nn.functional import grid_sample


def rotate(quaternion, vector):
    axis, scalar = quaternion[..., :3], quaternion[..., 3:]
    tangent = 2 * torch.linalg.cross(axis, vector)
    return vector + scalar * tangent + torch.linalg.cross(axis, tangent)


def smooth_unit(value):
    t = value.clamp(0, 1)
    return t.square() * (3 - 2 * t)


class HandSkirtClearance:
    def __init__(self, config, device):
        self.weight = float(config['weight'])
        self.margin = float(config['margin_m'])
        self.reference_allowance = float(config['reference_allowance_m'])
        if not math.isfinite(self.weight) or self.weight <= 0 or not 0 < self.margin <= .02:
            raise ValueError('Hand-skirt reward needs a positive weight and a margin up to 2 cm')
        if not 0 < self.reference_allowance <= .08:
            raise ValueError('Reference allowance must be a positive lateral distance up to 8 cm')
        self.surface = torch.as_tensor(config['envelope_m'], device=device, dtype=torch.float32)
        self.points = torch.as_tensor(config['hand_points_m'], device=device, dtype=torch.float32)
        if (self.surface.ndim != 2 or min(self.surface.shape) < 2 or not self.surface.isfinite().all()
                or self.surface.min() < 0 or self.points.ndim != 3 or self.points.shape[0] != 2
                or self.points.shape[-1] != 3 or not self.points.shape[1] or not self.points.isfinite().all()):
            raise ValueError('Invalid calibrated hand points or skirt surface')
        self.surface = self.surface[None, None]
        bounds = torch.as_tensor([config['y_bounds_m'], config['z_bounds_m']], device=device, dtype=torch.float32)
        span = bounds[:, 1] - bounds[:, 0]
        if not bounds.isfinite().all() or not (span > 0).all():
            raise ValueError('Invalid fore/aft or height bounds for the skirt surface')
        self.lower = bounds[:, 0]
        self.inverse_span = 2 / span
        self.sign = torch.tensor([1., -1.], device=device)[None, :, None]

    def surface_gap(self, pelvis, hands):
        axis, scalar = -pelvis[:, None, 3:6], pelvis[:, None, 6:7]
        hand_axis, hand_scalar = hands[..., 3:6], hands[..., 6:7]
        relative_rotation = torch.cat((
            scalar * hand_axis + hand_scalar * axis + torch.linalg.cross(axis, hand_axis),
            scalar * hand_scalar - (axis * hand_axis).sum(dim=-1, keepdim=True),
        ), dim=-1)
        inverse = torch.cat((axis, scalar), dim=-1)
        wrist = rotate(inverse, hands[..., :3] - pelvis[:, None, :3])
        local = wrist[:, :, None] + rotate(relative_rotation[:, :, None], self.points[None])
        grid = ((local[..., 1:] - self.lower) * self.inverse_span - 1).reshape(1, -1, self.points.shape[1], 2)
        surface = grid_sample(self.surface, grid, align_corners=True).reshape_as(local[..., 0])
        return (local[..., 0] * self.sign - surface).amin(dim=-1)

    def __call__(self, pelvis, hands, reference_pelvis, reference_hands, planned_command):
        gaps = self.surface_gap(torch.cat((pelvis, reference_pelvis)), torch.cat((hands, reference_hands)))
        gap, reference_gap = gaps.chunk(2, dim=0)
        # The time-union of deformed skirt shapes can extend well beyond a
        # natural hand trajectory. Bound the requested clearance relative to
        # this phase's existing reference, using the measured arm probe range.
        target = (reference_gap + self.reference_allowance).clamp_max(self.margin)
        deficit_m = (target - gap).clamp_min(0)
        # Fade smoothly above walk speed and during active turns. Deceleration
        # and a settled stand inherit the gap constraint through the planner.
        speed = planned_command[:, :2].norm(dim=-1)
        context = smooth_unit((.75 - speed) / .20) * smooth_unit((.45 - planned_command[:, 2].abs()) / .30)
        deficit = (deficit_m / .025).clamp_max(4)
        cost = torch.sqrt(1 + deficit.square()) - 1
        return -self.weight * context * cost.mean(dim=-1), gap, deficit_m, context
