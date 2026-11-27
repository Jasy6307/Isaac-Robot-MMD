"""Normalized force profiles sampled at the simulator's physics rate."""

import torch


PROFILES = {
    'impulse': (1,),
    'sustained': (1,) * 6,
    'ramp': (1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1),
    'double': (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
}


def profile_table(device):
    width = max(map(len, PROFILES.values())) + 1
    return torch.tensor([tuple(value / sum(ticks) for value in ticks) + (0.,) * (width - len(ticks))
                         for ticks in PROFILES.values()], device=device)


def horizontal_axes(quaternion):
    """World XY axes for Anon's local -Y forward direction."""
    x, y, z, w = quaternion.unbind(-1)
    forward = torch.stack((2 * (z*w - x*y), 2 * (x*x + z*z) - 1), dim=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)
    return forward, torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
