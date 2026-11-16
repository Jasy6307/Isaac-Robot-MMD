"""World velocity at sole points, with COM-based rigid-body velocities."""
import torch

def point_velocity(linear,angular,world_offsets,world_com_offsets):
    return linear[:,:,None,:]+torch.cross(angular[:,:,None,:].expand_as(world_offsets),world_offsets-world_com_offsets[:,:,None,:],dim=-1)


def displacement_slip_square(sole,previous_sole,near,previous_near,dt):
    """Squared planar displacement of the same grounded sole corners.

    Both states must belong to the same step, after any reset. Matching corner
    identities excludes heel/toe changes instead of mistaking them for sliding.
    """
    stable=near&previous_near
    square=((sole-previous_sole)[...,:2]/dt).square().sum(-1).clamp(max=1.)
    return (square*stable).sum((1,2))/stable.sum((1,2)).clamp_min(1)


def displacement_slip_speed(sole,previous_sole,near,previous_near,dt):
    """Mean grounded-corner speed, matching the geometric audit's statistic."""
    stable=near&previous_near
    speed=((sole-previous_sole)[...,:2]/dt).norm(dim=-1).clamp(max=1.)
    return (speed*stable).sum((1,2))/stable.sum((1,2)).clamp_min(1)
