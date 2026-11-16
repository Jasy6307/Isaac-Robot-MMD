"""Reward-only in-place-turn constraints, evaluated from physical state."""
import torch


def pivot_stability_terms(commands, reference_commands, sole_bottom, reference_contact, root_velocity, config):
    """Penalize missing support, late landing and horizontal drift.

    Costs are nonnegative and bounded. Expected swing feet are excluded from
    landing cost. Neither state, reference nor policy actions are modified.
    World-horizontal speed is rotation invariant and includes lateral drift.
    """
    active=(commands[:,:2].norm(dim=-1)<.05)&(reference_commands[:,:2].norm(dim=-1)<.10)&(commands[:,2].abs()>.10)
    tolerance=float(config.get('pivot_contact_tolerance_m',.012))
    scale=float(config.get('pivot_landing_scale_m',.04))
    speed_scale=float(config.get('pivot_drift_scale_m_s',.15))
    if min(scale,speed_scale)<=0:raise ValueError('Pivot reward scales must be positive')
    gap=(sole_bottom-tolerance).clamp_min(0)
    support=(gap.amin(-1)/scale).clamp(max=2).square()
    expected=reference_contact.clamp(0,1)
    landing=((gap/scale).clamp(max=2).square()*expected).sum(-1)/expected.sum(-1).clamp_min(1)
    # Preserve small mocap body sway and COM arcs during a planted-foot turn.
    drift_speed=(root_velocity[:,:2].norm(dim=-1)-float(config.get('pivot_drift_deadzone_m_s',.10))).clamp_min(0)
    drift=(drift_speed/speed_scale).square().clamp(max=4)
    costs={'support':support*active,'landing':landing*active,'drift':drift*active}
    rate=torch.zeros_like(drift)
    for name,cost in costs.items():
        weight=float(config.get('pivot_'+name+'_weight',0.))
        if weight<0:raise ValueError('Pivot penalty weights must be nonnegative')
        rate-=weight*cost
    return dict(active=active,**costs,reward_rate=rate)
