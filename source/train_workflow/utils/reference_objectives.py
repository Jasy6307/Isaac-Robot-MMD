"""Simulator-independent kernels for reference-conditioned dance objectives."""
import numpy as np
import torch

REWARD_VERSION = "reference-tracking-v1"


def tracking_score(normalized_squared_error):
    """Bounded precision reward with a non-exponential large-error tail."""
    return 1. / (1. + normalized_squared_error)


def contact_evidence(feet_world, fps, valid):
    """Two disjoint confidence channels; unknown contact is not a hard negative.

    Ankle origins are geometric proxies, not measured soles. Distances are in
    metres for this G1 model; all motions use the same tolerances.
    """
    from source.train_workflow.utils.motion.auto_jump import _causal_minimum
    feet = np.asarray(feet_world, float)
    valid = np.asarray(valid, bool)
    if feet.ndim != 3 or feet.shape[1:] != (2, 3) or len(feet) < 3:
        raise ValueError("Expected at least three frames of two foot positions")
    if valid.shape != (len(feet),) or not np.isfinite(feet).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid contact reference")
    floor = _causal_minimum(feet[:, :, 2].min(-1), max(3, round(.9 * fps)))
    clearance = np.maximum(feet[:, :, 2] - floor[:, None], 0.)
    vz = np.gradient(feet[:, :, 2], 1 / fps, axis=0)
    support = np.clip((.035 - clearance) / .025, 0., 1.) * np.clip((.25 - np.abs(vz)) / .15, 0., 1.)
    airborne = np.clip((clearance - .035) / .045, 0., 1.)
    return support * valid[:, None], airborne * valid[:, None]


def contact_score(contacts, support, airborne):
    """Fixed two-foot denominator: uncertain frames earn no contact bonus."""
    contacts = contacts.float()
    return (support * contacts + airborne * (1. - contacts)).mean(-1)


def exploration_strength(progress):
    """Early jump exploration vanishes halfway through the training budget."""
    return max(0., 1. - 2. * float(progress))
