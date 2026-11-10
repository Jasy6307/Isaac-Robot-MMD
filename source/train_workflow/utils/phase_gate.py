"""Smooth, reference-clock-driven gate; no dance-specific timestamps."""
import math
import torch


def jump_teacher_gate(seconds, schedules, *, pre_seconds=.6, post_seconds=.6,
                      ramp_seconds=.3):
    """Return a smooth residual-policy gate around reference flight phases.

    The historical defaults retain the original generic behavior.  Auto-jump
    tasks can pass a shorter pre-window so the normal policy owns the
    crouch/setup phase and the jump policy takes over at push-off.
    """
    for name, value in (("pre_seconds", pre_seconds), ("post_seconds", post_seconds),
                        ("ramp_seconds", ramp_seconds)):
        if not math.isfinite(float(value)) or float(value) < 0.:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(ramp_seconds) <= 0.:
        raise ValueError("ramp_seconds must be positive")
    gate=torch.zeros_like(seconds)
    for schedule in schedules:
        lo,hi=(schedule[0][0]-float(pre_seconds),
               schedule[-1][1]+float(post_seconds))
        ramp=torch.minimum((seconds-lo)/float(ramp_seconds),
                           (hi-seconds)/float(ramp_seconds)).clamp(0.,1.)
        gate=torch.maximum(gate,ramp.square()*(3.-2.*ramp))
    return gate
