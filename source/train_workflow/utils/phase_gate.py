"""Smooth, reference-clock-driven gate; no dance-specific timestamps."""
import torch


def jump_teacher_gate(seconds,schedules):
    gate=torch.zeros_like(seconds)
    for schedule in schedules:
        lo,hi=schedule[0][0]-.6,schedule[-1][1]+.6
        ramp=torch.minimum((seconds-lo)/.3,(hi-seconds)/.3).clamp(0.,1.)
        gate=torch.maximum(gate,ramp.square()*(3.-2.*ramp))
    return gate
