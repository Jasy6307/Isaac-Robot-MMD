"""Authored procedural gait style, not motion capture or a learned AMP prior.

Targets use radians and Anon's -Y forward convention. A low-weight style reward
allows physical balancing to deviate from these coordinated reference curves.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import math
import torch

UPPER_NAMES = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
               'neck_yaw_joint', 'neck_pitch_joint'] + [s+'_'+n+'_joint' for s in ('left', 'right')
               for n in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_flex')]


def cadence(commands):
    activity = torch.maximum(torch.linalg.vector_norm(commands[:, :2], dim=-1)/.4,
                             commands[:, 2].abs()/.6).clamp(0, 1)
    return 1.05+.2*activity


def upper_reference(phase, commands):
    speed = torch.linalg.vector_norm(commands[:, :2], dim=-1)
    activity = torch.maximum(speed/.4, commands[:, 2].abs()/.6).clamp(0, 1)
    wave = torch.cos(2*math.pi*phase)
    second = torch.cos(4*math.pi*phase)
    q = torch.zeros((len(phase), 13), device=phase.device)
    q[:, 0] = .055*activity*wave+.035*commands[:, 2]
    q[:, 1] = .025*activity*wave
    q[:, 2] = .035*speed/.4+.007*activity*second
    q[:, 3] = -.6*q[:, 0]+.08*commands[:, 2]
    q[:, 4] = -.85*q[:, 2]
    for k, sign in [(5, 1.), (9, -1.)]:
        q[:, k] = sign*(.75*commands[:, 0]+.08*speed)*wave
        q[:, k+1] = sign*(.75+.025*activity+.018*activity*torch.sin(2*math.pi*phase))
        q[:, k+2] = sign*.025*activity*torch.sin(2*math.pi*phase)
        q[:, k+3] = -sign*(.14+.09*activity+.025*activity*wave)
    return q
