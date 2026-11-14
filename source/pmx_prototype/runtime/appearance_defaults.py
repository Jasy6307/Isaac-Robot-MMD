"""Approved look 08; shared by character preview and recording entry points."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source


DEFAULT_LOOK = 'mmd-toon'
LOOK_CHOICES = ('original', 'soft-pbr', DEFAULT_LOOK)
ENVIRONMENT_SCALE = .75
BANGS_SHADOW_LEVEL = .65
CEL_BRIGHTNESS_SCALE = .88
FACE_SHAPING = True


def resolve_appearance_args(args):
    """Fill unspecified options using the approved look or the base look values."""
    refined = args.look == DEFAULT_LOOK
    for name, approved, base_value in (
        ('environment_scale', ENVIRONMENT_SCALE, 1.),
        ('bangs_shadow_level', BANGS_SHADOW_LEVEL, .72),
        ('cel_brightness_scale', CEL_BRIGHTNESS_SCALE, 1.),
        ('face_shaping', FACE_SHAPING, False),
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, approved if refined else base_value)
    return args
