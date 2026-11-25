"""Shared cel-studio settings exposed as the default mmd-toon look."""

import sys as _sys
from pathlib import Path as _Path
from dataclasses import dataclass, replace
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source


DEFAULT_LOOK = 'mmd-toon'
TOON_STYLE = 'cel-studio'
LOOK_CHOICES = ('original', 'soft-pbr', DEFAULT_LOOK)
ENVIRONMENT_SCALE = .75
BANGS_SHADOW_LEVEL = .68
CEL_BRIGHTNESS_SCALE = .88
FACE_SHAPING = True
TOON_INK = (.010, .007, .014)
FABRIC_SHADOW_PALETTE = 2
FABRIC_SHADOW_DARKEN = .07


@dataclass(frozen=True)
class CelMaterial:
    brightness: float = .5104
    threshold: float = .26
    edge_width: float = .03
    shadow_level: float = .66
    shadow_tint: tuple = (.88, .91, 1.)
    hair_light: float = 0.
    rim_light: float = 0.
    matte_fill: float = 0.
    simple_band: bool = False
    shadow_palette: int = 0


_GENERIC = CelMaterial()
_FACE = CelMaterial(.537, .39, .115, .77, (1., .925, .89), shadow_palette=3)
_SKIN = CelMaterial(.527, .29, .08, .76, (1., .925, .88), shadow_palette=3)
_HAIR = CelMaterial(.511, .24, .039, .65, (.89, .79, .91), hair_light=.11, rim_light=.068, shadow_palette=1)
_TOP = CelMaterial(.537, .32, .032, .58, (.76, .87, 1.), rim_light=.05, matte_fill=.036, shadow_palette=FABRIC_SHADOW_PALETTE)
_SKIRT = CelMaterial(.529, .30, .033, .68, (.87, .91, 1.), rim_light=.025, matte_fill=.030, shadow_palette=FABRIC_SHADOW_PALETTE)
_SHOES = CelMaterial(.556, .26, .043, .62, (.83, .89, 1.), rim_light=.03, matte_fill=.020)
_VEIL = CelMaterial(.517, .27, .055, .71, (.89, .925, 1.), rim_light=.02, shadow_palette=FABRIC_SHADOW_PALETTE)
# Keep small accessory and makeup color blocks as in the approved cel-studio sample.
_DETAIL = CelMaterial(threshold=.25, shadow_level=.64, shadow_tint=(1., 1., 1.), simple_band=True)


def cel_material(name, face_shaping):
    """Match material roles by their labels, independently of mesh/material numbering."""
    if any(key in name for key in ('face', '顔', '脸')):
        return _FACE if face_shaping else replace(
            _FACE, threshold=-.10, shadow_level=.82, shadow_tint=(1., 1., 1.))
    if any(key in name for key in ('眉', '睫', '眼白', '眼影', '项链', '耳环', '颈环',
                                   'kouzi', 'button', 'necklace', 'accessor', 'outline')):
        return _DETAIL
    if any(key in name for key in ('hair', '发', '髪', 'bang', 'fringe')):
        return _HAIR
    if any(key in name for key in ('skin', '脖子', '指甲', '腿', '肌', 'neck')):
        return _SKIN
    if any(key in name for key in ('鞋', '靴', 'shoe', 'boot')):
        return _SHOES
    if any(key in name for key in ('裙纱', 'veil')):
        return _VEIL
    if any(key in name for key in ('裙', 'dress', 'skirt')):
        return _SKIRT
    if any(key in name for key in ('上衣', 'clothes', 'jacket', 'shirt', 'blouse', 'trim', 'mutsmi_white')):
        return _TOP
    return _GENERIC


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
