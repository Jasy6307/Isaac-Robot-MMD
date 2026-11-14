"""Fit tapered sphere chains to bare-leg vertices, then measure cross-leg gaps.

This is a collision proxy, not triangle-mesh intersection detection. The upper
quarter of each thigh is excluded because both legs join the same pelvis there.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import numpy as np


def fit_leg_proxies(model):
    names = {b['name']: i for i, b in enumerate(model['bones'])}
    positions = np.asarray([b['position'] for b in model['bones']])
    vertices = np.asarray([v[0] for v in model['vertices']])
    # Body and socks only; exclude skirt, clothing and inverted-hull outlines.
    indices, offset = [], 0
    for material in model['materials']:
        if material['name'] == '体':
            indices.extend(model['indices'][offset:offset + material['count']])
        offset += material['count']
    points = vertices[np.unique(indices)]
    result = []
    for side in ('左', '右'):
        for a, b, start in ((side+'足', side+'ひざ', .28),
                            (side+'ひざ', side+'足首', .12)):
            ia, ib = names[a], names[b]
            origin, vector = positions[ia], positions[ib]-positions[ia]
            length = np.linalg.norm(vector)
            t = (points-origin) @ vector / (length*length)
            radial = np.linalg.norm(points-origin-t[:, None]*vector, axis=1)
            same_side = points[:, 0] * np.sign(origin[0]) > 0
            stations = np.linspace(start, .95 if start == .28 else .85, 9)
            radii = []
            for station in stations:
                sample = radial[same_side & (abs(t-station) < .055)
                                & (radial < length*.4)]
                if len(sample) < 8:
                    raise ValueError(f'Not enough leg vertices to fit {a} at {station}')
                radii.append(float(np.percentile(sample, 95)))
            result.append(dict(side=side, a=ia, b=ib, t=stations,
                               radii=np.asarray(radii)))
    return result


def proxy_gaps(proxies, world_positions):
    """Return each frame's minimum signed cross-leg sphere surface gap."""
    centers = [world_positions[p['a']][:, None, :]*(1-p['t'][None, :, None])
               + world_positions[p['b']][:, None, :]*p['t'][None, :, None]
               for p in proxies]
    gaps = []
    for i in (0, 1):
        for j in (2, 3):
            distance = np.linalg.norm(centers[i][:, :, None]-centers[j][:, None], axis=-1)
            distance -= proxies[i]['radii'][None, :, None]+proxies[j]['radii'][None, None, :]
            gaps.append(distance.min(axis=(1, 2)))
    return np.min(gaps, axis=0)


def smooth_upper_envelope(values, radius=12):
    """A smooth majorant: never reduce any collision-required displacement."""
    values = np.asarray(values)
    result = values.copy()
    for shift in range(1, radius+1):
        weight = .5*(1+np.cos(np.pi*shift/(radius+1)))
        result[shift:] = np.maximum(result[shift:], values[:-shift]*weight)
        result[:-shift] = np.maximum(result[:-shift], values[shift:]*weight)
    return result
