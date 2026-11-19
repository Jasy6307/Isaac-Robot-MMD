"""Character-specific measured source soles; no guessed cross-character fallback."""
from dataclasses import dataclass
import unicodedata
import numpy as np


@dataclass(frozen=True)
class SoleSelection:
    mesh_path: str
    ankle_margin_m: float = .005
    left_sign: int = 1
    minimum_leg_skin_weight: float | None = None


def measure_source_soles(source, frames, fps, selection, *, start_seconds=0.):
    """Skin actual mesh vertices with USD's row-vector skinning transforms.

    The Anon selector is an explicit caller choice. Other characters need their
    own mesh and vertex selection, rather than silently inheriting this mask.
    """
    from pxr import UsdGeom, UsdSkel
    mesh = UsdGeom.Mesh.Get(source.stage, selection.mesh_path)
    if not mesh:
        raise ValueError(f'Missing source sole mesh: {selection.mesh_path}')
    binding = UsdSkel.BindingAPI(mesh)
    points = np.asarray(mesh.GetPointsAttr().Get())
    indices = binding.GetJointIndicesPrimvar()
    weights = binding.GetJointWeightsPrimvar()
    width = indices.GetElementSize()
    if width <= 0 or weights.GetElementSize() != width:
        raise ValueError('Unsupported skinning influence layout')
    ids = np.asarray(indices.ComputeFlattened()).reshape(-1, width)
    values = np.asarray(weights.ComputeFlattened()).reshape(-1, width)
    joints = binding.GetJointsAttr().Get()
    if joints and list(joints) != list(source.query.GetJointOrder()):
        raise ValueError('Mesh joint order requires explicit remapping')
    geom_bind = binding.GetGeomBindTransformAttr().Get()
    if geom_bind is not None and not np.allclose(np.asarray(geom_bind), np.eye(4)):
        raise ValueError('Nonidentity mesh bind transform needs a character adapter')
    if len(ids) != len(points) or len(values) != len(points):
        raise ValueError('Skinning data does not cover vertices')
    result, metadata = {}, {}
    if selection.minimum_leg_skin_weight is not None:
        names = [unicodedata.normalize('NFKC', name) for name in
                 UsdSkel.Skeleton.Get(source.stage, '/Character/Skeleton').GetJointNamesAttr().Get()]
    for side, prefix, sign in (('left', '左', selection.left_sign), ('right', '右', -selection.left_sign)):
        ankle = source.names[prefix + '足首']
        mask = (points[:, 0] * sign > 0) & (points[:, 2] < source.rest[ankle, 2, 3] + selection.ankle_margin_m)
        spatial_count = int(mask.sum())
        if selection.minimum_leg_skin_weight is not None:
            leg = np.asarray([name.startswith((prefix + '足', prefix + 'ひざ')) for name in names])
            mask &= (values * leg[ids]).sum(axis=1) >= selection.minimum_leg_skin_weight
        if not mask.any():
            raise ValueError(f'No {side} sole vertices; adapt this character')
        pts, ji, jw = points[mask], ids[mask], values[mask]
        hp = np.c_[pts, np.ones(len(pts))]
        floor = float(pts[:, 2].min())
        heights = []
        for i in range(frames):
            time = source.first + (start_seconds + i / fps) * source.fps
            if time > source.last + 1e-5:
                raise ValueError('Source timeline too short')
            transforms = np.asarray(source.query.ComputeSkinningTransforms(time))
            posed = (np.einsum('vi,vwij->vwj', hp, transforms[ji]) * jw[:, :, None]).sum(1)[:, :3]
            heights.append(float(posed[:, 2].min() - floor))
        result[side] = np.asarray(heights)
        metadata[side] = dict(vertex_count=int(mask.sum()), rest_floor_m=floor)
        if selection.minimum_leg_skin_weight is not None:
            metadata[side].update(spatial_candidates=spatial_count,
                                  minimum_leg_skin_weight=selection.minimum_leg_skin_weight)
    return result, metadata
