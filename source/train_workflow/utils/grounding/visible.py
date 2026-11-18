"""Visible geometry shared by source-contact projection and mesh-driven retries."""
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial import ConvexHull


@lru_cache(maxsize=1)
def _stage(usd):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd))
    paths = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for path in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()):
                paths[path.name] = path
    return stage, UsdGeom.XformCache(), paths


@lru_cache(maxsize=64)
def visible_hull(usd, link):
    from pxr import Usd, UsdGeom

    stage, cache, paths = _stage(usd)
    body = stage.GetPrimAtPath(paths[link])
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
    pieces = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if (prim.IsA(UsdGeom.Mesh) and '/visuals/' in str(prim.GetPath()) and
                UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            matrix = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
            pieces.append(points @ matrix[:3, :3].T + matrix[:3, 3])
    if not pieces:
        raise ValueError(f'No visible geometry for {link}')
    points = np.unique(np.concatenate(pieces), axis=0)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    normals = np.unique(np.round(hull.equations[:, :3], 6), axis=0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return vertices, normals, (vertices @ normals.T).max(0) + 1e-7


class MeshPairConstraints:
    """Add conservative supporting-plane distances when FCL finds a missed pair."""

    def __init__(self, model, usd):
        self.model = model
        self.usd = usd
        self.base_distance = model.distances
        self.base_count = len(model.pairs)
        self.entries = {}
        self.links = {}
        self.names = set()
        model.extra_distance_count = 0
        model.distances = self.distances

    def add(self, pair):
        a, b = sorted(pair)
        if (a, b) in self.names:
            return
        for name in (a, b):
            if name not in self.links:
                vertices, normal, offset = visible_hull(self.usd, name)
                center = (vertices.max(0)+vertices.min(0))/2
                radius = np.linalg.norm(vertices-center, axis=1).max()+1e-7
                self.links[name] = (*map(self.model._tensor, (vertices, normal, offset, center)), radius)
        label = [f'{a} / {b}', f'{b} / {a}']
        column = next((i for i, value in enumerate(self.model.pair_labels) if value in label), None)
        if column is None:
            column = self.base_count+self.model.extra_distance_count
            self.model.extra_distance_count += 1
        self.entries[column] = (a, b)
        self.names.add((a, b))

    def _distance(self, rotations, positions, a, b):
        va, na, da, ca, radius_a = self.links[a]
        vb, nb, db, cb, radius_b = self.links[b]
        ra, rb = rotations[a], rotations[b]
        pa, pb = positions[a], positions[b]
        center_a = (ra @ ca).reshape(-1, 3)+pa
        center_b = (rb @ cb).reshape(-1, 3)+pb
        # Spheres enclosing both meshes prove clearance far from interaction.
        # Close samples use the actual visible hulls in both directions.
        distance = torch.linalg.vector_norm(center_a-center_b, dim=-1)-radius_a-radius_b
        near = torch.nonzero(distance < .06).flatten()
        for batch in near.split(32):
            if not len(batch):
                continue
            a_in_b = (va @ ra[batch].transpose(-1, -2)+(pa[batch]-pb[batch])[:, None]) @ rb[batch]
            b_in_a = (vb @ rb[batch].transpose(-1, -2)+(pb[batch]-pa[batch])[:, None]) @ ra[batch]
            a_gap = ((a_in_b @ nb.T).amin(dim=1)-db).amax(dim=1)
            b_gap = ((b_in_a @ na.T).amin(dim=1)-da).amax(dim=1)
            distance[batch] = torch.maximum(a_gap, b_gap)
        return distance

    def distances(self, rotations, positions, pair_indices=None):
        if pair_indices is None:
            indices = np.arange(self.base_count+self.model.extra_distance_count)
            base = self.base_distance(rotations, positions)
        else:
            indices = np.asarray(pair_indices)
            base = self.base_distance(rotations, positions, indices[indices < self.base_count])
        extra = int((indices >= self.base_count).sum())
        result = torch.cat([base, base.new_empty((len(base), extra))], dim=1) if extra else base
        for col, pair in self.entries.items():
            match = np.flatnonzero(indices == col)
            if match.size:
                result[:, match[0]] = self._distance(rotations, positions, *pair)
        return result
