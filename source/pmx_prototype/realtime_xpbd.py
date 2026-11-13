"""Stateful XPBD: one VMD body pose in, one live secondary pose out.

Only PMX topology and the body animation are loaded. No secondary-motion cache
is read, and no future particle trajectories are generated at initialization.
"""
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdSkel

from convert_pmx import read_pmx
from secondary_motion import njit, prepare_system, simulate


@njit(cache=True)
def multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return np.array([w*X + x*W + y*Z - z*Y, w*Y - x*Z + y*W + z*X,
                     w*Z + x*Y - y*X + z*W, w*W - x*X - y*Y - z*Z])


@njit(cache=True)
def rotate(q, v):
    u = q[:3]
    cross = np.cross(u, v)
    return v + 2 * (q[3] * cross + np.cross(u, cross))


@njit(cache=True)
def fk(local_p, local_q, parents):
    p, q = local_p.copy(), local_q.copy()
    for i in range(len(parents)):
        parent = parents[i]
        if parent >= 0:
            p[i] = p[parent] + rotate(q[parent], local_p[i])
            q[i] = multiply(q[parent], local_q[i])
    return p, q


@njit(cache=True)
def swing(a, b):
    la, lb = np.sqrt(np.sum(a*a)), np.sqrt(np.sum(b*b))
    if min(la, lb) < 1e-12:
        return np.array([0., 0., 0., 1.])
    a, b = a/la, b/lb
    c = np.cross(a, b)
    q = np.array([c[0], c[1], c[2], 1+np.sum(a*b)])
    norm = np.sqrt(np.sum(q*q))
    if norm < 1e-8:
        axis = np.array([1., 0., 0.]) if abs(a[0]) < .9 else np.array([0., 1., 0.])
        c = np.cross(a, axis)
        q = np.array([c[0], c[1], c[2], 0.])
        norm = np.sqrt(np.sum(q*q))
    return q/norm


@njit(cache=True)
def reconstruct(lp, lq, wp, wq, parents, dynamic, node_for_bone, tangent):
    p, q = wp.copy(), wq.copy()
    for i in range(len(parents)):
        if dynamic[i]:
            p[i] = node_for_bone[i]
    for i in range(len(parents)):
        if dynamic[i] and tangent[i] >= 0:
            other = tangent[i]
            delta = swing(wp[other]-wp[i], p[other]-p[i])
            q[i] = multiply(delta, wq[i])
    for i in range(len(parents)):
        parent = parents[i]
        if parent >= 0 and not dynamic[i]:
            p[i] = p[parent] + rotate(q[parent], lp[i])
            q[i] = multiply(q[parent], lq[i])
    out_p, out_q = p.copy(), q.copy()
    for i in range(len(parents)):
        parent = parents[i]
        if parent >= 0:
            inverse = q[parent].copy()
            inverse[:3] *= -1
            out_p[i] = rotate(inverse, p[i]-p[parent])
            out_q[i] = multiply(inverse, q[i])
    return out_p, out_q, p, q


@njit(cache=True)
def collider_poses(wp, wq, bones, offsets, axes):
    a, b = np.empty_like(offsets), np.empty_like(offsets)
    for j in range(len(bones)):
        i = bones[j]
        center = wp[i] + rotate(wq[i], offsets[j])
        delta = rotate(wq[i], axes[j])
        a[j], b[j] = center-delta, center+delta
    return a, b


class RealtimeXPBD:
    def __init__(self, usd, pmx, substeps=4, iterations=12, gravity=9.81):
        self.source = Usd.Stage.Open(str(Path(usd).resolve()))
        if self.source.GetDefaultPrim().GetCustomDataByKey('secondaryMotionSolver'):
            raise ValueError('Realtime input must be the body dance USD, not a baked physics cache')
        if abs(self.source.GetTimeCodesPerSecond()-30) > 1e-6:
            raise ValueError('This solver expects the 30fps native VMD body animation')
        self.model = read_pmx(Path(pmx).resolve(), physics=True)
        self.skeleton = UsdSkel.Skeleton.Get(self.source, '/Character/Skeleton')
        self.animation = UsdSkel.Animation.Get(self.source, '/Character/Animation')
        self.joints = list(self.skeleton.GetJointsAttr().Get())
        self.names = list(self.skeleton.GetJointNamesAttr().Get())
        indices = {path: i for i, path in enumerate(self.joints)}
        self.parents = np.array([indices.get(path.rpartition('/')[0], -1) for path in self.joints])
        self.columns = np.array([indices[path] for path in self.animation.GetJointsAttr().Get()])
        self.rest_p = np.array([m.ExtractTranslation() for m in self.skeleton.GetRestTransformsAttr().Get()])
        self.end_frame = int(self.source.GetEndTimeCode())
        self.substeps, self.iterations, self.gravity = substeps, iterations, gravity
        lp, lq, wp, wq = self.body_pose(0)
        self.system = s = prepare_system(self.model, self.source, self.joints, self.names,
                                        self.parents, wp[None], wq[None])
        self.nodes = np.asarray(s['nodes'])
        self.dynamic = np.array([i in s['dynamic'] for i in range(len(self.joints))])
        self.tangent = np.full(len(self.joints), -1, dtype=np.int64)
        for i in s['dynamic']:
            children = [j for j, p in enumerate(self.parents) if p == i and self.dynamic[j]]
            if children:
                self.tangent[i] = children[0]
            elif self.parents[i] >= 0 and self.dynamic[self.parents[i]]:
                self.tangent[i] = self.parents[i]
        self.collider_bones, self.collider_offsets, self.collider_axes = [], [], []
        raw = np.array([v[0] for v in self.model['vertices']])
        for body in self.model['rigid_bodies']:
            if body['mode'] or body['bone'] < 0:
                continue
            i = s['pmx_to_usd'][body['bone']]
            center = (np.array(body['position'])[[0, 2, 1]] - [0, 0, raw[:, 1].min()]) * s['scale']
            axis = Rotation.from_euler('xyz', body['rotation']).apply([0, 1, 0])[[0, 2, 1]]
            half = body['size'][1]*s['scale']/2 if body['shape'] == 2 else 0
            self.collider_bones.append(i)
            self.collider_offsets.append(center-s['bone_rest'][i])
            self.collider_axes.append(axis*half)
        self.collider_bones = np.array(self.collider_bones, dtype=np.int64)
        self.collider_offsets = np.array(self.collider_offsets)
        self.collider_axes = np.array(self.collider_axes)
        self.rest_points = np.asarray(UsdGeom.Mesh.Get(self.source, '/Character/Body').GetPointsAttr().Get())
        self.weighted_vertices = np.array([any(s['pmx_to_usd'][b] in s['dynamic'] and w > .2
                                              for b, w in zip(v[3], v[4]) if b >= 0)
                                           for v in self.model['vertices']])
        self.reset(0)
        self.advance(0)  # JIT warm-up is excluded from timing and then reset.
        self.reset(0)

    def body_pose(self, frame):
        p = self.rest_p.copy()
        q = np.zeros((len(self.joints), 4))
        q[:, 3] = 1
        # Vt.QuatfArray's NumPy buffer is XYZW, unlike its printed WXYZ form.
        q[self.columns] = np.asarray(self.animation.GetRotationsAttr().Get(frame))
        p[self.columns] = np.asarray(self.animation.GetTranslationsAttr().Get(frame))
        wp, wq = fk(p, q, self.parents)
        return p, q, wp, wq

    def reset(self, frame=0):
        self.lp, self.lq, self.wp, self.wq = self.body_pose(frame)
        self.position = self.wp[self.nodes].copy()
        self.velocity = np.zeros_like(self.position)
        self.previous_targets = self.position.copy()
        self.ca, self.cb = collider_poses(self.wp, self.wq, self.collider_bones,
                                         self.collider_offsets, self.collider_axes)
        self.previous_a, self.previous_b = self.ca.copy(), self.cb.copy()
        self.frame = frame-1

    def advance(self, frame):
        if frame != self.frame+1:
            raise ValueError('Advance sequentially; call reset() before seeking or looping')
        self.lp, self.lq, self.wp, self.wq = self.body_pose(frame)
        target = self.wp[self.nodes]
        self.ca, self.cb = collider_poses(self.wp, self.wq, self.collider_bones,
                                         self.collider_offsets, self.collider_axes)
        s = self.system
        import time
        started = time.perf_counter()
        result, self.max_speed, self.velocity = simulate(
            np.stack([self.previous_targets, target]),
            np.stack([self.previous_a, self.ca]), np.stack([self.previous_b, self.cb]),
            s['radii'], s['invmass'], s['edge_array'], s['lengths'], s['compliance'],
            s['groups'], s['thickness'], self.substeps, self.iterations,
            self.position, self.velocity, True, self.gravity)
        self.solve_ms = (time.perf_counter()-started)*1000
        self.position = result[1]
        positions = self.wp.copy()
        positions[self.nodes] = self.position
        self.pose_p, self.pose_q, physical_p, physical_q = reconstruct(
            self.lp, self.lq, self.wp, self.wq, self.parents, self.dynamic, positions, self.tangent)
        if not np.isfinite(self.pose_p).all() or not np.isfinite(self.pose_q).all():
            raise RuntimeError('Live XPBD produced non-finite bones')
        if np.max(np.abs(physical_p[self.collider_bones]-self.wp[self.collider_bones])) > 1e-6:
            raise RuntimeError('Live XPBD changed the body dance')
        self.previous_targets = target.copy()
        self.previous_a, self.previous_b = self.ca.copy(), self.cb.copy()
        self.frame = frame
        return self.pose_p, self.pose_q


def interpolate_pose(p0, q0, p1, q1, alpha):
    """Render-only interpolation: linear translation and shortest-path SLERP."""
    dot = np.sum(q0*q1, axis=1)
    end = np.where((dot < 0)[:, None], -q1, q1)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    denominator = np.sin(angle)
    safe = np.maximum(denominator, 1e-12)
    w0 = np.where(angle > 1e-5, np.sin((1-alpha)*angle)/safe, 1-alpha)
    w1 = np.where(angle > 1e-5, np.sin(alpha*angle)/safe, alpha)
    q = q0*w0[:, None] + end*w1[:, None]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return p0*(1-alpha)+p1*alpha, q


class InterpolatedXPBD:
    """Independent render sampling over unchanged 30Hz / four-substep batches.

One adjacent physics pose is evaluated ahead for interpolation (at most 33ms).
Only two live poses are retained; no trajectory cache is created or loaded.
"""
    def __init__(self, solver):
        self.solver = solver
        self.reset()

    def reset(self):
        self.solver.reset(0)
        self.previous = self.current = None
        self.solve_ms = 0.0
        self.updates = 0

    def sample(self, frame):
        if frame < 0 or frame > self.solver.end_frame:
            raise ValueError('Render sample is outside the dance')
        desired = int(math.ceil(frame-1e-9))
        self.solve_ms = 0.0
        while self.solver.frame < desired:
            self.previous = self.current
            self.current = self.solver.advance(self.solver.frame+1)
            if self.previous is None:
                self.previous = self.current
            self.solve_ms += self.solver.solve_ms
            self.updates += 1
        if desired != self.solver.frame:
            raise ValueError('Render samples must advance monotonically; reset before seeking')
        if desired == 0 or abs(frame-desired) < 1e-9:
            return self.current
        return interpolate_pose(*self.previous, *self.current, frame-(desired-1))
