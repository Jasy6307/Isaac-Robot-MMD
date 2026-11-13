"""Skeleton-space upper-limb retargeting with USD joint limits.

No dance names, frame numbers, hand-tuned shoulder bases or pose gates. Source
segment directions are measured in the animated chest frame. Robot lengths
replace source lengths, and bounded IK fits elbow/wrist/palm together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length < 1e-9:
        raise ValueError("Degenerate skeleton landmark")
    return v / length


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def palm_frame(wrist, middle, index, pinky):
    """Columns: fingers, across palm, palm normal; same semantics on both rigs."""
    x = unit(middle - wrist)
    y = index - pinky
    y = unit(y - x * np.dot(x, y))
    return np.column_stack((x, y, np.cross(x, y)))


@dataclass(frozen=True)
class ArmTarget:
    elbow: np.ndarray
    wrist: np.ndarray
    palm: np.ndarray


@dataclass(frozen=True)
class SolverWeights:
    # Dimensionless residuals; one shared configuration for all clips.
    wrist: float = 4.0
    elbow: float = 2.0
    palm: float = 0.35
    continuity: float = 0.20


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    p0: np.ndarray
    r0: np.ndarray
    p1: np.ndarray
    r1: np.ndarray
    axis: np.ndarray | None
    lower: float
    upper: float
    velocity: float = np.inf


def read_joints(usd_path):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Cannot open robot: {usd_path}")
    def rotation(q):
        return Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parents or not children:
            continue
        axis = prim.GetAttribute("physics:axis").Get()
        lo, hi = prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()
        velocity = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
        joints.append(Joint(prim.GetName(), parents[0].name, children[0].name,
                            np.array(joint.GetLocalPos0Attr().Get()), rotation(joint.GetLocalRot0Attr().Get()),
                            np.array(joint.GetLocalPos1Attr().Get()), rotation(joint.GetLocalRot1Attr().Get()),
                            np.eye(3)["XYZ".index(axis)] if axis else None,
                            np.radians(lo) if lo is not None else -np.inf,
                            np.radians(hi) if hi is not None else np.inf,
                            np.radians(velocity) if velocity is not None else np.inf))
    return joints


class RobotArm:
    """Exact joint-anchor FK, in torso coordinates; radians are absolute angles."""
    def __init__(self, joints, side):
        prefix = "lh" if side == "left" else "rh"
        by_child = {j.child: j for j in joints}
        child = f"{prefix}_hand_base_link"
        self.chain = []
        visited = set()
        while child != "torso_link":
            if child in visited:
                raise ValueError("Cyclic robot arm hierarchy")
            visited.add(child)
            if child not in by_child:
                raise ValueError(f"Disconnected arm link: {child}")
            joint = by_child[child]
            self.chain.insert(0, joint)
            child = joint.parent
        self.moving = [j for j in self.chain if j.axis is not None]
        self.names = [j.name for j in self.moving]
        if len(self.names) != 7:
            raise ValueError(f"Expected seven arm joints, got {self.names}")
        self.lower = np.array([j.lower for j in self.moving])
        self.upper = np.array([j.upper for j in self.moving])
        self.velocity = np.array([j.velocity for j in self.moving])
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Robot arm must have finite, nonempty joint limits")
        self.elbow_name = f"{side}_elbow_joint"
        self.wrist_name = f"{side}_wrist_roll_joint"
        # Finger root joint anchors define the neutral semantic palm frame.
        finger = {j.name: j for j in joints}
        def anchor(name):
            j = finger[f"{prefix}_{name}_mcp_pitch"]
            if j.parent != f"{prefix}_hand_base_link":
                raise ValueError("Unexpected finger root hierarchy")
            return j.p0
        self.palm_local = palm_frame(np.zeros(3), anchor("middle"), anchor("index"), anchor("pinky"))
        neutral = self.fk(np.zeros(7))
        self.shoulder = neutral["origins"][0]
        self.upper_length = np.linalg.norm(neutral["elbow"] - self.shoulder)
        self.fore_length = np.linalg.norm(neutral["wrist"] - neutral["elbow"])
        self.reach = self.upper_length + self.fore_length

    def fk(self, q):
        r, p = np.eye(3), np.zeros(3)
        origins, axes = [], []
        elbow = wrist = None
        k = 0
        for joint in self.chain:
            origin = p + r @ joint.p0
            pre = r @ joint.r0
            if joint.name == self.elbow_name:
                elbow = origin.copy()
            if joint.name == self.wrist_name:
                wrist = origin.copy()
            if joint.axis is not None:
                origins.append(origin)
                axes.append(pre @ joint.axis)
                motion = Rotation.from_rotvec(joint.axis * q[k]).as_matrix()
                k += 1
            else:
                motion = np.eye(3)
            r = pre @ motion @ joint.r1.T
            p = origin - r @ joint.p1
        return dict(elbow=elbow, wrist=wrist, palm=r @ self.palm_local,
                    origins=np.asarray(origins), axes=np.asarray(axes))

    def residual_jacobian(self, q, target, previous, weights=SolverWeights()):
        state = self.fk(q)
        blocks, jac = [], []
        for key, count, weight in (("wrist", 4, weights.wrist), ("elbow", 3, weights.elbow)):
            blocks.append(weight * (state[key] - getattr(target, key)) / self.reach)
            j = np.zeros((3, 7))
            j[:, :count] = np.cross(state["axes"][:count], state[key] - state["origins"][:count]).T
            jac.append(weight * j / self.reach)
        blocks.append(weights.palm * (state["palm"] - target.palm).ravel() / np.sqrt(2))
        jac.append(weights.palm * np.column_stack([(skew(v) @ state["palm"]).ravel() for v in state["axes"]]) / np.sqrt(2))
        blocks.append(weights.continuity * (q - previous))
        jac.append(weights.continuity * np.eye(7))
        return np.concatenate(blocks), np.vstack(jac)

    def solve(self, target, previous, seed=None, weights=SolverWeights(), bounds=None):
        previous = np.clip(np.asarray(previous), self.lower, self.upper)
        lower, upper = (self.lower, self.upper) if bounds is None else bounds
        # A small interval guard avoids infeasible starts at exact speed limits.
        lower, upper = np.asarray(lower), np.asarray(upper)
        if np.any(upper - lower < 1e-10):
            raise ValueError("Empty arm IK bounds")
        def solve_from(start):
            return least_squares(lambda q: self.residual_jacobian(q, target, previous, weights)[0],
                                 np.clip(start, lower + 1e-12, upper - 1e-12),
                                 jac=lambda q: self.residual_jacobian(q, target, previous, weights)[1],
                                 bounds=(lower, upper), max_nfev=60,
                                 ftol=1e-8, xtol=1e-8, gtol=1e-8)
        best = solve_from(previous)
        # A second seed helps escape shoulder singularities; same rule for every pose.
        if seed is not None and best.cost > 0.002:
            other = solve_from(seed)
            if other.cost < best.cost:
                best = other
        if not np.isfinite(best.x).all():
            raise ValueError("Nonfinite arm solution")
        return best.x

    def solve_sequence(self, targets, seeds, fps, passes=2, weights=SolverWeights()):
        """Joint trajectory optimization followed by hard-bound temporal polish.

        The joint solve penalizes speed excess and acceleration, then establishes
        exact speed feasibility. Neighbour-bounded spatial refinements preserve
        that feasibility while improving the pose. No frame-specific rules.
        """
        from scipy import sparse
        if not np.isfinite(fps) or fps <= 0 or len(targets) != len(seeds) or not len(targets):
            raise ValueError("Invalid trajectory timing or targets")
        n = len(self.names)
        step = self.velocity / float(fps)
        # Fit all poses first, then optimize the entire trajectory jointly. A
        # causal speed-clamped solve can get trapped in a shoulder branch for
        # hundreds of frames; offline look-ahead allows the branch to unwind.
        spatial = SolverWeights(weights.wrist, weights.elbow, weights.palm, 0.)
        sequence = np.asarray([self.solve(t,q,seed=np.zeros(n),weights=spatial) for t,q in zip(targets,seeds)])
        count = len(sequence)
        if count > 1:
            d1 = sparse.kron(sparse.diags([-np.ones(count-1), np.ones(count-1)], [0,1], shape=(count-1,count)), sparse.eye(n), format='csr')
            d2 = (sparse.kron(sparse.diags([np.ones(count-2),-2*np.ones(count-2),np.ones(count-2)],[0,1,2],shape=(count-2,count)),sparse.eye(n),format='csr') if count>2 else sparse.csr_matrix((0,count*n)))
            velocity_weight = weights.continuity * fps/30.
            acceleration_weight = weights.continuity * (fps/30.)**2
            limit_scale = np.tile(step,count-1)
            cache_x = cache_r = cache_j = None
            def evaluate(flat):
                nonlocal cache_x, cache_r, cache_j
                if cache_x is not None and np.array_equal(cache_x,flat):
                    return cache_r,cache_j
                residuals, blocks = [], []
                for q,t in zip(flat.reshape(-1,n),targets):
                    r,j=self.residual_jacobian(q,t,q,spatial)
                    residuals.append(r[:-n]);blocks.append(j[:-n])
                diff=d1@flat
                violation=np.maximum(abs(diff)/limit_scale-1,0)
                cache_r=np.concatenate([*residuals,velocity_weight*diff,acceleration_weight*(d2@flat),50.*violation])
                limit_j=sparse.diags(50.*np.sign(diff)*(violation>0)/limit_scale)@d1
                cache_j=sparse.vstack([sparse.block_diag(blocks,format='csr'),velocity_weight*d1,acceleration_weight*d2,limit_j],format='csr')
                cache_x=flat.copy()
                return cache_r,cache_j
            optimized=least_squares(lambda x:evaluate(x)[0],sequence.ravel(),jac=lambda x:evaluate(x)[1],
                                    bounds=(np.tile(self.lower,count),np.tile(self.upper,count)),
                                    max_nfev=65,ftol=1e-6,xtol=1e-6,gtol=1e-6,
                                    tr_solver='lsmr',tr_options={'maxiter':80})
            sequence=optimized.x.reshape(-1,n)
            # Establish exact speed feasibility before the hard-bound polish.
            # This only removes the numerical/penalty tolerance, rather than
            # replacing trajectory optimization with a smoothing filter.
            for i in range(1,count):
                sequence[i]=np.clip(sequence[i],sequence[i-1]-step,sequence[i-1]+step)
        for _ in range(passes):
            for indices in (range(len(sequence)-1,-1,-1), range(len(sequence))):
                for i in indices:
                    neighbours = [sequence[j] for j in (i-1,i+1) if 0 <= j < len(sequence)]
                    if not neighbours:
                        continue
                    lower = np.maximum(self.lower, np.max(np.asarray(neighbours)-step, axis=0))
                    upper = np.minimum(self.upper, np.min(np.asarray(neighbours)+step, axis=0))
                    # Fixed neighbours can leave a zero-width interval. The current
                    # feasible pose is retained rather than relaxing a hard bound.
                    if np.any(upper-lower < 1e-10):
                        continue
                    sequence[i] = self.solve(targets[i], np.mean(neighbours,axis=0),
                                             seed=sequence[i], weights=weights, bounds=(lower,upper))
        return sequence


class SourceSkeleton:
    """Read a baked MMD UsdSkel; includes hierarchy and append rotations."""
    # Converted PMX: X=left, -Y=forward, Z=up. G1: X=forward,Y=left,Z=up.
    to_robot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

    def __init__(self, path):
        from pxr import Usd, UsdSkel
        self.stage = Usd.Stage.Open(str(Path(path)))
        if self.stage is None:
            raise ValueError(f"Cannot open source skeleton: {path}")
        self.skeleton = UsdSkel.Skeleton.Get(self.stage, "/Character/Skeleton")
        if not self.skeleton:
            raise ValueError("Expected converted PMX UsdSkel /Character/Skeleton")
        self.cache = UsdSkel.Cache()
        self.cache.Populate(UsdSkel.Root.Get(self.stage, "/Character"), Usd.PrimDefaultPredicate)
        self.query = self.cache.GetSkelQuery(self.skeleton)
        self.names = {unicodedata.normalize("NFKC", n): i for i, n in enumerate(self.skeleton.GetJointNamesAttr().Get())}
        self.chest = "上半身2" if "上半身2" in self.names else "上半身"
        self.fps = self.stage.GetTimeCodesPerSecond()
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid source animation FPS")
        self.first = self.stage.GetStartTimeCode()
        self.last = self.stage.GetEndTimeCode()
        self.rest = np.asarray(self.query.ComputeJointSkelTransforms(Usd.TimeCode.Default(), True)).transpose(0, 2, 1)
        self._palm_rest = {}
        for side, prefix in (("left", "左"), ("right", "右")):
            required = [self.chest, prefix + "腕", prefix + "ひじ", prefix + "手首",
                        prefix + "中指1", prefix + "人指1", prefix + "小指1"]
            missing = [n for n in required if n not in self.names]
            if missing:
                raise ValueError(f"Source skeleton lacks landmarks: {missing}")
            def pos(name): return self.rest[self.names[prefix + name], :3, 3]
            rest_frame = palm_frame(pos("手首"), pos("中指1"), pos("人指1"), pos("小指1"))
            self._palm_rest[side] = self.rest[self.names[prefix + "手首"], :3, :3].T @ rest_frame

    def frame(self, time_seconds):
        frame = self.first + time_seconds * self.fps
        if frame < self.first - 1e-5 or frame > self.last + 1e-5:
            raise ValueError("Source animation does not cover H5 timeline")
        return np.asarray(self.query.ComputeJointSkelTransforms(float(frame))).transpose(0, 2, 1)

    def target(self, matrices, side, arm):
        prefix = "左" if side == "left" else "右"
        chest = matrices[self.names[self.chest], :3, :3]
        convert = self.to_robot @ chest.T
        shoulder, elbow, wrist = [matrices[self.names[prefix + n], :3, 3] for n in ("腕", "ひじ", "手首")]
        u, v = convert @ unit(elbow - shoulder), convert @ unit(wrist - elbow)
        target_elbow = arm.shoulder + arm.upper_length * u
        target_wrist = target_elbow + arm.fore_length * v
        palm = convert @ matrices[self.names[prefix + "手首"], :3, :3] @ self._palm_rest[side]
        return ArmTarget(target_elbow, target_wrist, palm)


def target_errors(arm, q, target):
    state = arm.fk(q)
    def angle(a, b): return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1., 1.))))
    return dict(wrist_m=float(np.linalg.norm(state["wrist"] - target.wrist)),
                elbow_m=float(np.linalg.norm(state["elbow"] - target.elbow)),
                upper_deg=angle(state["elbow"] - arm.shoulder, target.elbow - arm.shoulder),
                fore_deg=angle(state["wrist"] - state["elbow"], target.wrist - target.elbow),
                palm_deg=float(np.degrees(Rotation.from_matrix(state["palm"].T @ target.palm).magnitude())))
