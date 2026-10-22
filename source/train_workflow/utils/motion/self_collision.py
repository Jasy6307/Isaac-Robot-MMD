"""Offline G1 reference-pose repair with USD-derived collision envelopes.

This is a conservative capsule/box approximation, not a mesh intersection
certificate. Root motion, waist and fingers stay unchanged. It does not run
PhysX or modify training settings.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
import torch


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    p0: np.ndarray
    r0: np.ndarray
    p1: np.ndarray
    r1: np.ndarray
    axis: str | None
    lower: float
    upper: float


@dataclass
class Shape:
    name: str
    kind: str
    a: np.ndarray  # capsule endpoint, or box center
    b: np.ndarray  # capsule endpoint, or box half extents
    radius: float = 0.0


def playback_baseline(names: list[str], cfg_path: Path) -> np.ndarray:
    """Read the same T-pose offsets used by replay without booting Isaac Sim."""
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    call = next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "G1_TPOSE_INIT_STATE" for t in node.targets)
    )
    defaults = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "joint_pos"))
    return np.array([
        next((v for expression, v in defaults.items() if re.fullmatch(expression, name)), 0.0)
        for name in names
    ], dtype=np.float32)


def _quat_matrix(quat) -> np.ndarray:
    return Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()


def _fit_capsule(name: str, points: np.ndarray) -> Shape:
    center = (points.min(0) + points.max(0)) / 2
    _, axes = np.linalg.eigh(np.cov(points.T))
    axis = axes[:, -1]
    projection = (points - center) @ axis
    radial = points - center - projection[:, None] * axis
    radius = float(np.linalg.norm(radial, axis=1).max())
    lo, hi = float(projection.min()), float(projection.max())
    if hi - lo > 2 * radius:
        lo, hi = lo + radius, hi - radius
    else:
        lo = hi = (lo + hi) / 2
    closest = center + np.clip(projection, lo, hi)[:, None] * axis
    radius = float(np.linalg.norm(points - closest, axis=1).max())
    return Shape(name, "capsule", center + lo * axis, center + hi * axis, radius)


class G1CollisionModel:
    def __init__(self, usd_path: Path, names: list[str], device: str):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Cannot read robot USD: {usd_path}")
        self.names = list(names)
        self.index = {name: i for i, name in enumerate(names)}
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._tensor = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device)
        self.joints: list[Joint] = []
        body_paths = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parent_paths, child_paths = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parent_paths or not child_paths:
                continue
            parent, child = parent_paths[0].name, child_paths[0].name
            body_paths[parent], body_paths[child] = parent_paths[0], child_paths[0]
            axis = prim.GetAttribute("physics:axis").Get()
            lower = prim.GetAttribute("physics:lowerLimit").Get()
            upper = prim.GetAttribute("physics:upperLimit").Get()
            self.joints.append(Joint(
                prim.GetName(), parent, child,
                np.array(joint.GetLocalPos0Attr().Get()), _quat_matrix(joint.GetLocalRot0Attr().Get()),
                np.array(joint.GetLocalPos1Attr().Get()), _quat_matrix(joint.GetLocalRot1Attr().Get()),
                axis, math.radians(lower) if lower is not None else -math.inf,
                math.radians(upper) if upper is not None else math.inf,
            ))

        self.arm_names = {
            side: [f"{side}_{part}_link" for part in (
                "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"
            )] + ["lh_hand_base_link" if side == "left" else "rh_hand_base_link"]
            for side in ("left", "right")
        }
        self.leg_names = {
            side: [f"{side}_{part}_link" for part in ("hip_yaw", "knee", "ankle_roll")]
            for side in ("left", "right")
        }
        central = ["torso_link", "head_link", "pelvis_contour_link"]
        wanted = central + sum(self.arm_names.values(), []) + sum(self.leg_names.values(), [])
        cache = UsdGeom.XformCache()

        def vertices(body: str, target_body: str) -> np.ndarray:
            pieces = []
            target_inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_paths[target_body]))).T)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(body_paths[body]), Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
                    continue
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                transform = target_inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
                pieces.append(points @ transform[:3, :3].T + transform[:3, 3])
            return np.concatenate(pieces) if pieces else np.empty((0, 3))

        self.shapes: list[Shape] = []
        for name in wanted:
            points = vertices(name, name)
            if name in ("lh_hand_base_link", "rh_hand_base_link"):
                prefix = "lh_" if name.startswith("lh_") else "rh_"
                # Include open fingers in a single conservative hand envelope.
                extra = [vertices(body, name) for body in body_paths if body.startswith(prefix) and body != name]
                points = np.concatenate([points, *extra])
            if not len(points):
                raise ValueError(f"Missing visual geometry for {name}")
            if name in ("torso_link", "pelvis_contour_link", "head_link", "lh_hand_base_link", "rh_hand_base_link"):
                lo, hi = points.min(0), points.max(0)
                shape = Shape(name, "box", (lo + hi) / 2, (hi - lo) / 2)
            else:
                shape = _fit_capsule(name, points)
            self.shapes.append(shape)

        # Only evaluate kinematic ancestors of the collision/foot bodies.
        by_child = {joint.child: joint for joint in self.joints}
        needed = set(wanted)
        for name in list(needed):
            while name in by_child:
                name = by_child[name].parent
                needed.add(name)
        remaining = [joint for joint in self.joints if joint.child in needed]
        ordered = []
        visited = {"pelvis"}
        while remaining:
            ready = [joint for joint in remaining if joint.parent in visited]
            if not ready:
                raise ValueError("Robot kinematic tree is disconnected")
            for joint in ready:
                ordered.append(joint)
                visited.add(joint.child)
                remaining.remove(joint)
        self.joints = ordered
        self.constants = {
            joint.name: tuple(self._tensor(v) for v in (joint.p0, joint.r0, joint.p1, joint.r1))
            for joint in self.joints
        }
        lower, upper = np.full(len(names), -math.inf), np.full(len(names), math.inf)
        for joint in self.joints:
            if joint.axis and joint.name in self.index:
                i = self.index[joint.name]
                lower[i], upper[i] = joint.lower, joint.upper
        self.lower, self.upper = self._tensor(lower), self._tensor(upper)
        self.adjustable = [i for i, name in enumerate(names) if name.startswith(("left_", "right_"))]
        self.shape_constants = {s.name: (self._tensor(s.a), self._tensor(s.b)) for s in self.shapes}
        indices = {s.name: i for i, s in enumerate(self.shapes)}
        pairs = []
        for arms in self.arm_names.values():
            pairs.extend((a, b) for a in arms for b in central)
        pairs.extend((a, b) for a in self.arm_names["left"] for b in self.arm_names["right"])
        pairs.extend((a, b) for a in sum(self.arm_names.values(), []) for b in sum(self.leg_names.values(), []))
        pairs.extend((a, b) for a in self.leg_names["left"] for b in self.leg_names["right"])
        self.pairs = [(indices[a], indices[b]) for a, b in pairs]
        self.pair_labels = [f"{a} / {b}" for a, b in pairs]
        capsule_pairs, box_pairs, box_box_pairs = [], [], []
        for i, (a, b) in enumerate(self.pairs):
            if self.shapes[a].kind == self.shapes[b].kind == "capsule":
                capsule_pairs.append((i, a, b))
            elif self.shapes[a].kind == self.shapes[b].kind == "box":
                box_box_pairs.append((i, a, b))
            else:
                cap, box = (a, b) if self.shapes[a].kind == "capsule" else (b, a)
                box_pairs.append((i, cap, box))
        self.capsule_pairs = torch.tensor(capsule_pairs, device=self.device)
        self.box_pairs = torch.tensor(box_pairs, device=self.device)
        self.box_box_pairs = torch.tensor(box_box_pairs, device=self.device)
        self.local_a = self._tensor(np.stack([s.a for s in self.shapes]))
        self.local_b = self._tensor(np.stack([s.b for s in self.shapes]))
        self.radii = self._tensor([s.radius for s in self.shapes])
        lengths = np.array([np.linalg.norm(s.b - s.a) for s in self.shapes])
        self.sample_radii = self._tensor([math.sqrt(s.radius**2 + (lengths[i] / 32)**2) for i, s in enumerate(self.shapes)])
        self.sample_t = torch.linspace(0, 1, 17, device=self.device)

        # Validate FK against the USD's neutral link transforms, independently
        # of the H5 pose offsets. USD matrices use the transpose convention.
        with torch.no_grad():
            rotations, positions = self.fk(torch.zeros(1, len(names), device=self.device))
            root_inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_paths["pelvis"]))).T)
            error = 0.0
            for name in positions:
                expected = root_inverse @ np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_paths[name]))).T
                error = max(error, float(np.max(np.abs(positions[name][0].cpu().numpy() - expected[:3, 3]))), float(np.max(np.abs(rotations[name][0].cpu().numpy() - expected[:3, :3]))))
        self.fk_validation_error = error
        # Authored link poses are rounded separately from joint anchors in this
        # USD (about 0.2 mm / 0.01 degree discrepancy at the extremities).
        if error > 1e-3:
            raise ValueError(f"USD neutral FK mismatch: {error:g}")

    def fk(self, q: torch.Tensor):
        count = len(q)
        eye = torch.eye(3, device=q.device).expand(count, -1, -1)
        rotations, positions = {"pelvis": eye}, {"pelvis": torch.zeros(count, 3, device=q.device)}
        for joint in self.joints:
            p0, r0, p1, r1 = self.constants[joint.name]
            parent_r, parent_p = rotations[joint.parent], positions[joint.parent]
            if joint.axis:
                if joint.name not in self.index:
                    raise ValueError(f"H5 missing robot joint: {joint.name}")
                theta = q[:, self.index[joint.name]]
                c, s = torch.cos(theta), torch.sin(theta)
                z, one = torch.zeros_like(theta), torch.ones_like(theta)
                if joint.axis == "X":
                    values = (one, z, z, z, c, -s, z, s, c)
                elif joint.axis == "Y":
                    values = (c, z, s, z, one, z, -s, z, c)
                else:
                    values = (c, -s, z, s, c, z, z, z, one)
                motion = torch.stack(values, dim=-1).reshape(count, 3, 3)
                child_r = parent_r @ r0 @ motion @ r1.T
            else:
                child_r = parent_r @ r0 @ r1.T
            child_p = parent_p + (parent_r @ p0[..., None]).squeeze(-1) - (child_r @ p1[..., None]).squeeze(-1)
            rotations[joint.child], positions[joint.child] = child_r, child_p
        return rotations, positions

    @staticmethod
    def _segment_distance(a, b, c, d):
        u, v, w = b - a, d - c, a - c
        uu, vv = (u * u).sum(-1).clamp_min(1e-12), (v * v).sum(-1).clamp_min(1e-12)
        uv, uw, vw = (u * v).sum(-1), (u * w).sum(-1), (v * w).sum(-1)
        denom = uu * vv - uv * uv
        s = torch.where(denom > 1e-12, (uv * vw - uw * vv) / denom.clamp_min(1e-12), torch.zeros_like(denom)).clamp(0, 1)
        t = (uv * s + vw) / vv
        s = torch.where(t < 0, (-uw / uu).clamp(0, 1), s)
        s = torch.where(t > 1, ((uv - uw) / uu).clamp(0, 1), s)
        t = t.clamp(0, 1)
        return torch.linalg.vector_norm(w + s[..., None] * u - t[..., None] * v, dim=-1)

    def distances(self, rotations, positions):
        r = torch.stack([rotations[s.name] for s in self.shapes], dim=1)
        p = torch.stack([positions[s.name] for s in self.shapes], dim=1)
        start = (r @ self.local_a[..., None]).squeeze(-1) + p
        end = (r @ self.local_b[..., None]).squeeze(-1) + p
        cp, bp, bb = self.capsule_pairs, self.box_pairs, self.box_box_pairs
        cap_dist = self._segment_distance(start[:, cp[:, 1]], end[:, cp[:, 1]], start[:, cp[:, 2]], end[:, cp[:, 2]]) - self.radii[cp[:, 1]] - self.radii[cp[:, 2]]
        a, b = start[:, bp[:, 1]], end[:, bp[:, 1]]
        points = a[:, :, None] + self.sample_t[None, None, :, None] * (b - a)[:, :, None]
        local = (points - p[:, bp[:, 2], None]) @ r[:, bp[:, 2]]
        delta = (local - self.local_a[bp[:, 2]][None, :, None]).abs() - self.local_b[bp[:, 2]][None, :, None]
        signed = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) + delta.amax(-1).clamp_max(0)
        box_dist = signed.amin(-1) - self.sample_radii[bp[:, 1]]
        ra, rb = r[:, bb[:, 1]], r[:, bb[:, 2]]
        ca = p[:, bb[:, 1]] + (ra @ self.local_a[bb[:, 1], :, None]).squeeze(-1)
        cb = p[:, bb[:, 2]] + (rb @ self.local_a[bb[:, 2], :, None]).squeeze(-1)
        axes_a, axes_b = ra.transpose(-1, -2), rb.transpose(-1, -2)
        cross = torch.linalg.cross(axes_a[:, :, :, None], axes_b[:, :, None, :], dim=-1).flatten(2, 3)
        axes = torch.cat((axes_a, axes_b, cross), dim=2)
        lengths = torch.linalg.vector_norm(axes, dim=-1)
        axes = axes / lengths.clamp_min(1e-8)[..., None]
        center_distance = ((cb - ca)[:, :, None] * axes).sum(-1).abs()
        a_extent = ((axes @ ra).abs() * self.local_b[bb[:, 1]][None, :, None]).sum(-1)
        b_extent = ((axes @ rb).abs() * self.local_b[bb[:, 2]][None, :, None]).sum(-1)
        separating_gap = center_distance - a_extent - b_extent
        box_box_dist = separating_gap.masked_fill(lengths < 1e-6, -torch.inf).amax(-1)
        # Restore the original pair order for diagnostics.
        order = torch.argsort(torch.cat((cp[:, 0], bp[:, 0], bb[:, 0])))
        return torch.cat((cap_dist, box_dist, box_box_dist), dim=1)[:, order]

    def report(self, q: torch.Tensor, margin: float) -> dict:
        with torch.no_grad():
            distances = self.distances(*self.fk(q))
            worst = distances.amin(-1)
            count = (distances < 0).sum(0)
            order = torch.argsort(count, descending=True)[:8].tolist()
            frames = torch.argsort(worst)[:8].tolist()
            return {
                "penetrating_frames": int((worst < -0.001).sum()),
                "margin_violation_frames": int((worst < margin).sum()),
                "worst_penetration_m": max(0.0, -float(worst.min())),
                "minimum_clearance_m": float(worst.min()),
                "worst_frames": frames,
                "pairs": [{"pair": self.pair_labels[i], "penetrating_frames": int(count[i]), "minimum_clearance_m": float(distances[:, i].min())} for i in order],
            }


def repair_poses(model: G1CollisionModel, original: np.ndarray, *, iterations: int = 600, margin: float = 0.004, initial_pose: np.ndarray | None = None, fixed_endpoint_frames: int = 0, preserve_motion: bool = False, collision_strength: float | None = None, learning_rate: float | None = None, progress=None):
    reference = model._tensor(original)
    adjustable = model.adjustable
    start = reference if initial_pose is None else model._tensor(initial_pose)
    if start.shape != reference.shape or not torch.isfinite(start).all():
        raise ValueError("Initial pose must match the finite reference trajectory")
    if fixed_endpoint_frames < 0 or 2 * fixed_endpoint_frames >= len(reference):
        raise ValueError("Fixed endpoints must leave interior frames to repair")
    initial = torch.maximum(torch.minimum(start, model.upper), model.lower)
    variables = torch.nn.Parameter(initial[:, adjustable].clone())
    if learning_rate is None:
        learning_rate = 0.018 if initial_pose is None else 0.003
    optimizer = torch.optim.Adam([variables], lr=learning_rate)
    with torch.no_grad():
        ref_r, ref_p = model.fk(reference)
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_p = torch.stack([ref_p[n] for n in foot_names], dim=1)
        foot_r = torch.stack([ref_r[n] for n in foot_names], dim=1)
    weights = model._tensor([6.0 if "hip" in model.names[i] or "knee" in model.names[i] or "ankle" in model.names[i] else 1.0 for i in adjustable])
    before = model.report(reference, margin)
    print(f"[REPAIR] Before: {before}", flush=True)
    history = []
    for step in range(iterations):
        # Reduce jitter near the clearance boundary as the optimizer converges.
        schedule = step / max(1, iterations - 1)
        optimizer.param_groups[0]["lr"] = learning_rate * (0.02 + 0.98 * (1 + math.cos(math.pi * schedule)) / 2)
        optimizer.zero_grad(set_to_none=True)
        q = reference.clone()
        q[:, adjustable] = variables
        rotations, positions = model.fk(q)
        distances = model.distances(rotations, positions)
        penetration = torch.relu(margin - distances)
        midpoint_distances = model.distances(*model.fk((q[:-1] + q[1:]) / 2))
        midpoint_penetration = torch.relu(margin - midpoint_distances)
        collision_weight = 250000.0 if initial_pose is not None or step >= iterations * 0.75 else 50000.0
        if collision_strength is not None:
            collision_weight = collision_strength
        ref_loss = (((variables - reference[:, adjustable]) ** 2) * weights).sum(-1).mean()
        correction = variables - reference[:, adjustable]
        temporal = (correction[1:] - correction[:-1]).square().sum(-1).mean()
        trajectory = correction if preserve_motion else variables
        acceleration = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]).square().sum(-1).mean()
        speed = torch.zeros((), device=variables.device) if preserve_motion else torch.relu((variables[1:] - variables[:-1]).abs() - 0.22).square().sum(-1).mean()
        feet = torch.stack([positions[n] for n in foot_names], dim=1)
        feet_rot = torch.stack([rotations[n] for n in foot_names], dim=1)
        foot_loss = (feet - foot_p).square().sum(-1).mean()
        foot_rot_loss = (feet_rot - foot_r).square().sum((-1, -2)).mean()
        collision_loss = penetration.square().sum(-1).mean() + midpoint_penetration.square().sum(-1).mean()
        loss = ref_loss + 4 * temporal + 0.3 * acceleration + 25 * speed + 2000 * foot_loss + 10 * foot_rot_loss + collision_weight * collision_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variables.copy_(torch.maximum(torch.minimum(variables, model.upper[adjustable]), model.lower[adjustable]))
            if fixed_endpoint_frames:
                variables[:fixed_endpoint_frames] = initial[:fixed_endpoint_frames, adjustable]
                variables[-fixed_endpoint_frames:] = initial[-fixed_endpoint_frames:, adjustable]
        if step % 50 == 0 or step == iterations - 1:
            entry = {"iteration": step, "loss": float(loss.detach()), "collision_loss": float(collision_loss.detach()), "foot_rms_m": float(foot_loss.detach().sqrt()), "worst_penetration_m": max(0.0, -float(distances.detach().min()))}
            history.append(entry)
            print(f"[REPAIR] {entry}", flush=True)
            if progress is not None:
                progress((step + 1) / iterations, entry)
    with torch.no_grad():
        result = reference.clone()
        result[:, adjustable] = variables
        after = model.report(result, margin)
        midpoints = (result[:-1] + result[1:]) / 2
        midpoint_report = model.report(midpoints, margin)
        rotations, positions = model.fk(result)
        feet = torch.stack([positions[n] for n in foot_names], dim=1)
        foot_displacement = torch.linalg.vector_norm(feet - foot_p, dim=-1)
        delta = result - reference
        report = {
            "model": "USD visual geometry conservative capsules/boxes; open-hand envelope; selected non-adjacent body pairs",
            "margin_m": margin,
            "fk_validation_error": model.fk_validation_error,
            "before": before, "after": after, "midpoints_after": midpoint_report,
            "joint_rms_change_deg": float(delta.square().mean().sqrt() * 180 / math.pi),
            "joint_max_change_deg": float(delta.abs().max() * 180 / math.pi),
            "foot_displacement_rms_m": float(foot_displacement.square().mean().sqrt()),
            "foot_displacement_max_m": float(foot_displacement.max()),
            "max_step_before_deg": float((reference[1:] - reference[:-1]).abs().max() * 180 / math.pi),
            "max_step_after_deg": float((result[1:] - result[:-1]).abs().max() * 180 / math.pi),
            "history": history,
        }
    return result.cpu().numpy(), report
