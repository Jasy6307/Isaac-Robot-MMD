"""Fixed-root Cartesian SQP, measured sole bounds and independent publication gate."""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from source.train_workflow.utils.motion.self_collision import G1CollisionModel
from source.train_workflow.utils.motion.joint_constraints import read_constraints, validate_joint_positions, constrain_joint_positions
from source.train_workflow.utils.motion.constrained_trajectory import refine_feasible_trajectory
from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg
from .geometry import apply_batch_geometry

FRACTIONS = (.25, .5, .75)
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
LINKS = ('lh_hand_base_link', 'rh_hand_base_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
         'left_elbow_link', 'right_elbow_link', 'left_hip_roll_link', 'right_hip_roll_link',
         'left_knee_link', 'right_knee_link', *FEET)


def sampled_values(q):
    return np.concatenate([q, *[q[:-1]*(1-f)+q[1:]*f for f in FRACTIONS]])


def sample_brackets(count):
    coordinates = sampled_values(np.arange(count, dtype=float))
    return coordinates, np.floor(coordinates).astype(int), np.ceil(coordinates).astype(int)


def height_bounds(reference_heights):
    lower = np.where(reference_heights > .02, reference_heights-.01, np.minimum(reference_heights, -.001))-2e-6
    upper = reference_heights+np.where(reference_heights < .02, .008, .025)
    return lower, upper


@dataclass(frozen=True)
class SmoothingSettings:
    window_frames: int = 60
    stride: int = 30
    sweeps: int = 2
    iterations: int = 10
    acceleration_weight: float = .002
    jerk_weight: float = .000003
    restoration_quadratic_weight: float = 2e8


class GroundProblem:
    def __init__(self, reference, robot_usd, *, device='cpu', geometry='batch',
                 sole_vertices=None, desired=None, active_derivatives=False):
        if geometry not in ('girls', 'batch', 'batch-hips', 'batch-hips-palm'):
            raise ValueError('Unknown geometry profile')
        self.reference, self.robot_usd = reference, robot_usd
        self.geometry = geometry
        self.model = G1CollisionModel(robot_usd, reference.names, device, articulated_hands=True)
        if geometry != 'girls':
            apply_batch_geometry(self.model, robot_usd, profile=geometry)
        self.columns = np.asarray(self.model.adjustable)
        self.count = len(reference.delta)
        self.coordinates, self.lo, self.hi = sample_brackets(self.count)
        alpha = self.coordinates-self.lo
        self.world_rot = Slerp(np.arange(self.count), Rotation.from_matrix(reference.root_rot))(self.coordinates).as_matrix()
        self.world_pos = reference.root_pos[self.lo]*(1-alpha[:, None])+reference.root_pos[self.hi]*alpha[:, None]
        self.active_derivatives = active_derivatives
        self.soles = (np.stack([RobotLeg(robot_usd, side).sole for side in ('left', 'right')])
                      if sole_vertices is None else tuple(sole_vertices))
        joints, mapping, _, constraints = read_constraints(robot_usd, reference.names)
        self.lower = np.array([joints[n].lower for n in reference.names])
        self.upper = np.array([joints[n].upper for n in reference.names])
        self.step = np.array([min(joints[n].velocity, abs(mapping[n][1])*constraints[mapping[n][0]][2])/reference.fps for n in reference.names])
        with torch.no_grad():
            rotation, position = self.model.fk(self.model._tensor(reference.q))
            self.target_pos = {n: position[n].cpu().numpy() for n in LINKS}
            self.target_rot = {n: rotation[n].cpu().numpy() for n in (*LINKS[:2], *FEET)}
        self.reference_heights, _ = self.sample_geometry(reference.q, distances=False)
        self.height_lower, self.height_upper = height_bounds(self.reference_heights)
        if desired is not None:
            # Source-classified stance/flight applies in the interior of each
            # state. Transition samples keep their continuous IK heights.
            contact = (desired[self.lo] == 0) & (desired[self.hi] == 0)
            flight = (desired > .025).all(axis=1)
            flight = flight[self.lo] & flight[self.hi]
            precise = (self.reference_heights > .15) | flight[:, None]
            self.height_lower = np.maximum(self.reference_heights-.012, np.minimum(self.reference_heights, -.0003))
            self.height_upper = self.reference_heights+.010
            tolerance = np.where(flight[:, None], .002, .003)
            self.height_lower = np.where(precise, self.reference_heights-tolerance, self.height_lower)
            self.height_upper = np.where(precise, self.reference_heights+tolerance, self.height_upper)
            self.height_lower[contact], self.height_upper[contact] = -.0003, .003
        self.exact_contact = desired is not None

    def floor_points(self, rotations, positions, indices):
        vertical = self.world_rot[indices, 2, :]
        z = self.world_pos[indices, 2]
        points = []
        for link, mesh in zip(FEET, self.soles):
            rotation, position = rotations[link].cpu().numpy(), positions[link].cpu().numpy()
            up = np.einsum('ni,bnij->bnj', vertical, rotation)
            height = np.einsum('ni,bni->bn', vertical, position) + z
            points.append(up @ mesh.T + height[..., None])
        return points

    def sample_geometry(self, q, *, distances=True):
        if np.shape(q) != self.reference.delta.shape or not np.isfinite(q).all():
            raise ValueError('Invalid joint trajectory')
        poses = sampled_values(q)
        heights, clearance = [], []
        with torch.no_grad():
            for start in range(0, len(poses), 128):
                end = min(len(poses), start+128)
                r, p = self.model.fk(self.model._tensor(poses[start:end]))
                z = self.floor_points({n:r[n][None] for n in FEET}, {n:p[n][None] for n in FEET},
                                      np.arange(start, end))
                heights.append(np.stack([side[0].min(-1) for side in z], axis=-1))
                if distances:
                    clearance.append(self.model.distances(r, p).min(-1).values.cpu().numpy())
        return np.concatenate(heights), np.concatenate(clearance) if distances else None

    def audit(self, q, *, root_channels_unchanged, desired=None, require_arm_improvement=True):
        """Check encoded H5 q; root equality is established by the caller's raw channels."""
        heights, clearance = self.sample_geometry(q)
        excess = np.maximum(self.height_lower-heights, heights-self.height_upper)
        bad_height = np.flatnonzero((excess > 6e-6).any(1))
        bad_proxy = np.flatnonzero(clearance < .0005-1e-7)
        limits = validate_joint_positions(q, self.reference.names, self.reference.fps, self.robot_usd)
        stance = (self.reference_heights[:self.count] < .015).any(1)
        flight = (self.reference_heights[:self.count] > .025).all(1)
        false_float = int(((heights[:self.count] > .025).all(1) & stance).sum())
        retained = int(((heights[:self.count] > .015).all(1) & flight).sum())
        report = dict(frames=self.count, fps=self.reference.fps, geometry_profile=self.geometry,
            minimum_proxy_clearance_m=float(clearance.min()), quarter_frame_samples=len(clearance),
            joint_constraints=limits, root_channels_unchanged=bool(root_channels_unchanged),
            foot_height_bound_bad_samples=len(bad_height),
            foot_height_bound_bad_frames=np.unique(np.r_[self.lo[bad_height], self.hi[bad_height]]).tolist(),
            proxy_bad_frames=np.unique(np.r_[self.lo[bad_proxy], self.hi[bad_proxy]]).tolist(),
            maximum_height_bound_violation_m=float(max(0, excess.max())),
            both_floating_on_reference_stance_frames=false_float,
            true_reference_flight_frames=int(flight.sum()), flight_frames_retained=retained)
        report['passed'] = bool(limits['passed'] and root_channels_unchanged and not len(bad_height) and
                                not len(bad_proxy) and not false_float and retained == int(flight.sum()))
        if desired is not None:
            from .ik import source_contact_report
            contact = source_contact_report(heights[:self.count], desired)
            report['source_contact'] = contact
            report['passed'] &= contact['false_floating_on_source_stance'] == 0 and contact['source_flight_frames'] == contact['source_flight_retained']
        temporal = {}
        for label, keys in (('arms', ('shoulder', 'elbow', 'wrist')), ('legs', ('hip_', 'knee', 'ankle'))):
            cols = [i for i, n in enumerate(self.reference.names) if any(k in n for k in keys)]
            def jerk(x):
                return float(np.sqrt(np.mean((np.diff(x[:, cols], n=3, axis=0)*self.reference.fps**3)**2))) if self.count > 3 else 0.
            before, after = jerk(self.reference.q), jerk(q)
            temporal[label] = dict(reference_jerk_rms=before, corrected_jerk_rms=after,
                                   reduction=1-after/before if before else 0.)
        report['temporal'] = temporal
        if require_arm_improvement:
            report['passed'] = report['passed'] and temporal['arms']['reduction'] > 0
        return report

    def sampler(self, lo, hi):
        model, count = self.model, self.count
        indices = np.concatenate([np.arange(lo, hi), *[count+i*(count-1)+np.arange(lo, hi-1) for i in range(3)]])
        weights = np.sqrt([8000, 8000, 4000, 4000, 1800, 1800, 200, 200, 2000, 2000, 12000, 12000])
        pair_count = self.pair_count
        def batch_sample(qb, active=None):
            batch, n, channels = qb.shape
            stride = 4*n-3
            samples = np.concatenate([qb, *[qb[:, :-1]*(1-f)+qb[:, 1:]*f for f in FRACTIONS]], 1)
            with torch.no_grad():
                sr, sp = model.fk(model._tensor(samples.reshape(-1, channels)))
                r = {name:sr[name].reshape(batch, stride, 3, 3) for name in self.target_rot}
                p = {name:sp[name].reshape(batch, stride, 3) for name in LINKS}
                position = np.concatenate([(p[name][:,:n].cpu().numpy()-self.target_pos[name][None, lo:hi])*weights[i] for i, name in enumerate(LINKS)], 2)
                orientation = np.concatenate([(r[name][:,:n].cpu().numpy()-self.target_rot[name][None, lo:hi]).reshape(batch, n, -1)*np.sqrt(20 if name in FEET else 60) for name in self.target_rot], 2)
                pair = ((p[LINKS[0]][:,:n]-p[LINKS[1]][:,:n]).cpu().numpy()-(self.target_pos[LINKS[0]][None, lo:hi]-self.target_pos[LINKS[1]][None, lo:hi]))*np.sqrt(8000)
                residual = np.concatenate([position, orientation, pair, (qb-self.reference.q[None, lo:hi])[:, :, self.columns]*2], 2)
                hands = np.stack([p[name][:,:n].cpu().numpy() for name in LINKS[:2]], 2)
                d = model.distances(sr, sp) if active is None else model.distances(sr, sp, active[active < pair_count])
                distances = d.cpu().numpy().reshape(batch, stride, -1)
                floor = self.floor_points(r, p, indices)
                if self.exact_contact:
                    floor = [z.min(-1, keepdims=True) for z in floor]
                lower = np.concatenate([z-self.height_lower[None, indices, i, None]+.0005
                                        for i,z in enumerate(floor)], -1)
                upper = self.height_upper[None, indices]-np.stack([z.min(-1) for z in floor], -1)+.0005
                if active is not None:
                    foot = np.concatenate([lower, upper], -1)[:, :, active[active >= pair_count]-pair_count]
                    return residual, hands, np.concatenate([distances, foot], -1)
                return residual.astype(float), hands.astype(float), np.concatenate([distances, lower, upper], -1).astype(float)
        def sample(q):
            return tuple(x[0] for x in batch_sample(q[None]))
        if self.active_derivatives:
            foot_columns = 4 if self.exact_contact else 2+sum(len(mesh) for mesh in self.soles)
            sample.active_jacobian = batched_central_difference(batch_sample, pair_count+foot_columns, chunk_size=5)
        else:
            sample.jacobian = batched_central_difference(batch_sample, chunk_size=5)
        return sample

    @property
    def pair_count(self):
        return len(self.model.pairs) + getattr(self.model, 'extra_distance_count', 0)

    def smooth(self, seed, settings=SmoothingSettings(), *, start=0, end=None, progress=None):
        current = np.asarray(seed).copy()
        if current.shape != self.reference.delta.shape:
            raise ValueError('Seed shape differs from reference')
        end = self.count if end is None else end
        if not 0 <= start < end <= self.count or end-start < 4:
            raise ValueError('Invalid smoothing range')
        if settings.window_frames < 4 or settings.stride < 1 or settings.stride > settings.window_frames-4:
            raise ValueError('Windows must overlap by at least four frames')
        if settings.sweeps < 1 or settings.iterations < 1:
            raise ValueError('Positive sweeps and iterations required')
        weights = (settings.acceleration_weight, settings.jerk_weight, settings.restoration_quadratic_weight)
        if not np.isfinite(weights).all() or min(weights) < 0:
            raise ValueError('Smoothing weights must be finite and nonnegative')
        length = min(end-start, settings.window_frames)
        starts = list(range(start, end-length+1, settings.stride))
        if starts[-1] != end-length:
            starts.append(end-length)
        fps = self.reference.fps
        accel = np.maximum(60., np.abs(np.diff(self.reference.q, n=2, axis=0)).max(0)*fps**2+.01)
        accel = np.maximum(accel, np.abs(np.diff(current, n=2, axis=0)).max(0)*fps**2+.01)
        reports = []
        for sweep in range(settings.sweeps):
            for lo in starts if sweep % 2 == 0 else list(reversed(starts)):
                hi = lo+length
                def feasible(q):
                    full = current.copy(); full[lo:hi] = q
                    return (np.isfinite(full).all() and not np.any(full < self.lower-1e-6) and
                            not np.any(full > self.upper+1e-6) and
                            not np.any(abs(np.diff(full, axis=0)) > self.step+1e-5) and
                            not np.any(abs(np.diff(full, n=2, axis=0))*fps**2 > accel+.02))
                result, report = refine_feasible_trajectory(current[lo:hi], self.columns,
                    self.lower[self.columns], self.upper[self.columns], self.step[self.columns],
                    accel[self.columns]/fps**2, fps, None, None, self.sampler(lo, hi), feasible,
                    iterations=settings.iterations, fixed_endpoints=(4 if lo else 0, 4 if hi < self.count else 0),
                    collision_fractions=FRACTIONS, restore_collisions=True,
                    temporal_weights=(0., settings.acceleration_weight, settings.jerk_weight),
                    restoration_quadratic_weight=settings.restoration_quadratic_weight)
                current[lo:hi] = result
                report.update(frames=[lo, hi], sweep=sweep)
                reports.append(report)
                if progress:
                    progress(current, report)
        return current, reports

    def escape_bias(self, seed, center, *, random_seed=12017, sweeps=3):
        """Deterministic local bias seed search; always staging, never acceptance."""
        q = np.asarray(seed).copy()
        if not 0 <= center < self.count:
            raise ValueError('Invalid escape center')
        lo, hi = max(0, center-48), min(self.count, center+49)
        current = q[lo:hi].copy()
        cols = [i for i, n in enumerate(self.reference.names) if any(k in n for k in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow'))]
        bump = np.exp(-((np.arange(hi-lo)-(center-lo))/10)**2)
        bump[:4] = 0; bump[-4:] = 0
        def score(x):
            with torch.no_grad():
                d = self.model.distances(*self.model.fk(self.model._tensor(sampled_values(x))))
                return float(torch.maximum(.0005-d, torch.zeros_like(d)).square().sum().cpu())
        best = score(current)
        rng = np.random.default_rng(random_seed)
        for _ in range(sweeps):
            if best == 0:
                break
            candidates = []
            for i in range(len(cols)):
                for v in (-.8, -.5, -.3, -.15, .15, .3, .5, .8):
                    bias = np.zeros(len(cols)); bias[i] = v; candidates.append(bias)
            candidates.extend(rng.normal(0, .25, size=(80, len(cols))))
            winner = None
            for bias in candidates:
                trial = current.copy(); trial[:, cols] += bump[:, None]*bias
                trial, _ = constrain_joint_positions(trial, self.reference.names, self.reference.fps, self.robot_usd)
                value = score(trial)
                if value < best-1e-10:
                    best, winner = value, trial.copy()
                    if best == 0:
                        break
            if winner is None:
                break
            current = winner
        q[lo:hi] = current
        q, _ = constrain_joint_positions(q, self.reference.names, self.reference.fps, self.robot_usd)
        return q


def batched_central_difference(batch_sample, rows=None, *, chunk_size):
    """Bound the allocation for finite differences, including on busy GPUs."""
    def jacobian(q, columns, epsilon, active=None):
        chunks = []
        for subset in np.array_split(columns, (len(columns)+chunk_size-1)//chunk_size):
            variants = np.repeat(q[None], len(subset)*2, axis=0)
            for i, col in enumerate(subset):
                variants[2*i, :, col] += epsilon
                variants[2*i+1, :, col] -= epsilon
            values = batch_sample(variants, active)
            chunks.append(tuple(np.moveaxis((v[::2]-v[1::2])/(2*epsilon), 0, -1) for v in values))
        residual, hand, distance = (np.concatenate(values, -1) for values in zip(*chunks))
        if active is not None:
            full_distance = np.zeros((len(distance), rows, len(columns)), dtype=distance.dtype)
            full_distance[:, active] = distance
            distance = full_distance
        return residual, hand, distance
    return jacobian
