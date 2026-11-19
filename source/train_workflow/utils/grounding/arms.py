"""Source-guided wrist/palm temporal refinement with deterministic warm starts."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from source.train_workflow.utils.motion.constrained_trajectory import refine_feasible_trajectory
from source.train_workflow.utils.retarget.upper_body_ik import RobotArm, read_joints
from .smoothing import FRACTIONS, batched_central_difference, sampled_values

PALM_LINKS = ('lh_hand_base_link', 'rh_hand_base_link')
HAND_LENGTH = .105


def windows(frames, count, *, padding, gap):
    """Return merged padded half-open intervals, including edge violations."""
    spans = []
    for frame in sorted({int(frame) for frame in frames}):
        lo, hi = max(0, frame-padding), min(count, frame+padding+1)
        if spans and lo <= spans[-1][1]+gap:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
    return spans


def temporal_scores(wrist, marker, source_wrist, source_marker):
    """Subtract the local PMX envelope before scoring high frequency and speed."""
    highs, speeds = [], []
    for value, target in ((wrist, source_wrist), (marker, source_marker)):
        ripple = np.linalg.norm(value-gaussian_filter1d(value, 1.1, axis=0), axis=-1)
        expected = np.linalg.norm(target-gaussian_filter1d(target, 1.1, axis=0), axis=-1)
        highs.append(np.maximum(ripple-1.15*maximum_filter1d(expected, 3, axis=0), 0).max(1))
        step = np.linalg.norm(np.diff(value, axis=0), axis=-1)
        expected_step = np.linalg.norm(np.diff(target, axis=0), axis=-1)
        speeds.append(np.maximum(step-1.10*maximum_filter1d(expected_step, 3, axis=0), 0).max(1))
    high = np.maximum(*highs)
    speed = np.maximum(*speeds)
    return np.maximum(high/.009, np.r_[speed[0], speed]/.027)


class ArmRefinement:
    def __init__(self, problem, source, old):
        self.problem = problem
        self.model = problem.model
        self.old = old
        joints = read_joints(problem.robot_usd)
        sides = ('left', 'right')
        arms = [RobotArm(joints, side) for side in sides]
        self.columns = np.array([problem.reference.names.index(name) for arm in arms for name in arm.names])
        self.lower = problem.lower[self.columns]
        self.upper = problem.upper[self.columns]
        self.speed = problem.step[self.columns]
        self.fps = problem.reference.fps
        self.accel = np.maximum(80/self.fps**2, np.abs(np.diff(old[:, self.columns], n=2, axis=0)).max(0)+.001)
        self.wrist = [next(j for j in arm.chain if j.name == arm.wrist_name) for arm in arms]
        self.elbow = [next(j for j in arm.chain if j.name == arm.elbow_name) for arm in arms]
        self.anchors = {j.name: self.model._tensor(j.p1) for j in (*self.wrist, *self.elbow)}
        self.palm_local = np.stack([arm.palm_local for arm in arms])
        targets = [[source.target(matrix, side, arm) for side, arm in zip(sides, arms)]
                   for matrix in (source.frame(i/self.fps) for i in range(len(old)))]
        self.goal_wrist = np.array([[target.wrist for target in row] for row in targets])
        self.goal_elbow = np.array([[target.elbow for target in row] for row in targets])
        self.goal_palm = np.array([[target.palm for target in row] for row in targets])
        self.goal_marker = self.goal_wrist+HAND_LENGTH*self.goal_palm[..., 0]
        self.goal_derivative = gaussian_filter1d(self.goal_marker, .55, axis=0)
        self.old_wrist, self.old_marker = self.trajectory(old)

    def landmarks(self, rotation, position):
        trunk = rotation['torso_link']
        origin = position['torso_link']
        def anchors(joints):
            return torch.stack([position[j.child]+rotation[j.child]@self.anchors[j.name]-origin
                                for j in joints], dim=1) @ trunk
        palm = trunk.transpose(-1, -2)[:, None] @ torch.stack([rotation[k] for k in PALM_LINKS], dim=1)
        return anchors(self.wrist), anchors(self.elbow), palm

    def trajectory(self, q):
        with torch.no_grad():
            r, p = self.model.fk(self.model._tensor(q))
            wrist, _, palm = self.landmarks(r, p)
            wrist = wrist.cpu().numpy()
            marker = wrist+HAND_LENGTH*(palm.cpu().numpy() @ self.palm_local)[..., 0]
        return wrist, marker

    def scores(self, q, interval=slice(None)):
        return temporal_scores(*self.trajectory(q), self.goal_wrist[interval], self.goal_marker[interval])

    def feasible(self, arm):
        return (np.isfinite(arm).all() and (arm >= self.lower-1e-6).all() and (arm <= self.upper+1e-6).all()
                and (np.abs(np.diff(arm, axis=0)) <= self.speed+1e-5).all()
                and (np.abs(np.diff(arm, n=2, axis=0)) <= self.accel+2e-5).all())

    def sampler(self, lo, hi):
        window = hi-lo
        interval = slice(lo, hi)
        blend = np.ones(window)
        if lo:
            blend[:8] = np.linspace(0, 1, min(8, window))
        if hi < len(self.old):
            blend[-8:] = np.minimum(blend[-8:], np.linspace(1, 0, min(8, window)))
        blend = blend[:, None, None]
        wrist_target = blend*self.goal_wrist[interval]+(1-blend)*self.old_wrist[interval]
        temporal_target = blend*self.goal_derivative[interval]+(1-blend)*self.old_marker[interval]

        def batch_sample(q, active=None):
            batch, count, channels = q.shape
            poses = np.concatenate([q, *[q[:, :-1]*(1-f)+q[:, 1:]*f for f in FRACTIONS]], axis=1)
            stride = poses.shape[1]
            with torch.no_grad():
                rot, pos = self.model.fk(self.model._tensor(poses.reshape(-1, channels)))
                wrist, elbow, palm = self.landmarks(rot, pos)
                wrist = wrist.reshape(batch, stride, 2, 3)[:, :count].cpu().numpy()
                elbow = elbow.reshape(batch, stride, 2, 3)[:, :count].cpu().numpy()
                palm = palm.reshape(batch, stride, 2, 3, 3)[:, :count].cpu().numpy() @ self.palm_local
                distance = self.model.distances(rot, pos, active).cpu().numpy().reshape(batch, stride, -1)
            hand = wrist+HAND_LENGTH*palm[..., 0]-temporal_target
            residual = np.concatenate([(wrist-wrist_target).reshape(batch, count, -1)*110,
                                       (elbow-self.goal_elbow[interval]).reshape(batch, count, -1)*38,
                                       (palm-self.goal_palm[interval]).reshape(batch, count, -1)*7,
                                       (q[:, :, self.columns]-self.old[interval][:, self.columns])*1.5], axis=2)
            return residual, hand.reshape(batch, count, 6), distance

        def sample(q):
            return tuple(values[0] for values in batch_sample(q[None]))
        sample.active_jacobian = batched_central_difference(batch_sample, self.problem.pair_count, chunk_size=4)
        return sample

    def solve(self, seed, lo, hi, *, progress=None):
        current = seed.copy()
        length = min(64, hi-lo)
        starts = list(range(lo, hi-length+1, 40))
        if starts[-1] != hi-length:
            starts.append(hi-length)
        for sweep, order in enumerate((starts, starts[::-1])):
            for a in order:
                b = a+length
                margin_lo, margin_hi = max(0, a-2), min(len(current), b+2)
                def feasible(window):
                    arm = current[margin_lo:margin_hi, self.columns].copy()
                    arm[a-margin_lo:b-margin_lo] = window[:, self.columns]
                    return self.feasible(arm)
                result, report = refine_feasible_trajectory(
                    current[a:b], self.columns, self.lower, self.upper, self.speed, self.accel, self.fps,
                    None, None, self.sampler(a, b), feasible, iterations=23,
                    fixed_endpoints=(4 if a else 0, 4 if b < len(current) else 0),
                    collision_fractions=FRACTIONS, restore_collisions=True,
                    temporal_weights=(0., .007, .000045), hand_temporal_weights=(11000, 72000, 90000))
                current[a:b] = result
                if progress:
                    progress(dict(start=a, end=b, sweep=sweep, accepted_steps=report['accepted_steps']))
        return current

    def warm_seeds(self, current, lo, hi):
        """Rank tapered seeds by jitter and provisional clearance deficit."""
        base = current[lo:hi].copy()
        taper = np.ones(len(base))
        ramp = .5-.5*np.cos(np.linspace(0, np.pi, min(8, len(base)//2)))
        if lo:
            taper[:len(ramp)] = ramp
            taper[:4] = 0
        if hi < len(current):
            taper[-len(ramp):] = ramp[::-1]
            taper[-4:] = 0
        ranked = []
        for sigma in (1.5, 2.4, 3.6, 5., 6.8):
            smooth = gaussian_filter1d(base[:, self.columns], sigma, axis=0)
            for amount in (.32, .60, .85, 1.):
                q = base.copy()
                q[:, self.columns] += taper[:, None]*amount*(smooth-q[:, self.columns])
                if not self.feasible(q[:, self.columns]):
                    continue
                with torch.no_grad():
                    distance = self.model.distances(*self.model.fk(self.model._tensor(sampled_values(q))))
                    deficit = (.0005-distance).clamp_min(0)
                    peak = float(deficit.max())
                    energy = float(deficit.square().sum())
                if peak <= .075:
                    score = float(self.scores(q, slice(lo, hi)).max())+18*peak+24*energy
                    ranked.append((score, sigma, amount, q))
        return sorted(ranked, key=lambda seed: seed[:3])
