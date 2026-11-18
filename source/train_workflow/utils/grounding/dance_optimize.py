"""Bounded, fully numeric retries and independent acceptance for G1 dance clips."""
import argparse

import h5py
import numpy as np
import torch

from source.train_workflow.utils.motion.mesh_audit import scan
from .arms import ArmRefinement, windows
from .motion import read_motion, sha256, write_motion
from .pipeline import require_same_reference
from .smoothing import GroundProblem, SmoothingSettings, sampled_values
from .visible import MeshPairConstraints

# Existing mechanical CAD overlaps of this fixed G1 O6 asset. Scan and report
# them along with all other contacts. Only these exact pairs are allowlisted.
CAD_PAIRS = {frozenset(pair) for pair in (
    ('left_ankle_roll_link', 'left_knee_link'), ('right_ankle_roll_link', 'right_knee_link'),
    ('left_hip_roll_link', 'pelvis'), ('right_hip_roll_link', 'pelvis'),
    ('left_hip_roll_link', 'pelvis_contour_link'), ('right_hip_roll_link', 'pelvis_contour_link'),
    ('torso_link', 'waist_yaw_link'),
)}


def save_candidate(reference, path, q):
    delta = reference.delta.copy()
    changed = q != reference.q
    delta[changed] = (q-reference.baseline)[changed]
    write_motion(reference, path, delta, attributes=dict(collision_optimization_applied=True,
                 without_collision_validated=False, staging_only=True))
    return read_motion(path)


def audit_mesh(candidate, folder, robot):
    report = scan(argparse.Namespace(h5=candidate.path, output=folder, stride=1,
                                    pair_scope='all', start_sample=0, end_sample=None))
    count = 4*len(candidate.q)-3
    if (report['h5_sha256'] != sha256(candidate.path) or report['usd_sha256'] != sha256(robot) or
            report['sample_count'] != count or report['start_sample'] != 0 or report['end_sample'] != count or
            report['sample_step_frames'] != .25 or report['pair_scope'] != 'all'):
        raise RuntimeError('Incomplete or mismatched visible mesh audit')
    unexpected = [hit for hit in report['intersections'] if frozenset(hit['pair']) not in CAD_PAIRS]
    return dict(report=str(folder/'mesh_scan.json'), tested_pairs=len(report['tested_pairs']), sampled_poses=count,
                existing_cad_pairs=[hit['pair'] for hit in report['intersections']
                                    if frozenset(hit['pair']) in CAD_PAIRS], unexpected=unexpected)


def _hard_checks(problem, candidate, desired):
    roots_equal = require_same_reference(problem.reference, candidate)
    return problem.audit(candidate.q, root_channels_unchanged=roots_equal, desired=desired,
                         require_arm_improvement=False)


def _collision_reference(run, folder, problem, desired, geometry, retries):
    ref = problem.reference
    settings = SmoothingSettings(window_frames=48, stride=30)
    progress = run.progress('collision_window')
    q, _ = problem.smooth(ref.q, settings, progress=lambda _, info: progress(info['frames']))
    for attempt in range(retries+1):
        attempt_dir = folder/f'collision_{attempt:02d}'
        candidate = save_candidate(ref, attempt_dir/'candidate.h5', q)
        audit = _hard_checks(problem, candidate, desired)
        mesh = audit_mesh(candidate, attempt_dir/'mesh', problem.robot_usd) if audit['passed'] else None
        run.report(f'collision_{attempt:02d}', dict(motion=audit, mesh=mesh, candidate=str(candidate.path)))
        if mesh is not None and not mesh['unexpected']:
            return candidate, audit, mesh
        if attempt == retries:
            break
        frames = audit['proxy_bad_frames'] + audit['foot_height_bound_bad_frames']
        if mesh is not None:
            for hit in mesh['unexpected']:
                geometry.add(hit['pair'])
                frames.extend(round(sample) for sample in hit['sample_coordinates'])
        if not frames:
            raise RuntimeError('Collision candidate failed joint, root or source-contact validation')
        q = candidate.q
        intervals = windows(frames, len(q), padding=52, gap=12)
        run.report(f'collision_retry_{attempt+1:02d}', dict(windows=intervals, added_mesh_pairs=sorted(geometry.names)))
        for lo, hi in intervals:
            if attempt:
                center = next(frame for frame in sorted(frames) if lo <= frame < hi)
                q = problem.escape_bias(q, center, random_seed=12017+attempt*701+center, sweeps=1)
            retry_settings = SmoothingSettings(window_frames=48, stride=28, iterations=16+4*attempt)
            q, _ = problem.smooth(q, retry_settings, start=lo, end=hi,
                                  progress=lambda _, info: progress(info['frames']))
    raise RuntimeError('Collision/grounding failed acceptance after the retry budget')


def _arm_quality(scores):
    excess = np.flatnonzero(scores > 1)
    return dict(maximum_score=float(scores.max()), flagged_frames=excess.tolist(),
                flagged_seconds=(excess/30).tolist())


def _rank(scores):
    return float(scores.max()), int((scores > 1).sum()), float(np.maximum(scores-1, 0).sum())


def _local_clearance(problem, q, lo, hi):
    with torch.no_grad():
        samples = sampled_values(q[max(0, lo-1):min(len(q), hi+1)])
        return float(problem.model.distances(*problem.model.fk(problem.model._tensor(samples))).min())


def _arm_motion(run, folder, problem, baseline, source, desired, geometry, retries):
    arms = ArmRefinement(problem, source, baseline.q)
    other = np.setdiff1d(np.arange(len(baseline.names)), arms.columns)
    current = baseline.q.copy()
    original = _arm_quality(arms.scores(current))
    progress = run.progress('arm_window')
    mesh_frames = []
    for attempt in range(retries+1):
        scores = arms.scores(current)
        frames = np.r_[np.flatnonzero(scores > 1), np.asarray(mesh_frames, dtype=int)]
        intervals = windows(frames, len(current), padding=15 if attempt else 12, gap=14)
        decisions = []
        for lo, hi in sorted(intervals, key=lambda span: -scores[slice(*span)].max()):
            seed = current.copy()
            seed_parameters = None
            if attempt:
                alternatives = arms.warm_seeds(current, lo, hi)
                if alternatives:
                    _, sigma, amount, pose = alternatives[min(attempt-1, len(alternatives)-1)]
                    seed[lo:hi] = pose
                    seed_parameters = dict(sigma_frames=sigma, blend=amount)
            interval = slice(max(0, lo-4), min(len(current), hi+4))
            before = _rank(arms.scores(current)[interval])
            try:
                proposal = arms.solve(seed, lo, hi, progress=progress)
            except ValueError as error:
                if seed_parameters is None or str(error) != 'Refinement requires a feasible seed':
                    raise
                decisions.append(dict(frames=[lo, hi], seed=seed_parameters, selected=False, reason=str(error)))
                continue
            after = _rank(arms.scores(proposal)[interval])
            clearance = _local_clearance(problem, proposal, lo, hi)
            needs_mesh_repair = any(lo <= frame < hi for frame in mesh_frames)
            accepted = bool(clearance >= .0005-1e-7 and (after < before or (needs_mesh_repair and after[0] <= 1)))
            if accepted:
                current = proposal
            decisions.append(dict(frames=[lo, hi], seconds=[lo/arms.fps, hi/arms.fps], seed=seed_parameters,
                                  prior_rank=before, proposed_rank=after, minimum_clearance_m=clearance,
                                  selected=accepted))

        attempt_dir = folder/f'arms_{attempt:02d}'
        candidate = save_candidate(baseline, attempt_dir/'candidate.h5', current)
        if not np.array_equal(candidate.delta[:, other], baseline.delta[:, other]):
            raise RuntimeError('Arm pass changed a non-arm channel')
        quality = _arm_quality(arms.scores(candidate.q))
        audit = _hard_checks(problem, candidate, desired)
        mesh = None
        mesh_frames = []
        if audit['passed'] and not quality['flagged_frames']:
            mesh = audit_mesh(candidate, attempt_dir/'mesh', problem.robot_usd)
            for hit in mesh['unexpected']:
                geometry.add(hit['pair'])
                mesh_frames.extend(round(sample) for sample in hit['sample_coordinates'])
        run.report(f'arms_{attempt:02d}', dict(source_envelope_threshold=1, original=original,
                   final=quality, motion=audit, mesh=mesh, decisions=decisions, candidate=str(candidate.path)))
        if mesh is not None and not mesh['unexpected']:
            return candidate, audit, mesh, quality
        if not quality['flagged_frames'] and not mesh_frames:
            raise RuntimeError('Arm candidate failed encoded motion acceptance')
        current = candidate.q
    raise RuntimeError('Arm stabilization failed acceptance after the retry budget; see flagged seconds')


def optimize(run, reference, source, desired, soles, robot, device, retries):
    folder = run.directory/'optimization'
    folder.mkdir()
    problem = GroundProblem(reference, robot, device=device, sole_vertices=soles,
                            desired=desired, active_derivatives=True)
    geometry = MeshPairConstraints(problem.model, robot)
    run.phase('collision')
    baseline, base_audit, base_mesh = _collision_reference(run, folder, problem, desired, geometry, retries)
    run.phase('arm_stabilization')
    final, audit, mesh, quality = _arm_motion(run, folder, problem, baseline, source, desired, geometry, retries)
    for label, candidate, motion_report, mesh_report in (('collision_baseline', baseline, base_audit, base_mesh),
                                                         ('arm_stabilized', final, audit, mesh)):
        # Rewrite headers on a fresh copy. FCL verified the exact encoded motion
        # arrays, so verify that those channels survive the copy unchanged.
        path = folder/f'{label}.h5'
        write_motion(candidate, path, candidate.delta, attributes=dict(collision_optimization_applied=True,
                     without_collision_validated=True, arm_motion_validated=label == 'arm_stabilized',
                     staging_only=False, validation_candidate_sha256=sha256(candidate.path)))
        encoded = read_motion(path)
        if not np.array_equal(encoded.delta, candidate.delta) or not require_same_reference(candidate, encoded):
            raise RuntimeError(f'Certified motion changed while exporting {label}')
        run.report(f'{label}_validation', dict(path=str(path), sha256=sha256(path), motion=motion_report,
                   mesh=mesh_report, temporal=quality if label == 'arm_stabilized' else None))
    return folder/'collision_baseline.h5', folder/'arm_stabilized.h5'
