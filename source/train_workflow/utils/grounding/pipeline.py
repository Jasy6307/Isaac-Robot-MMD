"""Exclusive-output stages for repaired IK and independently certified smoothing."""
from pathlib import Path
import json
import numpy as np
import h5py

from source.train_workflow.utils.retarget.upper_body_ik import SourceSkeleton
from source.train_workflow.utils.motion.joint_constraints import validate_joint_positions
from .motion import read_motion, write_motion, sha256
from .sole import SoleSelection, measure_source_soles
from .ik import prepare_targets, solve_side, sole_heights, source_contact_report, stance_root_feedback
from .smoothing import GroundProblem, SmoothingSettings


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def require_same_reference(reference, candidate):
    if reference.names != candidate.names or reference.fps != candidate.fps or reference.delta.shape != candidate.delta.shape:
        raise ValueError('Candidate names, frame count and FPS must match the repaired IK reference')
    with h5py.File(reference.path) as ref, h5py.File(candidate.path) as other:
        for key in ('record_baseline_joint_pos', 'record_root_anchor_pos', 'record_root_anchor_quat_wxyz'):
            if not np.array_equal(ref.attrs[key], other.attrs[key]):
                raise ValueError(f'Candidate changed {key}')
        return all(np.array_equal(ref[key][:len(reference.delta)], other[key][:])
                   for key in ('root_pos_delta', 'root_quat_delta_wxyz'))


def prepare(input_h5, source_usd, robot_usd, output_dir, *, frames=None, source_leg_mode,
            character, mesh_path='/Character/Body', start_seconds=0., minimum_leg_skin_weight=None):
    """Prepare explicit source mode and Anon-only measured-sole targets.

    Source baking is intentionally separate: caller must choose FK/IK from VMD
    metadata, and the baked source sidecar must agree with that choice.
    """
    if character != 'anon':
        raise ValueError('Only the calibrated Anon sole adapter is registered; adapt other characters explicitly')
    if source_leg_mode not in ('fk', 'ik'):
        raise ValueError('Choose source FK or IK explicitly')
    sidecar = Path(source_usd).with_suffix('.json')
    if not sidecar.is_file() or json.loads(sidecar.read_text(encoding='utf-8')).get('leg_mode') != source_leg_mode:
        raise ValueError('Baked source sidecar must confirm the selected source leg mode')
    if start_seconds != 0:
        raise ValueError('This stage consumes prefix clips; preclip both source and H5 for other ranges')
    motion = read_motion(input_h5, frames)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(input=str(motion.path), source=str(Path(source_usd).resolve()), robot=str(Path(robot_usd).resolve()),
                  frames=len(motion.delta), fps=motion.fps, character=character, source_leg_mode=source_leg_mode,
                  mesh_path=mesh_path, input_sha256=sha256(input_h5), source_sha256=sha256(source_usd), robot_sha256=sha256(robot_usd))
    write_json(output_dir/'config.json', config)
    source = SourceSkeleton(source_usd)
    selection = SoleSelection(mesh_path, minimum_leg_skin_weight=minimum_leg_skin_weight)
    heights, metadata = measure_source_soles(source, len(motion.delta), motion.fps, selection)
    np.savez_compressed(output_dir/'source_sole.npz', **heights)
    write_json(output_dir/'sole_selection.json', metadata)
    data = prepare_targets(source, motion, heights, robot_usd)
    np.savez_compressed(output_dir/'targets.npz', **data)
    return data


def load_prepared(directory):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    for field in ('input', 'source', 'robot'):
        if sha256(config[field]) != config[field+'_sha256']:
            raise ValueError(f'Prepared {field} changed; create a new run')
    return config


def solve(directory, side):
    if side not in ('left', 'right'):
        raise ValueError('Choose left or right')
    directory = Path(directory)
    config = load_prepared(directory)
    output = directory/(side+'_solved.npz')
    if output.exists():
        raise FileExistsError(output)
    with np.load(directory/'targets.npz') as targets:
        q, cols = solve_side(targets, side, config['robot'], config['fps'])
    with output.open('xb') as stream:
        np.savez_compressed(stream, q=q, cols=cols)


def finalize_ik(directory, *, root_feedback=False):
    directory = Path(directory)
    config = load_prepared(directory)
    output = directory/'fixed_ik.h5'
    if output.exists():
        raise FileExistsError(output)
    motion = read_motion(config['input'], config['frames'])
    with np.load(directory/'targets.npz') as data:
        root_pos, root_rot = data['root_pos'].copy(), data['root_rot'].copy()
        desired = np.c_[data['left_desired'], data['right_desired']]
    delta = motion.delta.copy()
    for side in ('left', 'right'):
        with np.load(directory/(side+'_solved.npz')) as solved:
            delta[:, solved['cols']] = (solved['q']-motion.baseline[solved['cols']])
    heights = sole_heights(delta+motion.baseline, motion.names, root_pos, root_rot, config['robot'])
    if root_feedback:
        offset = stance_root_feedback(heights, desired)
        root_pos[:, 2] += offset
        heights = sole_heights(delta+motion.baseline, motion.names, root_pos, root_rot, config['robot'])
    report = source_contact_report(heights, desired)
    limits = validate_joint_positions(delta+motion.baseline, motion.names, motion.fps, config['robot'])
    report.update(joint_constraints=limits, frames=len(delta), fps=motion.fps, root_feedback=root_feedback,
                  worst_penetration_m=float(max(0, -heights.min())), source_leg_mode=config['source_leg_mode'])
    if not limits['passed'] or report['false_floating_on_source_stance'] or report['source_flight_frames'] != report['source_flight_retained']:
        write_json(directory/'fixed_ik.failed.json', report)
        raise ValueError('Repaired IK failed joint or source-contact acceptance; see failed report')
    write_motion(motion, output, delta, root_pos=root_pos, root_rot=root_rot, attributes=dict(
        lower_body_retarget_method='source-sole-pelvis-frame-v2', root_trajectory_changed=True,
        collision_optimization_applied=False, without_collision_validated=False, staging_only=False,
        source_leg_mode=config['source_leg_mode']))
    write_json(directory/'fixed_ik.validation.json', report)
    return report


def audit_motion(reference_h5, candidate_h5, robot_usd, *, geometry='batch', device='cpu', targets=None):
    reference, candidate = read_motion(reference_h5), read_motion(candidate_h5)
    root_equal = require_same_reference(reference, candidate)
    desired = None
    if targets:
        with np.load(targets) as data:
            desired = np.c_[data['left_desired'], data['right_desired']]
        if desired.shape != (len(reference.delta), 2):
            raise ValueError('Source targets do not cover reference')
    problem = GroundProblem(reference, robot_usd, geometry=geometry, device=device)
    report = problem.audit(candidate.q, root_channels_unchanged=root_equal, desired=desired)
    report.update(reference=str(reference.path), candidate=str(candidate.path),
                  reference_sha256=sha256(reference_h5), candidate_sha256=sha256(candidate_h5),
                  mesh_validation='not run; selected envelopes do not certify full visible meshes')
    return report


def smooth_motion(reference_h5, seed_h5, robot_usd, output_dir, *, geometry='batch', device='cpu',
                  settings=SmoothingSettings(), start=0, end=None, escape_centers=(), targets=None):
    """Staging is always separate; publish final H5 only after encoded hard gates.

    FCL mesh scan is a separate explicit stage. Final here certifies envelopes,
    joint limits, root channels, and sole bounds, not zero mesh intersections.
    """
    reference, seed = read_motion(reference_h5), read_motion(seed_h5)
    if not require_same_reference(reference, seed):
        raise ValueError('Seed must preserve the repaired IK root channels')
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    problem = GroundProblem(reference, robot_usd, geometry=geometry, device=device)
    current = seed.q
    for center in escape_centers:
        current = problem.escape_bias(current, center)
    def progress(q, report):
        np.savez_compressed(folder/'checkpoint.npz', q=q)
        (folder/'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    current, windows = problem.smooth(current, settings, start=start, end=end, progress=progress)
    delta = reference.delta.copy()
    changed = current != reference.q
    delta[changed] = (current-reference.baseline)[changed]
    candidate = folder/'candidate.h5'
    write_motion(reference, candidate, delta, attributes=dict(staging_only=True,
        without_collision_validated=False, collision_optimization_applied=True))
    report = audit_motion(reference_h5, candidate, robot_usd, geometry=geometry, device=device, targets=targets)
    report['windows'] = windows
    report['smoothing_settings'] = vars(settings)
    write_json(folder/'validation.json', report)
    if not report['passed']:
        raise ValueError('Staging failed hard acceptance; no final H5 published')
    write_motion(reference, folder/'final.h5', delta, attributes=dict(staging_only=False,
        without_collision_validated=True, collision_optimization_applied=True,
        collision_geometry_profile=geometry,
        collision_validation='selected envelopes >=0.5mm at frames/quarter samples; fixed root; joint limits/rates/mimic; sole bounds; FCL separate'))
    return report
