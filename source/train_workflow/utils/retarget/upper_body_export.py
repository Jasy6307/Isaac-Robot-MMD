"""In-memory upper-body solve shared by live recording and H5 postprocessing."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from source.train_workflow.utils.retarget.upper_body_ik import (
    RobotArm, SourceSkeleton, SolverWeights, read_joints, target_errors,
)


def summarize(rows):
    return {key: dict(mean=float(np.mean([r[key] for r in rows])),
                      p95=float(np.percentile([r[key] for r in rows],95)),
                      max=float(max(r[key] for r in rows))) for key in rows[0]}


def solve_upper_body_arrays(delta, names, baseline, fps, source_animation, robot_usd,
                            indices=None, independent=False):
    delta = np.asarray(delta)
    baseline = np.asarray(baseline, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != len(names) or baseline.shape != (len(names),):
        raise ValueError('Invalid motion arrays or baseline')
    if not len(delta) or not np.isfinite(delta).all() or not np.isfinite(baseline).all():
        raise ValueError('Empty or nonfinite motion')
    if not np.isfinite(fps) or fps <= 0 or len(set(names)) != len(names):
        raise ValueError('Invalid FPS or duplicate joints')
    frames = np.arange(len(delta))
    indices = frames if indices is None else np.asarray(indices,dtype=int)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(delta)):
        raise ValueError('Invalid frame selection')
    source = SourceSkeleton(source_animation)
    if (len(frames)-1)/fps > (source.last-source.first)/source.fps+1e-5:
        raise ValueError('Source animation is shorter than motion')
    joints = read_joints(robot_usd)
    result = delta.copy()
    weights = SolverWeights()
    report = dict(method='skeleton-space-bounded-upper-body-ik-v1', weights=asdict(weights),
                  source_animation=str(Path(source_animation).resolve()),
                  robot_usd=str(Path(robot_usd).resolve()),
                  fps=fps, frames=len(frames), evaluated_frames=indices.tolist(),
                  collision_constraints=False, arms={})
    matrices = [source.frame(float(i) / fps) for i in indices]
    for side in ('left', 'right'):
        arm = RobotArm(joints, side)
        columns = [names.index(n) for n in arm.names]
        before, after = [], []
        targets = [source.target(matrix, side, arm) for matrix in matrices]
        originals = delta[np.ix_(indices, columns)].astype(float) + baseline[columns]
        if independent:
            solutions = np.asarray([arm.solve(t, q, seed=np.zeros(7), weights=weights) for t,q in zip(targets,originals)])
        else:
            print(f'UPPER_IK {side} trajectory {len(indices)} frames', flush=True)
            solutions = arm.solve_sequence(targets, originals, fps, weights=weights)
        for k, (i, target, original, solved) in enumerate(zip(indices, targets, originals, solutions)):
            result[i, columns] = solved - baseline[columns]
            before.append(target_errors(arm, original, target))
            after.append(target_errors(arm, solved, target))
            if k % 300 == 0:
                print(f'UPPER_IK {side} {k}/{len(indices)}', flush=True)
        violations = int(np.count_nonzero((solutions < arm.lower - 1e-7) | (solutions > arm.upper + 1e-7)))
        speed_violations = (int(np.count_nonzero(abs(np.diff(solutions,axis=0))*fps > arm.velocity+1e-5))
                            if not independent else None)
        report['arms'][side] = dict(before=summarize(before), after=summarize(after),
                                    limit_violations=violations,
                                    speed_limit_violations=speed_violations,
                                    max_step_deg=float(np.degrees(abs(np.diff(solutions, axis=0))).max()) if len(solutions)>1 else 0,
                                    joint_names=arm.names,
                                    per_frame=[dict(frame=int(i), before=b, after=a) for i,b,a in zip(indices,before,after)])
        if violations or speed_violations:
            raise RuntimeError('Joint limit violation')
    return result, report
