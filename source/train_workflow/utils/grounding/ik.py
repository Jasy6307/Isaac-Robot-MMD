"""Approved pelvis-frame and measured-sole target repair, retaining RobotLeg IK."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg, LegTarget
from source.train_workflow.utils.retarget.upper_body_ik import unit, SolverWeights


def pelvis_rotations(source_rotations, to_robot, alignment):
    source_rotations = np.asarray(source_rotations)
    rotations = np.asarray(alignment).T @ source_rotations @ np.asarray(to_robot).T
    identity = np.asarray(alignment) @ rotations @ to_robot @ np.swapaxes(source_rotations, -1, -2)
    if not np.allclose(identity, np.eye(3), atol=1e-6):
        raise ValueError('Pelvis coordinate identity failed')
    return rotations


def prepare_targets(source, motion, sole_heights, robot_usd, *, alignment=None, start_seconds=0.):
    alignment = Rotation.from_euler('z', 180, degrees=True).as_matrix() if alignment is None else alignment
    count = len(motion.delta)
    if count < 9:
        raise ValueError('Approved pelvis smoothing needs at least nine frames')
    matrices = [source.frame(start_seconds + i / motion.fps) for i in range(count)]
    rw = pelvis_rotations([m[source.names['下半身'], :3, :3] for m in matrices], source.to_robot, alignment)
    rp = motion.root_pos.copy()
    data, candidates, confidence = {}, [], []
    for side, prefix in (('left', '左'), ('right', '右')):
        leg = RobotLeg(robot_usd, side)
        ix = [source.names[n] for n in ('下半身', prefix+'足', prefix+'ひざ', prefix+'足首', prefix+'つま先')]
        rest = source.rest
        length = np.linalg.norm(rest[ix[2], :3, 3] - rest[ix[1], :3, 3]) + np.linalg.norm(rest[ix[3], :3, 3] - rest[ix[2], :3, 3])
        forward = rest[ix[4], :3, 3] - rest[ix[3], :3, 3]
        forward[2] = 0
        forward = unit(forward)
        up = np.array([0., 0., 1.])
        local = rest[ix[3], :3, :3].T @ np.column_stack((forward, np.cross(up, forward), up))
        knees, ankles, orientations, bottom = [], [], [], []
        for i, matrix in enumerate(matrices):
            convert = source.to_robot @ matrix[ix[0], :3, :3].T
            hip, knee, ankle = [matrix[j, :3, 3] for j in ix[1:4]]
            k = leg.shoulder + leg.upper_length * (convert @ unit(knee - hip))
            a = k + leg.fore_length * (convert @ unit(ankle - knee))
            orient = convert @ matrix[ix[3], :3, :3] @ local
            knees.append(k); ankles.append(a); orientations.append(orient)
            bottom.append(((leg.sole @ (orient @ leg.palm_local.T).T + a) @ rw[i].T)[:, 2].min())
        heights = np.asarray(sole_heights[side])
        if heights.shape != (count,) or not np.isfinite(heights).all():
            raise ValueError('Measured sole heights must cover the clip')
        desired = np.maximum(heights - .003, 0) * (leg.reach / length)
        candidates.append(desired - np.array(bottom))
        confidence.append(1 / (.015 + desired)**2)
        data.update({side+'_knee': np.asarray(knees), side+'_ankle': np.asarray(ankles),
                     side+'_orient': np.asarray(orientations), side+'_desired': desired,
                     side+'_bottom': np.asarray(bottom), side+'_cols': np.array([motion.names.index(n) for n in leg.names])})
    candidates, confidence = np.array(candidates), np.array(confidence)
    rp[:, 2] = savgol_filter((candidates * confidence).sum(0) / confidence.sum(0), 9, 3)
    for side in ('left', 'right'):
        shift = data[side+'_desired'] - (data[side+'_bottom'] + rp[:, 2])
        data[side+'_ankle'] += np.einsum('nji,nj->ni', rw, np.c_[np.zeros((count, 2)), shift])
    data.update(root_pos=rp, root_rot=rw, q=motion.q)
    return data


def solve_side(data, side, robot_usd, fps):
    leg = RobotLeg(robot_usd, side)
    cols = data[side+'_cols']
    targets = [LegTarget(k, a, o, r, p, bool(h < .01)) for k, a, o, r, p, h in zip(
        data[side+'_knee'], data[side+'_ankle'], data[side+'_orient'], data['root_rot'], data['root_pos'], data[side+'_desired'])]
    solved = leg.solve_sequence(targets, data['q'][:, cols], fps, passes=2, weights=SolverWeights(8., .7, .6, .25))
    return solved, cols


def sole_heights(q, names, root_pos, root_rot, robot_usd):
    heights = []
    for side in ('left', 'right'):
        leg = RobotLeg(robot_usd, side)
        cols = [names.index(n) for n in leg.names]
        side_heights = []
        for row, rotation, position in zip(q[:, cols], root_rot, root_pos):
            state = leg.fk(row)
            side_heights.append(((leg.sole @ state['body_rotation'].T + state['wrist']) @ rotation.T + position)[:, 2].min())
        heights.append(side_heights)
    return np.asarray(heights).T


def source_contact_report(heights, desired):
    support = np.asarray(desired) < .01
    flight = (np.asarray(desired) > .025).all(1)
    return dict(height_rmse_m=float(np.sqrt(np.mean((heights-desired)**2))),
                false_floating_on_source_stance=int(((heights > .025).all(1) & support.any(1)).sum()),
                source_flight_frames=int(flight.sum()),
                source_flight_retained=int(((heights > .015).all(1) & flight).sum()))


def stance_root_feedback(heights, desired):
    """Approved Deep Blue Town bounded minimum-curvature Z feedback, explicitly opt-in."""
    import clarabel
    from scipy import sparse
    heights, desired = np.asarray(heights), np.asarray(desired)
    count = len(heights)
    stance, flight = (desired < .01).any(1), (desired > .025).all(1)
    bottom, wanted = heights.min(1), desired.min(1)
    lo = -.001 - bottom
    lo[flight] = np.maximum(lo[flight], .016-bottom[flight])
    hi = np.maximum(wanted+.015, .02) - bottom
    hi[stance] = wanted[stance]+.003-bottom[stance]
    if count < 4 or np.any(hi < lo):
        raise ValueError('Infeasible source-contact feedback bounds')
    d2 = sparse.diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count))
    d3 = sparse.diags([-np.ones(count-3), 3*np.ones(count-3), -3*np.ones(count-3), np.ones(count-3)], [0, 1, 2, 3], shape=(count-3, count))
    p = sparse.triu(2*(sparse.eye(count)+80*(d2.T@d2)+500*(d3.T@d3)), format='csc')
    a = sparse.vstack([sparse.eye(count), -sparse.eye(count)], format='csc')
    settings = clarabel.DefaultSettings(); settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    solution = clarabel.DefaultSolver(p, np.zeros(count), a, np.r_[hi, -lo], [clarabel.NonnegativeConeT(2*count)], settings).solve()
    if str(solution.status) not in ('Solved', 'AlmostSolved'):
        raise ValueError(f'Root feedback failed: {solution.status}')
    offset = np.asarray(solution.x)
    if not np.isfinite(offset).all() or np.any(offset < lo-1e-7) or np.any(offset > hi+1e-7):
        raise ValueError('Root feedback bounds failed')
    return offset
