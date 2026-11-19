"""Project source-classified support onto the G1 visible soles before root lock."""
import clarabel
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation, Slerp

from source.train_workflow.utils.motion.constrained_trajectory import difference_matrix, local_rows
from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg
from .smoothing import FEET, FRACTIONS, sample_brackets, sampled_values
from .visible import visible_hull


def robot_sole_vertices(robot):
    return tuple(visible_hull(robot, link)[0] for link in FEET)


def _leg_fk(leg, angles):
    rotation = np.broadcast_to(np.eye(3), (len(angles), 3, 3))
    position = np.zeros((len(angles), 3))
    origins, axes = [], []
    for joint in leg.chain:
        origin = position + rotation @ joint.p0
        pre = rotation @ joint.r0
        if joint.axis is not None:
            column = len(axes)
            origins.append(origin)
            axes.append(pre @ joint.axis)
            pre = pre @ Rotation.from_rotvec(angles[:, column, None]*joint.axis).as_matrix()
        rotation = pre @ joint.r1.T
        position = origin-rotation @ joint.p1
    return rotation, position, np.stack(origins, 1), np.stack(axes, 1)


def _sole_geometry(legs, meshes, leg_angles, root_z, vertical, *, derivatives):
    heights = np.empty((len(leg_angles), 2))
    jac = np.zeros((len(leg_angles), 2, 13)) if derivatives else None
    for side, (leg, mesh) in enumerate(zip(legs, meshes)):
        cols = slice(side*6, (side+1)*6)
        rotation, position, origins, axes = _leg_fk(leg, leg_angles[:, cols])
        direction = np.einsum('ni,nij->nj', vertical, rotation)
        dots = direction @ mesh.T
        heights[:, side] = dots.min(1) + np.einsum('ni,ni->n', vertical, position) + root_z
        if derivatives:
            vertex = mesh[dots.argmin(1)]
            point = np.einsum('nij,nj->ni', rotation, vertex) + position
            axes_cross = np.cross(axes, point[:, None]-origins)
            jac[:, side, 1+cols.start:1+cols.stop] = np.einsum('ni,nji->nj', vertical, axes_cross)*.1
            jac[:, side, 0] = .02
    return heights, jac


def project_contact(q, names, fps, root_pos, root_rot, desired, robot, meshes):
    """Fit only source-contact frames; keep both legs and root exact in flight.

    Variables are root Z and twelve leg angles. Quarter-frame shoe vertex
    constraints are linearized, with a nonlinear check on each trial. The
    caller checks the final, encoded trajectory separately.
    """
    legs = [RobotLeg(robot, side) for side in ('left', 'right')]
    columns = np.array([names.index(name) for leg in legs for name in leg.names])
    leg_angles = q[:, columns].astype(float)
    count = len(q)
    contact = desired == 0
    movable = contact.any(1)
    coordinates, low, high = sample_brackets(count)
    samples = contact[low] & contact[high]
    vertical = Slerp(np.arange(count), Rotation.from_matrix(root_rot))(coordinates).as_matrix()[:, 2, :]
    source_legs = sampled_values(leg_angles)
    root_z = sampled_values(root_pos[:, 2])
    scale = np.r_[.02, np.full(12, .1)]

    def geometry(x, derivatives=False):
        edit = sampled_values(x)*scale
        return _sole_geometry(legs, meshes, source_legs+edit[:, 1:], root_z+edit[:, 0], vertical,
                              derivatives=derivatives)

    x = np.zeros((count, 13))
    reference, _ = geometry(x)
    lower = np.maximum(reference-.021, np.minimum(reference, -.0003))
    upper = reference+.023
    lower[samples], upper[samples] = -.00017, .00285
    checked = movable[low] | movable[high]

    def excess(heights):
        return np.maximum(np.maximum(lower-heights, heights-upper), 0)*checked[:, None]

    if movable.any():
        free = np.flatnonzero(np.repeat(movable, 13))
        eye = sparse.eye(count*13, format='csc')[:, free]
        d1 = difference_matrix(count, 13)
        d2 = difference_matrix(count, 13, 2)
        d3 = difference_matrix(count, 13, 3)
        weights = np.tile([.55, *([1, 1, 1, 1, 3, 3]*2)], count)[free]
        smooth = (sparse.eye(len(free))*1.04 + 5*d1[:, free].T@d1[:, free] +
                  14*d2[:, free].T@d2[:, free] + 3*d3[:, free].T@d3[:, free])
        w = sparse.diags(np.sqrt(weights))
        penalty = sparse.triu(2*w@smooth@w, format='csc')
        joint_lower = np.concatenate([leg.lower for leg in legs])
        joint_upper = np.concatenate([leg.upper for leg in legs])
        lo = np.c_[np.full(count, -2.5), np.maximum((joint_lower-leg_angles)/.1, -3.5)]
        hi = np.c_[np.full(count, 2.5), np.minimum((joint_upper-leg_angles)/.1, 3.5)]
        step = np.r_[.020, np.concatenate([leg.velocity for leg in legs])/fps]
        initial_step = np.c_[np.zeros(count-1), np.diff(leg_angles, axis=0)]
        diff = d1[:, free]
        bound = sparse.vstack([eye[free], -eye[free], diff, -diff], format='csc')
        bound_rhs = np.r_[hi.ravel()[free], -lo.ravel()[free],
                          ((step-initial_step)/scale).ravel(), ((step+initial_step)/scale).ravel()]
        active = np.zeros_like(samples)
        active[:count] = movable[:, None]
        options = clarabel.DefaultSettings()
        options.verbose = False
        options.max_threads = 2
        options.tol_feas = 1e-8
        for iteration in range(42):
            heights, jac = geometry(x, True)
            violation = excess(heights)
            print('SOLE_PROJECTION', iteration, float(violation.max()*1000), flush=True)
            if violation.max() < .00012:
                break
            active |= violation > 2e-6
            chosen = np.where(active)
            local = local_rows(jac, *chosen, count, 13, FRACTIONS)[:, free]/.005
            offset = local @ x.ravel()[free]
            rhs = np.r_[(upper-heights)[chosen]/.005 + offset, (heights-lower)[chosen]/.005 - offset]
            trust = max(.25, .8*.96**iteration)
            base = bound_rhs.copy()
            base[:2*len(free)] = np.minimum(base[:2*len(free)],
                                            np.r_[x.ravel()[free], -x.ravel()[free]]+trust)
            slack_count = len(chosen[0])
            slack = -sparse.eye(slack_count)
            matrix = sparse.bmat([[bound, sparse.csc_matrix((bound.shape[0], slack_count))],
                                  [local, slack], [-local, slack]], format='csc')
            cost = sparse.block_diag([penalty, 1e6*sparse.eye(slack_count)], format='csc')
            proposal = clarabel.DefaultSolver(cost, np.zeros(cost.shape[0]), matrix, np.r_[base, rhs],
                                               [clarabel.NonnegativeConeT(len(base)+len(rhs))], options).solve()
            if str(proposal.status) not in ('Solved', 'AlmostSolved'):
                raise RuntimeError(f'Visible sole projection QP failed: {proposal.status}')
            update = np.zeros_like(x)
            update.ravel()[free] = proposal.x[:len(free)]
            best = (float(np.sum(violation**2)), x)
            for alpha in (1., .5, .25, .125, .0625):
                trial = x+alpha*(update-x)
                trial_violation = excess(geometry(trial)[0])
                score = float(np.sum(trial_violation**2))
                if score < best[0]:
                    best = score, trial
            x = best[1]

    result = np.asarray(q, dtype=float).copy()
    result[:, columns] += x[:, 1:]*scale[1:]
    root = root_pos.copy()
    root[:, 2] += x[:, 0]*scale[0]
    heights, _ = geometry(x)
    violations = np.flatnonzero((excess(heights) > .00015).any(1))
    report = dict(contact_samples=int(samples.sum()), no_contact_frames=int((~movable).sum()),
                  foot_height_on_contact_m=[float(heights[samples].min()), float(heights[samples].max())]
                  if samples.any() else None, maximum_root_adjustment_m=float(max(abs(root[:, 2]-root_pos[:, 2]))),
                  failed_frames=np.unique(np.r_[low[violations], high[violations]]).tolist(),
                  passed=not len(violations) and np.all(heights[samples] >= -.0003) and np.all(heights[samples] <= .003))
    report['passed'] = bool(report['passed'])
    return result, root, report
