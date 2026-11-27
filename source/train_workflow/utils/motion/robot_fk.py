"""Joint-anchor FK shared by training, validation, and replay framing."""
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


@lru_cache(maxsize=8)
def _chain(usd_path, links):
    from source.train_workflow.utils.retarget.upper_body_ik import read_joints

    by_child = {joint.child: joint for joint in read_joints(usd_path)}
    ordered = {}
    for link in links or by_child:
        chain, pending = [], set()
        while link != "pelvis" and link not in ordered:
            if link in pending or link not in by_child:
                raise ValueError(f"Disconnected dance robot link: {link} ({usd_path})")
            pending.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.parent
        ordered.update((joint.child, joint) for joint in reversed(chain))
    return tuple(ordered.values())


def robot_fk(usd_path, names, q, *, links=()):
    """Return link rotations and translations in the pelvis frame."""
    indices = {name: i for i, name in enumerate(names)}
    rotations = {"pelvis": np.broadcast_to(np.eye(3), (len(q), 3, 3))}
    positions = {"pelvis": np.zeros((len(q), 3))}
    for joint in _chain(str(usd_path), tuple(links)):
        parent_r = rotations[joint.parent]
        r = parent_r @ joint.r0
        if joint.axis is not None:
            if joint.name not in indices:
                raise ValueError(f"Motion missing robot joint: {joint.name}")
            r = r @ Rotation.from_rotvec(q[:, indices[joint.name], None] * joint.axis).as_matrix()
        r = r @ joint.r1.T
        rotations[joint.child] = r
        positions[joint.child] = positions[joint.parent] + parent_r @ joint.p0 - r @ joint.p1
    return rotations, positions
