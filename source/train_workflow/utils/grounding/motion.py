"""Explicit H5 motion boundaries for the repaired-grounding pipeline."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass
class Motion:
    path: Path
    names: list[str]
    delta: np.ndarray
    baseline: np.ndarray
    root_pos: np.ndarray
    root_rot: np.ndarray
    fps: float

    @property
    def q(self):
        return self.delta + self.baseline


def read_motion(path, frames=None):
    path = Path(path).resolve()
    with h5py.File(path) as stream:
        names = list(stream['joint_names'].asstr()[:])
        total = len(stream['joint_pos_delta'])
        count = total if frames is None else frames
        if not 2 <= count <= total:
            raise ValueError('Expected 2..input frame count')
        delta = stream['joint_pos_delta'][:count]
        baseline = np.asarray(stream.attrs['record_baseline_joint_pos'])
        root = stream['root_pos_delta'][:count] + stream.attrs['record_root_anchor_pos']
        anchor = np.asarray(stream.attrs['record_root_anchor_quat_wxyz'])
        rotation = (Rotation.from_quat(stream['root_quat_delta_wxyz'][:count][:, [1, 2, 3, 0]]) *
                    Rotation.from_quat(anchor[[1, 2, 3, 0]])).as_matrix()
        fps = float(stream.attrs['fps'])
    if not np.isfinite(fps) or fps <= 0 or len(set(names)) != len(names):
        raise ValueError('Invalid motion timing or joint names')
    if delta.shape != (count, len(names)) or baseline.shape != (len(names),) or root.shape != (count, 3):
        raise ValueError('Incompatible motion channels')
    if not all(np.isfinite(x).all() for x in (delta, baseline, root, rotation)):
        raise ValueError('Nonfinite motion')
    return Motion(path, names, delta, baseline, root, rotation, fps)


def continuous_quaternions(matrices, anchor_wxyz):
    encoded = (Rotation.from_matrix(matrices) *
               Rotation.from_quat(np.asarray(anchor_wxyz)[[1, 2, 3, 0]]).inv()).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(encoded)):
        if encoded[i-1] @ encoded[i] < 0:
            encoded[i] *= -1
    return encoded


def write_motion(reference, output, delta, *, root_pos=None, root_rot=None, attributes=None):
    """Create exclusively, preserving all unchanged channels byte-for-byte as arrays.

    Root data is copied without re-encoding for collision/smoothing outputs.
    Publication callers must validate the encoded delta, not only absolute q.
    """
    output = Path(output)
    delta = np.asarray(delta)
    if delta.shape != reference.delta.shape or not np.isfinite(delta).all():
        raise ValueError('Invalid output joint channels')
    for value, shape in ((root_pos, reference.root_pos.shape), (root_rot, reference.root_rot.shape)):
        if value is not None and (np.shape(value) != shape or not np.isfinite(value).all()):
            raise ValueError('Invalid output root channels')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(reference.path) as source, h5py.File(output, 'x') as target:
        for key, value in source.attrs.items():
            if not key.startswith('collision_') and key not in ('without_collision_validated', 'arm_motion_validated'):
                target.attrs[key] = value
        count = len(reference.delta)
        total = len(source['joint_pos_delta'])
        def copy_group(src, dst):
            for key, value in src.items():
                if isinstance(value, h5py.Group):
                    child = dst.create_group(key)
                    copy_group(value, child)
                else:
                    data = value[:count] if key != 'joint_names' and value.ndim and value.shape[0] == total else value[()]
                    child = dst.create_dataset(key, data=data, dtype=value.dtype)
                for attr, data in value.attrs.items():
                    child.attrs[attr] = data
        copy_group(source, target)
        target['joint_pos_delta'][:] = delta
        if root_pos is not None:
            target['root_pos_delta'][:] = root_pos - source.attrs['record_root_anchor_pos']
        if root_rot is not None:
            target['root_quat_delta_wxyz'][:] = continuous_quaternions(
                root_rot, source.attrs['record_root_anchor_quat_wxyz'])
        target.attrs['grounding_protocol'] = 'source-sole-fixed-root-v1'
        target.attrs['grounding_reference_sha256'] = sha256(reference.path)
        target.attrs['clip_end_seconds'] = count / reference.fps
        for key, value in (attributes or {}).items():
            target.attrs[key] = value
