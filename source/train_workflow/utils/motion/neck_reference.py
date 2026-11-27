"""Provenance for the verified early Deep Blue Town neck track."""
from hashlib import sha256
from pathlib import Path

import numpy as np


NECK_JOINTS = ("neck_pitch_joint", "neck_yaw_joint")
_DEEP_BLUE_BODY_HASH = "38dcbcb590bdfa2df21a565c9f73aa55006e21f6e9680efa8e7e04cbba842bba"
_DEEP_BLUE_TRACK = Path(__file__).resolve().parents[4] / "assets/g1_plus_o6/deepbluetown_neck_0_90.npz"


def verified_neck_reference(h5_path, fps, count):
    """Return the companion track only for its exact synchronized body reference."""
    if fps != 30 or count != 2700:
        return None
    with open(h5_path, "rb") as stream:
        if sha256(stream.read()).hexdigest() != _DEEP_BLUE_BODY_HASH:
            return None
    with np.load(_DEEP_BLUE_TRACK, allow_pickle=False) as track:
        angles = track["angles"]
        if (track["fps"].item() != fps or tuple(track["joint_names"]) != NECK_JOINTS
                or angles.shape != (count, len(NECK_JOINTS)) or not np.isfinite(angles).all()):
            raise ValueError(f"Invalid G1+ neck reference: {_DEEP_BLUE_TRACK}")
        return angles
