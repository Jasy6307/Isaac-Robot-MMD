"""Experimental leg-only takeoff reference shaping; never edits the source H5."""
import torch


def smoothstep(value):
    value = value.clamp(0., 1.)
    return value.square() * (3. - 2. * value)


def shape_takeoff_reference(reference, joint_names, control_hz, limits=None, events=None):
    if events is not None:
        result = reference.clone()
        for start, peak, release, finish, anchor_time in events:
            result = _shape_event(result, joint_names, control_hz, limits,
                                  start, peak, release, finish, anchor_time)
        return result
    return _shape_event(reference, joint_names, control_hz, limits, 16.86, 17.00, 17.08, 17.22, 16.90)


def _shape_event(reference, joint_names, control_hz, limits, start, peak, release, finish, anchor_time):
    shaped = reference.clone()
    seconds = torch.arange(len(reference), dtype=reference.dtype, device=reference.device) / control_hz
    # Smooth extension pulse before the original tucked/closed pose.
    extension = smoothstep((seconds - start) / (peak-start)) * (1. - smoothstep((seconds - peak) / (release+.05-peak)))
    hold = smoothstep((seconds - anchor_time) / .05) * (1. - smoothstep((seconds - release) / (finish-release)))
    anchor = min(len(reference) - 1, int(round(anchor_time * control_hz)))
    for side in ("left", "right"):
        for joint, offset in (("knee", -.45), ("hip_pitch", .10), ("ankle_pitch", -.08)):
            col = joint_names.index(f"{side}_{joint}_joint")
            values = reference[:, col] + extension * offset
            if limits is not None:
                values = values.clamp(limits[col, 0], limits[col, 1])
            shaped[:, col] = torch.where(extension > 0., values, reference[:, col])
        col = joint_names.index(f"{side}_hip_roll_joint")
        values = reference[:, col] * (1. - hold) + reference[anchor, col] * hold
        if limits is not None:
            values = values.clamp(limits[col, 0], limits[col, 1])
        shaped[:, col] = torch.where(hold > 0., values, reference[:, col])
    return shaped
