"""Reference-only jump hypotheses. No dance names or reviewed time labels.

Kinematic clearance is evidence of intent, never proof of physical flight.
Low-confidence events may be oversampled but cannot enable flight rewards.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

VERSION = 6


def _causal_minimum(values, span):
    """Running minimum that never uses future samples.

    A centered minimum makes a later crouch look like an earlier takeoff.  The
    reference classifier must only compare a frame with its past, so the
    baseline is deliberately causal.
    """
    values = np.asarray(values, float)
    out = np.empty_like(values)
    for index in range(len(values)):
        out[index] = values[max(0, index - span + 1):index + 1].min()
    return out


def analyze_signals(root_z, feet, fps, valid=None):
    root_z, feet = np.asarray(root_z, float), np.asarray(feet, float)
    n = len(root_z)
    if (root_z.ndim != 1 or not np.isfinite(fps) or fps <= 0 or n < 3 or feet.shape != (n, 2, 3)
            or not np.isfinite(root_z).all() or not np.isfinite(feet).all()):
        raise ValueError("Expected finite root[N], feet[N,2,3], N>=3 and positive fps")
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != (n,):
        raise ValueError("Invalid root validity mask")
    sigma = max(.5, fps * .035)
    z = gaussian_filter1d(root_z, sigma)
    p = gaussian_filter1d(feet, sigma, axis=0)
    span = max(3, int(round(.9 * fps)) | 1)
    # Local floor is inferred, not a measured sole/ground contact plane.  Keep
    # this baseline causal as well; a later squat must not manufacture earlier
    # foot clearance.
    floor = _causal_minimum(p[:, :, 2].min(1), span)
    clearance = p[:, :, 2].min(1) - floor
    rise = z - _causal_minimum(z, span)
    vz = np.gradient(z, 1 / fps)
    gap = np.linalg.norm(p[:, 0, :2] - p[:, 1, :2], axis=-1)
    closing = -np.gradient(gap, 1 / fps)
    # Require two independent cues. Closing alone and root bobbing alone fail.
    active = valid & (clearance > .025) & ((rise > .025) | (closing > .35))
    runs = []
    for index in np.flatnonzero(active):
        if runs and index - runs[-1][-1] <= max(1, int(.08 * fps)):
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    events = []
    for run in runs:
        lo, hi = run[0], run[-1] + 1
        if (hi - lo) / fps < .065 or (hi - lo) / fps > .8:
            continue
        before = max(0, lo - int(.3 * fps))
        up = float(vz[before:hi].max())
        root_rise = float(rise[lo:hi].max())
        clear = float(clearance[lo:hi].max())
        close = float(closing[lo:hi].max())
        # A flight request requires bilateral clearance for a real interval,
        # not a single lifted-foot sample.  This rejects small hops and
        # crouches whose feet remain effectively supported.
        airborne = clearance[lo:hi] > .08
        longest_air = run = 0
        for value in airborne:
            run = run + 1 if value else 0
            longest_air = max(longest_air, run)
        airborne_s = longest_air / fps
        # Stronger evidence is required to actually request flight.
        # Modest body bobbing is not sufficient to impose bilateral flight.
        # Smaller possible jumps remain unknown rather than hard negatives.
        # A high jump must show a measurable bilateral flight interval.  The
        # stricter 10 cm / 150 ms gates keep a small hop followed by a squat
        # out of the hard flight-reward path.
        fast = up > .45 and root_rise > .10 and clear > .10 and airborne_s >= .15
        slow_large = up > .35 and root_rise > .12 and clear > .10 and airborne_s >= .15
        high = (fast or slow_large) and valid[before:hi].all()
        confidence = "high" if high else "review"
        start, end = lo / fps, hi / fps
        duration = n / fps
        schedule = [[max(0., start - .30), max(0., start - .15)],
                    [max(0., start - .15), start],
                    [start, end], [end, min(duration, end + .25)]]
        events.append(dict(id=len(events), confidence=confidence,
                           flight_evidence=("strong" if high else "moderate" if
                               up > .15 and root_rise > .035 and clear > .04 else "weak"),
                           flight_reward=bool(high), phases=schedule,
                           target_upward_velocity=float(np.clip(up, .2, 1.2)),
                           evidence=dict(clearance_m=clear, root_rise_m=root_rise,
                                         upward_mps=up, closing_mps=close,
                                         bilateral_airborne_s=airborne_s)))
    foot_vz = np.gradient(p[:, :, 2], 1 / fps, axis=0)
    support = ((p[:, :, 2] - floor[:, None]) < .012) & (np.abs(foot_vz) < .12) & valid[:, None]
    for event in events:
        lo, hi = event["phases"][0][0] - .1, event["phases"][-1][1] + .1
        support[max(0, int(lo * fps)):min(n, int(np.ceil(hi * fps)))] = False
    # Positive single-support evidence: one ankle is clearly higher while the
    # lower ankle stays near the inferred floor. Unknown/bilateral flight is
    # never converted into a request to lift a particular foot.
    gap_z = p[:, :, 2] - p[:, ::-1, 2]
    lower_near_floor = (p[:, :, 2].min(1) - floor) < .035
    swing = (gap_z > .08) & lower_near_floor[:, None] & valid[:, None]
    for event in events:
        if event["flight_reward"]:
            lo, hi = event["phases"][0][0], event["phases"][-1][1]
            swing[max(0, int(lo * fps)):min(n, int(np.ceil(hi * fps)))] = False
    from source.train_workflow.utils.reference_objectives import contact_evidence
    contact_support, contact_air = contact_evidence(p, fps, valid)
    return dict(version=VERSION, fps=float(fps), frames=n, duration_s=n / fps,
                contact_support_confidence=contact_support.tolist(), contact_air_confidence=contact_air.tolist(),
                swing_feet=swing.astype(int).tolist(), swing_gap_m=np.maximum(gap_z, 0.).tolist(),
                support_feet=support.astype(int).tolist(), support_windows=[],
                events=events, limitations=[
                    "Candidates describe reference intent, not confirmed jumps or failures.",
                    "Ankle-origin local clearance is not measured sole contact.",
                    "Review candidates only affect sampling; no automatic reference shaping."])


def apply_review(report, review):
    """Data-bound human evidence; never silently transfer between dances."""
    if review.get("source_sha256") != report.get("source_sha256"):
        raise ValueError("Review belongs to a different reference file")
    for item in review.get("support_windows", []):
        lo, hi = float(item["start_s"]), float(item["end_s"])
        if not 0 <= lo < hi <= report["duration_s"] or not item.get("reason"):
            raise ValueError("Support review requires an in-range interval and reason")
        report["support_windows"].append(dict(start_s=lo, end_s=hi, reason=item["reason"]))
        for event in report["events"]:
            a, b = event["phases"][2]
            if a < hi and b > lo:
                event.update(flight_reward=False, confidence="reviewed_support")
    report["review"] = review
    return report


def analyze_h5(path, robot_usd, frames=None):
    """USD FK uses the reference's recorded baseline, not a learned policy."""
    import h5py
    import torch
    from scipy.spatial.transform import Rotation
    from .self_collision import G1CollisionModel, playback_baseline

    path = Path(path).resolve()
    if frames is not None and frames < 3:
        raise ValueError("Analysis requires at least three source frames")
    root = Path(__file__).resolve().parents[4]
    with h5py.File(path) as f:
        names = list(f["joint_names"].asstr()[:])
        count = min(len(f["joint_pos_delta"]), frames or len(f["joint_pos_delta"]))
        q = f["joint_pos_delta"][:count]
        baseline = np.asarray(f.attrs.get("record_baseline_joint_pos",
                              playback_baseline(names, root / "source/my_task/g1_replay_env_cfg.py")))
        pos = f["root_pos_delta"][:count]
        quat = f["root_quat_delta_wxyz"][:count]
        valid = f["root_valid"][:count].astype(bool)
        fps = float(f.attrs["fps"])
        anchor = np.asarray(f.attrs.get("record_root_anchor_quat_wxyz", [.70710678, 0, 0, .70710678]))
    if not np.isfinite(q).all() or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("Non-finite reference")
    model = G1CollisionModel(Path(robot_usd), names, "cpu")
    with torch.no_grad():
        _, positions = model.fk(torch.tensor(q + baseline, dtype=torch.float32))
        feet = torch.stack([positions[f"{side}_ankle_roll_link"] for side in ("left", "right")], 1).numpy()
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]) * Rotation.from_quat(anchor[[1, 2, 3, 0]])
    feet = np.einsum("nij,nkj->nki", rotation.as_matrix(), feet) + pos[:, None]
    report = analyze_signals(pos[:, 2], feet, fps, valid)
    report.update(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  robot_usd=str(Path(robot_usd).resolve()))
    return report


def event_schedules(report):
    return tuple(tuple(tuple(pair) for pair in event["phases"])
                 for event in report["events"] if event["flight_reward"])
