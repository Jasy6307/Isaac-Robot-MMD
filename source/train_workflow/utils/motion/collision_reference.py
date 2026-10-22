"""Reusable G1 reference repair: collision projection followed by local spike repair.

The certification covers the selected USD-derived envelopes at frames and
midpoints, with a 1 mm tolerance. It is not an exact mesh/PhysX guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import torch

from .self_collision import G1CollisionModel, playback_baseline, repair_poses

ROOT = Path(__file__).resolve().parents[4]
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def _feet(model, q):
    _, p = model.fk(q)
    return torch.stack([p[n] for n in FEET], dim=1)


def _windows(frames, count, padding):
    windows = []
    for frame in sorted(set(frames)):
        lo, hi = max(0, int(frame)-padding), min(count, int(frame)+padding+2)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(hi, windows[-1][1])
        else:
            windows.append([lo, hi])
    return windows


def _audit(model, q):
    q = model._tensor(q)
    reports = {"frames": model.report(q, .004)}
    if len(q) > 1:
        reports["midpoints"] = model.report((q[:-1]+q[1:])/2, .004)
    return reports


def _bad_frames(model, q, pairs=None):
    with torch.no_grad():
        q = model._tensor(q)
        distance = model.distances(*model.fk(q))
        if pairs is not None:
            distance = distance[:, pairs]
        bad = set(torch.where(distance.amin(-1) < -.001)[0].tolist())
        if len(q) > 1:
            distance = model.distances(*model.fk((q[:-1]+q[1:])/2))
            if pairs is not None:
                distance = distance[:, pairs]
            mid = torch.where(distance.amin(-1) < -.001)[0].tolist()
            bad.update(mid)
            bad.update(i+1 for i in mid)
    return bad


def _spikes(model, raw, repaired, fps):
    if len(raw) < 3:
        return []
    with torch.no_grad():
        feet = [_feet(model, model._tensor(q)) for q in (raw, repaired)]
        velocity = [torch.linalg.vector_norm(p[1:]-p[:-1], dim=-1).amax(-1).numpy() for p in feet]
        acceleration = [torch.linalg.vector_norm(p[2:]-2*p[1:-1]+p[:-2], dim=-1).amax(-1).numpy() for p in feet]
    scale = 30 / fps
    bad = set(np.flatnonzero((velocity[1] > .12*scale) & (velocity[1] > 2.5*velocity[0])).tolist())
    bad.update((np.flatnonzero((acceleration[1] > .15*scale**2) & (acceleration[1] > 2.5*acceleration[0]))+1).tolist())
    return sorted(bad)


def _stable_leg_seed(model, raw, current, legs, fps, iterations, notify):
    """Generate a stable donor only in a detected pathological leg window."""
    current, raw = model._tensor(current), model._tensor(raw)
    fixed = min(6, max(1, (len(current)-1)//4))
    seed = gaussian_filter1d(current[:, legs].numpy(), sigma=2*fps/30, axis=0)
    t = np.linspace(0, 1, len(seed))[:, None]
    bridge = current[fixed, legs].numpy()*(1-t)+current[-fixed-1, legs].numpy()*t
    seed = .2*seed+.8*bridge
    variable = torch.nn.Parameter(model._tensor(seed))
    with torch.no_grad():
        variable[:fixed] = current[:fixed, legs]
        variable[-fixed:] = current[-fixed:, legs]
        foot_target = model._tensor(gaussian_filter1d(_feet(model, raw).numpy(), sigma=1.5*fps/30, axis=0))
        center_y = foot_target[:, :, 1].mean(1)
        half_gap = ((foot_target[:, 0, 1]-foot_target[:, 1, 1]).abs()/2).clamp_min(.09)
        foot_target[:, 0, 1] = center_y+half_gap
        foot_target[:, 1, 1] = center_y-half_gap
    optimizer = torch.optim.Adam([variable], lr=.008)
    fade = torch.sin(torch.linspace(0, np.pi, len(current))).square()
    for step in range(iterations):
        optimizer.param_groups[0]["lr"] = .008*(.02+.98*(1+math.cos(math.pi*step/max(1, iterations-1)))/2)
        optimizer.zero_grad(set_to_none=True)
        q = current.clone()
        q[:, legs] = variable
        _, positions = model.fk(q)
        feet = torch.stack([positions[n] for n in FEET], dim=1)
        velocity = feet[1:]-feet[:-1]
        acceleration = velocity[1:]-velocity[:-1]
        samples = torch.cat([q, (q[:-1]+q[1:])/2])
        distances = model.distances(*model.fk(samples))
        collision = torch.relu(.004-distances).square().sum(-1).mean()
        tracking = (feet-foot_target).square().sum(-1).mean()
        speed = torch.relu(torch.linalg.vector_norm(velocity, dim=-1)-.045*30/fps).square().mean()
        knee_gap = positions["left_knee_link"][:, 1]-positions["right_knee_link"][:, 1]
        ankle_gap = feet[:, 0, 1]-feet[:, 1, 1]
        order = (fade*(torch.relu(.16-knee_gap).square()+torch.relu(.18-ankle_gap).square())).mean()
        joint_acc = (variable[2:]-2*variable[1:-1]+variable[:-2]).square().sum(-1).mean()
        loss = 1000000*collision+2000*tracking+200000*acceleration.square().sum(-1).mean()+1000000*speed+100*joint_acc+.5*(variable-raw[:, legs]).square().sum(-1).mean()+50000*order
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(model.lower[legs], model.upper[legs])
            variable[:fixed] = current[:fixed, legs]
            variable[-fixed:] = current[-fixed:, legs]
        if step % 100 == 0 or step == iterations-1:
            notify((step+1)/iterations)
    result = current.clone()
    result[:, legs] = variable.detach()
    return result.numpy()


def _project_leg_patch(model, current, donor, alpha, legs, rotation, position, iterations, notify):
    current = model._tensor(current)
    desired = current.clone()
    desired[:, legs] += model._tensor(alpha[:, None])*(model._tensor(donor)[:, legs]-current[:, legs])
    fixed = min(2, max(1, (len(current)-1)//4))
    desired[:fixed] = current[:fixed]
    desired[-fixed:] = current[-fixed:]
    rotation, position = model._tensor(rotation), model._tensor(position)
    with torch.no_grad():
        target = (rotation[:, None]@_feet(model, desired)[..., None]).squeeze(-1)+position[:, None]
        target_velocity = target[1:]-target[:-1]
        target_acc = target_velocity[1:]-target_velocity[:-1]
    variable = torch.nn.Parameter(desired[:, legs].clone())
    optimizer = torch.optim.Adam([variable], lr=.004)
    weight = model._tensor(1+30*(1-alpha)**2)
    for step in range(iterations):
        optimizer.param_groups[0]["lr"] = .004*(.02+.98*(1+math.cos(math.pi*step/max(1, iterations-1)))/2)
        optimizer.zero_grad(set_to_none=True)
        q = current.clone()
        q[:, legs] = variable
        feet = (rotation[:, None]@_feet(model, q)[..., None]).squeeze(-1)+position[:, None]
        velocity = feet[1:]-feet[:-1]
        acceleration = velocity[1:]-velocity[:-1]
        samples = torch.cat([q, (q[:-1]+q[1:])/2])
        distances = model.distances(*model.fk(samples))
        collision = torch.relu(.004-distances).square().sum(-1).mean()
        correction = variable-desired[:, legs]
        curvature = (correction[2:]-2*correction[1:-1]+correction[:-2]).square().sum(-1).mean()
        loss = 1000000*collision+10000*(weight[:, None]*(feet-target).square().sum(-1)).mean()+10*(weight*correction.square().sum(-1)).mean()+40000*(velocity-target_velocity).square().sum(-1).mean()+40000*(acceleration-target_acc).square().sum(-1).mean()+100*curvature
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            variable.clamp_(model.lower[legs], model.upper[legs])
            variable[:fixed] = current[:fixed, legs]
            variable[-fixed:] = current[-fixed:, legs]
        if step % 100 == 0 or step == iterations-1:
            notify((step+1)/iterations)
    result = current.clone()
    result[:, legs] = variable.detach()
    return result.numpy()


def repair_reference_h5(source, output, robot_usd, progress=None, iterations=600):
    """Repair a recorded raw H5. Never overwrites source or an existing output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output:
        raise FileExistsError(f"Output already exists: {output}")
    if iterations < 1:
        raise ValueError("Iterations must be positive")
    torch.set_num_threads(4)
    def notify(stage, fraction=0.0, detail=""):
        if progress is not None:
            progress(stage, float(fraction), detail)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with h5py.File(source) as f:
        names = list(f["joint_names"].asstr()[:])
        delta = f["joint_pos_delta"][:]
        frames = f["frames"][:]
        fps = float(f.attrs["fps"])
        baseline = np.asarray(f.attrs.get("record_baseline_joint_pos", playback_baseline(names, ROOT/"source/my_task/g1_replay_env_cfg.py")), dtype=np.float32)
        anchor = np.asarray(f.attrs.get("record_root_anchor_quat_wxyz", [.70710678, 0, 0, .70710678]))
        root_quat, root_pos, root_valid = f["root_quat_delta_wxyz"][:], f["root_pos_delta"][:], f["root_valid"][:].astype(bool)
    if delta.ndim != 2 or len(delta) < 1 or delta.shape[1] != len(names) or len(set(names)) != len(names) or baseline.shape != (len(names),) or not np.isfinite(delta).all() or not np.isfinite(baseline).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid recorded reference trajectory")
    if frames.shape != (len(delta),) or root_quat.shape != (len(delta), 4) or root_pos.shape != (len(delta), 3) or root_valid.shape != (len(delta),) or anchor.shape != (4,):
        raise ValueError("Inconsistent recorded reference channel dimensions")
    if not np.isfinite(root_quat[root_valid]).all() or not np.isfinite(root_pos[root_valid]).all() or not np.isfinite(anchor).all() or np.linalg.norm(anchor) < 1e-8 or np.any(np.linalg.norm(root_quat[root_valid], axis=1) < 1e-8):
        raise ValueError("Invalid recorded root transforms")
    notify("geometry", 0, "Reading robot geometry")
    model = G1CollisionModel(Path(robot_usd), names, "cpu")
    missing = sorted({j.name for j in model.joints if j.axis} - set(names))
    if missing:
        raise ValueError(f"Recorded reference is missing G1 joints: {missing}")
    raw = delta+baseline
    padded = np.concatenate([raw, np.repeat(raw[-1:], max(0, 3-len(raw)), axis=0)])
    before = _audit(model, raw)
    notify("collision", 0, "Repairing self-collision while preserving motion")
    if any(r["penetrating_frames"] for r in before.values()) or (model._tensor(raw[:, model.adjustable]) < model.lower[model.adjustable]).any() or (model._tensor(raw[:, model.adjustable]) > model.upper[model.adjustable]).any():
        repaired, initial_report = repair_poses(model, padded, iterations=iterations, preserve_motion=True, progress=lambda fraction, _entry: notify("collision", fraction))
        repaired = repaired[:len(raw)]
    else:
        repaired, initial_report = raw.copy(), {"skipped": "input already passes geometry and limits"}
    # Polish remaining frame/midpoint violations locally, never by dance name.
    collision_windows = []
    for attempt in range(2):
        bad = _bad_frames(model, repaired)
        if not bad:
            break
        windows = _windows(bad, len(raw), max(8, round(14*fps/30)))
        for index, (lo, hi) in enumerate(windows):
            notify("polish", index/max(1, len(windows)), f"Collision window {lo}-{hi-1}")
            current, reference = repaired[lo:hi], raw[lo:hi]
            if len(current) < 3:
                current = np.concatenate([current, np.repeat(current[-1:], 3-len(current), axis=0)])
                reference = np.concatenate([reference, np.repeat(reference[-1:], 3-len(reference), axis=0)])
            fixed = 0 if lo == 0 or hi == len(raw) else min(3, (len(current)-1)//3)
            projected, _ = repair_poses(model, reference, initial_pose=current, iterations=450, fixed_endpoint_frames=fixed, preserve_motion=True, collision_strength=1000000, progress=lambda f, _e: notify("polish", (index+f)/len(windows)))
            repaired[lo:hi] = projected[:hi-lo]
            collision_windows.append([lo, hi])
    # A collision-free pair of endpoints can still interpolate through the body.
    # Re-seed only unresolved windows to choose a coherent route around it.
    for attempt in range(2):
        bad = _bad_frames(model, repaired)
        if not bad:
            break
        windows = _windows(bad, len(raw), max(16, round(16*fps/30)))
        arm_links = set(sum(model.arm_names.values(), []))
        arm_pairs = [i for i, (a, b) in enumerate(model.pairs) if model.shapes[a].name in arm_links or model.shapes[b].name in arm_links]
        arm_bad = _bad_frames(model, repaired, arm_pairs)
        for index, (lo, hi) in enumerate(windows):
            current = repaired[lo:hi].copy()
            if len(current) < 3:
                continue
            smooth = gaussian_filter1d(current[:, model.adjustable], sigma=(2+attempt)*fps/30, axis=0)
            blend = np.zeros(hi-lo, dtype=np.float32)
            for frame in bad:
                if lo <= frame < hi:
                    distance = np.abs(np.arange(lo, hi)-frame)/max(4, round(8*fps/30))
                    blend = np.maximum(blend, np.clip(1-distance, 0, 1))
            seed = current.copy()
            seed[:, model.adjustable] += blend[:, None]*(smooth-current[:, model.adjustable])
            if any(lo <= frame < hi for frame in arm_bad):
                # A neutral, outward arm seed can leave a trapped interpolation
                # branch. This is only a solver seed; reference tracking pulls
                # it back toward the original dance after collision projection.
                arms = [i for i in model.adjustable if not any(s in names[i] for s in ("hip_", "knee_", "ankle_"))]
                seed[:, arms] += blend[:, None]*(baseline[arms]-seed[:, arms])
            fixed = 0 if lo == 0 or hi == len(raw) else min(3, (len(current)-1)//3)
            seed[:fixed] = current[:fixed]
            if fixed:
                seed[-fixed:] = current[-fixed:]
            notify("polish", index/max(1, len(windows)), f"Repairing frame transition {lo}-{hi-1}")
            projected, _ = repair_poses(model, raw[lo:hi], initial_pose=seed, iterations=700, fixed_endpoint_frames=fixed, preserve_motion=True, collision_strength=2000000, learning_rate=.012, progress=lambda f, _e: notify("polish", (index+f)/len(windows)))
            repaired[lo:hi] = projected
            collision_windows.append([lo, hi])
    legs = [i for i, n in enumerate(names) if any(s in n for s in ("hip_", "knee_", "ankle_"))]
    pre_spike = repaired.copy()
    spike_frames = _spikes(model, raw, repaired, fps)
    # Only unresolved leg/leg crossings call for a leg donor. Arms are fixed
    # during this stage, so unrelated failures must not be mislabeled leg spikes.
    with torch.no_grad():
        leg_pairs = [i for i, (a, b) in enumerate(model.pairs) if model.shapes[a].name in sum(model.leg_names.values(), []) and model.shapes[b].name in sum(model.leg_names.values(), [])]
        leg_bad = torch.where(model.distances(*model.fk(model._tensor(repaired)))[:, leg_pairs].amin(-1) < -.001)[0].tolist()
        if len(repaired) > 1:
            mid = torch.where(model.distances(*model.fk(model._tensor((repaired[:-1]+repaired[1:])/2)))[:, leg_pairs].amin(-1) < -.001)[0].tolist()
            leg_bad += mid + [i+1 for i in mid]
    spike_frames = sorted(set(spike_frames) | set(leg_bad))
    cores = _windows(spike_frames, len(raw), max(2, round(2*fps/30)))
    padding = max(16, round(16*fps/30))
    patch_windows = _windows([i for lo, hi in cores for i in (lo, hi-2)], len(raw), padding)
    alpha = np.zeros(len(raw), dtype=np.float32)
    for lo, hi in cores:
        alpha[lo:hi] = 1
        for begin, end, rising in ((max(0, lo-padding), lo, True), (hi, min(len(raw), hi+padding), False)):
            t = np.linspace(0, 1, end-begin, endpoint=not rising)
            smooth = t*t*(3-2*t)
            alpha[begin:end] = np.maximum(alpha[begin:end], smooth if rising else 1-smooth)
    for i in range(len(root_valid)):
        if not root_valid[i]:
            root_quat[i] = root_quat[i-1] if i else [1, 0, 0, 0]
            root_pos[i] = root_pos[i-1] if i else [0, 0, 0]
    rotation = (Rotation.from_quat(root_quat[:, [1, 2, 3, 0]])*Rotation.from_quat(anchor[[1, 2, 3, 0]])).as_matrix()
    for index, (lo, hi) in enumerate(patch_windows):
        first, last = max(0, lo-padding), min(len(raw), hi+padding)
        if last-first < 3:
            continue
        notify("spikes", index/max(1, len(patch_windows)), f"Repairing isolated spike {lo}-{hi-1}")
        donor = _stable_leg_seed(model, raw[first:last], repaired[first:last], legs, fps, 1000, lambda f: notify("spikes", (index+.6*f)/len(patch_windows)))
        repaired[lo:hi] = _project_leg_patch(model, repaired[lo:hi], donor[lo-first:hi-first], alpha[lo:hi], legs, rotation[lo:hi], root_pos[lo:hi], 1000, lambda f: notify("spikes", (index+.6+.4*f)/len(patch_windows)))
    allowed = np.zeros(len(raw), dtype=bool)
    for lo, hi in patch_windows:
        allowed[lo:hi] = True
    assert np.array_equal(pre_spike[~allowed], repaired[~allowed])
    untouched = [i for i, n in enumerate(names) if i not in model.adjustable]
    result = delta.copy()
    result[:, model.adjustable] = (repaired[:, model.adjustable]-baseline[model.adjustable]).astype(delta.dtype)
    assert np.array_equal(result[:, untouched], delta[:, untouched])
    notify("validate", 0, "Checking complete motion and splice boundaries")
    after = _audit(model, result+baseline)
    remaining_spikes = _spikes(model, raw, result+baseline, fps)
    report = {"schema": "g1_collision_reference_v1", "source_sha256": source_hash, "frames": len(raw), "fps": fps, "validation": "selected USD envelopes at frames and midpoints, tolerance 1mm", "before": before, "after": after, "initial_repair": initial_report, "collision_windows": collision_windows, "spike_frames": spike_frames, "spike_windows": patch_windows, "remaining_spike_frames": remaining_spikes, "spike_changed_frames": int(np.any(pre_spike != repaired, axis=1).sum()), "normal_spike_stage_frames_preserved": int((~allowed).sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".collision.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(r["penetrating_frames"] for r in after.values()) or remaining_spikes:
        np.savez_compressed(output.with_suffix(".candidate.npz"), joint_pos=repaired, baseline=baseline)
        raise RuntimeError(f"Reference did not pass collision/spike validation. See {report_path}")
    if not np.isfinite(result).all() or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Non-finite result or source changed during repair")
    fd, temporary = tempfile.mkstemp(prefix=".collision-", suffix=".h5", dir=output.parent)
    os.close(fd)
    reserved = False
    try:
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r+") as f:
            f["joint_pos_delta"][:] = result
            f.attrs["without_collision_validated"] = True
            f.attrs["collision_validation"] = report["validation"]
            f.attrs["collision_reference_source_sha256"] = source_hash
        # Exclusive reservation protects existing outputs; replace is atomic.
        with output.open("xb"):
            pass
        reserved = True
        os.replace(temporary, output)
        reserved = False
    finally:
        if reserved:
            output.unlink(missing_ok=True)
        if os.path.exists(temporary):
            os.unlink(temporary)
    notify("done", 1, f"Saved {output.name}")
    return report
