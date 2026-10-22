"""Window-based evidence summaries. Candidates are not jump/failure labels."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def longest_run(mask, dt):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    durations = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * dt
    return float(durations.max()) if len(durations) else 0.


def window_metrics(data, start, end):
    t = data["time_s"]
    keep = (t >= start) & (t <= end)
    if keep.sum() < 10:
        raise ValueError(f"Not enough samples in {start}..{end}")
    dt = float(np.median(np.diff(t[keep])))
    actual, ref = data["root_z_m"][keep], data["ref_root_z_m"][keep]
    error = actual - ref
    result = dict(start_s=float(start), end_s=float(end), samples=int(keep.sum()),
                  height_rmse_m=float(np.sqrt(np.mean(error ** 2))),
                  height_error_p95_m=float(np.quantile(np.abs(error), .95)),
                  actual_height_range_m=float(np.ptp(actual)), reference_height_range_m=float(np.ptp(ref)),
                  both_feet_no_contact_fraction=float(data["both_feet_no_contact"][keep].mean()),
                  longest_no_contact_s=longest_run(data["both_feet_no_contact"][keep] > 0, dt))
    # Fixed overlap for all shifts. Positive lag means actual follows reference.
    # Held reference timing itself introduces uncertainty up to one control step.
    margin = int(round(.3 / dt))
    result.update(lag_candidate_s=None, lag_correlation=None, aligned_height_rmse_m=None,
                  same_overlap_unshifted_rmse_m=None)
    if len(actual) > 2 * margin + 10 and np.std(ref) > .003:
        base = ref[margin:-margin]
        result["same_overlap_unshifted_rmse_m"] = float(np.sqrt(np.mean((actual[margin:-margin] - base) ** 2)))
        scores = []
        for shift in range(-margin, margin + 1):
            shifted = actual[margin + shift:len(actual) - margin + shift]
            corr = float(np.corrcoef(base, shifted)[0, 1]) if np.std(shifted) > 1e-8 else -1.
            scores.append((corr, shift, float(np.sqrt(np.mean((shifted - base) ** 2)))))
        corr, shift, rmse = max(scores, key=lambda item: item[0])
        result["lag_correlation"] = corr
        if corr >= .5 and abs(shift) < margin:
            result.update(lag_candidate_s=shift * dt, aligned_height_rmse_m=rmse)
    joints = [key[:-len("_ref_rad")] for key in data if key.endswith("_ref_rad")]
    if joints:
        residual = np.stack([data[n + "_target_rad"][keep] - data[n + "_ref_rad"][keep] for n in joints])
        tracking = np.stack([data[n + "_actual_rad"][keep] - data[n + "_target_rad"][keep] for n in joints])
        clipping = np.stack([np.abs(data[n + "_computed_Nm"][keep] - data[n + "_applied_Nm"][keep]) > 1.
                             for n in joints])
        result.update(target_reference_rms_rad=float(np.sqrt(np.mean(residual ** 2))),
                      actual_target_rms_rad=float(np.sqrt(np.mean(tracking ** 2))),
                      torque_clipping_fraction=float(clipping.mean()))
    return result


def analyze(data, windows=None):
    t = data["time_s"]
    scanned = [window_metrics(data, float(start), float(start + 2.))
               for start in np.arange(t[0], t[-1] - 2., .5)]
    # Rank sustained segment errors, never reference height maxima. A candidate
    # must exceed both the run-relative percentile and an absolute noise floor.
    candidates = []
    if scanned:
        height_threshold = max(.02, float(np.quantile([r["height_rmse_m"] for r in scanned], .9)))
        tracking_threshold = max(.04, float(np.quantile([r.get("actual_target_rms_rad", 0) for r in scanned], .9)))
        for row in scanned:
            reasons = []
            if row["height_rmse_m"] >= height_threshold:
                reasons.append("sustained_height_tracking_error")
            if row.get("actual_target_rms_rad", 0) >= tracking_threshold:
                reasons.append("joint_target_tracking_error")
            if reasons:
                row["candidate_reasons"] = reasons
                row["ranking_score"] = max(row["height_rmse_m"] / height_threshold,
                                           row.get("actual_target_rms_rad", 0) / tracking_threshold)
                candidates.append(row)
        selected = []
        for row in sorted(candidates, key=lambda r: r["ranking_score"], reverse=True):
            if all(abs(row["start_s"] - other["start_s"]) >= 2. for other in selected):
                selected.append(row)
            if len(selected) >= 10:
                break
        candidates = selected
    annotated = []
    for label, start, end in windows or []:
        annotated.append(dict(label=label, source="user_selected_review_window",
                              **window_metrics(data, start, end)))
    return dict(method="2s windows / 0.5s stride; sustained tracking errors, not height peaks",
                limitations=["Candidates are review priorities, not detected jumps or confirmed failures.",
                             "No reference contact or foot trajectory is recorded: expected flight cannot be inferred.",
                             "No-contact includes all ankle-link normal contacts; it is not proof of flight.",
                             "Lag is a correlation candidate, can be ambiguous in periodic motion; reference is held per control step.",
                             "Target-reference RMS has no sign/direction: it cannot prove residual cancellation.",
                             "Torque clipping fraction alone does not establish motor capacity as cause."],
                annotated_windows=annotated, review_candidates=candidates)


def export_analysis(data, directory, windows=None):
    directory = Path(directory)
    result = analyze(data, windows)
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Segment review", "", result["method"], "",
             "Candidates are not confirmed jumps/failures. User labels are observations, not detector conclusions.", "",
             "| Window | Z RMSE (cm) | Candidate lag (ms) | Target/ref RMS (rad) | Actual/target RMS (rad) |",
             "|---|---:|---:|---:|---:|"]
    reviewed = result["annotated_windows"] or result["review_candidates"][:4]
    for row in reviewed:
        lag = row["lag_candidate_s"]
        lag_text = "unresolved" if lag is None else f"{lag * 1000:.0f}"
        lines.append(f"| {row.get('label', 'candidate')} {row['start_s']:.2f}–{row['end_s']:.2f}s | "
                     f"{100 * row['height_rmse_m']:.2f} | {lag_text} | "
                     f"{row.get('target_reference_rms_rad', 0):.3f} | {row.get('actual_target_rms_rad', 0):.3f} |")
    lines += ["", *["- " + text for text in result["limitations"]]]
    (directory / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if reviewed:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(reviewed), 3, figsize=(17, 3.1 * len(reviewed)), squeeze=False)
        t = data["time_s"]
        for axrow, row in zip(axes, reviewed):
            keep = (t >= row["start_s"]) & (t <= row["end_s"])
            for key in ("root_z_m", "ref_root_z_m"):
                axrow[0].plot(t[keep], data[key][keep], label=key)
            for side in ("left", "right"):
                axrow[1].plot(t[keep], data[f"{side}_contact_N"][keep], label=side + " contact N")
            axrow[2].plot(t[keep], 100 * (data["root_z_m"][keep] - data["ref_root_z_m"][keep]), label="height error cm")
            axrow[0].set_title(row.get("label", "review candidate"))
            for ax in axrow:
                ax.legend(fontsize=8)
                ax.grid(alpha=.25)
                ax.set_xlabel("rollout seconds")
        fig.tight_layout()
        fig.savefig(directory / "segment_review.png", dpi=140)
        plt.close(fig)
    return result
