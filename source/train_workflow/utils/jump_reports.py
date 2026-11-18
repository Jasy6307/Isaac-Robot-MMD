"""Consolidated jump reports; historical measurement rules retained."""
from __future__ import annotations
from source.paths import REPO_ROOT
from source.train_workflow.utils.jump_analysis import load_telemetry, new_diagnostic_output, require_new_files
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right")
          for joint in ("hip_pitch", "knee", "ankle_pitch")]


DECOMPOSITION_WINDOWS = [("observed issue: 17s", 16, 18), ("observed issue: 21s", 20, 22),
           ("observed issue: 48s", 47, 49), ("observed acceptable: 53s", 52, 54)]


def decompose_jump_tracking_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    output = new_diagnostic_output('jump_decomposition', args.output_dir)
    require_new_files(output/'decomposition.json', *(output/f'decomposition_{start}_{end}.png' for _, start, end in DECOMPOSITION_WINDOWS))
    data = load_telemetry(args.report)
    t = data["time_s"]
    summaries = []
    for label, start, end in DECOMPOSITION_WINDOWS:
        mask = (t >= start) & (t <= end)
        ascending = mask & (data["ref_root_vz_m_s"] > .15)
        fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
        for side, column in (("left", 0), ("right", 1)):
            for row, joint in enumerate(("hip_pitch", "knee", "ankle_pitch")):
                name = f"{side}_{joint}_joint"
                for kind in ("ref", "target", "actual"):
                    axes[row, column].plot(t[mask], data[f"{name}_{kind}_rad"][mask], label=kind)
                axes[row, column].set_ylabel(name + " (rad)")
            axes[3, column].plot(t[mask], data[f"{side}_contact_N"][mask], label="ankle contact N")
            for key in ("ref_root_z_m", "root_z_m"):
                axes[4, column].plot(t[mask], data[key][mask], label=key)
        for ax in axes.flat:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(label + ": held reference / commanded target / measured angle")
        fig.tight_layout()
        fig.savefig(output / f"decomposition_{start}_{end}.png", dpi=140)
        plt.close(fig)
        summary = dict(label=label, start_s=start, end_s=end, ascent_samples=int(ascending.sum()), joints={})
        for name in JOINTS:
            ref = data[f"{name}_ref_rad"]
            target = data[f"{name}_target_rad"]
            actual = data[f"{name}_actual_rad"]
            summary["joints"][name] = dict(
                ascent_mean_target_minus_ref_rad=float((target-ref)[ascending].mean()),
                ascent_mean_actual_minus_target_rad=float((actual-target)[ascending].mean()),
                ascent_actual_target_rms_rad=float(np.sqrt(np.mean((actual-target)[ascending]**2))),
                window_target_reference_rms_rad=float(np.sqrt(np.mean((target-ref)[mask]**2))),
                window_actual_target_rms_rad=float(np.sqrt(np.mean((actual-target)[mask]**2))),
            )
        summaries.append(summary)
    snapshots = []
    for timestamp in (16.5, 16.7, 16.95, 17.14, 20.5, 20.81, 47.5, 47.71, 48.2, 52.7, 52.91):
        index = int(np.argmin(np.abs(t-timestamp)))
        snapshots.append(dict(time_s=float(t[index]),
            ref_z=float(data["ref_root_z_m"][index]), actual_z=float(data["root_z_m"][index]),
            contact_N=[float(data[f"{side}_contact_N"][index]) for side in ("left", "right")],
            joint_angles={name: [float(data[f"{name}_{kind}_rad"][index]) for kind in ("ref", "target", "actual")]
                          for name in JOINTS}))
    result = dict(windows=summaries, snapshots=snapshots,
                  note="Ascent means reference root vz > 0.15m/s, not confirmed physical takeoff. Angles are radians.")
    (output / "decomposition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print('OUTPUT', output, file=sys.stderr)
    print(json.dumps(result, indent=2))


COLORS = {"prepare_open": "#b8d9f3", "push_off": "#ffcf83",
          "planned_flight_close": "#bbdfaf", "landing_recovery": "#d6c2e8"}


def mark_jump_phases_main(argv=None):
    root = Path(REPO_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument("--annotations", type=Path,
                        default=root / "source/train_workflow/annotations/v7_jump_phases.review.json")
    args = parser.parse_args(argv)
    output = new_diagnostic_output('jump_phases', args.output_dir)
    require_new_files(output/'phase_review.json', output/'phase_review.png')
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    metadata = json.loads((args.report / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dance"] != annotations["dance"]:
        raise ValueError("Annotation dance does not match report")
    d = load_telemetry(args.report)
    t = d["time_s"]
    events = annotations["events"]
    fig, axes = plt.subplots(4, len(events), figsize=(19, 12), squeeze=False)
    summaries = []
    for col, event in enumerate(events):
        start, end = event["context"]
        mask = (t >= start) & (t <= end)
        if not mask.any():
            raise ValueError(f"No data in context {event['id']}")
        for key in ("ref_root_z_m", "root_z_m"):
            axes[0, col].plot(t[mask], d[key][mask], label=key)
        for side in ("left", "right"):
            for kind, style in (("ref", "--"), ("actual", "-")):
                key = f"{side}_hip_roll_joint_{kind}_rad"
                axes[1, col].plot(t[mask], d[key][mask], style, label=f"{side} {kind}")
            axes[2, col].plot(t[mask], d[f"{side}_contact_N"][mask], label=f"{side} normal contact N")
            axes[3, col].step(t[mask], d[f"{side}_contact"][mask] + (1.3 if side == "right" else 0),
                              where="post", label=f"{side} contact")
        axes[3, col].step(t[mask], d["both_feet_no_contact"][mask] + 2.6,
                          where="post", color="black", label="both no-contact candidate")
        axes[0, col].set_title(event["id"])
        for ax, label in zip(axes[:, col], ("root Z (m)", "hip roll (rad; not foot spacing)",
                                           "ankle normal contact (N)", "contact indicators (offset lanes)")):
            for phase in event["phases"]:
                if not start <= phase["start"] < phase["end"] <= end:
                    raise ValueError("Phase outside context")
                ax.axvspan(phase["start"], phase["end"], color=COLORS[phase["phase"]], alpha=.4)
            ax.set_xlim(start, end)
            ax.set_ylabel(label)
            ax.set_xlabel("rollout seconds")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=.2)
        for phase in event["phases"]:
            samples = (t >= phase["start"]) & (t < phase["end"])
            summaries.append(dict(event=event["id"], **phase,
                                  actual_no_contact_fraction=float(d["both_feet_no_contact"][samples].mean()),
                                  actual_left_contact_fraction=float(d["left_contact"][samples].mean()),
                                  actual_right_contact_fraction=float(d["right_contact"][samples].mean())))
    fig.suptitle("PROVISIONAL intended phases vs measured contact — not confirmed flight labels", fontsize=15)
    fig.legend(handles=[Patch(color=color, alpha=.5, label=phase) for phase, color in COLORS.items()],
               loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output / "phase_review.png", dpi=150)
    plt.close(fig)
    result = dict(annotations=annotations, phase_contact_summary=summaries)
    (output / "phase_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print('OUTPUT', output, file=sys.stderr)
    print(json.dumps(summaries, indent=2))


def plot_jump_reference_change_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=Path)
    parser.add_argument('shaped', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    require_new_files(args.output, args.output.with_suffix('.json'))
    old, new = (load_telemetry(args.original), load_telemetry(args.shaped))
    np.testing.assert_allclose(old['ref_time_s'], new['ref_time_s'], atol=1e-06)
    np.testing.assert_array_equal(old['ref_root_z_m'], new['ref_root_z_m'])
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    t = new['ref_time_s']
    changes = {}
    for col, side in enumerate(('left', 'right')):
        for row, joint in enumerate(('hip_roll', 'knee', 'hip_pitch', 'ankle_pitch')):
            name = f'{side}_{joint}_joint_ref_rad'
            axes[row, col].plot(t, old[name], label='original reference')
            axes[row, col].plot(t, new[name], label='V2 reference')
            axes[row, col].axvspan(16.93, 17.08, color='orange', alpha=0.15, label='push-off')
            axes[row, col].axvspan(17.08, 17.3, color='green', alpha=0.1, label='intended flight/close')
            axes[row, col].set_ylabel(f'{side} {joint} (rad)')
            axes[row, col].set_xlim(16.65, 17.45)
            axes[row, col].legend(fontsize=7)
            axes[row, col].grid(alpha=0.2)
            changes[name] = float(np.max(np.abs(new[name] - old[name])))
    for ax in axes[-1]:
        ax.set_xlabel('original motion seconds')
    fig.suptitle('Experimental leg-only reference change; root-height reference unchanged')
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    print(json.dumps(changes, indent=2))
