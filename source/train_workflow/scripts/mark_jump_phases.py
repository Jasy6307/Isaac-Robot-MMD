"""Overlay review-only intended phases and actual contact on existing telemetry."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {"prepare_open": "#b8d9f3", "push_off": "#ffcf83",
          "planned_flight_close": "#bbdfaf", "landing_recovery": "#d6c2e8"}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--annotations", type=Path,
                        default=root / "source/train_workflow/annotations/v7_jump_phases.review.json")
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    metadata = json.loads((args.report / "metadata.json").read_text(encoding="utf-8"))
    if metadata["dance"] != annotations["dance"]:
        raise ValueError("Annotation dance does not match report")
    arrays = np.load(args.report / "samples.npz")
    d = dict(zip(arrays["columns"], arrays["values"].T))
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
    fig.savefig(args.report / "phase_review.png", dpi=150)
    plt.close(fig)
    result = dict(annotations=annotations, phase_contact_summary=summaries)
    (args.report / "phase_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
