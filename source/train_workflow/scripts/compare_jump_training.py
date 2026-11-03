"""Compare pre/post specialist telemetry using identical ground-only contacts."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    rows = []
    for path in sorted(root.glob("episode_*")):
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["contact_source"] != "ground-only filtered ankle normal forces":
            raise ValueError("Comparison requires the same ground-only contact measurement")
        archive = np.load(path / "samples.npz")
        d = dict(zip(archive["columns"], archive["values"].T))
        mask = (d["ref_time_s"] >= 17.08) & (d["ref_time_s"] < 17.30)
        if mask.sum() < 10:
            raise ValueError("Missing intended flight interval")
        air = (d["both_feet_no_contact"] > 0) & mask
        changes = np.diff(np.r_[False, air, False].astype(int))
        durations = (np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)) * np.median(np.diff(d["time_s"]))
        rows.append(dict(seed=metadata["seed"], episode=str(path), end_reason=metadata["end_reason"],
                         phase_no_contact_fraction=float(d["both_feet_no_contact"][mask].mean()),
                         phase_longest_no_contact_s=float(durations.max()) if len(durations) else 0.,
                         phase_max_root_z_m=float(d["root_z_m"][mask].max()),
                         phase_height_rmse_m=float(np.sqrt(np.mean((d["root_z_m"][mask]-d["ref_root_z_m"][mask])**2)))))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-label", default="Before: warm start")
    parser.add_argument("--after-label", default="After: 300 PPO iterations")
    args = parser.parse_args()
    before, after = summarize(args.before), summarize(args.after)
    if not before or not after or [x["seed"] for x in before] != [x["seed"] for x in after]:
        raise ValueError("Need matching nonempty seed sets")
    result = dict(before=before, after=after,
                  note="No-contact and root height are diagnostic metrics, not proof of a visually correct jump.")
    for key in ("phase_no_contact_fraction", "phase_longest_no_contact_s", "phase_max_root_z_m", "phase_height_rmse_m"):
        result[key] = dict(before=float(np.mean([x[key] for x in before])),
                           after=float(np.mean([x[key] for x in after])))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for column, (label, rows) in enumerate(((args.before_label, before), (args.after_label, after))):
        archive = np.load(Path(rows[0]["episode"]) / "samples.npz")
        data = dict(zip(archive["columns"], archive["values"].T))
        seconds = data["time_s"] + 15.7
        for key in ("ref_root_z_m", "root_z_m"):
            axes[0, column].plot(seconds, data[key], label=key)
        for side in ("left", "right"):
            axes[1, column].plot(seconds, data[f"{side}_contact_N"], label=f"{side} ground contact N")
        axes[0, column].set_title(label + " / seed42")
        for ax in axes[:, column]:
            ax.axvspan(17.08, 17.30, color="green", alpha=.12, label="intended flight/close")
            ax.set_xlim(16.5, 17.6)
            ax.legend(fontsize=8)
            ax.grid(alpha=.25)
        axes[1, column].set_xlabel("original motion seconds")
    axes[0, 0].set_ylabel("root height (m)")
    axes[1, 0].set_ylabel("ground normal force (N)")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key not in ("before", "after")}, indent=2))
