"""Decompose an existing jump report into reference/target/actual leg motion."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right")
          for joint in ("hip_pitch", "knee", "ankle_pitch")]
WINDOWS = [("observed issue: 17s", 16, 18), ("observed issue: 21s", 20, 22),
           ("observed issue: 48s", 47, 49), ("observed acceptable: 53s", 52, 54)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    arrays = np.load(args.report / "samples.npz")
    data = dict(zip(arrays["columns"], arrays["values"].T))
    t = data["time_s"]
    summaries = []
    for label, start, end in WINDOWS:
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
        fig.savefig(args.report / f"decomposition_{start}_{end}.png", dpi=140)
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
    (args.report / "decomposition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
