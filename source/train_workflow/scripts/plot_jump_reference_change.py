"""Compare recorded original and V2 reference tables on the same motion clock."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("shaped", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    def load(path):
        archive = np.load(path / "samples.npz")
        return dict(zip(archive["columns"], archive["values"].T))
    old, new = load(args.original), load(args.shaped)
    np.testing.assert_allclose(old["ref_time_s"], new["ref_time_s"], atol=1.e-6)
    np.testing.assert_array_equal(old["ref_root_z_m"], new["ref_root_z_m"])
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    t = new["ref_time_s"]
    changes = {}
    for col, side in enumerate(("left", "right")):
        for row, joint in enumerate(("hip_roll", "knee", "hip_pitch", "ankle_pitch")):
            name = f"{side}_{joint}_joint_ref_rad"
            axes[row, col].plot(t, old[name], label="original reference")
            axes[row, col].plot(t, new[name], label="V2 reference")
            axes[row, col].axvspan(16.93, 17.08, color="orange", alpha=.15, label="push-off")
            axes[row, col].axvspan(17.08, 17.30, color="green", alpha=.1, label="intended flight/close")
            axes[row, col].set_ylabel(f"{side} {joint} (rad)")
            axes[row, col].set_xlim(16.65, 17.45)
            axes[row, col].legend(fontsize=7)
            axes[row, col].grid(alpha=.2)
            changes[name] = float(np.max(np.abs(new[name] - old[name])))
    for ax in axes[-1]:
        ax.set_xlabel("original motion seconds")
    fig.suptitle("Experimental leg-only reference change; root-height reference unchanged")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    print(json.dumps(changes, indent=2))
