"""Plot motion-start weights as a curve chart."""

from __future__ import annotations

import argparse
import csv
import json
import os


def _load_csv_weights(path: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    weights: list[float] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            weights.append(float(row["weight"]))
    if not steps:
        raise ValueError(f"No rows found in CSV: {path}")
    return steps, weights


def _load_json_meta(path: str) -> tuple[float | None, list[tuple[int, int]]]:
    if not path:
        return None, []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    threshold = payload.get("threshold")
    ranges_raw = payload.get("high_ranges", [])
    ranges: list[tuple[int, int]] = []
    for item in ranges_raw:
        ranges.append((int(item["start"]), int(item["end"])))
    return (float(threshold) if threshold is not None else None), ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot motion-start weight curve.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV: step,weight")
    parser.add_argument("--csv2", type=str, default=None, help="Optional second CSV for comparison")
    parser.add_argument("--json", type=str, default=None, help="Optional JSON summary for threshold/ranges")
    parser.add_argument("--title", type=str, default="Motion Start Weights", help="Chart title")
    parser.add_argument("--label1", type=str, default="weight-1", help="Legend label for --csv")
    parser.add_argument("--label2", type=str, default="weight-2", help="Legend label for --csv2")
    parser.add_argument(
        "--x_divisor",
        type=float,
        default=1.0,
        help="Display x as step/x_divisor (e.g. 2.0 for 60fps->30fps frame index).",
    )
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    csv_path = os.path.abspath(args.csv)
    json_path = os.path.abspath(args.json) if args.json else None
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    steps, weights = _load_csv_weights(csv_path)
    threshold, ranges = _load_json_meta(json_path) if json_path else (None, [])
    x_divisor = float(args.x_divisor)
    if x_divisor <= 0.0:
        raise ValueError(f"--x_divisor must be > 0, got {x_divisor}")
    x_steps = [float(s) / x_divisor for s in steps]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.plot(x_steps, weights, linewidth=1.8, label=args.label1)
    if args.csv2:
        csv2_path = os.path.abspath(args.csv2)
        steps2, weights2 = _load_csv_weights(csv2_path)
        if len(steps2) != len(steps):
            raise ValueError(
                f"--csv and --csv2 length mismatch: {len(steps)} vs {len(steps2)}"
            )
        x_steps2 = [float(s) / x_divisor for s in steps2]
        ax.plot(x_steps2, weights2, linewidth=1.6, linestyle="--", label=args.label2)
    ax.set_title(args.title)
    if abs(x_divisor - 1.0) < 1e-8:
        ax.set_xlabel("Start step")
    else:
        ax.set_xlabel(f"Start frame (step/{x_divisor:g})")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.9, 3.1)
    ax.grid(True, alpha=0.25)

    if threshold is not None:
        ax.axhline(y=threshold, linestyle="--", linewidth=1.2, label=f"threshold={threshold:.3f}")

    for i, (st, ed) in enumerate(ranges):
        ax.axvspan(
            float(st) / x_divisor,
            float(ed) / x_divisor,
            alpha=0.16,
            color="tab:red",
            label="high-weight range" if i == 0 else None,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[INFO] Saved plot: {out_path}")


if __name__ == "__main__":
    main()

