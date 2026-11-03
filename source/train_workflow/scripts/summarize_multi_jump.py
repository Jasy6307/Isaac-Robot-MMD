"""Summarize three annotated jump windows in clip/full evaluation reports."""
import argparse
import json
from pathlib import Path
import numpy as np

WINDOWS = (("17s",17.08,17.30),("21s",20.75,20.95),("48s",48.18,48.38))


def summarize(root):
    result = {name: [] for name, _, _ in WINDOWS}
    for path in sorted(root.glob("episode_*")):
        archive = np.load(path / "samples.npz")
        d = dict(zip(archive["columns"], archive["values"].T))
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["contact_source"] != "ground-only filtered ankle normal forces":
            raise ValueError("Ground-only telemetry required")
        for name, start, end in WINDOWS:
            mask = (d["ref_time_s"] >= start) & (d["ref_time_s"] < end)
            if mask.sum() < 10:
                continue
            air = (d["both_feet_no_contact"] > 0) & mask
            edges = np.diff(np.r_[False,air,False].astype(int))
            duration = (np.flatnonzero(edges==-1)-np.flatnonzero(edges==1))*np.median(np.diff(d["time_s"]))
            result[name].append(dict(seed=meta["seed"], report=str(path), end_reason=meta["end_reason"],
                no_contact_fraction=float(d["both_feet_no_contact"][mask].mean()),
                longest_air_s=float(duration.max()) if len(duration) else 0.,
                height_rmse_m=float(np.sqrt(np.mean((d["root_z_m"][mask]-d["ref_root_z_m"][mask])**2)))))
    return {name: dict(episodes=rows, mean={key:float(np.mean([row[key] for row in rows]))
                   for key in ("no_contact_fraction","longest_air_s","height_rmse_m")} if rows else {})
            for name,rows in result.items()}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("reports",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.reports)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({name:value["mean"] for name,value in result.items()},indent=2))
