"""Matched three-seed turn metrics plus unchanged full-motion acceptance."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.scripts.analyze_turn import control_rows,spectral_rms


JOINT_PARTS=("hip_yaw","hip_roll","ankle_pitch","ankle_roll")


def read_arm(directory):
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    physical=[];command=[];reference=[];tracking=[];error_hf=[];seed_rows=[]
    for seed in summary["seeds"]:
        window=seed["windows"][1]
        selected=[j for j in window["joints"] if any(part in j["joint"] for part in JOINT_PARTS)]
        actual=float(np.sqrt(np.mean([j["actual"]["hf_rms_rad"]**2 for j in selected])))
        target=float(np.sqrt(np.mean([j["command"]["hf_rms_rad"]**2 for j in selected])))
        ref=float(np.sqrt(np.mean([j["reference"]["hf_rms_rad"]**2 for j in selected])))
        with np.load(seed["source"]) as z:data=dict(zip(z["columns"].tolist(),z["values"].T))
        meta=json.loads(Path(seed["source"]).with_name("metadata.json").read_text(encoding="utf-8"))
        hz=float(meta["control_hz"])
        mask=(data["time_s"]>=12)&(data["time_s"]<14)
        err=np.stack([data[j["joint"]+"_actual_rad"][mask]-data[j["joint"]+"_ref_rad"][mask] for j in window["joints"]])
        rms=float(np.sqrt(np.mean(err**2)))
        indices=control_rows(data,hz)
        indices=indices[(data["ref_time_s"][indices]>=12)&(data["ref_time_s"][indices]<14)]
        error_spectrum=[spectral_rms(data[j["joint"]+"_actual_rad"][indices]-data[j["joint"]+"_ref_rad"][indices],hz) for j in selected]
        error_hf.append(float(np.sqrt(np.mean(np.square(error_spectrum)))))
        physical.append(actual);command.append(target);reference.append(ref);tracking.append(rms)
        seed_rows.append(dict(seed=seed["seed"],actual_hf_rms_deg=float(np.rad2deg(actual)),
            command_hf_rms_deg=float(np.rad2deg(target)),leg_reference_rmse_rad=rms))
    return dict(actual_hf_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(physical))))),
        command_hf_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(command))))),
        reference_hf_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(reference))))),
        leg_reference_rmse_rad=float(np.sqrt(np.mean(np.square(tracking)))),
        tracking_error_hf_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(error_hf))))),seeds=seed_rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--treatment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    metrics={name:read_arm(directory) for name,directory in
             (("baseline",args.baseline),("control",args.control/"diagnostics"),("treatment",args.treatment/"diagnostics"))}
    full={name:json.loads((directory/"validation.json").read_text(encoding="utf-8")) for name,directory in
          (("control",args.control),("treatment",args.treatment))}
    b,c,t=(metrics[name] for name in ("baseline","control","treatment"))
    change={name:dict(actual_reduction_fraction=1-t["actual_hf_rms_deg"]/other["actual_hf_rms_deg"],
                     command_reduction_fraction=1-t["command_hf_rms_deg"]/other["command_hf_rms_deg"])
            for name,other in (("baseline",b),("control",c))}
    # These declared experiment criteria do not change original acceptance gates.
    checks=dict(original_full_acceptance=full["treatment"]["passed"],
        actual_reduction_vs_control=change["control"]["actual_reduction_fraction"]>=.15,
        actual_reduction_vs_baseline=change["baseline"]["actual_reduction_fraction"]>=.15,
        command_reduction_vs_control=change["control"]["command_reduction_fraction"]>=.15,
        command_reduction_vs_baseline=change["baseline"]["command_reduction_fraction"]>=.15,
        local_tracking_preserved=t["leg_reference_rmse_rad"]<=b["leg_reference_rmse_rad"]*1.15)
    report=dict(window_s=[12.,14.],band_hz=[4.,12.],metrics=metrics,treatment_changes=change,
        full_acceptance={name:dict(passed=record["passed"],seeds=[dict(seed=r["seed"],passed=r["acceptance_passed"],
            score=r["acceptance_score"],failures=[f for trial in r["results"] for f in trial["acceptance"]["failures"]])
            for r in record["records"]]) for name,record in full.items()},
        criteria=checks,provisional_improvement=all(checks.values()),
        limitation="One matched short continuation pair, not proof across training seeds; human visual review still required")
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in zip(axes,("actual_hf_rms_deg","command_hf_rms_deg","leg_reference_rmse_rad"),
                           ("Actual joint 4-12 Hz RMS (deg)","Command 4-12 Hz RMS (deg)","Leg/reference RMSE (rad)")):
        values=[metrics[n][key] for n in metrics]
        ax.bar(list(metrics),values,color=["#8899a6","#e49b40","#3c927c"])
        for i,value in enumerate(values):ax.text(i,value,f"{value:.4f}",ha="center",va="bottom")
        ax.set_title(label);ax.grid(axis="y",alpha=.2);ax.set_ylim(0,max(values)*1.2)
    fig.suptitle("12-14 s supported turn | 3-seed RMS | identical low-LR continuation protocol")
    fig.tight_layout();fig.savefig(args.output/"comparison.png",dpi=150);plt.close(fig)
    print(json.dumps(dict(criteria=checks,changes=change),indent=2))


if __name__=="__main__":main()
