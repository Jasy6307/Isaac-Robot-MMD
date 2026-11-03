"""Compare matched physics telemetry; window results do not certify a full dance."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion_quality import MotionQualityMonitor
from source.train_workflow.utils.phase_gate import jump_teacher_gate


def read(directory, schedules):
    z=np.load(Path(directory)/"samples.npz")
    d=dict(zip(z["columns"],z["values"].T))
    joints=[key[:-8] for key in d if key.endswith("_ref_rad")]
    ref=np.stack([d[n+"_ref_rad"] for n in joints],axis=-1)
    actual=np.stack([d[n+"_actual_rad"] for n in joints],axis=-1)
    target=np.stack([d[n+"_target_rad"] for n in joints],axis=-1)
    d["leg_rms_rad"]=np.sqrt(np.mean((actual-ref)**2,axis=-1))
    d["target_reference_rms_rad"]=np.sqrt(np.mean((target-ref)**2,axis=-1))
    d["actual_target_rms_rad"]=np.sqrt(np.mean((actual-target)**2,axis=-1))
    d["height_error_m"]=np.abs(d["root_z_m"]-d["ref_root_z_m"])
    d["clipping_fraction"]=np.mean(np.stack([
        np.abs(d[n+"_computed_Nm"]-d[n+"_applied_Nm"])>1. for n in joints]),axis=0)
    seconds=torch.tensor(d["ref_time_s"],dtype=torch.float32)
    normal=1.-jump_teacher_gate(seconds,schedules)
    monitor=MotionQualityMonitor(1,"cpu",float(np.median(d["sample_dt_s"])))
    for i,sec in enumerate(seconds):
        metrics={k:torch.tensor([d[k][i]],dtype=torch.float32) for k in ("leg_rms_rad","height_error_m")}
        metrics.update(tilt_error_rad=torch.tensor([0.]),normal_weight=normal[i:i+1])
        monitor.update(metrics,torch.tensor([True]),sec.reshape(1))
    quality=monitor.summary(0)
    quality["scope"]="recorded window, leg/height only; reference-relative tilt unavailable in older telemetry"
    return d,quality


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--trial",required=True)
    parser.add_argument("--candidates",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report=json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    schedules=tuple(tuple(tuple(p) for p in e["phases"]) for e in report["events"] if e["flight_reward"])
    old,oldq=read(args.baseline,schedules)
    new,newq=read(args.trial,schedules)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    summary=dict(baseline_quality=oldq,trial_quality=newq,
        note="Telemetry comparison; changes in control targets precede/overlap loss of balance, not a causal ablation.",
        baseline=str(Path(args.baseline).resolve()),trial=str(Path(args.trial).resolve()))
    rows=[]
    for label,data in (("baseline",old),("trial",new)):
        t=data["ref_time_s"]
        for sec in range(int(np.ceil(t.min())),int(np.floor(t.max()))):
            keep=(t>=sec)&(t<sec+1)
            rows.append(dict(model=label,start_s=sec,tilt_max_deg=float(np.rad2deg(data["root_tilt_rad"][keep]).max()),
                height_rmse_m=float(np.sqrt(np.mean(data["height_error_m"][keep]**2))),
                leg_rms_rad=float(np.sqrt(np.mean(data["leg_rms_rad"][keep]**2))),
                target_reference_rms_rad=float(np.sqrt(np.mean(data["target_reference_rms_rad"][keep]**2))),
                torque_clipping_fraction=float(data["clipping_fraction"][keep].mean())))
    summary["per_second"]=rows
    (output/"comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (output/"per_second.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(13,12),sharex=True)
    for label,data in (("baseline",old),("trial",new)):
        t=data["ref_time_s"]
        for ax,key in zip(axes,("root_z_m","root_tilt_rad","leg_rms_rad","target_reference_rms_rad","clipping_fraction")):
            values=np.rad2deg(data[key]) if key=="root_tilt_rad" else data[key]
            ax.plot(t,values,label=label,alpha=.85)
    axes[0].plot(old["ref_time_s"],old["ref_root_z_m"],"k--",label="reference")
    for ax,label in zip(axes,("Root height (m)","Body tilt (deg)","Actual/reference leg RMS (rad)",
                              "Target/reference leg RMS (rad)","Clipped joint fraction")):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc="upper right")
    axes[-1].set_xlabel("Reference motion time (s)")
    fig.tight_layout();fig.savefig(output/"comparison.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
    for label,data in (("baseline",old),("trial",new)):
        for ax,key in zip(axes,("root_z_m","root_tilt_rad","leg_rms_rad")):
            values=np.rad2deg(data[key]) if key=="root_tilt_rad" else data[key]
            ax.plot(data["ref_time_s"],values,label=label)
    axes[0].plot(old["ref_time_s"],old["ref_root_z_m"],"k--",label="reference")
    for ax,label in zip(axes,("Root height (m)","Body tilt (deg)","Leg tracking RMS (rad)")):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc="upper right")
    axes[-1].set_xlabel("Reference motion time (s)")
    fig.tight_layout();fig.savefig(output/"comparison_overview.png",dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!="per_second"},indent=2))


if __name__=="__main__":main()
