"""Consolidated turn analysis; historical measurement rules retained."""
from __future__ import annotations
from source.train_workflow.utils.jump_analysis import load_telemetry, new_diagnostic_output, require_new_files
import argparse
import json
from pathlib import Path
import numpy as np


def spectral_rms(values,hz,lo=4.,hi=12.):
    values=np.asarray(values,dtype=float)
    if len(values)<8:raise ValueError("Too few samples for spectral comparison")
    t=np.arange(len(values),dtype=float)
    trend=np.polyval(np.polyfit(t,values,1),t)
    window=np.hanning(len(values));spectrum=np.fft.rfft((values-trend)*window)
    freq=np.fft.rfftfreq(len(values),1/hz)
    weights=np.full(len(spectrum),2.);weights[0]=1.
    if len(values)%2==0:weights[-1]=1.
    power=weights*np.abs(spectrum)**2/(len(values)*np.sum(window**2))
    band=(freq>=lo)&(freq<=hi)
    return float(np.sqrt(power[band].sum()))


def control_rows(data,hz):
    """Last actual physics state per *complete* held-reference control interval."""
    keys=np.rint(data["ref_time_s"]*hz).astype(int)
    unique,counts=np.unique(keys,return_counts=True)
    expected=round(1/hz/float(np.median(data["sample_dt_s"])))
    valid=unique[counts==expected]
    return np.array([np.flatnonzero(keys==k)[-1] for k in valid],dtype=int)


def summarize(data,meta,start,end):
    hz=float(meta["control_hz"]);idx=control_rows(data,hz)
    idx=idx[(data["ref_time_s"][idx]>=start)&(data["ref_time_s"][idx]<end)]
    if len(idx)<20:raise ValueError("Incomplete diagnostic window")
    physical=(data["time_s"]>=start)&(data["time_s"]<end)
    joints=[]
    for name in meta["leg_joint_names"]:
        ref=data[f"{name}_ref_rad"][idx];target=data[f"{name}_target_rad"][idx];actual=data[f"{name}_actual_rad"][idx]
        residual=target-ref
        row=dict(joint=name)
        for label,values in (("reference",ref),("command",target),("residual",residual),("actual",actual)):
            row[label]=dict(hf_rms_rad=spectral_rms(values,hz),
                step_rms_rad=float(np.sqrt(np.mean(np.diff(values)**2))),
                second_difference_rms_rad=float(np.sqrt(np.mean(np.diff(values,n=2)**2))),
                max_step_rad=float(np.max(np.abs(np.diff(values)))))
        limit=data[f"{name}_limit_Nm"][physical]
        torque=data[f"{name}_applied_Nm"][physical]
        row.update(actual_command_rms_rad=float(np.sqrt(np.mean((actual-target)**2))),
                   torque_p95_Nm=float(np.quantile(np.abs(torque),.95)),
                   effort_limit_hit_fraction=float((np.abs(torque)>=.98*limit).mean()))
        joints.append(row)
    feet={}
    for side in ("left","right"):
        force=data[f"{side}_contact_N"][physical];contact=force>=meta["contact_threshold_N"]
        speed=np.hypot(data[f"{side}_foot_vx_m_s"][physical],data[f"{side}_foot_vy_m_s"][physical])
        feet[side]=dict(contact_fraction=float(contact.mean()),contact_switches=int(np.count_nonzero(np.diff(contact.astype(int)))),
                        normal_force_range_N=[float(force.min()),float(force.max())],
                        foot_origin_horizontal_speed_p95_m_s=float(np.quantile(speed,.95)),
                        foot_yaw_rate_range_rad_s=[float(data[f"{side}_foot_wz_rad_s"][physical].min()),
                                                   float(data[f"{side}_foot_wz_rad_s"][physical].max())])
    return dict(seed=meta["seed"],start_s=start,end_s=end,control_samples=len(idx),
        feet=feet,joints=sorted(joints,key=lambda j:j["actual"]["hf_rms_rad"],reverse=True))


def plot(data,meta,summary,output,start,end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask=(data["time_s"]>=start)&(data["time_s"]<=end);t=data["time_s"][mask]
    selected=[j["joint"] for j in summary["joints"][:4]]
    fig,axes=plt.subplots(4,2,figsize=(16,12),sharex=True)
    for i,name in enumerate(selected):
        for field,color in (("ref","#777777"),("target","#e69500"),("actual","#1673b1")):
            axes[i,0].plot(t,data[f"{name}_{field}_rad"][mask],label=field,color=color,lw=1.3)
        axes[i,0].set_ylabel(name.replace("_joint","")+" (rad)")
        axes[i,1].plot(t,data[f"{name}_target_rad"][mask]-data[f"{name}_ref_rad"][mask],label="policy residual (rad)",color="#c74457")
        twin=axes[i,1].twinx()
        twin.plot(t,data[f"{name}_velocity_rad_s"][mask],label="actual velocity (rad/s)",color="#1673b1",alpha=.55)
        twin.set_ylabel("actual velocity (rad/s)");twin.legend(loc="lower right",fontsize=7)
        axes[i,1].set_ylabel("command - reference (rad)")
        for ax in axes[i]:ax.grid(alpha=.25);ax.legend(fontsize=8,loc="upper left");ax.axvspan(12,14,color="#eeeeee",zorder=-10)
    for ax in axes[-1]:ax.set_xlabel("rollout time (s)")
    fig.suptitle(f"Seed {meta['seed']}: reference / held command / physical motion | 120 Hz telemetry")
    fig.tight_layout();fig.savefig(output/f"seed_{meta['seed']}_joints.png",dpi=140);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(15,12),sharex=True)
    for side in ("left","right"):
        axes[0].plot(t,data[f"{side}_contact_N"][mask],label=side)
        axes[1].plot(t,np.hypot(data[f"{side}_foot_vx_m_s"][mask],data[f"{side}_foot_vy_m_s"][mask]),label=side)
        axes[2].plot(t,data[f"{side}_foot_wz_rad_s"][mask],label=side)
        yaw=np.unwrap(data[f"{side}_foot_yaw_rad"])
        axes[3].plot(t,yaw[mask],label=side+" foot")
    axes[3].plot(t,np.unwrap(data["root_yaw_rad"])[mask],label="actual root",ls="--")
    ref=np.unwrap(data["ref_root_delta_yaw_rad"])
    ref=ref-ref[0]+np.unwrap(data["root_yaw_rad"])[0]
    axes[3].plot(t,ref[mask],label="reference root (offset aligned)",color="black",ls=":")
    for name in selected:axes[4].plot(t,data[f"{name}_applied_Nm"][mask],label=name.replace("_joint",""))
    labels=["ground normal force (N)","foot link origin speed (m/s)","foot yaw velocity (rad/s)","unwrapped yaw (rad)","applied joint torque (Nm)"]
    for ax,label in zip(axes,labels):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(loc="upper left",fontsize=8);ax.axvspan(12,14,color="#eeeeee",zorder=-10)
    axes[-1].set_xlabel("rollout time (s)");fig.suptitle(f"Seed {meta['seed']}: contact and turning response (foot speed is not sole slip)")
    fig.tight_layout();fig.savefig(output/f"seed_{meta['seed']}_contact.png",dpi=140);plt.close(fig)


def analyze_turn_main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    parser.add_argument('--output-dir', type=Path)
    args=parser.parse_args(argv);summaries=[]
    output = new_diagnostic_output('turn_analysis', args.output_dir)
    require_new_files(output/'summary.json')
    for path in sorted(args.directory.glob("seed_*/episode_*/samples.npz")):
        meta=json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        data = load_telemetry(path)
        windows=[summarize(data,meta,a,b) for a,b in ((10.,12.),(12.,14.),(14.,16.))]
        summaries.append(dict(seed=meta["seed"],source=str(path.resolve()),windows=windows))
        require_new_files(output/f"seed_{meta['seed']}_joints.png", output/f"seed_{meta['seed']}_contact.png")
        plot(data,meta,windows[1],output,11.,15.)
    if not summaries:raise ValueError("No turn telemetry found")
    payload=dict(spectral_band_hz=[4.,12.],method="Detrended Hann-weighted RMS at the control rate; held targets deduplicated before differentiation",
        limitations=["Observational correlation does not establish one-way causality in closed-loop control",
                     "Foot-link-origin speed is not contact-point slip speed",
                     "Revalidation uses 16 envs, not the original 8192-env training acceptance scene"],seeds=summaries)
    (output/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(output/"summary.json")


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


def compare_turn_smoothing_main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--treatment",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument('--control-diagnostics', type=Path)
    parser.add_argument('--treatment-diagnostics', type=Path)
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    require_new_files(args.output/'comparison.json', args.output/'comparison.png')
    metrics={name:read_arm(directory) for name,directory in
             (("baseline",args.baseline),("control",args.control_diagnostics or args.control/"diagnostics"),("treatment",args.treatment_diagnostics or args.treatment/"diagnostics"))}
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
