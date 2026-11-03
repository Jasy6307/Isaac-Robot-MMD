"""Recheck saved real poses/contact traces; no simulator or policy execution."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion_feedback import KinematicMotionAnalyzer,MotionFeedback


class DataLoader(yaml.SafeLoader):pass
def tagged(loader,suffix,node):
    if isinstance(node,yaml.SequenceNode):return loader.construct_sequence(node)
    if isinstance(node,yaml.MappingNode):return loader.construct_mapping(node)
    return loader.construct_scalar(node)
DataLoader.add_multi_constructor("tag:yaml.org,2002:python/",tagged)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source_run",type=Path,required=True)
    parser.add_argument("--capture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(min(8,torch.get_num_threads()))
    cfg=yaml.load((args.source_run/"params/env.yaml").read_text(encoding="utf-8"),Loader=DataLoader)
    report=cfg["auto_jump_report"];hz=report["control_hz"];count=round(report["duration_s"]*hz)
    paths=sorted(args.capture.glob("validation_media/iter_*/seed_*/poses.npz"))
    traces=sorted((args.capture/"traces").glob("trace_*.npz"))
    if len(paths)!=3 or len(traces)!=3:raise ValueError("Exactly three real pose and control/contact traces required")
    with np.load(paths[0]) as z:names=json.loads(str(z["metadata"]))["joint_names"]
    source=Path(cfg["actions"]["joint_pos"]["motion_h5_path"])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=report["source_sha256"]:raise ValueError("Source reference changed")
    init=cfg["scene"]["robot"]["init_state"]
    baseline=np.array([next((v for expression,v in init["joint_pos"].items() if re.fullmatch(expression,n)),0.) for n in names],dtype=np.float32)
    with h5py.File(source) as f:
        original=f["joint_names"].asstr()[:].tolist();q=f["joint_pos_delta"][:]
        pos=f["root_pos_delta"][:];quat=f["root_quat_delta_wxyz"][:];fps=float(f.attrs["fps"])
    frame=np.minimum(np.arange(count)*fps/hz,len(q)-1);a=np.floor(frame).astype(int);b=np.minimum(a+1,len(q)-1)
    alpha=(frame-a).astype(np.float32)[:,None]
    order=[original.index(n) for n in names]
    qref=baseline+(q[a][:,order]+(q[b][:,order]-q[a][:,order])*alpha)
    rootpos=pos[a]+(pos[b]-pos[a])*alpha;rootquat=quat[a]+(quat[b]-quat[a])*alpha
    rootquat/=np.linalg.norm(rootquat,axis=-1,keepdims=True)
    def table(values):
        data=torch.as_tensor(values,dtype=torch.float32)
        return lambda steps:data[steps.clamp(0,len(data)-1)]
    buf=NS(num_steps=count,q_ref_abs=table(qref),root_pos_delta=table(rootpos),root_quat_wxyz=table(rootquat))
    robot=NS(joint_names=names,data=NS(default_root_state=NS(torch=torch.tensor([list(init["pos"])+list(init["rot"])],dtype=torch.float32))))
    env=NS(step_dt=1/hz,device="cpu",scene={"robot":robot},cfg=NS(scene=NS(robot=NS(spawn=NS(usd_path=cfg["scene"]["robot"]["spawn"]["usd_path"])))))
    analyzer=KinematicMotionAnalyzer(env,report,buf);records=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for path,trace_path in zip(paths,traces):
        with np.load(path) as z:poses=z["poses"];meta=json.loads(str(z["metadata"]))
        with np.load(trace_path) as z:
            trace=z["trace"]
            if int(z["seed"])!=meta["seed"]:raise ValueError("Seed recordings are misaligned")
        result=analyzer.assess(poses,None,trace)
        result["analysis_source"]="saved_actual_validation_poses_and_ground_contact_trace"
        # No original joint targets in the legacy pose archive; command metrics
        # stay explicitly unavailable. Actual-motion metrics use real samples.
        (args.output/f"seed_{meta['seed']}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        records.append(dict(seed=meta["seed"],results=[dict(motion_diagnostics=result)]))
        print(f"[RECHECK] seed={meta['seed']} complete={result['complete']} score={result['score']} candidates={len(result['candidates'])}")
        print("candidate starts",[r["start_s"] for r in result["candidates"]])
    feedback=MotionFeedback(int(np.ceil(report["duration_s"])));feedback.update(records)
    (args.output/"feedback.json").write_text(json.dumps(feedback.state_dict(),indent=2),encoding="utf-8")


if __name__=="__main__":main()
