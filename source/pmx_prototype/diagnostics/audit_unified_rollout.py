"""Audit a no-reset unified rollout against its exact full-body reference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import argparse,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from full_motion_reference import FullMotionReference
from unified_walk_run_reference import prepare,sample

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();data=np.load(a.rollout);cfg=json.loads(a.config.read_text(encoding='utf-8'))
if data['dones'].any():raise ValueError('Exact phase audit requires no resets; do not reconstruct through falls')
torch.set_num_threads(4);motion=FullMotionReference(cfg['full_motion_path'],'cpu');prepare(motion)
cmd=data['commands'][:,0];dt=float(data['dt']);filtered=np.zeros(3);commands=[];rates=[];headings=[];heading=0.
for command in cmd:
    delta=(command-filtered)*(1-np.exp(-dt/.35));delta[:2]*=min(1.,.6*dt/max(np.linalg.norm(delta[:2]),1e-6));delta[2]=np.clip(delta[2],-1.2*dt,1.2*dt)
    filtered+=delta;commands.append(filtered.copy());rates.append(delta/dt);heading+=dt*filtered[2];headings.append(heading)
with torch.inference_mode():
    ref=sample(motion,torch.tensor(data['phases'].reshape(-1)),torch.tensor(np.array(commands),dtype=torch.float32),cfg,torch.tensor(np.array(rates),dtype=torch.float32))
actual_links=list(data['link_names']);actual_pelvis=actual_links.index('pelvis');ref_pelvis=motion.links.index('pelvis')
actual_height=data['poses'][:,0,actual_pelvis,2];target_height=ref['positions'][:,ref_pelvis,2].numpy()
target_rotation=Rotation.from_euler('Z',np.array(headings)[:,None])*Rotation.from_quat(ref['rotations'][:,ref_pelvis].numpy())
actual_rotation=Rotation.from_quat(data['poses'][:,0,actual_pelvis,3:]);rotation_error=(target_rotation.inv()*actual_rotation).as_rotvec()
qref=ref['q'][:,[motion.names.index(n) for n in data['dof_names']]].numpy();q=data['joints'][:,0]
labels=data['labels'];edges=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)];reports=[]
for start,end in zip(edges[:-1],edges[1:]):
    first=start+round(2/dt)
    if end-first<20:continue
    x=actual_height[first:end];y=target_height[first:end];best=(float('inf'),0)
    for lag in range(-round(.2/dt),round(.2/dt)+1):
        left=x[max(lag,0):min(len(x)+lag,len(x))];right=y[max(-lag,0):min(len(y)-lag,len(y))]
        error=np.mean(((left-left.mean())-(right-right.mean()))**2)
        if error<best[0]:best=(float(error),lag)
    reports.append(dict(label=str(labels[start]),start_frame=int(start),seconds=(end-start)*dt,
                        height_rmse_m=float(np.sqrt(np.mean((x-y)**2))),height_mean_bias_m=float(np.mean(x-y)),
                        actual_height_q05_q95_span_m=float(np.ptp(np.quantile(x,[.05,.95]))),reference_height_q05_q95_span_m=float(np.ptp(np.quantile(y,[.05,.95]))),
                        best_height_delay_seconds=best[1]*dt,height_demeaned_rmse_after_delay_m=float(np.sqrt(best[0])),
                        mean_pelvis_orientation_error_rad=float(np.linalg.norm(rotation_error[first:end],axis=-1).mean()),
                        mean_pelvis_yaw_error_rad=float(rotation_error[first:end,2].mean()),joint_rmse_rad=float(np.sqrt(np.mean((q[first:end]-qref[first:end])**2)))))
result=dict(rollout=str(a.rollout.resolve()),config=str(a.config.resolve()),segments=reports,diagnostic_only=True,
            link_alignment='Physical and reference link/joint names mapped independently; tensor index orders are not assumed equal')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
for r in reports:
    if r['label']=='run_backward':print(json.dumps(r),flush=True)
