"""Verify captured stride differences survive deterministic physical execution.

Phase interpolation here is measurement only, never used to modify a rollout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from full_motion_reference import FullMotionReference

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True)
p.add_argument('--bank',type=Path,required=True)
p.add_argument('--source-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();d=np.load(a.rollout);m=FullMotionReference(a.bank,'cpu');meta=json.loads(a.source_report.read_text())
t=np.arange(len(d['phases']))*float(d['dt']);mask=(t>=8)&(d['commands'][:,0,0]>.1)
phase=d['phases'][mask,0];progress=np.unwrap(phase*2*np.pi)/(2*np.pi);q=d['joints'][mask,0]
names=list(d['dof_names']);order=[m.names.index(n) for n in names]
anchors=(np.array(meta['source_left_swing_peaks'])-meta['source_first'])/(meta['source_last']-meta['source_first'])
actual=[];reference=[];labels=[];durations=[]
for loop in range(int(np.floor(progress[0])),int(np.ceil(progress[-1]))):
    for j in range(len(anchors)-1):
        lo=loop+anchors[j];hi=loop+anchors[j+1]
        if lo<progress[0] or hi>progress[-1]:continue
        grid=lo+np.arange(64)/64*(hi-lo)
        actual.append(np.stack([np.interp(grid,progress,q[:,k]) for k in range(len(names))],-1))
        r=m.sample(torch.tensor(grid%1,dtype=torch.float32),torch.full((64,),2,dtype=torch.long))
        reference.append(r['q'][:,order].numpy());labels.append(j)
        durations.append(float(np.interp(hi,progress,t[mask])-np.interp(lo,progress,t[mask])))
assert len(actual)>=12,'Need at least two complete long passages for variation acceptance'
actual=np.array(actual);reference=np.array(reference);labels=np.array(labels)
group_actual=np.stack([actual[labels==j].mean(0) for j in range(len(anchors)-1)])
group_reference=np.stack([reference[labels==j].mean(0) for j in range(len(anchors)-1)])
upper=[i for i,n in enumerate(names) if any(s in n for s in ['waist','shoulder','elbow'])]
interior=(np.arange(64)/64>=.2)&(np.arange(64)/64<=.8)
da=(group_actual[:,:,upper]-group_actual[:,:,upper].mean(0))[:,interior]
dr=(group_reference[:,:,upper]-group_reference[:,:,upper].mean(0))[:,interior]
corr=float(np.corrcoef(da.ravel(),dr.ravel())[0,1]);rms=lambda x:float(np.degrees(np.sqrt(np.mean(x*x))))
links=list(d['link_names']);height=d['poses'][mask,0,links.index('left_ankle_roll'),2];dt=float(d['dt'])
peaks,_=find_peaks(height,distance=round(1.2/dt),prominence=.035)
peak_time=[];peak_progress=[]
for k in peaks:
    den=height[k-1]-2*height[k]+height[k+1]
    offset=float(np.clip(.5*(height[k-1]-height[k+1])/den,-.5,.5)) if abs(den)>1e-8 else 0.
    pt=t[mask][k]+offset*dt;peak_time.append(pt);peak_progress.append(float(np.interp(pt,t[mask],progress)))
physical_periods=np.diff(peak_time)
source_periods=np.diff(anchors)*float(m.periods[2])
rs=m.sample(torch.tensor(phase,dtype=torch.float32),torch.full((len(phase),),2,dtype=torch.long))
rh=rs['positions'][:,m.links.index('left_ankle_roll'),2].numpy()
rp,_=find_peaks(rh,distance=round(1.2/dt),prominence=.035);reference_times=[]
for k in rp:
    den=rh[k-1]-2*rh[k]+rh[k+1];off=float(np.clip(.5*(rh[k-1]-rh[k+1])/den,-.5,.5)) if abs(den)>1e-8 else 0.
    reference_times.append(t[mask][k]+off*dt)
reference_times=np.array(reference_times);pairs=[]
for pt in peak_time:
    if not len(reference_times):continue
    j=int(np.abs(reference_times-pt).argmin())
    if abs(reference_times[j]-pt)<.3:pairs.append((j,pt,reference_times[j]))
actual_intervals=[];expected=[]
for left,right in zip(pairs[:-1],pairs[1:]):
    if right[0]-left[0]==1:
        actual_intervals.append(right[1]-left[1]);expected.append(right[2]-left[2])
actual_intervals=np.array(actual_intervals);expected=np.array(expected)
mae=float(np.mean(np.abs(actual_intervals-expected))) if len(expected) else None
timing_correlation=float(np.corrcoef(actual_intervals,expected)[0,1]) if len(expected)>5 else None
result=dict(complete_reference_strides_binned=len(actual),physical_strides_detected=len(physical_periods),distinct_source_cycles=len(anchors)-1,source_stride_periods_seconds=source_periods.tolist(),reference_cycle_intervals_in_rollout_clock_seconds=durations,physical_stride_periods_seconds=physical_periods.tolist(),reference_ankle_stride_periods_seconds=expected.tolist(),physical_stride_timing_mae_seconds=mae,physical_stride_timing_correlation=timing_correlation,reference_upper_step_difference_rms_deg=rms(dr),actual_upper_step_difference_rms_deg=rms(da),captured_step_difference_correlation=corr,amplitude_ratio=rms(da)/max(rms(dr),1e-8),measurement='Reference-progress bins isolate each source cycle for joint variation; actual and reference ankle height maxima independently measure matching physical events with quadratic sub-frame interpolation',passed_variation_gate=bool(corr>.4 and .5<rms(da)/max(rms(dr),1e-8)<2. and mae is not None and mae<.12))
result['variation_phase_interval']=[.2,.8]
result['variation_excludes_loop_bridge']=True
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
