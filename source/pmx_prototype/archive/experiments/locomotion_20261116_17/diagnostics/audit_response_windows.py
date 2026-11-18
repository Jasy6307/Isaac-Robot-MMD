"""Separate planning, response and steady errors in saved matched probes."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117'


def first_dwell(flags,dt,dwell=1.):
    length=round(dwell/dt)
    times=[]
    for e in range(flags.shape[1]):
        hits=np.flatnonzero(np.convolve(flags[:,e].astype(int),np.ones(length,dtype=int),mode='valid')==length)
        times.append(float(hits[0]*dt) if len(hits) else None)
    finite=[t for t in times if t is not None]
    return dict(by_env_seconds=times,reached=len(finite),total=len(times),median_seconds=float(np.median(finite)) if finite else None,
        maximum_seconds=max(finite) if finite else None)


def main():
    schedule=json.loads((OUT/'transition_schedule.json').read_text());result=[]
    for case,segments in schedule.items():
        data={name:np.load(OUT/'transitions'/f'{name}_{case}.npz') for name in ['old','new']}
        dt=float(np.diff(data['old']['time']).mean());offset=0
        logical={}
        for name,z in data.items():
            h=z['headings'];v=z['root_velocity']
            logical[name]=np.stack([np.sin(h)*v[:,:,0]-np.cos(h)*v[:,:,1],
                np.cos(h)*v[:,:,0]+np.sin(h)*v[:,:,1],v[:,:,5]],-1)
        for index,(seconds,cmd) in enumerate(segments):
            n=round(seconds/dt);end=offset+n
            valid=~(data['old']['done'][:end].any(0)|data['new']['done'][:end].any(0))
            row=dict(case=case,segment=index,command=cmd,seconds=seconds,
                valid_envs=int(valid.sum()),total_envs=len(valid),models={})
            target=np.array(cmd);speed=np.linalg.norm(target[:2]);stationary=np.abs(target).sum()<.001
            linear_limit=max(.06,speed*.22);yaw_limit=.1 if stationary else .2
            for name,z in data.items():
                physical=logical[name][offset:end,valid];planned=z['ref_commands'][offset:end,valid]
                if not valid.any():
                    row['models'][name]=dict(falls=int(z['done'][offset:end].sum()),valid_metrics=False)
                    continue
                windows={}
                for label,begin,finish in [('prefix_5_seconds',2.,min(5.,seconds)),('whole_original_exclusion',2.,seconds),
                    ('last_3_seconds',max(0.,seconds-3),seconds),('later_start_3_seconds',3.,seconds)]:
                    if finish<=begin:continue
                    chosen=physical[round(begin/dt):round(finish/dt)]
                    plan_window=planned[round(begin/dt):round(finish/dt)]
                    planning_bias=(plan_window[:,:,:2]-target[:2]).mean(0)
                    tracking_bias=(chosen[:,:,:2]-plan_window[:,:,:2]).mean(0)
                    bias=np.linalg.norm((chosen[:,:,:2]-target[:2]).mean(0),axis=-1)
                    yaw=np.abs((chosen[:,:,2]-target[2]).mean(0))
                    within=(bias<linear_limit)&(yaw<yaw_limit)
                    if stationary:within&=np.linalg.norm(chosen[:,:,:2],axis=-1).mean(0)<.08
                    windows[label]=dict(start=begin,end=finish,velocity_error_mean=float(bias.mean()),velocity_error_max=float(bias.max()),
                        yaw_error_mean=float(yaw.mean()),yaw_error_max=float(yaw.max()),
                        planning_bias_vector_mean=planning_bias.mean(0).tolist(),
                        physical_vs_planned_bias_vector_mean=tracking_bias.mean(0).tolist(),
                        original_tolerance_fraction=float(within.mean()))
                plan_flag=(np.linalg.norm(planned[:,:,:2]-target[:2],axis=-1)<linear_limit)&(abs(planned[:,:,2]-target[2])<yaw_limit)
                plan_close=(np.linalg.norm(planned[:,:,:2]-target[:2],axis=-1)<.02)&(abs(planned[:,:,2]-target[2])<.02)
                width=round(1/dt)
                mean=np.stack([physical[i-width+1:i+1].mean(0) for i in range(width-1,n)])
                physical_flag=(np.linalg.norm(mean[:,:,:2]-target[:2],axis=-1)<linear_limit)&(abs(mean[:,:,2]-target[2])<yaw_limit)
                linear_flag=np.linalg.norm(mean[:,:,:2]-target[:2],axis=-1)<linear_limit
                if stationary:
                    mean_speed=np.stack([np.linalg.norm(physical[i-width+1:i+1,:,:2],axis=-1).mean(0) for i in range(width-1,n)])
                    physical_flag&=mean_speed<.08
                    linear_flag&=mean_speed<.08
                padded=np.concatenate([np.zeros((width-1,int(valid.sum())),dtype=bool),physical_flag])
                linear_padded=np.concatenate([np.zeros((width-1,int(valid.sum())),dtype=bool),linear_flag])
                row['models'][name]=dict(falls=int(z['done'][offset:end].sum()),valid_metrics=True,windows=windows,
                    plan_within_original_tolerance=first_dwell(plan_flag,dt),plan_within_002=first_dwell(plan_close,dt),
                    linear_response_within_original_tolerance=first_dwell(linear_padded,dt),
                    physical_within_original_tolerance=first_dwell(padded,dt))
            result.append(row);offset=end
    summary=dict(training=False,rows=result,
        method='Physical linear response: one-second trailing mean, first entry into original tracking tolerances sustained for one second. Yaw steady bias is reported separately: gait oscillations can prevent a short combined velocity/yaw dwell despite a passing long-window yaw bias. Planning-near-target uses0.02 only as a timing diagnostic, never as an acceptance threshold.',
        official_gates_changed=False,limitation='Response metrics use the matched subset with no prior fall before segment end; all falls are separately retained, not excused. Alternative windows are sensitivity diagnostics only.')
    (OUT/'response_window_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps([dict(case=r['case'],segment=r['segment'],command=r['command'],valid=r['valid_envs'],
        metrics={name:dict(falls=v['falls'],response=v.get('physical_within_original_tolerance',{}).get('median_seconds'),
            plan=v.get('plan_within_002',{}).get('median_seconds'),windows=v.get('windows')) for name,v in r['models'].items()}) for r in result],indent=2))


if __name__=='__main__':main()
