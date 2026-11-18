"""Replay recorded commands/references without simulator or policy updates."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from full_motion_reference import FullMotionReference
from unified_walk_run_reference import components,sample,velocity_at_offset

ROOT=source_layout.PROJECT_ROOT
OUT=ROOT/'output/anon_locomotion/transition_chain_20261116'


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    folder=ROOT/'output/anon_locomotion/retention_strength_20261116'
    result={}
    for name in ['baseline','original']:
        d=np.load(folder/(name+'_lite')/'rollout.npz')
        cfg=json.loads((folder/(name+'_lite')/'report.json').read_text(encoding='utf-8'))['env_config']
        motion=FullMotionReference(cfg['full_motion_path'],'cpu')
        motion.gait_cycles=torch.tensor(motion.metadata_cycles,dtype=torch.float32)
        motion.gait_offsets=torch.tensor(cfg['unified_contact_offsets'],dtype=torch.float32)
        raw=torch.tensor(d['commands'][:,0],dtype=torch.float32);dt=float(d['dt'])
        smooth=torch.zeros_like(raw);rate=torch.zeros_like(raw);current=torch.zeros(3)
        for i in range(len(raw)):
            delta=(raw[i]-current)*(1-np.exp(-dt/.35))
            delta[:2]*=(.6*dt/delta[:2].norm().clamp_min(1e-6)).clamp(max=1)
            delta[2]=delta[2].clamp(-1.2*dt,1.2*dt)
            current=current+delta;smooth[i]=current;rate[i]=delta/dt
        phase=torch.tensor(d['phases'][:,0])
        ref=sample(motion,phase,smooth,cfg,rate)
        order=[motion.names.index(n) for n in d['dof_names']]
        leg=[i for i,n in enumerate(d['dof_names']) if any(k in n for k in ['hip','knee','ankle'])]
        manifest=json.loads((Path(cfg['asset']).parent/'manifest.json').read_text(encoding='utf-8'))
        limits={j['name']:np.deg2rad(j['limits_deg']) for j in manifest['joints']}
        lower=np.array([limits[n][0]+.01 for n in d['dof_names']]);upper=np.array([limits[n][1]-.01 for n in d['dof_names']])
        nominal=ref['q'][:,order].numpy()
        residual=np.clip(d['actions'][:,0],-2,2)*cfg.get('imitation_residual_scale',.3)
        target=np.clip(nominal+residual,lower,upper)
        base=motion.links.index(cfg['reference_root_link_name'])
        v=velocity_at_offset(ref['velocities'][:,base],ref['rotations'][:,base],torch.tensor(cfg['reference_root_com_local_m']))
        refvel=torch.stack([-v[:,1],v[:,0]],-1).numpy()
        previous_phase=torch.cat([torch.zeros(1),phase[:-1]])
        previous_smooth=torch.cat([torch.zeros(1,3),smooth[:-1]])
        previous_rate=torch.cat([torch.zeros(1,3),rate[:-1]])
        pre=sample(motion,previous_phase,previous_smooth,cfg,previous_rate)
        recorded=d['observations'][:,0,105:136]
        reconstructed=pre['q'][:,order].numpy()
        identity=float(abs(recorded-reconstructed).max())
        ids,weights,frequency=components(motion,smooth,cfg)
        rows={}
        actual=d['velocities'][:,0,:2]
        for label in ['run_diagonal_135','reverse_turn']:
            ix=np.flatnonzero(d['labels']==label);times=np.arange(len(ix))*dt
            row=dict(seconds=len(ix)*dt,windows={},timeline=[])
            for title,start in [('scored',2.),('last_second',len(ix)*dt-1.)]:
                chosen=ix[times>=start]
                terms=dict(command_smoothing=(smooth.numpy()[chosen,:2]-raw.numpy()[chosen,:2]).mean(0),
                    reference_motion=(refvel[chosen]-smooth.numpy()[chosen,:2]).mean(0),
                    policy_physics=(actual[chosen]-refvel[chosen]).mean(0),
                    total=(actual[chosen]-raw.numpy()[chosen,:2]).mean(0))
                row['windows'][title]={k:dict(vector=v.tolist(),norm=float(np.linalg.norm(v))) for k,v in terms.items()}
                row['windows'][title]['leg_target_tracking_rmse_rad']=float(np.sqrt(np.mean((d['joints'][chosen,0][:,leg]-target[chosen][:,leg])**2)))
                row['windows'][title]['leg_reference_tracking_rmse_rad']=float(np.sqrt(np.mean((d['joints'][chosen,0][:,leg]-nominal[chosen][:,leg])**2)))
                assert np.max(abs(terms['total']-terms['command_smoothing']-terms['reference_motion']-terms['policy_physics']))<1e-5
            for t in np.arange(0,len(ix)*dt,.5):
                i=ix[min(len(ix)-1,round(t/dt))]
                row['timeline'].append(dict(t=float(t),smooth=smooth[i].tolist(),actual=actual[i].tolist(),reference_root_velocity=refvel[i].tolist(),
                    gait_frequency=float(frequency[i]),run_weight=float(weights[i,2:4].sum()),pivot_weight=float(weights[i,5]),
                    clips=[str(motion.clip_names[int(c)]) for c in ids[i]],weights=weights[i].tolist()))
            rows[label]=row
        result[name]=dict(reference_q_reconstruction_max_error=identity,segments=rows)
    (OUT/'reference_chain.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,item in result.items():
        print(name,'reference reconstruction error',item['reference_q_reconstruction_max_error'])
        for label,row in item['segments'].items():print(label,json.dumps(row['windows']))


if __name__=='__main__':main()
