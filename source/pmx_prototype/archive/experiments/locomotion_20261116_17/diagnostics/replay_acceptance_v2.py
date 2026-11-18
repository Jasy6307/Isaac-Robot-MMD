"""Re-score archived trajectories without rewriting historical reports."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(SOURCE),str(SOURCE/'diagnostics')]
import source_layout
from evaluate_unified_walk_run import command,linear_tracking_error
from locomotion_acceptance import score_segment,summarize,specification
from command_plan import command_increment


def replay(folder):
    old=json.loads((folder/'report.json').read_text());z=np.load(folder/'rollout.npz');dt=float(z['dt'])
    cfg=old['env_config'];vel=z['velocities'].mean(1)
    root=list(z['link_names']).index(cfg['reference_root_link_name']);pos=z['poses'][:,0,root,:3]
    delta=np.vstack([np.zeros((1,3)),np.diff(pos,axis=0)])/dt
    heading=z['headings'][:,0]
    geo=np.stack([np.sin(heading)*delta[:,0]-np.cos(heading)*delta[:,1],np.cos(heading)*delta[:,0]+np.sin(heading)*delta[:,1]],-1)
    current=torch.zeros((1,3));items=[];start=0
    for segment in old['segments']:
        n=round(segment['seconds']/dt);initial=current[0].numpy().copy();planned=[];requested=[]
        for t in range(n):
            cmd=command(segment['label'],t*dt,segment['seconds'],segment['command'])
            requested.append(cmd);current+=command_increment(torch.tensor([cmd],dtype=torch.float32),current,dt,cfg)
            if z['dones'][start+t].any():current.zero_()
            planned.append(current[0].numpy().copy())
        raw=np.asarray(requested,dtype=np.float32);item=copy.deepcopy(segment)
        # Check the replayed geometric metric when no reset discontinuity occurs.
        if not z['dones'][start:start+n].any():
            error=linear_tracking_error(geo[start+60:start+n]-raw[60:,:2],segment['label'].startswith('ramp_') or segment['label']=='direction_sweep',dt)
            if abs(error-segment['geometric_velocity_error'])>2e-4:raise ValueError('Root-position replay mismatch')
        item['legacy_passed']=segment.get('legacy_passed',segment['passed'])
        item['response_steady']=score_segment(item,vel[start:start+n],np.asarray(planned),raw,geo[start:start+n],initial,cfg,dt)
        item['passed']=item['response_steady']['passed'];items.append(item);start+=n
    return dict(source_report=str(folder/'report.json'),replay_only=True,training=False,acceptance_protocol='response_steady_v2',
        acceptance_protocol_sha256=hashlib.sha256((SOURCE/'training/locomotion_acceptance.py').read_bytes()).hexdigest(),
        source_complete_scope=old['complete_scope'],source_diagnostic=old.get('diagnostic',False),
        specification=specification(),summary=summarize(items),segments=items,
        caveat='Archived short stops may have insufficient steady observation. Such segments remain inconclusive; this replay cannot promote a model or manufacture missing samples.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);result=replay(args.report_folder)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary']))
