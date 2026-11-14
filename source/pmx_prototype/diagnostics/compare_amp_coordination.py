"""Score measured rollouts with the same frozen AMP prior and expert manifold."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from amp_motion_prior import AMPMotionPrior,features
from full_motion_reference import FullMotionReference
from evaluate_amp_forward import coordination

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(4)
cfg=json.loads((args.checkpoint.parent/'env_config.json').read_text());state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
d=np.load(args.candidate);names=list(d['dof_names']);links=list(d['link_names'])
keys=['waist_pitch','head','left_knee_pitch','right_knee_pitch','left_ankle_roll','right_ankle_roll','left_hand','right_hand']
m=FullMotionReference(cfg['amp_expert_paths'][0],'cpu')
env=SimpleNamespace(cfg=cfg,device='cpu',num_envs=1,dt=1/30,names=names,q_order=[m.names.index(n) for n in names],pelvis_reference_id=m.links.index('pelvis'),base_reference_id=m.links.index(links[0]),key_ref_ids=[m.links.index(n) for n in keys])
prior=AMPMotionPrior(env);prior.load_state_dict(state['amp_state']);prior.discriminator.eval()

def score(path):
    data=np.load(path);pose=torch.tensor(data['poses'][:,0]);q=torch.tensor(data['joints'][:,0]);qd=torch.tensor(data['joint_velocities'][:,0]);velocity=torch.tensor(data['root_velocities'][:,0]);own_links=list(data['link_names']);pelvis=own_links.index('pelvis')
    f=features(q,qd,pose[:,pelvis,:3],pose[:,pelvis,3:],velocity,pose[:,[own_links.index(n) for n in keys],:3])
    windows=torch.stack([f[i:len(f)-3+i] for i in range(4)],1).flatten(1)
    t=np.arange(3,len(f))*float(data['dt']);mask=torch.tensor((t>=8)&(t<30))
    with torch.no_grad():
        x=prior.normalize(windows[mask]);expert=prior.normalize(prior.expert)
        logits=prior.discriminator(x).squeeze(-1)
        distances=[]
        for chunk in x.split(128):distances.append(torch.cdist(chunk,expert).amin(-1)/np.sqrt(x.shape[1]))
    return dict(coordination=coordination(data),amp_discriminator_score=float(logits.mean()),amp_style_reward=float((1-.25*(logits-1).square()).clamp(0,1).mean()),expert_nearest_window_rmse=float(torch.cat(distances).mean()))

result=dict(baseline=score(args.baseline),candidate=score(args.candidate),expert_metrics=[])
for path in cfg['amp_expert_paths']:
    m=FullMotionReference(path,'cpu');time=torch.arange(37*30)/30;s=m.sample(time/m.period)
    data=dict(poses=torch.cat([s['positions'],s['rotations']],-1)[:,None].numpy(),joints=s['q'][:,env.q_order,None].transpose(1,2).numpy(),commands=np.tile([.4,0,0],(len(time),1,1)),dt=1/30,link_names=m.links,dof_names=names)
    result['expert_metrics'].append(coordination(data))
result['note']='Same frozen discriminator for both measured trajectories; these style metrics do not replace visual review.'
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
