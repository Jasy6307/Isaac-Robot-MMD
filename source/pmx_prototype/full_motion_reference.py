"""GPU interpolation of complete mocap physical states, including contacts."""
import numpy as np
import torch


class FullMotionReference:
    def __init__(self,path,device):
        d=np.load(path);self.names=list(d['joint_names']);self.links=list(d['link_names'])
        self.bank=d['q'].ndim==3
        self.period=float(d['periods'][0]) if self.bank else float(d['period'])
        self.speed=float(d['speeds'][0]) if self.bank else float(d['speed'])
        self.periods=torch.tensor(d['periods'] if self.bank else [self.period],device=device,dtype=torch.float32)
        self.commands=torch.tensor(d['commands'] if self.bank else [d['command'] if 'command' in d else [self.speed,0.,0.]],device=device,dtype=torch.float32)
        self.clip_names=list(d['clip_names']) if self.bank else ['forward_native']
        self.data={k:torch.tensor(d[k],device=device) for k in ['q','qd','positions','rotations','velocities','contacts']}
        for key in ['gait_sin','gait_cos']:
            if key in d:self.data[key]=torch.tensor(d[key],device=device)
        self.expert_valid_range=d['expert_valid_range'].tolist() if 'expert_valid_range' in d else None
        self.idle=torch.tensor(d['idle'],device=device)
        self.count=d['q'].shape[1] if self.bank else len(d['q'])

    def sample(self,phase,clip_ids=None):
        x=(phase%1)*self.count;a=x.long()%self.count;b=(a+1)%self.count;f=x-a
        result={}
        if clip_ids is None:clip_ids=torch.zeros(len(phase),device=phase.device,dtype=torch.long)
        for key,value in self.data.items():
            va,vb=(value[clip_ids,a],value[clip_ids,b]) if self.bank else (value[a],value[b])
            weight=f.reshape((len(f),)+(1,)*(va.ndim-1))
            if key=='rotations':vb=torch.where(((va*vb).sum(-1)<0)[...,None],-vb,vb)
            result[key]=va*(1-weight)+vb*weight
        result['rotations']=torch.nn.functional.normalize(result['rotations'],dim=-1)
        return result
