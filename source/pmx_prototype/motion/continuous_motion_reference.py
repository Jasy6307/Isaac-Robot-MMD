"""Continuous command-conditioned blending of complete captured states."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import torch
import numpy as np


def align_contacts(motion):
    contacts=motion.data['contacts'].cpu().numpy();reference=contacts[2]
    offsets=[]
    for clip in contacts:
        errors=[np.mean((np.roll(clip,-k,axis=0)-reference)**2) for k in range(motion.count)]
        k=int(np.argmin(errors));offsets.append((k if k<=motion.count//2 else k-motion.count)/motion.count)
    motion.phase_offsets=torch.tensor(offsets,device=motion.commands.device,dtype=torch.float32)
    return offsets


def components(motion,commands):
    f,l,y=commands.unbind(-1)
    ids=torch.stack([torch.where(f>=0,2,4),torch.where(l>=0,5,6),torch.where(y>=0,9,10),torch.full_like(f,3,dtype=torch.long)],-1)
    speed=commands[:,:2].norm(dim=-1)
    weights=torch.stack([f.abs()/.4,l.abs()/.35,y.abs()/.6*(1-.6*(speed/.4).clamp(0,1)),torch.zeros_like(f)],-1)
    weights[:,3]=(1-weights.sum(-1)).clamp_min(0)
    rate=weights.sum(-1).clamp_min(1)
    weights=weights/rate[:,None]
    frequency=(weights/motion.periods[ids]).sum(-1)*rate
    return ids,weights,frequency


def sample_continuous(motion,phase,commands):
    ids,weights,frequency=components(motion,commands)
    shifts=motion.phase_offsets[ids] if hasattr(motion,'phase_offsets') else torch.zeros_like(weights)
    shifts=shifts-(shifts*weights).sum(-1,keepdim=True)
    result={};anchor=None
    for k in range(4):
        state=motion.sample(phase+shifts[:,k],ids[:,k]);scale=frequency*motion.periods[ids[:,k]]
        state['qd']=state['qd']*scale[:,None]
        v=state['velocities'].clone();source=motion.commands[ids[:,k]]
        mean=torch.stack([source[:,1],-source[:,0],torch.zeros_like(scale)],-1)
        yaw=torch.zeros_like(mean);yaw[:,2]=source[:,2]
        v[:,:,:3]-=mean[:,None]+torch.cross(yaw[:,None].expand_as(state['positions']),state['positions'],dim=-1)
        v[:,:,3:]-=yaw[:,None]
        state['velocities']=v*scale[:,None,None]
        if anchor is None:anchor=state['rotations']
        state['rotations']=torch.where(((anchor*state['rotations']).sum(-1)<0)[...,None],-state['rotations'],state['rotations'])
        for name,value in state.items():
            w=weights[:,k].reshape((len(phase),)+(1,)*(value.ndim-1))
            result[name]=result.get(name,0)+w*value
    result['rotations']=torch.nn.functional.normalize(result['rotations'],dim=-1)
    mean=torch.stack([commands[:,1],-commands[:,0],torch.zeros_like(frequency)],-1)
    yaw=torch.zeros_like(mean);yaw[:,2]=commands[:,2]
    result['velocities'][:,:,:3]+=mean[:,None]+torch.cross(yaw[:,None].expand_as(result['positions']),result['positions'],dim=-1)
    result['velocities'][:,:,3:]+=yaw[:,None]
    return result
