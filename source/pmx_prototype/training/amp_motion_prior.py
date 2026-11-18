"""Least-squares adversarial full-body motion prior with a real/fake replay buffer.

The discriminator sees four consecutive heading-invariant physical states.
Only rewards are added; this module never writes joint targets or body states.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import torch
from torch import nn
from full_motion_reference import FullMotionReference


def inverse_rotate(q,v):
    xyz=-q[...,:3];w=q[...,3:]
    t=2*torch.cross(xyz,v,dim=-1)
    return v+w*t+torch.cross(xyz,t,dim=-1)


def features(q,qd,pelvis_p,pelvis_q,velocity,keypoints):
    gravity=torch.zeros_like(pelvis_p);gravity[:,2]=-1
    relative=keypoints-pelvis_p[:,None]
    return torch.cat([pelvis_p[:,2:3],inverse_rotate(pelvis_q,gravity),
        inverse_rotate(pelvis_q,velocity[:,:3]),inverse_rotate(pelvis_q,velocity[:,3:]),
        q,qd*.1,inverse_rotate(pelvis_q[:,None].expand(-1,relative.shape[1],-1),relative).flatten(1)],-1)


class AMPMotionPrior:
    def __init__(self,env):
        self.env=env;self.device=env.device;self.pending=[];self.replay=None
        self.mirror=None
        if env.cfg.get('amp_mirror_augmentation',False):
            from lateral_symmetry import AnatomicalMirror
            self.mirror=AnatomicalMirror.from_env(env)
        self.window=4;windows=[]
        for path in env.cfg['amp_expert_paths']:
            motion=FullMotionReference(path,self.device)
            phase=torch.arange(4096,device=self.device)/4096
            if motion.expert_valid_range:
                left,right=motion.expert_valid_range
                left+=3*env.dt/motion.period
                phase=left+phase*(right-left)
            sequence=[]
            for offset in range(self.window):
                state=motion.sample(phase+(offset-3)*env.dt/motion.period)
                sequence.append(features(state['q'][:,env.q_order],state['qd'][:,env.q_order],
                    state['positions'][:,env.pelvis_reference_id],state['rotations'][:,env.pelvis_reference_id],
                    env.reference_root_velocity(state) if env.cfg.get('reference_velocity_at_com') else state['velocities'][:,env.base_reference_id],state['positions'][:,env.key_ref_ids]))
            windows.append(torch.stack(sequence,1))
        expert=torch.cat(windows,0)
        if self.mirror is not None:
            # Mirror raw temporal features before normalization. Natural step
            # variation and temporal order are retained in both copies.
            expert=torch.cat([expert,self.mirror.amp(expert)],0)
        self.feature_size=expert.shape[-1]
        self.mean=expert.mean((0,1));self.scale=expert.std((0,1)).clamp_min(.05)
        self.feature_weights=torch.ones(self.feature_size,device=self.device)
        self.feature_weights[1:10]=.6
        for i,name in enumerate(getattr(env,'names',motion.names)):
            weight=1.8 if 'waist' in name else 1.3 if any(k in name for k in ['shoulder','elbow','forearm','wrist']) else .5
            self.feature_weights[10+i]=weight;self.feature_weights[10+env.num_dofs+i]=weight
        key_start=10+2*env.num_dofs
        self.feature_weights[key_start:key_start+18]=.6;self.feature_weights[key_start+18:key_start+24]=1.5
        self.expert=expert.flatten(1).detach()
        self.discriminator=nn.Sequential(nn.Linear(self.feature_size*4,256),nn.LeakyReLU(.2),
            nn.Linear(256,128),nn.LeakyReLU(.2),nn.Linear(128,1)).to(self.device)
        self.optimizer=torch.optim.Adam(self.discriminator.parameters(),lr=1e-4,weight_decay=1e-4)
        self.history=torch.zeros((env.num_envs,4,self.feature_size),device=self.device)
        self.valid=torch.zeros(env.num_envs,device=self.device,dtype=torch.bool)
        self.last_score=0.;self.last_reward=0.;self.stats={}

    def normalize(self,x):
        return (((x.reshape(-1,4,self.feature_size)-self.mean)/self.scale)*self.feature_weights).clamp(-8,8).flatten(1)

    def score(self,flat):
        score=self.discriminator(self.normalize(flat)).squeeze(-1)
        if self.mirror is not None:
            mirrored=self.mirror.amp(flat.reshape(-1,4,self.feature_size)).flatten(1)
            # PPO reuses advantages for mirrored transitions. Its style reward
            # must therefore be invariant even while the discriminator still
            # has a learned directional bias. Only the training reward is
            # symmetrized; policy outputs and physical targets are untouched.
            score=.5*(score+self.discriminator(self.normalize(mirrored)).squeeze(-1))
        return score

    def observe(self):
        e=self.env
        x=features(e.q,e.qd,e.poses[:,e.pelvis_id,:3],e.poses[:,e.pelvis_id,3:],e.root_vel,e.poses[:,e.key_ids,:3])
        self.history=torch.cat([self.history[:,1:],x[:,None]],1)
        self.history[~self.valid]=x[~self.valid,None].expand(-1,4,-1);self.valid[:]=True
        flat=self.history.flatten(1)
        score=self.score(flat)
        style=(1-.25*(score-1).square()).clamp(0,1)
        self.last_score=score.mean();self.last_reward=style.mean()
        if not e.evaluation:
            moving=(e.commands[:,:2].norm(dim=-1)>.05).nonzero(as_tuple=False).flatten()
            if len(moving):
                ids=moving[torch.randint(len(moving),(min(512,len(moving)),),device=self.device)]
                self.pending.append(flat[ids].detach())
        return style

    def reset(self,ids):self.valid[ids]=False

    def update(self):
        if not self.pending:return {}
        fresh=torch.cat(self.pending,0).clone();self.pending=[]
        if self.mirror is not None:
            fresh=torch.cat([fresh,self.mirror.amp(fresh.reshape(-1,4,self.feature_size)).flatten(1)],0)
        self.replay=fresh if self.replay is None else torch.cat([self.replay,fresh],0)[-65536:]
        stats=[]
        for _ in range(8):
            real=self.expert[torch.randint(len(self.expert),(1024,),device=self.device)]
            fake=self.replay[torch.randint(len(self.replay),(1024,),device=self.device)]
            real=self.normalize(real).detach().clone().requires_grad_(True);fake=self.normalize(fake).detach()
            r=self.discriminator(real);f=self.discriminator(fake)
            adversarial=.5*((r-1).square().mean()+(f+1).square().mean())
            gradient=torch.autograd.grad(r.sum(),real,create_graph=True)[0]
            gp=gradient.square().sum(-1).mean()
            loss=adversarial+10*gp
            self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),10);self.optimizer.step()
            stats.append(torch.stack([loss.detach(),gp.detach(),r.detach().mean(),f.detach().mean()]))
        values=torch.stack(stats).mean(0).cpu().tolist()
        self.stats=dict(zip(['amp_discriminator_loss','amp_gradient_penalty','amp_real_score','amp_fake_score'],values))
        return self.stats

    def state_dict(self):
        return dict(discriminator=self.discriminator.state_dict(),optimizer=self.optimizer.state_dict(),mean=self.mean,scale=self.scale,feature_weights=self.feature_weights)

    def load_state_dict(self,state):
        self.discriminator.load_state_dict(state['discriminator']);self.optimizer.load_state_dict(state['optimizer'])
        self.mean=state['mean'].to(self.device);self.scale=state['scale'].to(self.device)
        if 'feature_weights' in state:self.feature_weights=state['feature_weights'].to(self.device)


def install_amp(runner,env):
    if not hasattr(env,'amp'):return
    update=runner.alg.update;save=runner.alg.save
    def wrapped_update():
        losses=update();losses.update(env.amp.update());return losses
    def wrapped_save():
        state=save();state['amp_state']=env.amp.state_dict();return state
    runner.alg.update=wrapped_update;runner.alg.save=wrapped_save
