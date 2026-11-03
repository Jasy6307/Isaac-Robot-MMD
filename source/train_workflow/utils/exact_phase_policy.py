"""A single deployment policy with exact, phase-gated residual composition.

The base and jump weights are embedded in the checkpoint. There are no external
teacher files or H5 reads in forward(). Normal-motion outputs are preserved.
"""
import copy
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization


class ExactPhaseResidualPolicy(MLPModel):
    def __init__(self,*args,freeze_base=True,freeze_normalization=True,**kwargs):
        super().__init__(*args,**kwargs)
        if self.obs_dim != 163:
            raise ValueError("Exact conditional residual policy requires 163 observations")
        self.jump_mlp = copy.deepcopy(self.mlp)
        self.jump_mlp[0] = torch.nn.Linear(162,self.mlp[0].out_features)
        self.jump_normalizer = EmpiricalNormalization(162)
        self.freeze_base=freeze_base
        self.freeze_normalization=freeze_normalization
        for parameter in self.mlp.parameters():parameter.requires_grad_(not freeze_base)
        if self.distribution is not None and freeze_base:
            for parameter in self.distribution.parameters():parameter.requires_grad_(False)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        raw=obs if torch.is_tensor(obs) else torch.cat([obs[name] for name in self.obs_groups],dim=-1)
        base_raw=raw.clone()
        base_raw[:,153:]=0.
        base=self.mlp(self.obs_normalizer(base_raw))
        jump=self.jump_mlp(self.jump_normalizer(raw[:,:162]))
        mean=base+raw[:,-1:].clamp(0.,1.)*(jump-base)
        if self.distribution is not None and stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def update_normalization(self,obs):
        if self.freeze_normalization:return
        raw=obs if torch.is_tensor(obs) else torch.cat([obs[name] for name in self.obs_groups],dim=-1)
        normal=raw[:,-1] < .01
        flight=raw[:,-1] > .01
        if bool(normal.any()):
            base=raw[normal].clone();base[:,153:]=0.
            self.obs_normalizer.update(base)
        if bool(flight.any()):self.jump_normalizer.update(raw[flight,:162])

    def as_jit(self):
        raise NotImplementedError("Use the PT deployment bundle; base-only export is incomplete")

    def as_onnx(self,verbose=False):
        raise NotImplementedError("Use the PT deployment bundle; base-only export is incomplete")
