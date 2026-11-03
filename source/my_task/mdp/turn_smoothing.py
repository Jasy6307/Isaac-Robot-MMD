"""Opt-in reference-gated residual smoothness; physical targets are not filtered."""
from __future__ import annotations
import numpy as np
import torch


def reference_turn_gate(quaternions_wxyz, hip_yaw, dt, schedules=()):
    """Body/hip heading-rate evidence with ramps and protected jump margins."""
    q=np.asarray(quaternions_wxyz,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or len(q)<3 or not np.isfinite(q).all() or not np.isfinite(dt) or dt<=0:
        raise ValueError("Finite quaternion table, >=3 frames and positive timestep required")
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1.e-8)
    w,x,y,z=q.T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    kernel=np.array([1,2,3,2,1],dtype=float)/9
    smooth=np.convolve(np.pad(yaw,(2,2),mode="edge"),kernel,mode="valid")
    speed=np.abs(np.gradient(smooth,dt))
    hip_yaw=np.asarray(hip_yaw,dtype=float)
    if hip_yaw.size:
        hip=np.unwrap(hip_yaw,axis=0)
        hip=np.stack([np.convolve(np.pad(h,(2,2),mode="edge"),kernel,mode="valid") for h in hip.T],axis=-1)
        speed=np.maximum(speed,.35*np.abs(np.gradient(hip,dt,axis=0)).max(-1))
    gate=np.clip((speed-.4)/.8,0,1)
    radius=max(1,round(.15/dt))
    padded=np.pad(gate,(radius,radius),mode="edge")
    gate=np.max(np.stack([padded[i:i+len(gate)] for i in range(2*radius+1)]),axis=0)
    gate=np.convolve(np.pad(gate,(2,2),mode="edge"),kernel,mode="valid")
    seconds=np.arange(len(gate))*dt
    for schedule in schedules:
        gate[(seconds>=schedule[0][0]-.3-1.e-6)&(seconds<=schedule[-1][1]+.3+1.e-6)]=0.
    return gate.astype(np.float32)


class ResidualHistory:
    def __init__(self, shape,device):
        self.previous=torch.zeros(shape,device=device)
        self.previous_delta=torch.zeros_like(self.previous)
        self.last_step=torch.full((shape[0],),-100000,device=device,dtype=torch.long)
        self.count=torch.zeros(shape[0],device=device,dtype=torch.long)
        self.previous_gate=torch.zeros(shape[0],device=device)
        self.previous_previous_gate=torch.zeros_like(self.previous_gate)

    def update(self,residual,steps,episode_steps,gate,dt):
        contiguous=(steps==self.last_step+1)&(episode_steps>0)
        valid_first=contiguous&(self.count>=1)
        valid_second=contiguous&(self.count>=2)
        delta=residual-self.previous
        acceleration=delta-self.previous_delta
        # Normalize physical derivatives, not raw policy action scale.
        self.rate=(delta/(dt*4.5)).square().clamp(max=25.)*valid_first[:,None]
        self.acceleration=(acceleration/(dt*dt*225.)).square().clamp(max=25.)*valid_second[:,None]
        self.rate*=torch.minimum(gate,self.previous_gate)[:,None]
        self.acceleration*=torch.minimum(torch.minimum(gate,self.previous_gate),self.previous_previous_gate)[:,None]
        self.previous_previous_gate=self.previous_gate.clone()
        self.previous_gate=gate.detach().clone()
        self.previous=residual.detach().clone();self.previous_delta=delta.detach().clone()
        self.last_step=steps.detach().clone()
        self.count=torch.where(contiguous,self.count+1,torch.ones_like(self.count))


def capture_residual(env,term,q_ref):
    """Called at action processing: q_ref is the exact baseline used by q_cmd."""
    from source.my_task.motion_reference import get_or_create_motion_buffer,motion_steps
    state=getattr(env,"_turn_smoothing_state",None)
    if state is None:
        selected=[]
        for joint,column in zip(term._residual_joint_ids.tolist(),term._residual_action_cols.tolist()):
            name=term._asset.joint_names[joint]
            if any(part in name for part in ("ankle_pitch","ankle_roll","hip_yaw","hip_roll")):
                selected.append((joint,column,0 if name.startswith("left_") else 1))
        if not selected:raise ValueError("Turn smoothing needs residual-controlled leg joints")
        action=env.cfg.actions.joint_pos
        buf=get_or_create_motion_buffer(env,action.motion_h5_path,action.motion_window_seconds)
        all_steps=torch.arange(buf.num_steps,device=env.device)
        names=term._asset.joint_names
        hips=[i for i,n in enumerate(names) if "hip_yaw_joint" in n]
        gate=reference_turn_gate(buf.root_quat_wxyz(all_steps).cpu().numpy(),
            buf.q_ref_abs(all_steps)[:,hips].cpu().numpy(),env.step_dt,env.cfg.jump_phase_schedules)
        state=ResidualHistory((env.num_envs,len(selected)),env.device)
        state.columns=[s[1] for s in selected];state.sides=[s[2] for s in selected]
        state.gate=torch.as_tensor(gate,device=env.device)
        expected=env.cfg.auto_jump_report.get("support_feet")
        state.expected=torch.as_tensor(expected,device=env.device,dtype=torch.bool) if expected else None
        env._turn_smoothing_state=state
    steps=motion_steps(env)
    current_gate=state.gate[steps.clamp(0,len(state.gate)-1)]
    residual=term.processed_actions[:,state.columns]-q_ref[:,state.columns]
    state.update(residual,steps,env.episode_length_buf,current_gate,env.step_dt)
    state.reference_steps=steps.detach().clone()


def _cost(env,kind):
    state=getattr(env,"_turn_smoothing_state",None)
    if state is None:return torch.zeros(env.num_envs,device=env.device)
    from source.my_task.mdp.jump import full_step_ground_contacts
    contacts=full_step_ground_contacts(env)
    if state.expected is not None:
        fps=env.cfg.auto_jump_report["fps"]
        indices=(state.reference_steps.float()*env.step_dt*fps).long().clamp(0,len(state.expected)-1)
        contacts=contacts&state.expected[indices]
    supported=contacts[:,state.sides]
    return (getattr(state,kind)*supported).sum(-1)/supported.sum(-1).clamp_min(1)


def residual_rate(env):return _cost(env,"rate")
def residual_acceleration(env):return _cost(env,"acceleration")


def configure(cfg,rate_weight=1.,acceleration_weight=.5):
    from isaaclab.managers import RewardTermCfg
    cfg.turn_smoothing_enabled=True
    cfg.turn_smoothing_settings=dict(version=1,rate_weight=rate_weight,acceleration_weight=acceleration_weight,
        velocity_scale_rad_s=4.5,acceleration_scale_rad_s2=225.,jump_margin_s=.3)
    cfg.rewards.turn_residual_rate=RewardTermCfg(func=residual_rate,weight=-rate_weight)
    cfg.rewards.turn_residual_acceleration=RewardTermCfg(func=residual_acceleration,weight=-acceleration_weight)
