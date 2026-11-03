"""Reference-relative tracking quality, independent of simulator and dance time."""
from __future__ import annotations

import math
import torch


QUALITY_LIMITS = dict(leg_rms_rad=.45, height_error_m=.18, tilt_error_rad=.60,
                      persistence_s=.40)


def up_vector_xyzw(quaternion):
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)


def tracking_tail_cost(leg_rms, height_error, tilt_error, normal_weight):
    """A monotone cost beyond the near-zero region of exp tracking rewards.

    Use actual/reference error, not target/reference residual magnitude: a
    nonzero residual is often necessary for correct physical tracking.
    """
    leg = ((leg_rms-.20)/.20).clamp(0.,5.)
    height = ((height_error-.08)/.08).clamp(0.,5.)
    tilt = ((tilt_error-.20)/.35).clamp(0.,5.)
    return (leg+height+tilt)*normal_weight.clamp(0.,1.)


class MotionQualityMonitor:
    def __init__(self, num_envs, device, dt, limits=None):
        self.limits = dict(QUALITY_LIMITS if limits is None else limits)
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0 or any(not math.isfinite(x) or x <= 0 for x in self.limits.values()):
            raise ValueError("Quality thresholds and timestep must be finite and positive")
        self.required_steps = math.ceil(self.limits["persistence_s"]/self.dt-1.e-8)
        self.run = torch.zeros((num_envs,4),dtype=torch.long,device=device)
        self.longest = torch.zeros_like(self.run)
        self.maximum = torch.zeros((num_envs,3),device=device)
        self.samples = torch.zeros(num_envs,dtype=torch.long,device=device)
        self.first_issue = torch.full((num_envs,),float("inf"),device=device)
        self.invalid = torch.zeros(num_envs,dtype=torch.bool,device=device)

    def update(self, metrics, active, seconds):
        values = torch.stack([metrics[k] for k in ("leg_rms_rad","height_error_m","tilt_error_rad")],dim=-1).detach()
        finite = torch.isfinite(values).all(-1)
        thresholds = values.new_tensor([self.limits[k] for k in ("leg_rms_rad","height_error_m","tilt_error_rad")])
        flags = values > thresholds
        # Strong jump objectives deliberately allow larger leg deviations.
        flags[:,0] &= metrics["normal_weight"] >= .99
        flags |= ~finite[:,None]
        flags = torch.cat((flags,flags.any(-1,keepdim=True)),dim=-1)
        self.invalid |= active & ~finite
        self.samples += active.long()
        self.run = torch.where(active[:,None],torch.where(flags,self.run+1,0),self.run)
        self.longest = torch.maximum(self.longest,self.run)
        safe = torch.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
        self.maximum = torch.maximum(self.maximum,torch.where(active[:,None],safe,0.))
        failing = active & ((self.run >= self.required_steps).any(-1) | ~finite)
        onset = seconds - (self.run.max(-1).values.float()-1).clamp_min(0)*self.dt
        self.first_issue = torch.where(failing,torch.minimum(self.first_issue,onset),self.first_issue)

    def summary(self, index):
        samples = int(self.samples[index])
        longest = self.longest[index].detach().cpu().tolist()
        maximum = self.maximum[index].detach().cpu().tolist()
        first = float(self.first_issue[index])
        return dict(passed=samples>0 and max(longest)<self.required_steps and not bool(self.invalid[index]),
            samples=samples,first_issue_s=first if math.isfinite(first) else None,
            longest_excess_s=dict(zip(("leg","height","tilt","any"),[n*self.dt for n in longest])),
            maximum=dict(zip(("leg_rms_rad","height_error_m","tilt_error_rad"),maximum)),
            nonfinite=bool(self.invalid[index]),limits=dict(self.limits))


def validation_passed(records):
    # Missing quality data is unknown, never an implicit pass.
    return bool(records) and all(r["completed"]==r["tested"] and
        r.get("quality_passed") is True for r in records)
