"""Stage-bounded learning rates using RSL-RL's native mini-batch KL scheduler.

The learning_rate property bounds every scheduler assignment before RSL-RL
copies it to the optimizer, including assignments inside a PPO update.
No optimizer, loss, KL formula, or third-party library code is replaced.
"""
from __future__ import annotations

import math

from rsl_rl.algorithms import PPO


# Bound the short stage inside every mini-batch as well as at chunk entry.
STAGE_LR_BOUNDS = {
    "short": (1.e-5, 1.e-3),
    "long": (3.e-5, 2.e-4),
    "full": (1.e-5, 5.e-5),
}


class BoundedAdaptivePPO(PPO):
    def __init__(self, *args, **kwargs):
        self.lr_min, self.lr_max = STAGE_LR_BOUNDS["short"]
        self.lr_stage = "short"
        super().__init__(*args, **kwargs)

    @property
    def learning_rate(self):
        return self._bounded_learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rate must be finite and positive")
        self._bounded_learning_rate = min(self.lr_max, max(self.lr_min, value))
        if hasattr(self, "optimizer"):
            for group in self.optimizer.param_groups:
                group["lr"] = self._bounded_learning_rate

    def configure_learning_rate(self, stage, *, repair=False):
        if stage not in STAGE_LR_BOUNDS:
            raise ValueError(f"Unknown learning-rate stage: {stage}")
        if self.desired_kl is None or not math.isfinite(self.desired_kl) or self.desired_kl <= 0:
            raise ValueError("Bounded KL adaptation requires a positive desired_kl")
        self.lr_stage = stage
        self.lr_min, self.lr_max = (1.e-5, 1.e-5) if repair else STAGE_LR_BOUNDS[stage]
        self.schedule = "fixed" if repair else "adaptive"
        # Carry the learned rate across chunks/stages; only project to new bounds.
        rate = self.learning_rate
        if stage == "short":
            rate = min(rate, 1.e-3)
        self.learning_rate = rate
        return self.learning_rate_policy()

    def learning_rate_policy(self):
        return dict(stage=self.lr_stage, schedule=self.schedule, lower=self.lr_min,
                    upper=self.lr_max, desired_kl=self.desired_kl)

    def save(self):
        saved = super().save()
        saved["bounded_learning_rate"] = dict(self.learning_rate_policy(),
                                               value=self.learning_rate, version=1)
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        restored_iteration = super().load(loaded_dict, load_cfg, strict)
        # Loading weights without the optimizer must keep the fresh LR policy.
        if load_cfg is None or load_cfg.get("optimizer", False):
            policy = loaded_dict.get("bounded_learning_rate")
            if policy:
                self.configure_learning_rate(policy["stage"], repair=policy["schedule"] == "fixed")
                self.learning_rate = policy["value"]
            else:
                # Older PPO checkpoints only store the optimizer's current LR.
                self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return restored_iteration
