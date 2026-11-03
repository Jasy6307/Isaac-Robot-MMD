"""Scoped, per-replica random streams for validation (no training changes)."""
from __future__ import annotations

import hashlib
import torch


PROTOCOL = "parallel-seed-streams-v1"


def uniform_like(env, data, stream, env_ids=None):
    rng = getattr(env, "_validation_random", None)
    return torch.rand_like(data) if rng is None else rng.uniform_like(data, stream, env_ids)


class ValidationRandom:
    """Each seed/term owns a generator, independent of env index/batch size."""
    def __init__(self, env, seeds):
        self.env = env
        self.seeds = list(seeds)
        self.generators = {}
        self.patched = []

    def uniform_like(self, data, stream, env_ids=None):
        result = torch.rand_like(data)
        ids = range(min(len(data), len(self.seeds))) if env_ids is None else env_ids.detach().cpu().tolist()
        rows = {env_id: row for row, env_id in enumerate(ids) if env_id < len(self.seeds)}
        for env_id, row in rows.items():
            key = (self.seeds[env_id], stream, str(data.device))
            if key not in self.generators:
                seed = int.from_bytes(hashlib.sha256(f"{PROTOCOL}:{key[0]}:{stream}".encode()).digest()[:8], "little") % (2**63-1)
                self.generators[key] = torch.Generator(device=data.device).manual_seed(seed)
            result[row] = torch.rand(data[row].shape, device=data.device, dtype=data.dtype,
                                     generator=self.generators[key])
        return result

    def __enter__(self):
        if getattr(self.env, "_validation_random", None) is not None:
            raise RuntimeError("Nested validation RNG scope")
        # Manager owns resolved term configs, which need not be cfg's originals.
        manager = self.env.observation_manager
        try:
            for group, terms in manager._group_obs_term_cfgs.items():
                for name, term in zip(manager._group_obs_term_names[group], terms):
                    noise = term.noise
                    if noise is None:
                        continue
                    if not all(hasattr(noise, k) for k in ("n_min", "n_max", "operation")):
                        raise ValueError(f"Unsupported validation noise: {group}.{name}")
                    original = noise.func
                    def apply(data, cfg, stream=f"obs:{group}:{name}"):
                        random = self.uniform_like(data, stream)
                        lo=torch.as_tensor(cfg.n_min,device=data.device,dtype=data.dtype)
                        hi=torch.as_tensor(cfg.n_max,device=data.device,dtype=data.dtype)
                        value = random * (hi-lo) + lo
                        if cfg.operation == "add": return data+value
                        if cfg.operation == "scale": return data*value
                        if cfg.operation == "abs": return value
                        raise ValueError(f"Unsupported noise operation: {cfg.operation}")
                    self.patched.append((noise, original))
                    noise.func = apply
            self.env._validation_random = self
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for noise, original in reversed(self.patched): noise.func = original
        self.patched.clear()
        if getattr(self.env, "_validation_random", None) is self:
            del self.env._validation_random
