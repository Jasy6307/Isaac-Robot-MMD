"""Expand a C1 checkpoint by eight appended observations without changing outputs."""
import argparse
import copy
from pathlib import Path
import torch


def expand_checkpoint(checkpoint, extra=8):
    result = copy.deepcopy(checkpoint)
    for group in ("actor_state_dict", "critic_state_dict"):
        state = result[group]
        weight = state["mlp.0.weight"]
        if weight.shape[1] != 153:
            raise ValueError(f"Expected 153 C1 observations, got {weight.shape[1]}")
        state["mlp.0.weight"] = torch.cat((weight, torch.zeros(weight.shape[0], extra, dtype=weight.dtype)), dim=1)
        for name, fill in (("_mean", 0.), ("_var", 1.), ("_std", 1.)):
            key = "obs_normalizer." + name
            value = state[key]
            state[key] = torch.cat((value, torch.full((1, extra), fill, dtype=value.dtype)), dim=1)
        # Preserve old running statistics at initialization. New entries start at
        # mean=0/std=1 and subsequently adapt in the specialist experiment.
    result.pop("optimizer_state_dict", None)
    result["iter"] = 0
    result["infos"] = {"migration": "append vz, contacts2, phase4, zerror; preserve old inputs; fresh optimizer"}
    return result


def restart_specialist(checkpoint, std=.45):
    result = copy.deepcopy(checkpoint)
    if result["actor_state_dict"]["mlp.0.weight"].shape[1] != 161:
        raise ValueError("Specialist restart requires 161 observations")
    result["actor_state_dict"]["distribution.std_param"].fill_(std)
    result.pop("optimizer_state_dict", None)
    result["iter"] = 0
    result["infos"] = {"migration": "V2 specialist warm start, fresh optimizer, exploration std=" + str(std)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--restart-specialist", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    result = restart_specialist(source) if args.restart_specialist else expand_checkpoint(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output.resolve())
