"""Preserve two validated controllers exactly inside one conditional policy."""
import argparse
import copy
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.scripts.prepare_jump_checkpoint import expand_checkpoint


def compose(normal,jump):
    result=copy.deepcopy(jump)
    normal_expanded=expand_checkpoint(normal,extra=10)
    actor=normal_expanded["actor_state_dict"]
    for key,value in jump["actor_state_dict"].items():
        if key.startswith("mlp."):
            actor["jump_mlp."+key[len("mlp."):]]=value.clone()
        elif key.startswith("obs_normalizer."):
            actor["jump_normalizer."+key[len("obs_normalizer."):]]=value.clone()
    critic=result["critic_state_dict"]
    weight=critic["mlp.0.weight"]
    if weight.shape[1]!=162:raise ValueError("Jump controller must have 162 inputs")
    critic["mlp.0.weight"]=torch.cat((weight,torch.zeros_like(weight[:,:1])),1)
    for suffix,fill in (("_mean",0.),("_var",1.),("_std",1.)):
        key="obs_normalizer."+suffix
        critic[key]=torch.cat((critic[key],torch.full_like(critic[key][:,:1],fill)),1)
    result["actor_state_dict"]=actor
    result.pop("optimizer_state_dict",None)
    result["infos"]=dict(result.get("infos") or {},exact_phase_residual=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("normal",type=Path);p.add_argument("jump",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=compose(torch.load(a.normal,map_location="cpu",weights_only=False),torch.load(a.jump,map_location="cpu",weights_only=False))
    torch.save(result,a.output);print(a.output.resolve())
