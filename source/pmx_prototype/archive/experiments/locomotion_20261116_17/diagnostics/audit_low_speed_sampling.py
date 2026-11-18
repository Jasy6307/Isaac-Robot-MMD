"""CPU-only sampling-event audit using the production command sampler."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
sys.path[:0]=[str(SOURCE),str(SOURCE/'motion')]
from unified_walk_run_reference import sample_commands

OUT=ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117'
CONFIGS={
    'original_scratch_recipe':ROOT/'output/anon_locomotion/scratch_v8_unified_20261116_12288/config_override.json',
    'sampling_only_7600':ROOT/'output/anon_locomotion/pivot_ablation_preserved_20261116/sampling_only.json',
    'latest_scratch_recipe':ROOT/'output/anon_locomotion/scratch_split_20261117/config.json',
}


def environment(cfg,iteration,n=12288):
    return SimpleNamespace(num_envs=n,device='cpu',cfg=cfg,evaluation=False,iteration=iteration,
        commands=torch.zeros(n,3),continuous_ref_commands=torch.zeros(n,3),command_age=torch.zeros(n))


def flags(commands):
    speed=commands[:,:2].norm(dim=-1)
    return dict(idle=commands.abs().sum(-1)<.001,pivot=(speed<.001)&(commands[:,2].abs()>.001),
        low_moving=(speed>=.02)&(speed<.4),low_020_040=(speed>=.2)&(speed<.4),
        benchmark_030_neighborhood=(speed>=.25)&(speed<=.35)&(commands[:,0]>0)&
            (commands[:,1].abs()<.03)&(commands[:,2].abs()<.05),
        running=speed>=.85)


def main():
    torch.set_num_threads(2);rows={}
    for name,path in CONFIGS.items():
        cfg=json.loads(path.read_text());stages={}
        for iteration in [0,1500,5674]:
            torch.manual_seed(42);env=environment(cfg,iteration);ids=torch.arange(env.num_envs)
            count={};weighted={};total=0;hold_total=0.
            for sample in range(128):
                sample_commands(env,ids,on_reset=sample==0)
                masks=flags(env.commands);hold=env.unified_command_hold
                for key,mask in masks.items():
                    count[key]=count.get(key,0)+int(mask.sum())
                    weighted[key]=weighted.get(key,0.)+float(hold[mask].sum())
                total+=env.num_envs;hold_total+=float(hold.sum())
            stages[str(iteration)]=dict(generated_commands=total,event_fraction={k:v/total for k,v in count.items()},
                hold_weighted_fraction_without_falls={k:v/hold_total for k,v in weighted.items()})
        dedicated=sum(cfg.get(k,0.) for k in ['unified_backward_env_fraction','unified_right_env_fraction'])
        pivots=cfg.get('unified_pivot_sampling',cfg.get('unified_pivot_training',False))
        if pivots:dedicated+=cfg.get('unified_pivot_env_fraction',.1)+cfg.get('unified_reversal_env_fraction',.1)
        dedicated+=cfg.get('unified_transition_env_fraction',0.)
        rows[name]=dict(source=str(path),remaining_general_pool=1-dedicated,stages=stages)
    cfg=json.loads(CONFIGS['latest_scratch_recipe'].read_text())
    variants=[]
    for fraction in [0.,.9]:
        altered=dict(cfg,continuous_low_speed_fraction=fraction)
        torch.manual_seed(123);env=environment(altered,5674)
        sample_commands(env,torch.arange(env.num_envs),on_reset=True)
        variants.append(env.commands)
    result=dict(training=False,method='Production sampler on CPU,12288 environment IDs,128 resampling events per stage.',
        limitation='These are generated-command proportions, not historical successful rollout-frame occupancy. Hold weighting ignores falls and asynchronous episode resets.',
        recipes=rows,legacy_low_speed_fraction_flag_changes_commands=not torch.equal(*variants))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'sampling_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
