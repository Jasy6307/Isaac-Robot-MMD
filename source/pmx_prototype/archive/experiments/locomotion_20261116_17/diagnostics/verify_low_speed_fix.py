"""CPU sampling verification for the repaired recipe; never starts training."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[3];SOURCE=ROOT/'source/pmx_prototype'
sys.path[:0]=[str(SOURCE),str(SOURCE/'diagnostics')]
from audit_low_speed_sampling import environment,flags
from unified_walk_run_reference import sample_commands


def main():
    torch.set_num_threads(2)
    out=ROOT/'output/anon_locomotion/low_speed_fix_20261117';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'configs/training/unified/low_speed_balanced_v2.json').read_text())
    torch.manual_seed(42);e=environment(cfg,5674);ids=torch.arange(e.num_envs);count={};total=0
    for step in range(128):
        sample_commands(e,ids,on_reset=step==0)
        for name,mask in flags(e.commands).items():count[name]=count.get(name,0)+int(mask.sum())
        total+=e.num_envs
    before=json.loads((ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117/sampling_audit.json').read_text())['recipes']['latest_scratch_recipe']['stages']['5674']['event_fraction']
    result=dict(training=False,generated_commands=total,before=before,after={k:v/total for k,v in count.items()},
        pool_fractions=dict(backward=.2,lateral=.25,pivot=.08,reversal=.07,run_transition=.05,sustained_low_speed=.15,low_speed_stop=.1,general=.1),
        limitation='Generated command proportions, not historical rollout occupancy; the low_moving bin excludes the exact0.4 boundary.')
    assert abs(sum(result['pool_fractions'].values())-1)<1e-8
    assert result['after']['low_moving']>.15
    assert result['after']['benchmark_030_neighborhood']>before['benchmark_030_neighborhood']*20
    (out/'sampling_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
