"""Verify pivot sampling/reward wiring with existing policy inference only."""
import argparse
import json
from pathlib import Path
import runpy
import sys

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout


def validate(env,runner,output,checkpoint):
    import torch
    import pivot_stability
    output.mkdir(parents=True,exist_ok=True)
    original=pivot_stability.pivot_stability_terms
    cache={}
    def audited(*values):
        result=original(*values)
        for key,value in result.items():cache[key]=value.clone()
        return result
    pivot_stability.pivot_stability_terms=audited
    infer=runner.get_inference_policy()
    rows=[]
    with torch.inference_mode():
        for label,command in [('standing',[0,0,0]),('walking',[.3,0,0]),('pivot_right',[0,0,-.6]),('pivot_left',[0,0,.6])]:
            env.reset(env.ids,command=command)
            env.refresh()
            active_steps=0;nonzero_steps=0;max_penalty=0.;fall_count=0
            for _ in range(120):
                env.set_command(command)
                _,reward,done,extra=env.step(infer(env.get_observations()))
                assert torch.isfinite(reward).all()
                active=cache['active'];denominator=active.sum().clamp_min(1)
                for key in ('support','landing','drift','reward_rate'):
                    expected=cache[key].sum()/denominator
                    assert torch.allclose(extra['log']['Pivot/'+key],expected)
                active_steps+=int(active.sum())
                nonzero_steps+=int((cache['reward_rate']<0).sum())
                max_penalty=max(max_penalty,float(-cache['reward_rate'].min()))
                fall_count+=int(done.sum())
            if label in ('standing','walking'):
                assert active_steps==0 and nonzero_steps==0
            else:
                assert active_steps>0 and nonzero_steps>0
            rows.append(dict(case=label,active_environment_steps=active_steps,
                             nonzero_penalty_environment_steps=nonzero_steps,max_penalty_rate=max_penalty,
                             falls=fall_count))
    report=dict(training=False,checkpoint=str(checkpoint),config=str(args.config),passed=True,
                note='Checks reward activation/log wiring with an existing policy, not improved policy quality.',cases=rows)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PIVOT_WIRING_VALIDATED '+json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import evaluate_unified_walk_run
    evaluate_unified_walk_run.evaluate=validate
    sys.argv=[str(SOURCE/'train_anon_locomotion.py'),'--mode','eval','--headless',
              '--num-envs','4','--checkpoint',str(args.checkpoint),'--config-override',str(args.config),
              '--output',str(args.output)]
    runpy.run_path(str(SOURCE/'train_anon_locomotion.py'),run_name='__main__')
