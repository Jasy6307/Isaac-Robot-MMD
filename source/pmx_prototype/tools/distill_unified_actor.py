"""Offline physical-teacher distillation into one unified actor; no runtime teachers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT
import argparse,copy,json,shutil,time
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from policy_normalization_rebase import rebase_state
from source_bundle import copy_sources

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=3000)
args=p.parse_args();torch.manual_seed(42);torch.set_num_threads(4)
device='cuda:0';out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
data=np.load(args.data);x=torch.tensor(data['observations'],device=device);y=torch.tensor(data['actions'],device=device)
groups=torch.tensor(data['groups'],device=device);n=len(x)
state=torch.load(args.checkpoint,map_location=device,weights_only=False)
agent=json.loads((args.checkpoint.parent/'agent_config.json').read_text())
observations=TensorDict({'policy':x[:1]},batch_size=[1])
actor_options={key:value for key,value in agent['actor'].items() if key!='class_name'}
actor=MLPModel(observations,agent['obs_groups'],'actor',y.shape[1],**actor_options).to(device)
before=None
with torch.no_grad():
    actor.load_state_dict(state['actor_state_dict']);actor.eval();before=actor(TensorDict({'policy':x},batch_size=[n])).clone()
    for role in ['actor','critic']:
        statistics=dict(state[role+'_state_dict'])
        mean=x.mean(0,keepdim=True);variance=x.var(0,unbiased=False,keepdim=True).clamp_min(.06**2)
        statistics.update({'obs_normalizer._mean':mean,'obs_normalizer._var':variance,'obs_normalizer._std':variance.sqrt(),
                           'obs_normalizer.count':torch.full_like(statistics['obs_normalizer.count'],float(n))})
        state[role+'_state_dict']=rebase_state(state[role+'_state_dict'],statistics)
    actor.load_state_dict(state['actor_state_dict']);actor.obs_normalizer.until=0;actor.eval()
    parity=float((actor(TensorDict({'policy':x},batch_size=[n]))-before).abs().max())
    if parity>1e-4:raise RuntimeError(f'Functional normalization rebase failed: {parity}')

# Hold out contiguous temporal blocks, rather than interleaved adjacent frames.
validation=torch.arange(n,device=device)%300>=240
train=[((groups==g)&~validation).nonzero().flatten() for g in [0,1]]
valid=[((groups==g)&validation).nonzero().flatten() for g in [0,1]]
optimizer=torch.optim.Adam(actor.mlp.parameters(),lr=1e-4)
best=None;best_score=float('inf');history=[];started=time.perf_counter()
for step in range(args.steps):
    indices=torch.cat([ids[torch.randint(len(ids),(1024,),device=device)] for ids in train])
    predicted=actor(TensorDict({'policy':x[indices]},batch_size=[len(indices)]))
    loss=(predicted-y[indices]).square().mean()
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(),1.);optimizer.step()
    if step%50==0 or step==args.steps-1:
        with torch.no_grad():
            error=[float((actor(TensorDict({'policy':x[ids]},batch_size=[len(ids)]))-y[ids]).square().mean().sqrt()) for ids in valid]
        item=dict(step=step,loss=float(loss),low_validation_action_rmse=error[0],run_validation_action_rmse=error[1],elapsed_seconds=time.perf_counter()-started)
        history.append(item);score=sum(error)
        if score<best_score:best_score=score;best=copy.deepcopy(actor.state_dict())
        (out/'status.json').write_text(json.dumps(item,indent=2));print('DISTILL',json.dumps(item),flush=True)
        if max(error)<.018 and step>=300:break
actor.load_state_dict(best);state['actor_state_dict']=best
torch.save(state,out/'policy_final.pt')
cfg=json.loads((args.checkpoint.parent/'env_config.json').read_text())
for key in ['full_motion_path']:
    target=out/Path(cfg[key]).name;shutil.copyfile(cfg[key],target);cfg[key]=str(target)
experts=[]
for i,path in enumerate(cfg['amp_expert_paths']):
    target=out/f'amp_expert_{i}.npz';shutil.copyfile(path,target);experts.append(str(target))
cfg['amp_expert_paths']=experts;cfg['train_freeze_obs_normalization']=True
cfg['offline_single_actor_distillation']=dict(source_checkpoint=str(args.checkpoint.resolve()),dataset=str(args.data.resolve()),runtime_teachers=False)
(out/'env_config.json').write_text(json.dumps(cfg,indent=2));(out/'agent_config.json').write_text(json.dumps(agent,indent=2))
seconds=time.perf_counter()-started
(out/'training_time.json').write_text(json.dumps(dict(training_seconds=seconds,completed_steps=step+1,definition='Offline actor optimization plus save; excludes physical teacher data collection'),indent=2))
(out/'distillation.json').write_text(json.dumps(dict(normalization_rebase_max_action_delta=parity,history=history,best_validation_sum_rmse=best_score,samples=n,one_actor=True),indent=2))
copy_sources(out,['tools/distill_unified_actor.py','tools/collect_unified_teachers.py','anon_imitation_env.py','unified_walk_run_reference.py','policy_normalization_rebase.py'])
print('DISTILL_COMPLETE',seconds,flush=True)
