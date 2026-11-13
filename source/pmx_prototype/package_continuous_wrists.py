"""Publish locally only after both continuous and boundary evaluations pass."""
import json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'output/anon_locomotion/continuous_wrists'
reports=[json.load(open(work/folder/'report.json')) for folder in ['final_eval','boundary_eval']]
assert all(r['passed_continuous_gate'] and r['runtime_compensation_disabled'] and r['total_falls']==0 for r in reports)
assert len({r['checkpoint'] for r in reports})==1
checkpoint=Path(reports[0]['checkpoint']).resolve();run=checkpoint.parent
out=ROOT/'output/anon_locomotion/release_v6_continuous_wrists';out.mkdir(exist_ok=True,parents=True)
shutil.copy2(checkpoint,out/'policy.pt')
shutil.copy2(work/'final_eval/policy_jit.pt',out/'policy_jit.pt')
for name in ['agent_config.json','full_motion.npz']:shutil.copy2(run/name,out/name)
cfg=json.load(open(run/'env_config.json'));cfg['full_motion_path']=str(out/'full_motion.npz');cfg['demo_pivots']=True
paths=[]
for i,source in enumerate(cfg['amp_expert_paths']):
    dest=out/f'amp_expert_{i}.npz';shutil.copy2(source,dest);paths.append(str(dest))
cfg['amp_expert_paths']=paths;cfg.pop('runtime_arm_clearance',None)
(out/'env_config.json').write_text(json.dumps(cfg,indent=2))
for name in ['ATTRIBUTION.txt','SOURCES.json']:shutil.copy2(work/name,out/name)
result=dict(checkpoint=str(out/'policy.pt'),policy_sha256=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest(),dofs=31,observations=248,num_envs=8192,translation_max_m_s=.4,yaw_max_rad_s=.6,evaluations=reports,runtime_arm_compensation=False)
(work/'result.json').write_text(json.dumps(result,indent=2));print(out)
