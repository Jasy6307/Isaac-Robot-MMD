"""Publish one unified actor only after full CPU and GPU acceptance."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse,hashlib,json,shutil
from pathlib import Path
from source_bundle import copy_sources

ROOT=PROJECT_ROOT
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu-report',type=Path,required=True)
p.add_argument('--cpu-report',type=Path,required=True)
p.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/release_v6_unified_walk_run')
a=p.parse_args()
reports=[json.loads(path.read_text(encoding='utf-8')) for path in [a.gpu_report,a.cpu_report]]
assert all(r['passed_unified_gate'] and r['complete_scope'] and r['total_falls']==0 for r in reports),'Full walk/run acceptance has not passed'
assert len({r['checkpoint'] for r in reports})==1,'CPU/GPU must evaluate the same checkpoint'
assert {r['env_config'].get('simulation_device','cuda:0') for r in reports}=={'cpu','cuda:0'}
checkpoint=Path(reports[0]['checkpoint']).resolve();run=checkpoint.parent
jit=a.cpu_report.parent/'policy_jit.pt'
assert jit.is_file(),'CPU evaluation JIT export is missing'
cfg=json.loads((run/'env_config.json').read_text(encoding='utf-8'))
assert cfg['unified_walk_run'] and cfg['max_speed']==1.2 and cfg['max_yaw']==.6 and not cfg.get('amp_forward_only')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
shutil.copy2(checkpoint,out/'policy.pt');shutil.copy2(run/'agent_config.json',out/'agent_config.json')
shutil.copy2(jit,out/'policy_jit.pt')
shutil.copy2(cfg['full_motion_path'],out/'full_motion.npz');cfg['full_motion_path']=str(out/'full_motion.npz')
paths=[]
for i,source in enumerate(cfg['amp_expert_paths']):
    destination=out/f'amp_expert_{i}.npz';shutil.copy2(source,destination);paths.append(str(destination))
cfg['amp_expert_paths']=paths;cfg.pop('amp_reset_discriminator',None);cfg['runtime_inference_only']=False
(out/'env_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
for name in ['ATTRIBUTION.txt','reference_report.json','acceptance_plan.json','experiment_plan.json']:
    source=ROOT/'output/anon_locomotion/unified_walk_run'/name
    if source.exists():shutil.copy2(source,out/name)
result=dict(checkpoint=str(out/'policy.pt'),source_checkpoint=str(checkpoint),policy_sha256=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest(),
            num_actions=31,num_observations=248,num_envs=8192,max_speed_m_s=1.2,max_yaw_rad_s=.6,policy_count=1,
            runtime_compensation_removed=True,full_acceptance=reports)
(out/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
copy_sources(out, ['train_anon_locomotion.py', 'evaluate_continuous.py',
                   'build_anon_v3.py', 'build_anon_v4.py'])
print(out)
