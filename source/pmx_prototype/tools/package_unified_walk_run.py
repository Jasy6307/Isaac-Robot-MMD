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
p.add_argument('--baseline-cpu-report',type=Path,help='Require a stance-slip improvement against this CPU baseline')
p.add_argument('--baseline-gpu-report',type=Path,help='Require a stance-slip improvement against this GPU baseline')
p.add_argument('--minimum-slip-reduction',type=float,default=.2)
p.add_argument('--output',type=Path,default=ROOT/'output/anon_locomotion/release_v6_unified_walk_run')
a=p.parse_args()
reports=[json.loads(path.read_text(encoding='utf-8')) for path in [a.gpu_report,a.cpu_report]]
assert all(r['passed_unified_gate'] and r['complete_scope'] and r['total_falls']==0 for r in reports),'Full walk/run acceptance has not passed'
plan=json.loads((ROOT/'output/anon_locomotion/unified_walk_run/acceptance_plan.json').read_text(encoding='utf-8'))
assert all(r.get('evaluator_sha256')==plan['evaluator_sha256'] for r in reports),'Acceptance evaluator differs from the recorded plan'
assert all(r.get('flight_acceptance_method')=='Both physical foot support loads below 1 N' for r in reports),'Force-based running acceptance is required'
assert all(r.get('velocity_command_frame')=='Controller reference heading' for r in reports),'Command-heading velocity measurement is required'
assert all(all(s.get('force_sample_hz')==120 for s in r['segments']) for r in reports),'120 Hz support-force sampling is required'
assert len({r['checkpoint'] for r in reports})==1,'CPU/GPU must evaluate the same checkpoint'
assert {r['env_config'].get('simulation_device','cuda:0') for r in reports}=={'cpu','cuda:0'}
checkpoint=Path(reports[0]['checkpoint']).resolve();run=checkpoint.parent
slip_comparisons={}
if a.baseline_cpu_report or a.baseline_gpu_report:
    assert a.baseline_cpu_report and a.baseline_gpu_report,'Both backend baselines are required'
    from compare_stance_slip import compare
    for backend,baseline,candidate in [('cpu',a.baseline_cpu_report,a.cpu_report),('gpu',a.baseline_gpu_report,a.gpu_report)]:
        result=compare(baseline,candidate,a.minimum_slip_reduction)
        assert result['passed'],f'{backend}: directional stance-slip improvement has not passed'
        slip_comparisons[backend]=result
jit=a.cpu_report.parent/'policy_jit.pt'
assert jit.is_file(),'CPU evaluation JIT export is missing'
cfg=json.loads((run/'env_config.json').read_text(encoding='utf-8'))
assert cfg['unified_walk_run'] and cfg['max_speed']==1.2 and cfg['max_yaw']==.6 and not cfg.get('amp_forward_only')
reference=ROOT/'output/anon_locomotion/unified_walk_run/direction_reference_v2'
motion_hash=hashlib.sha256(Path(cfg['full_motion_path']).read_bytes()).hexdigest()
assert motion_hash==hashlib.sha256((reference/'full_motion.npz').read_bytes()).hexdigest(),'The accepted model does not use the validated cycle-corrected reference bank'
out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
shutil.copy2(checkpoint,out/'policy.pt');shutil.copy2(run/'agent_config.json',out/'agent_config.json')
shutil.copy2(jit,out/'policy_jit.pt')
shutil.copy2(cfg['full_motion_path'],out/'full_motion.npz');cfg['full_motion_path']=str(out/'full_motion.npz')
paths=[]
for i,source in enumerate(cfg['amp_expert_paths']):
    destination=out/f'amp_expert_{i}.npz';shutil.copy2(source,destination);paths.append(str(destination))
cfg['amp_expert_paths']=paths;cfg.pop('amp_reset_discriminator',None);cfg['runtime_inference_only']=False
if cfg.get('train_teacher_anchor_path'):
    shutil.copy2(cfg['train_teacher_anchor_path'],out/'training_teacher_anchor.npz')
    cfg['train_teacher_anchor_path']=str(out/'training_teacher_anchor.npz')
cfg['reference_kind']='complete_100STYLE_unified_directional_walk_run_v2'
cfg['reference_revision']='direction_reference_v2'
cfg['unified_command_frame_metrics']=True
cfg['reference_provenance_path']=str(out/'reference_provenance.json')
(out/'env_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
for name in ['ATTRIBUTION.txt','acceptance_plan.json','experiment_plan.json']:
    source=ROOT/'output/anon_locomotion/unified_walk_run'/name
    if source.exists():shutil.copy2(source,out/name)
shutil.copy2(reference/'reference_report.json',out/'reference_report.json')
preparation=out/'reference_preparation';preparation.mkdir(exist_ok=True)
for name in ['backward_raw.json','backward_balanced.json','left_raw.json','left_balanced.json']:
    shutil.copy2(reference/name,preparation/name)
provenance=dict(full_motion_sha256=motion_hash,source_reference_bank=str(reference/'full_motion.npz'),
                backward_cycle_anchors=json.loads((reference/'backward_raw.json').read_text(encoding='utf-8')),
                left_cycle_anchors=json.loads((reference/'left_raw.json').read_text(encoding='utf-8')),
                source_bvh_sha256={name:hashlib.sha256((ROOT/'media/mocap/100style'/name).read_bytes()).hexdigest()
                                   for name in ['Neutral_FR.bvh','Neutral_BR.bvh','Neutral_SR.bvh']})
(out/'reference_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
result=dict(checkpoint=str(out/'policy.pt'),source_checkpoint=str(checkpoint),policy_sha256=hashlib.sha256((out/'policy.pt').read_bytes()).hexdigest(),
            num_actions=31,num_observations=248,num_envs=8192,max_speed_m_s=1.2,max_yaw_rad_s=.6,policy_count=1,
            runtime_compensation_removed=True,full_motion_sha256=motion_hash,full_acceptance=reports)
(out/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if slip_comparisons:
    (out/'stance_slip_acceptance.json').write_text(json.dumps(slip_comparisons,indent=2),encoding='utf-8')
copy_sources(out, ['train_anon_locomotion.py', 'evaluate_continuous.py',
                   'build_anon_v3.py', 'build_anon_v4.py',
                   'compare_stance_slip.py', 'compare_locomotion_video.py'])
print(out)
