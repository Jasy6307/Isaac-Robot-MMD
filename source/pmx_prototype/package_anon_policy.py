"""Package a selected checkpoint, controller config, style library and validation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    dest=args.destination.resolve(); dest.mkdir(parents=True,exist_ok=True)
    run=args.checkpoint.parent
    config=json.loads((run/'env_config.json').read_text(encoding='utf-8'))
    report=json.loads((args.evaluation/'report.json').read_text(encoding='utf-8'))
    if not any(report.get(k,False) for k in ['passed_initial_demo','passed_forward_gate','passed_bank_gate']):
        raise ValueError('Checkpoint did not pass control validation')
    if config.get('amp_enabled') and not report.get('passed_amp_naturalness_gate'):
        raise ValueError('AMP release requires coordination and visual acceptance in addition to control validation')
    if report.get('checkpoint') and Path(report['checkpoint']).resolve()!=args.checkpoint.resolve():
        raise ValueError('Evaluation belongs to a different checkpoint')
    if config.get('full_motion_path'):
        config.update(report.get('env_config',{}))
    shutil.copyfile(args.checkpoint,dest/'policy.pt')
    shutil.copyfile(args.evaluation/'policy_jit.pt',dest/'residual_actor_jit.pt')
    if config.get('mocap_library'):
        shutil.copyfile(config['mocap_library'],dest/'mocap_library.npz')
        config['mocap_library']=str(dest/'mocap_library.npz')
    if config.get('full_motion_path'):
        shutil.copyfile(config['full_motion_path'],dest/'full_motion.npz')
        config['full_motion_path']=str(dest/'full_motion.npz')
    if config.get('amp_expert_paths'):
        paths=[]
        for i,path in enumerate(config['amp_expert_paths']):
            target=dest/f'amp_expert_{i}.npz';shutil.copyfile(path,target);paths.append(str(target))
        config['amp_expert_paths']=paths
        config['max_yaw']=0.
        for name in ['amp_motion_prior.py','evaluate_amp_forward.py','compare_amp_coordination.py']:
            shutil.copyfile(HERE/name,dest/name)
    (dest/'env_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    shutil.copyfile(run/'agent_config.json',dest/'agent_config.json')
    for name in ['anon_locomotion_env.py','train_anon_locomotion.py','natural_walk_reference.py',
                 'mocap_walk_reference.py','physical_character.py','build_anon_v3.py','anon_v5_config.json','anon_v5_flow_config.json',
                 'anon_imitation_env.py','full_motion_reference.py','build_full_motion.py','build_motion_bank.py',
                 'continuous_motion_reference.py','continuous_command_suite.py','evaluate_continuous.py']:
        shutil.copyfile(HERE/name,dest/name)
    for name,target in [('report.json','evaluation.json'),('gait_comparison.json','gait_comparison.json')]:
        if (args.evaluation/name).exists():shutil.copyfile(args.evaluation/name,dest/target)
    for name in ['ATTRIBUTION.txt','SOURCES.json','reference_processing.json']:
        shutil.copyfile(ROOT/'output/anon_locomotion/release_v4_100style'/name,dest/name)
    if config.get('full_motion_path'):
        helpers=['full_body_retarget.py','plant_mocap_feet.py','balance_full_motion.py',
                 'smooth_full_motion.py','mirror_full_motion.py','build_turn_motion.py',
                 'retarget_100style.py','summarize_anon_training_time.py']
        for name in helpers:shutil.copyfile(HERE/name,dest/name)
        processing=dict(source='100STYLE Neutral, CC BY 4.0',
            reference='Complete joint, pelvis, trunk, velocity and contact states',
            processing=['Anatomical full-body retargeting','Captured contact timing and foot planting',
                        'Mass-aware balance adjustment for backward and sideways motion',
                        'Bounded joint velocity/acceleration cleanup','Whole-state transition blending'],
            procedural_foot_gait=False,root_assistance=False,
            command_interface='Continuous command-conditioned complete-state blending with bounded command acceleration' if config.get('continuous_commands') else 'Nearest captured/retargeted command in the frozen reference bank; continuous arbitrary-direction interpolation is not implemented',
            initialization='Captured state at episode reset; continuous evaluation starts from stationary idle')
        (dest/'reference_processing.json').write_text(json.dumps(processing,indent=2),encoding='utf-8')
        if config.get('amp_enabled'):
            processing.update(source='100STYLE Neutral FW; two stable cycles plus their averaged forward reference',
                method='Four-state full-body least-squares adversarial motion prior with R1 gradient penalty and replay',
                discriminator_feature_size=84,temporal_frames=4,
                runtime_arm_compensation=False,spatial_clearance_only=True,
                command_interface='One forward gait at 0.4 m/s plus stand/stop; other directions not trained in this release',
                upper_nominal_reference_mix=config.get('amp_upper_reference_mix'),upper_drive_scale=config.get('upper_drive_scale'))
            (dest/'reference_processing.json').write_text(json.dumps(processing,indent=2),encoding='utf-8')
    info=dict(selected_checkpoint=str(args.checkpoint.resolve()),
              policy_sha256=hashlib.sha256((dest/'policy.pt').read_bytes()).hexdigest(),
              evaluation=str(args.evaluation.resolve()),
              method=report.get('method','PPO with 100STYLE style and sole slip constraints'),
              runtime='Full-motion imitation' if config.get('full_motion_path') else '89 observation inputs; 25 residual actions; controller filter state reset on episode reset')
    (dest/'release.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__': main()
