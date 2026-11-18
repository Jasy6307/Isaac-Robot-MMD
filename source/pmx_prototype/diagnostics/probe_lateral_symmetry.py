"""Paired lateral inference and mirror diagnostics; performs no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout


def diagnose(env,runner,output,checkpoint):
    import numpy as np
    import torch
    from lateral_symmetry import AnatomicalMirror
    from amp_motion_prior import features
    from unified_walk_run_reference import heading_velocity
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    mirror=AnatomicalMirror.from_env(env);actor=runner.alg.actor;actor.eval()
    pairs=env.num_envs//2
    left_phase=torch.arange(pairs,device=env.device)/4
    phase=torch.cat([left_phase,left_phase+mirror.lateral_phase_delta])
    commands=torch.zeros_like(env.commands);commands[:pairs,1]=1.2;commands[pairs:,1]=-1.2
    env.max_episode_length=100000
    def reset():
        env.reset(env.ids,phase=phase,heading=torch.zeros_like(phase),command=[0,0,0])
        # reset() samples evaluation idle commands. Initialize each physical
        # state from its own explicit lateral command and paired clock below.
        env.commands[:]=commands;env.continuous_ref_commands[:]=commands;env.continuous_ref_rate.zero_()
        ref=env.sample_reference(phase)
        q=ref['q'][:,env.q_order];qd=ref['qd'][:,env.q_order]
        state=env.initial_root.clone();state[:,:3]=env.origins+ref['positions'][:,env.base_reference_id]
        state[:,3:]=ref['rotations'][:,env.base_reference_id]
        env.robot.set_root_transforms(state,env.ids);env.robot.set_root_velocities(env.reference_root_velocity(ref),env.ids)
        env.robot.set_dof_positions(q,env.ids);env.robot.set_dof_velocities(qd,env.ids);env.robot.set_dof_position_targets(q,env.ids)
        env.refresh();return ref
    ref=reset();obs=env.get_observations()
    with torch.no_grad():
        predicted=actor(obs);exact_mirror=obs[:pairs].clone()
        exact_mirror['policy']=mirror.observations(obs['policy'][:pairs])
        predicted_mirror=actor(exact_mirror)
        gap=predicted_mirror-mirror.joints(predicted[:pairs])
        raw_gap=obs['policy'][pairs:]-exact_mirror['policy']
        qgap=ref['q'][pairs:,env.q_order]-mirror.joints(ref['q'][:pairs,env.q_order])
        x=features(env.q,env.qd,env.poses[:,env.pelvis_id,:3],env.poses[:,env.pelvis_id,3:],env.root_vel,env.poses[:,env.key_ids,:3])
        flat=x[:pairs,None,:].expand(-1,4,-1).flatten(1)
        mirrored=mirror.amp(flat.reshape(pairs,4,-1)).flatten(1)
        scores=env.amp.discriminator(env.amp.normalize(flat)).flatten()
        mirror_scores=env.amp.discriminator(env.amp.normalize(mirrored)).flatten()
    result=dict(training=False,checkpoint=str(checkpoint),checkpoint_sha256=digest,pairs=pairs,
        phases=left_phase.cpu().tolist(),phase_delta_cycles=mirror.lateral_phase_delta,
        reference_joint_mirror_max_rad=float(qgap.abs().max()),initial_raw_obs_mirror_rms=float(raw_gap.square().mean().sqrt()),
        initial_policy_equivariance_rms=float(gap.square().mean().sqrt()),
        initial_policy_equivariance_max=float(gap.abs().max()),
        amp_exact_mirror_score_difference_mean=float((scores-mirror_scores).mean()),
        amp_exact_mirror_score_difference_abs_mean=float((scores-mirror_scores).abs().mean()),cases=[])
    if args.verify_hooks:
        from lateral_symmetry import configure,install_normalization,augmentation,paired_reference_std
        from rsl_rl.extensions import Symmetry
        from amp_motion_prior import AMPMotionPrior
        models=[runner.alg.actor,runner.alg.critic]
        before=[{k:v.clone() for k,v in model.state_dict().items()} for model in models]
        original_updates=[model.update_normalization for model in models]
        old_cfg=dict(env.cfg);old_evaluation=env.evaluation;old_prior=env.amp
        env.cfg.update(train_lateral_mirror=True,train_mirror_augmentation=True,train_mirror_loss_coeff=.05,amp_mirror_augmentation=True)
        settings={'algorithm':{}}
        configure(settings,env);install_normalization(runner,env)
        env.evaluation=False
        training_obs=env.get_observations()
        paired,_=augmentation(env,training_obs)
        ext=Symmetry(env=env,**settings['algorithm']['symmetry_cfg'])
        with torch.enable_grad():
            loss=ext.compute_loss(actor,type('Batch',(),{'observations':paired})(),env.num_envs)
            (.05*loss).backward()
            gradient=sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)
        runner.alg.optimizer.zero_grad(set_to_none=True)
        # Exercise real expert loading and raw AMP augmentation without updating
        # either optimizer or any saved model.
        paired_prior=AMPMotionPrior(env)
        half=len(paired_prior.expert)//2
        expert_gap=(paired_prior.expert[half:]-mirror.amp(paired_prior.expert[:half].reshape(-1,4,96)).flatten(1)).abs().max()
        paired_prior.load_state_dict(old_prior.state_dict())
        amp_identity=(paired_prior.discriminator(paired_prior.normalize(flat))-old_prior.discriminator(old_prior.normalize(flat))).abs().max()
        reward_score_gap=(paired_prior.score(flat)-paired_prior.score(mirrored)).abs().max()
        reference_q=env.motion.data['q'].flatten(0,1)[:,env.q_order]
        symmetric_scale=paired_reference_std(reference_q,mirror,env.cfg.get('amp_phase_std_floor',.06))
        scale_gap=(symmetric_scale-symmetric_scale[mirror.index]).abs().max()
        identity=all(torch.equal(value,before[i][key]) for i,model in enumerate(models) for key,value in model.state_dict().items())
        checks=dict(policy_channels=training_obs['policy'].shape[-1],eligible_samples=int(training_obs['mirror_eligible'].sum()),
            augmented_samples=len(paired),mirror_loss=float(loss.detach()),mirror_gradient_l1=gradient,
            model_and_normalizer_unchanged=identity,amp_expert_mirror_max_error=float(expert_gap),
            amp_checkpoint_score_identity_max_error=float(amp_identity.detach()),
            augmented_amp_reward_score_mirror_max_error=float(reward_score_gap.detach()),
            phase_reward_paired_scale_max_error=float(scale_gap),optimizer_steps=0)
        (output/'hook_verification.json').write_text(json.dumps(checks,indent=2))
        assert identity and gradient>0 and float(expert_gap)==0 and float(amp_identity.detach())==0 and float(scale_gap)==0 and float(reward_score_gap.detach())==0
        result['hook_verification']=checks
        for model,method in zip(models,original_updates):model.update_normalization=method
        env.cfg=old_cfg;env.evaluation=old_evaluation;env.amp=old_prior
        del paired_prior
        print('LATERAL_HOOK_VERIFIED '+json.dumps(checks),flush=True)
    print('LATERAL_INITIAL_DIAGNOSTICS '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    traces={}
    for mode in ['paired_actions','native_policy']:
        reset();rows={key:[] for key in ['velocity','reward','done','actions','q','reference_q','amp_score_gap','amp_style_gap']}
        duration=1. if mode=='paired_actions' else 12.
        for step in range(round(duration/env.dt)):
            env.commands[:]=commands
            with torch.no_grad():
                obs=env.get_observations();action=actor(obs)
                if mode=='paired_actions':action[pairs:]=mirror.joints(action[:pairs])
                _,reward,done,_=env.step(action)
                left=env.amp.history[:pairs].flatten(1);right=mirror.amp(env.amp.history[:pairs]).flatten(1)
                dl=env.amp.discriminator(env.amp.normalize(left)).flatten();dr=env.amp.discriminator(env.amp.normalize(right)).flatten()
                dg=dl-dr;style_gap=(1-.25*(dl-1).square()).clamp(0,1)-(1-.25*(dr-1).square()).clamp(0,1)
                current=env.sample_reference(env.phase)
                values=dict(velocity=heading_velocity(env.root_vel[:,:3],env.reference_heading),reward=reward,done=done,
                    actions=action,q=env.q,reference_q=current['q'][:,env.q_order],amp_score_gap=dg,amp_style_gap=style_gap)
                for key,value in values.items():rows[key].append(value.cpu().numpy().copy())
        arrays={key:np.asarray(value) for key,value in rows.items()};np.savez_compressed(output/(mode+'.npz'),**arrays)
        a=0 if mode=='paired_actions' else round(6/env.dt)
        mean=arrays['velocity'][a:].mean(0)
        result['cases'].append(dict(mode=mode,seconds=duration,
            left_speed_mean=float(mean[:pairs,1].mean()),right_speed_mean=float(-mean[pairs:,1].mean()),
            left_falls=int(arrays['done'][:,:pairs].sum()),right_falls=int(arrays['done'][:,pairs:].sum()),
            paired_reward_gap_mean=float((arrays['reward'][:,:pairs]-arrays['reward'][:,pairs:]).mean()),
            paired_reward_gap_abs_mean=float(abs(arrays['reward'][:,:pairs]-arrays['reward'][:,pairs:]).mean()),
            amp_exact_mirror_score_gap_abs_mean=float(abs(arrays['amp_score_gap']).mean()),
            amp_exact_mirror_style_gap_abs_mean=float(abs(arrays['amp_style_gap']).mean())))
        print('LATERAL_CASE '+json.dumps(result['cases'][-1]),flush=True)
    result.update(wall_seconds=time.perf_counter()-started,checkpoint_unchanged=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==digest)
    (output/'report.json').write_text(json.dumps(result,indent=2));print('LATERAL_MIRROR_PROBE '+json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-hooks',action='store_true',help='Verify real training hooks/gradients with zero optimizer steps')
    parser.add_argument('--config-override',type=Path)
    args=parser.parse_args()
    import evaluate_unified_walk_run
    evaluate_unified_walk_run.evaluate=diagnose
    sys.argv=[str(SOURCE/'train_anon_locomotion.py'),'--mode','eval','--headless','--num-envs','32',
        '--checkpoint',str(args.checkpoint),'--output',str(args.output)]
    if args.config_override:sys.argv+=['--config-override',str(args.config_override)]
    runpy.run_path(str(SOURCE/'train_anon_locomotion.py'),run_name='__main__')
