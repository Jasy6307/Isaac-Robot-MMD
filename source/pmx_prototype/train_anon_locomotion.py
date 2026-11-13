"""Train or evaluate Anon's physically simulated residual walking policy."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['train', 'eval', 'smoke', 'reference'], default='train')
parser.add_argument('--num-envs', type=int, default=2048)
parser.add_argument('--iterations', type=int, default=1800)
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--run-dir', type=Path)
parser.add_argument('--output', type=Path, default=ROOT/'output/anon_locomotion/evaluation')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--zero-policy', action='store_true')
parser.add_argument('--residual-scale', type=float)
parser.add_argument('--suite', choices=['demo', 'compass'], default='demo')
parser.add_argument('--linear-speed', type=float, default=.25, help='Speed for the 16-direction compass suite')
parser.add_argument('--yaw-speed', type=float, default=.35)
parser.add_argument('--v2', action='store_true', help='Enable physical arm swing, smoother reference and natural-gait costs')
parser.add_argument('--arm-swing-gain', type=float, default=None)
parser.add_argument('--curriculum-iterations', type=int)
parser.add_argument('--v3', action='store_true', help='25-DOF learned upper body with authored procedural motion reference')
parser.add_argument('--mocap-library', type=Path, help='Retargeted 100STYLE periodic style library; requires a full body policy')
parser.add_argument('--config-override',type=Path,help='JSON environment settings applied after checkpoint config')
args = parser.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting', 'fast_shutdown': True,
                     'extra_args': ['--/app/asyncRendering=false']})
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from anon_locomotion_env import AnonLocomotionEnv


def agent_config():
    return dict(num_steps_per_env=48, save_interval=100, obs_groups={'actor':['policy'], 'critic':['policy']},
                actor=dict(class_name='MLPModel', hidden_dims=[256, 128, 128], activation='elu',
                           obs_normalization=True, distribution_cfg=dict(class_name='GaussianDistribution', init_std=.5, std_type='log')),
                critic=dict(class_name='MLPModel', hidden_dims=[256, 128, 128], activation='elu', obs_normalization=True),
                algorithm=dict(class_name='PPO', value_loss_coef=1., use_clipped_value_loss=True,
                               clip_param=.2, entropy_coef=.006, num_learning_epochs=5, num_mini_batches=4,
                               learning_rate=.0007, schedule='adaptive', gamma=.99, lam=.95,
                               desired_kl=.01, max_grad_norm=1., rnd_cfg=None, symmetry_cfg=None),
                logger='tensorboard', check_for_nan=True)


def load_policy(runner, env, checkpoint):
    state = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    if hasattr(env,'amp') and 'amp_state' in state and not env.cfg.get('amp_reset_discriminator',False):env.amp.load_state_dict(state['amp_state'])
    old_inputs=state['actor_state_dict']['mlp.0.weight'].shape[1]
    new_inputs=env.get_observations()['policy'].shape[1]
    if env.cfg.get('migration_old_dof_names') and state['actor_state_dict']['mlp.6.bias'].numel()!=env.num_actions:
        from migrate_articulated_policy import migrate
        return migrate(runner,env,state,env.cfg['migration_old_dof_names'])
    if state['actor_state_dict']['mlp.6.bias'].numel()==env.num_actions and old_inputs<new_inputs:
        for role,model in [('actor',runner.alg.actor),('critic',runner.alg.critic)]:
            target=model.state_dict()
            for name,old in state[role+'_state_dict'].items():
                value=target[name]
                if value.shape==old.shape:value.copy_(old)
                else:
                    value.fill_(1. if name.endswith('_var') or name.endswith('_std') else 0.)
                    value[tuple(slice(0,n) for n in old.shape)]=old
                if name=='obs_normalizer.count':value.fill_(10000.)
            model.load_state_dict(target)
        runner.current_learning_iteration=int(state['iter'])
        return dict(kind='append_observations_zero_new_weights',old_inputs=old_inputs,new_inputs=new_inputs,optimizer='fresh')
    if state['actor_state_dict']['mlp.6.bias'].numel() == env.num_actions:
        runner.load(str(checkpoint), map_location='cuda:0')
        return None
    if not env.full_body or state['actor_state_dict']['mlp.6.bias'].numel() != 12:
        raise ValueError('Unsupported checkpoint/action mapping')
    for role, model in [('actor', runner.alg.actor), ('critic', runner.alg.critic)]:
        target = model.state_dict()
        for name, old in state[role+'_state_dict'].items():
            value = target[name]
            if value.shape == old.shape:
                value.copy_(old)
            else:
                if name == 'distribution.log_std_param': value.fill_(-1.89712)
                elif name.endswith('_var') or name.endswith('_std'): value.fill_(1.)
                else: value.zero_()
                value[tuple(slice(0, n) for n in old.shape)] = old
            if name == 'obs_normalizer.count': value.fill_(10000.)
        model.load_state_dict(target)
    runner.current_learning_iteration = int(state['iter'])
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    x = torch.randn(16, 50, device=env.device)
    old_obs = TensorDict({'policy': x}, batch_size=[16])
    old_actor = MLPModel(old_obs, {'actor': ['policy']}, 'actor', 12,
                         hidden_dims=[256, 128, 128], activation='elu', obs_normalization=True,
                         distribution_cfg=dict(class_name='GaussianDistribution', init_std=.5, std_type='log')).to(env.device)
    old_actor.load_state_dict(state['actor_state_dict']); old_actor.eval(); runner.alg.actor.eval()
    with torch.inference_mode():
        old_a = old_actor(old_obs)
        new_a = runner.alg.actor(TensorDict({'policy': torch.cat([x, torch.zeros(16, 39, device=env.device)], -1)}, batch_size=[16]))
        parity = float((old_a-new_a[:, :12]).abs().max())
        upper_zero = float(new_a[:, 12:].abs().max())
    if parity > 1e-5 or upper_zero > 1e-5: raise RuntimeError(f'Migration parity failed: {parity}, {upper_zero}')
    return dict(source=str(checkpoint), source_actions=12, target_actions=25,
                source_observations=50, target_observations=89,
                leg_output_max_error=parity, initial_upper_output_max_abs=upper_zero,
                mapping='Preserve first 50 observation channels and first 12 leg outputs; zero new inputs/upper means',
                optimizer='fresh Adam state for expanded parameter tensors')


def export_reference(env):
    if not env.full_body: raise ValueError('Reference preview requires --v3')
    schedule = [('stand', 2., (0, 0, 0)), ('forward', 6., (.3, 0, 0)),
                ('stop', 2., (0, 0, 0)), ('backward', 4., (-.2, 0, 0)),
                ('left', 4., (0, .2, 0)), ('right', 4., (0, -.2, 0)),
                ('turn_left', 4., (.2, 0, .4)), ('final_stop', 3., (0, 0, 0))]
    poses, joints, commands, velocities, phases, labels = [], [], [], [], [], []
    env.reset(env.ids); env.phase.zero_(); env.motion_commands.zero_()
    for label, seconds, cmd in schedule:
        env.commands[:] = torch.tensor(cmd, device=env.device)
        for _ in range(round(seconds/env.dt)):
            p, q, v, phase = env.reference_step()
            poses.append(p.cpu().numpy()); joints.append(q.cpu().numpy())
            commands.append(np.tile(cmd, (env.num_envs, 1))); velocities.append(v.cpu().numpy())
            phases.append(phase.cpu().numpy()); labels.append(label)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'rollout.npz', poses=np.array(poses), joints=np.array(joints),
                        commands=np.array(commands), velocities=np.array(velocities), phases=np.array(phases),
                        labels=labels, dones=np.zeros((len(poses), env.num_envs), dtype=bool),
                        link_names=env.links, dof_names=env.names, dt=env.dt, kind='procedural_reference')
    q = np.array(joints)
    limits = env.limits.cpu().numpy()
    report = dict(kind='authored_procedural_reference_not_mocap', seconds=len(poses)*env.dt,
                  dofs=25, finite=bool(np.isfinite(q).all()),
                  joint_limit_violation_rad=float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1]))),
                  physics_steps_during_preview=0, source='natural_walk_reference.py + anatomical leg IK',
                  meaning='Kinematic reference animation, not evidence of physical policy stability', env_config=env.cfg)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ANON_REFERENCE '+json.dumps(report), flush=True)


def evaluate(env, runner):
    if env.cfg.get('full_motion_path'):
        return evaluate_imitation(env,runner)
    env.max_episode_length = round(120./env.dt)
    policy = runner.get_inference_policy() if runner and not args.zero_policy else None
    # Continuous changes, including diagonals, reverse, both turn signs and final stops.
    schedule = [('stand', 3., (0, 0, 0)), ('forward', 5., (.25, 0, 0)),
                ('stop', 3., (0, 0, 0)), ('backward', 5., (-.18, 0, 0)),
                ('left', 5., (0, .18, 0)), ('right', 5., (0, -.18, 0)),
                ('diagonal', 5., (.18, .14, 0)), ('turn_left', 5., (.15, 0, .35)),
                ('turn_right', 5., (.15, 0, -.35)), ('turn_in_place', 5., (0, 0, .35)),
                ('final_stop', 4., (0, 0, 0))]
    if args.suite == 'compass':
        import math
        schedule = [('stand', 2., (0, 0, 0))]
        for k in range(16):
            theta = 2*math.pi*k/16
            schedule.append((f'direction_{k*22.5:g}', 3., (args.linear_speed*math.cos(theta), args.linear_speed*math.sin(theta), 0.)))
            if k % 4 == 3: schedule.append(('stop', 2., (0, 0, 0)))
        schedule += [('turn_left', 4., (.15, 0, args.yaw_speed)), ('turn_right', 4., (.15, 0, -args.yaw_speed)),
                     ('turn_in_place', 4., (0, 0, args.yaw_speed)), ('turn_in_place_right', 4., (0, 0, -args.yaw_speed)),
                     ('final_stop', 3., (0, 0, 0))]
    poses, qtrace, commands, velocities, phases, labels, dones = [], [], [], [], [], [], []
    metrics = []
    with torch.inference_mode():
        env.reset(env.ids)
        env.commands.zero_()
        for label, seconds, cmd in schedule:
            env.commands[:] = torch.tensor(cmd, device=env.device)
            errors, yaw_errors, falls, speeds = [], [], 0, []
            displacement_start = env.robot.get_root_transforms()[:, :3].clone()
            for step in range(round(seconds/env.dt)):
                obs = env.get_observations()
                action = policy(obs) if policy else torch.zeros_like(env.actions)
                _, reward, done, extras = env.step(action)
                physical_poses, physical_q, physical_velocity, physical_phase = env.recorded_state
                poses.append(physical_poses.detach().cpu().numpy())
                qtrace.append(physical_q.detach().cpu().numpy())
                commands.append(np.tile(cmd, (env.num_envs, 1)))
                velocities.append(physical_velocity.detach().cpu().numpy())
                phases.append(physical_phase.detach().cpu().numpy())
                labels.append(label); dones.append(done.detach().cpu().numpy())
                if step >= 30:
                    errors.append(float(torch.linalg.vector_norm(physical_velocity[:, :2]-torch.tensor(cmd[:2], device=env.device), dim=-1).mean()))
                    yaw_errors.append(float((physical_velocity[:, 2]-cmd[2]).abs().mean()))
                    speeds.append(float(torch.linalg.vector_norm(physical_velocity[:, :2], dim=-1).mean()))
                falls += int((done & ~extras['time_outs']).sum())
                # Keep deterministic commands after any safety reset.
                env.commands[:] = torch.tensor(cmd, device=env.device)
            displacement = env.robot.get_root_transforms()[:, :3]-displacement_start
            metrics.append(dict(segment=label, command=cmd, seconds=seconds, falls=falls,
                                linear_error_m_s=float(np.mean(errors)), yaw_error_rad_s=float(np.mean(yaw_errors)),
                                mean_speed_m_s=float(np.mean(speeds)),
                                mean_displacement_m=displacement.mean(0).tolist()))
            print('ANON_EVAL '+json.dumps(metrics[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'rollout.npz', poses=np.array(poses), joints=np.array(qtrace),
                        commands=np.array(commands), velocities=np.array(velocities), phases=np.array(phases),
                        labels=labels, dones=np.array(dones), link_names=env.links, dof_names=env.names, dt=env.dt)
    total_falls = sum(m['falls'] for m in metrics)
    moving = [m for m in metrics if m['segment'] not in ('stand', 'stop', 'final_stop')]
    stops = [m for m in metrics if m['segment'] in ('stand', 'stop', 'final_stop')]
    report = dict(checkpoint=str(args.checkpoint), seed=args.seed, num_envs=env.num_envs,
                  suite=args.suite,
                  total_seconds=sum(s[1] for s in schedule), total_falls=total_falls,
                  mean_moving_linear_error=float(np.mean([m['linear_error_m_s'] for m in moving])),
                  max_stop_speed=float(max(m['mean_speed_m_s'] for m in stops)), segments=metrics,
                  method=('PPO full-body residual with authored procedural pose/velocity reference rewards; no root forcing'
                          if env.full_body else 'PPO residual around phase-conditioned analytical foot-placement reference; no root forcing'),
                  env_config=env.cfg)
    report['max_moving_linear_error'] = max(m['linear_error_m_s'] for m in moving)
    if env.mocap:
        report['method'] = 'PPO full-body residual with 100STYLE Neutral mocap pose/velocity style rewards; no root forcing'
    report['max_segment_yaw_error'] = max(m['yaw_error_rad_s'] for m in metrics)
    report['passed_initial_demo'] = (total_falls == 0 and report['max_moving_linear_error'] < .10
                                     and report['max_segment_yaw_error'] < .15 and report['max_stop_speed'] < .08)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ANON_EVAL_RESULT '+json.dumps(report), flush=True)


def evaluate_imitation(env,runner):
    if env.motion.bank:return evaluate_motion_bank(env,runner)
    env.max_episode_length=round(120/env.dt)
    policy=runner.get_inference_policy() if runner and not args.zero_policy else None
    env.reset(env.ids)
    traces={k:[] for k in ['poses','joints','commands','velocities','phases','dones']}
    logs={};falls=0
    with torch.inference_mode():
        for step in range(900):
            actions=policy(env.get_observations()) if policy else torch.zeros_like(env.actions)
            _,_,done,extras=env.step(actions)
            p,q,v,phase=env.recorded_state
            for key,value in [('poses',p),('joints',q),('commands',env.commands),('velocities',v),('phases',phase),('dones',done)]:
                traces[key].append(value.cpu().numpy())
            falls+=int(done.sum())
            for key,value in extras['log'].items():
                if key.startswith('Imitation/'):logs.setdefault(key,[]).append(float(value))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'rollout.npz',**{k:np.array(v) for k,v in traces.items()},
                        labels=['forward']*900,link_names=env.links,dof_names=env.names,dt=env.dt)
    metrics={k:float(np.mean(v[30:])) for k,v in logs.items()}
    speed=float(np.mean(np.array(traces['velocities'])[30:,:,0]))
    report=dict(checkpoint=str(args.checkpoint),total_seconds=30.,total_falls=falls,mean_forward_speed_m_s=speed,
                target_forward_speed_m_s=env.motion.speed,metrics=metrics,env_config=env.cfg,
                initialization='Captured phase-zero physical joint/root state and velocities; no root forcing after reset',
                method='Complete mocap joint/velocity/keypoint/pelvis/contact imitation, no procedural gait')
    report['passed_forward_gate']=(falls==0 and abs(speed-env.motion.speed)<.15
        and metrics['Imitation/joint_rmse_rad']<.22 and metrics['Imitation/keypoint_rmse_m']<.065
        and metrics['Imitation/pelvis_height_rmse_m']<.035 and metrics['Imitation/contact_accuracy']>.75)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ANON_IMITATION_EVAL '+json.dumps(report),flush=True)


def evaluate_motion_bank(env,runner):
    if env.cfg.get('amp_forward_only'):
        from evaluate_amp_forward import evaluate
        return evaluate(env,runner,args.output,args.checkpoint)
    if env.cfg.get('continuous_commands'):
        from evaluate_continuous import evaluate
        return evaluate(env,runner,args.output,args.checkpoint)
    env.max_episode_length=round(600/env.dt)
    env.cfg['imitation_evaluation_initial_clip']=int(env.motion.commands.norm(dim=-1).argmin())
    env.reset(env.ids)
    policy=runner.get_inference_policy() if runner and not args.zero_policy else None
    schedule=[('forward',6.,(.79330927,0,0)),('forward',6.,(.6,0,0)),('forward',8.,(.4,0,0)),('stop',4.,(0,0,0)),
              ('forward',6.,(.4,0,0)),('forward',6.,(.6,0,0)),('forward',6.,(.79330927,0,0)),('final_stop',4.,(0,0,0))]
    if len(env.motion.clip_names)>4:
        schedule += [('backward',6.,(-.4,0,0)),('stop',2.,(0,0,0)),('left',6.,(0,.35,0)),('right',6.,(0,-.35,0)),('stop',3.,(0,0,0))]
    if bool((env.motion.commands[:,2].abs()>.01).any()):
        schedule += [('turn_left',4.,(.15,0,.6)),('turn_right',4.,(.15,0,-.6)),('turn_in_place',4.,(0,0,.6)),('turn_in_place_right',4.,(0,0,-.6)),('final_stop',3.,(0,0,0))]
    if env.cfg.get('imitation_command_scope')=='translation_040_yaw_060':
        schedule=[('forward',10.,(.4,0,0)),('stop',4.,(0,0,0)),('forward_restart',6.,(.4,0,0)),
                  ('backward',6.,(-.4,0,0)),('stop',3.,(0,0,0)),('left',6.,(0,.35,0)),
                  ('right',6.,(0,-.35,0)),('stop',3.,(0,0,0)),('turn_left',4.,(.15,0,.6)),
                  ('turn_right',4.,(.15,0,-.6)),('turn_in_place',4.,(0,0,.6)),
                  ('turn_in_place_right',4.,(0,0,-.6)),('final_stop',3.,(0,0,0))]
    traces={k:[] for k in ['poses','joints','commands','velocities','phases','dones']};labels=[];segments=[]
    with torch.inference_mode():
        for label,seconds,cmd in schedule:
            if env.cfg.get('imitation_eval_reset_each_segment',False):
                requested=torch.tensor(cmd,device=env.device)
                env.cfg['imitation_evaluation_initial_clip']=int(((env.motion.commands-requested).square().sum(-1)).argmin())
                env.reset(env.ids)
            falls=0;speeds=[];lateral=[];logs={};world_speeds=[];yaw_speeds=[]
            for step in range(round(seconds/env.dt)):
                env.set_command(cmd)
                action=policy(env.get_observations()) if policy else torch.zeros_like(env.actions)
                _,_,done,extras=env.step(action);p,q,v,phase=env.recorded_state
                for key,value in [('poses',p),('joints',q),('commands',env.commands),('velocities',v),('phases',phase),('dones',done)]:
                    traces[key].append(value.cpu().numpy())
                labels.append(label);falls+=int(done.sum())
                if step>=30:
                    speeds.append(float(v[:,0].mean()));lateral.append(float(v[:,1].abs().mean()))
                    theta=env.reference_heading;c=torch.cos(theta);s=torch.sin(theta)
                    vx,vy=env.root_vel[:,0],env.root_vel[:,1]
                    world_speeds.append([float((s*vx-c*vy).mean()),float((c*vx+s*vy).mean())])
                    yaw_speeds.append(float(env.root_vel[:,5].mean()))
                    for key,value in extras['log'].items():
                        if key.startswith('Imitation/'):logs.setdefault(key,[]).append(float(value))
            item=dict(label=label,command=cmd,falls=falls,mean_forward_speed=float(np.mean(speeds)),
                      mean_lateral_speed=float(np.mean(lateral)),mean_heading_velocity=np.mean(world_speeds,axis=0).tolist(),
                      mean_yaw_rate=float(np.mean(yaw_speeds)),metrics={k:float(np.mean(v)) for k,v in logs.items()})
            segments.append(item);print('ANON_BANK_SEGMENT '+json.dumps(item),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'rollout.npz',**{k:np.array(v) for k,v in traces.items()},labels=labels,
                        link_names=env.links,dof_names=env.names,dt=env.dt)
    report=dict(checkpoint=str(args.checkpoint),total_seconds=sum(s[1] for s in schedule),total_falls=sum(s['falls'] for s in segments),
                segments=segments,env_config=env.cfg,method='Full mocap reference bank; continuous state blending; no procedural foot gait or root forcing',
                evaluation_mode='independent_reference_initialized_segments' if env.cfg.get('imitation_eval_reset_each_segment',False) else 'continuous_command_sequence')
    report['passed_bank_gate']=(report['total_falls']==0 and all(np.linalg.norm(np.array(s['mean_heading_velocity'])-np.array(s['command'][:2]))<(.08 if np.linalg.norm(s['command'][:2])==0 else .12) for s in segments)
        and all(abs(s['mean_yaw_rate']-s['command'][2])<.2 for s in segments)
        and all(s['metrics']['Imitation/keypoint_rmse_m']<.075 for s in segments))
    if env.cfg.get('imitation_eval_reset_each_segment',False):
        report['passed_steady_gate']=report['passed_bank_gate'];report['passed_bank_gate']=False
    if env.cfg.get('learn_arm_clearance',False):
        report['runtime_compensation_disabled']=True
        report['runtime_compensation_removed']=True
        report['passed_arm_clearance_gate']=report['runtime_compensation_disabled'] and all(s['metrics'].get('Imitation/near_hip_clearance_deficit_m',1.)<.012 for s in segments if s['label'] in ['forward','forward_restart'])
        report['passed_bank_gate'] &= report['passed_arm_clearance_gate']
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ANON_BANK_EVAL '+json.dumps(report),flush=True)


def main():
    torch.set_num_threads(8)
    config = {}
    if args.checkpoint:
        config = json.loads((args.checkpoint.parent/'env_config.json').read_text(encoding='utf-8'))
        # Discriminator reset is a one-run data migration option, not inherited.
        config.pop('amp_reset_discriminator',None)
        config.pop('train_reset_actor_head',None)
        config.pop('train_leg_std',None)
    if args.residual_scale is not None: config['residual_scale'] = args.residual_scale
    if args.v2: config.update(arm_swing=True, natural_gait=True, max_speed=.4, max_yaw=.6,
                              lateral_stance_gain=.15, pelvis_height_control=True, natural_gait_version=2)
    if args.v3:
        config.update(full_body_reference=True, arm_swing=True, natural_gait=True, max_speed=.4, max_yaw=.6,
                      lateral_stance_gain=.15, pelvis_height_control=True, natural_gait_version=3,
                      style_head_upright_weight=.35,
                      boundary_command_probability=.2,
                      reference_kind='authored_procedural_full_body_v1_not_mocap')
    if args.arm_swing_gain is not None: config['arm_swing_gain'] = args.arm_swing_gain
    if args.mocap_library:
        library_path=args.mocap_library.resolve()
        if args.mode == 'train' and args.run_dir:
            args.run_dir.mkdir(parents=True,exist_ok=True)
            frozen=args.run_dir.resolve()/'mocap_library.npz'
            if frozen != library_path: shutil.copyfile(library_path,frozen)
            library_path=frozen
        config.update(full_body_reference=True, mocap_library=str(library_path),
                      reference_kind='100STYLE_Neutral_mocap_style', natural_gait_version=4)
    if args.curriculum_iterations is not None: config['curriculum_iterations'] = args.curriculum_iterations
    if args.config_override:
        config.update(json.loads(args.config_override.read_text(encoding='utf-8')))
    if config.get('full_motion_path') and args.mode=='train' and args.run_dir:
        args.run_dir.mkdir(parents=True,exist_ok=True)
        source=Path(config['full_motion_path']).resolve();frozen=args.run_dir.resolve()/'full_motion.npz'
        if source!=frozen:shutil.copyfile(source,frozen)
        config['full_motion_path']=str(frozen)
    if config.get('amp_expert_paths') and args.mode=='train' and args.run_dir:
        paths=[]
        for i,path in enumerate(config['amp_expert_paths']):
            source=Path(path).resolve();destination=args.run_dir.resolve()/f'amp_expert_{i}.npz'
            if source!=destination:shutil.copyfile(source,destination)
            paths.append(str(destination))
        config['amp_expert_paths']=paths
    environment=AnonLocomotionEnv
    if config.get('full_motion_path'):
        from anon_imitation_env import AnonImitationEnv
        environment=AnonImitationEnv
    env = environment(args.num_envs, args.seed, app, evaluation=args.mode != 'train', config=config)
    if args.mode == 'smoke':
        started = time.perf_counter()
        smoke_falls=0
        for _ in range(180):
            obs, rewards, done, extras = env.step(torch.zeros_like(env.actions))
            smoke_falls+=int(done.sum())
        print('ANON_SMOKE '+json.dumps(dict(seconds=time.perf_counter()-started,
                    height=float(env.root[:, 2].mean()), error=float(torch.linalg.vector_norm(env.logical_vel[:, :2]-env.commands[:, :2], dim=-1).mean()),
                    observations=list(obs['policy'].shape), falls=smoke_falls,finite=bool(torch.isfinite(obs['policy']).all()))), flush=True)
    elif args.mode == 'reference':
        export_reference(env)
    elif args.mode == 'eval':
        runner = OnPolicyRunner(env, agent_config(), log_dir=None, device='cuda:0') if args.checkpoint else None
        if runner: load_policy(runner, env, args.checkpoint)
        evaluate(env, runner)
        if runner: runner.export_policy_to_jit(str(args.output), 'policy_jit.pt')
    else:
        run = args.run_dir or ROOT/'logs/rsl_rl/anon_locomotion'/datetime.now().strftime('%Y%m%d_%H%M%S')
        run.mkdir(parents=True, exist_ok=True)
        (run/'env_config.json').write_text(json.dumps(env.cfg, indent=2), encoding='utf-8')
        cfg = agent_config()
        if env.cfg.get('full_motion_path'):
            cfg['actor']['distribution_cfg']['init_std']=.2
            cfg['algorithm']['entropy_coef']=.003
        if 'train_entropy_coef' in env.cfg:
            cfg['algorithm']['entropy_coef']=env.cfg['train_entropy_coef']
        (run/'agent_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        for filename in ('anon_locomotion_env.py', 'train_anon_locomotion.py', 'natural_walk_reference.py', 'build_anon_v3.py', 'mocap_walk_reference.py','anon_imitation_env.py','full_motion_reference.py','build_full_motion.py','continuous_motion_reference.py','evaluate_continuous.py','continuous_command_suite.py'):
            shutil.copyfile(HERE/filename, run/filename)
        shutil.copyfile(HERE/'anon_config_compat.py',run/'anon_config_compat.py')
        if env.cfg.get('amp_enabled'):shutil.copyfile(HERE/'amp_motion_prior.py',run/'amp_motion_prior.py')
        if env.cfg.get('articulated_wrists'):
            for filename in ['build_anon_v4.py','physical_character.py','migrate_articulated_policy.py']:
                shutil.copyfile(HERE/filename,run/filename)
        runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=str(run), device='cuda:0')
        from amp_motion_prior import install_amp
        install_amp(runner,env)
        if args.checkpoint:
            migration = load_policy(runner, env, args.checkpoint)
            if migration: (run/'migration.json').write_text(json.dumps(migration, indent=2), encoding='utf-8')
            runner.current_learning_iteration += 1
            env.iteration = runner.current_learning_iteration
        if env.cfg.get('train_upper_std_cap'):
            parameter=dict(runner.alg.actor.named_parameters())['distribution.log_std_param']
            with torch.no_grad():
                before=parameter.exp().cpu().tolist()
                ids=env.upper_ids
                parameter[ids]=parameter[ids].clamp(max=float(np.log(env.cfg['train_upper_std_cap'])))
            adjustment=dict(before_std=before,after_std=parameter.detach().exp().cpu().tolist(),scope='Training exploration only; actor mean unchanged')
            (run/'exploration_adjustment.json').write_text(json.dumps(adjustment,indent=2),encoding='utf-8')
        if env.cfg.get('train_reset_actor_head',False):
            parameters=dict(runner.alg.actor.named_parameters())
            with torch.no_grad():
                for name in ['mlp.6.weight','mlp.6.bias']:parameters[name].zero_()
            for opt_name in ['optimizer','actor_optimizer']:
                optimizer=getattr(runner.alg,opt_name,None)
                if optimizer:
                    for name in ['mlp.6.weight','mlp.6.bias']:optimizer.state.pop(parameters[name],None)
            (run/'actor_head_reset.json').write_text(json.dumps(dict(reason='New running residual controller; start from nominal PD instead of walking action bias')),encoding='utf-8')
        if env.cfg.get('train_leg_std'):
            parameter=dict(runner.alg.actor.named_parameters())['distribution.log_std_param']
            with torch.no_grad():parameter[env.leg_ids]=float(np.log(env.cfg['train_leg_std']))
        floor=env.cfg.get('imitation_normalizer_std_floor',0.)
        if floor:
            for model in [runner.alg.actor,runner.alg.critic]:
                state=model.state_dict()
                for name,value in state.items():
                    if name.startswith('obs_normalizer'):
                        if name.endswith('_var'):value.clamp_(min=floor**2)
                        elif name.endswith('_std'):value.clamp_(min=floor)
                        elif name.endswith('count'):value.fill_(10000.)
                model.load_state_dict(state)
        original_log = runner.logger.log
        started = time.perf_counter()
        target_iteration = runner.current_learning_iteration+args.iterations
        class StopTraining(Exception):
            pass
        def log(**kwargs):
            env.iteration = kwargs['it']
            original_log(**kwargs)
            status = dict(iteration=kwargs['it'], total=target_iteration, elapsed_seconds=time.perf_counter()-started,
                          num_envs=env.num_envs,gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                          tracking_error=float(torch.linalg.vector_norm(env.logical_vel[:, :2]-env.commands[:, :2], dim=-1).mean()),
                          height=float(env.root[:, 2].mean()), run_dir=str(run))
            if hasattr(env,'learned_clearance_deficit'):status['clearance_deficit_m']=float(env.learned_clearance_deficit)
            if hasattr(env,'amp'):status.update(env.amp.stats)
            (run/'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            if kwargs['it'] % 10 == 0: print('爱音训练 '+json.dumps(status, ensure_ascii=False), flush=True)
            if (run/'stop.request').is_file(): raise StopTraining
        runner.logger.log = log
        print('ANON_TRAIN_RUN '+str(run), flush=True)
        try:
            runner.learn(args.iterations, init_at_random_ep_len=True)
        except StopTraining:
            print('ANON_TRAIN_STOP_REQUEST: saving current policy', flush=True)
            runner.logger.stop_logging_writer()
        runner.save(str(run/'policy_final.pt'))
        runner.export_policy_to_jit(str(run), 'policy_jit.pt')
        timing=dict(training_seconds=time.perf_counter()-started,
                    completed_iteration=int(env.iteration),
                    definition='Training loop plus final checkpoint/JIT export; excludes scene setup, evaluation and video recording')
        (run/'training_time.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
        print('ANON_TRAIN_TIME '+json.dumps(timing),flush=True)
    env.close()


result = 0
try:
    main()
except Exception:
    import traceback
    traceback.print_exc(); sys.stderr.flush(); result = 1
finally:
    app.close(exit_code=result)
sys.exit(result)
