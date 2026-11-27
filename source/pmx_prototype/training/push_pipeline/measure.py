"""Evaluate an unchanged Anon checkpoint under calibrated waist pushes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'source/pmx_prototype'))
import source_layout
from trials import recovery_confirmation


COMMANDS = {
    'stand': (0., 0., 0.),
    'walk_forward': (.4, 0., 0.),
    'walk_backward': (-.4, 0., 0.),
    'walk_left': (0., .4, 0.),
    'walk_right': (0., -.4, 0.),
    'run_forward': (1.2, 0., 0.),
    'walk_turn_left': (.4, 0., .4),
}
DIRECTIONS = ('front', 'front_left', 'left', 'back_left',
              'back', 'back_right', 'right', 'front_right')
LEVELS = (.10, .20, .30, .40, .525, .65, .80, 1.00, 1.25)


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda:0', 'cpu'], default='cuda:0')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--only', choices=COMMANDS, nargs='+')
    parser.add_argument('--levels', type=float, nargs='+')
    parser.add_argument('--phases', type=int, default=8)
    parser.add_argument('--angle-offset-deg', type=float, default=0.)
    parser.add_argument('--phase-offset', type=float, default=0.)
    parser.add_argument('--lead-seconds', type=float, default=4.)
    parser.add_argument('--force-profile', choices=('impulse', 'sustained', 'ramp', 'double'), default='impulse')
    parser.add_argument('--rehearsal', action='store_true', help='Save pre-action observations and teacher means from recovered trials')
    parser.add_argument('--trial-ids', type=int, nargs='+')
    parser.add_argument('--match-reference', type=Path)
    args = parser.parse_args()
    if bool(args.trial_ids) != bool(args.match_reference) or (args.trial_ids and args.rehearsal):
        parser.error('Pose capture requires trial IDs and a matching reference, separately from rehearsal')
    levels = args.levels or ((.525, 1.25) if args.pilot else LEVELS)
    if args.phases < 2 or any(level <= 0 for level in levels) or len(set(levels)) != len(levels):
        parser.error('Supply at least two phases and unique positive impulse levels')
    args.output.mkdir(parents=True, exist_ok=False)

    from isaaclab.app import AppLauncher
    app = AppLauncher(dict(headless=True, device=args.device, disable_viewport_updates=True,
                           limit_cpu_threads=2, kit_args='--/persistent/physics/numThreads=0')).app
    env = None
    exit_code = 1
    try:
        import numpy as np
        import torch
        from anon_environment import create_environment
        from rsl_rl.runners import OnPolicyRunner

        torch.set_num_threads(2)
        cases = args.only or (['stand', 'run_forward'] if args.pilot else list(COMMANDS))
        phases = 2 if args.pilot else args.phases
        directions = range(0, 8, 2) if args.pilot else range(8)
        design = [(-1, 0.)] + [(d, level) for level in levels for d in directions]
        entries = [(case, phase, direction, level) for case in cases for phase in range(phases)
                   for direction, level in design]
        count = len(entries)
        config_path = args.checkpoint.parent / 'env_config.json'
        agent_path = args.checkpoint.parent / 'agent_config.json'
        inputs = {str(path.resolve()): sha256(path) for path in (args.checkpoint, config_path, agent_path)}
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config.update(simulation_device=args.device, runtime_inference_only=True,
                      runtime_manual_reset_only=True, amp_reward_diagnostics=False,
                      measure_ground_contact=False, unified_substep_support_forces=False,
                      unified_command_frame_metrics=True)
        config.pop('unified_eval_parallel_envs', None)
        if not (config.get('unified_walk_run') and config.get('continuous_commands')):
            raise ValueError('The probe requires a unified continuous walk/run checkpoint')
        env = create_environment(count, 42, app, evaluation=True, config=config)
        env.max_episode_length = round(60 / env.dt)
        runner = OnPolicyRunner(env, json.loads(agent_path.read_text(encoding='utf-8')),
                                log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location=args.device)
        infer = runner.get_inference_policy()
        metadata, records, trace = run_trials(env, infer, entries, phases, len(design), np, torch, args)
        if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
            raise RuntimeError('A checkpoint input changed during the measurement')
        metadata.update(inputs_sha256=inputs, device=args.device, cases=cases,
                        direction_names=DIRECTIONS, levels_m_s=levels, trials=count,
                        asset=env.cfg['asset'], reference=env.cfg['full_motion_path'],
                        angle_offset_degrees=args.angle_offset_deg, phase_offset_cycles=args.phase_offset,
                        minimum_onset_seconds=args.lead_seconds)
        with (args.output / 'trials.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        if args.rehearsal:
            teacher = trace.pop('rehearsal')
            trials = teacher['trial_ids']
            accepted = np.asarray([row['recovered'] for row in records])[trials]
            impulses = np.asarray([row['impulse_m_s'] for row in records])[trials]
            groups = np.where(impulses == 0., 0, np.where(impulses == .20, 2, 1))
            output = args.output / 'teacher.npz'
            np.savez_compressed(output, groups=groups[accepted], **{key: values[accepted] for key, values in teacher.items()})
            metadata['teacher'] = dict(sha256=sha256(output), counts=np.bincount(groups[accepted], minlength=3).tolist(),
                                       accepted_trials=int(np.unique(trials[accepted]).size), post_seconds=1.5,
                                       group_strengths={'0': [0.], '1': [level for level in levels if level != .20],
                                                        '2': [level for level in levels if level == .20]})
        if args.trial_ids:
            selected = args.trial_ids
            with np.load(args.match_reference / 'telemetry.npz') as reference:
                for key in ('onset', 'first_fall', 'commands'):
                    np.testing.assert_array_equal(trace[key][selected], reference[key][selected])
                np.testing.assert_allclose(trace['force_direction_xy'][selected],
                                           reference['force_direction_xy'][selected], atol=.001, rtol=0.)
                difference = float(np.max(np.abs(trace['frames'][:, selected] - reference['frames'][:, selected])))
                if not np.isfinite(difference) or difference > .03:
                    raise RuntimeError(f'Selected trajectories differ from measured acceptance by {difference}')
            metadata.update(captured_trial_ids=selected, acceptance_reference=str(args.match_reference.resolve()),
                            max_trace_difference=difference)
            np.savez_compressed(args.output / 'poses.npz', poses=trace['clip_poses'],
                                link_names=env.links, dt=env.dt, trial_ids=selected,
                                frames=trace['frames'][:, selected], onset=trace['onset'][selected],
                                first_fall=trace['first_fall'][selected], commands=trace['commands'][selected],
                                force_direction_xy=trace['force_direction_xy'][selected],
                                left_xy=trace['left_xy'][selected])
        else:
            np.savez_compressed(args.output / 'telemetry.npz', **trace)
        (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'PUSH_TRIALS_DONE {count} {args.output}', flush=True)
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        app.close(exit_code=exit_code)


def run_trials(env, infer, entries, phases, group_size, np, torch, args):
    from anon_character_push import PUSH_DELTA_SPEED_MAX, PUSH_DELTA_SPEED_MIN
    from anon_push_waveforms import PROFILES

    profile_name = getattr(args, 'force_profile', 'impulse')
    profile_ticks = PROFILES[profile_name]
    weights = [value / sum(profile_ticks) for value in profile_ticks]

    @torch.inference_mode()
    def collect():
        device = env.device
        cfg = env.cfg
        control_dt = env.dt
        hz = cfg['physics_hz']
        mass = env.robot.get_masses().to(device)
        total_mass = mass.sum(dim=1)
        link = env.links.index('waist_pitch')
        forces = torch.zeros((env.num_envs, env.robot.max_links, 3), device=device)
        experiment = getattr(args, 'action_experiment', None)
        task = env.unwrapped.task
        scheduled = torch.zeros((env.num_envs, 2), device=device)
        ticks = torch.full((env.num_envs,), len(weights), dtype=torch.long, device=device)
        envelope = torch.tensor([*weights, 0.], device=device)

        def shaped_substep():
            forces[:, link, :2] = scheduled * envelope[ticks, None]
            env.robot.apply_forces_and_torques_at_position(forces, None, None, env.ids, True)
            ticks.add_(1).clamp_(max=len(weights))

        if profile_name != 'impulse':
            task._apply_disturbance_substep = shaped_substep

        def apply(ids, impulse_velocity):
            if profile_name != 'impulse':
                scheduled[ids] = total_mass[ids, None] * hz * impulse_velocity
                ticks[ids] = 0
                return
            forces.zero_()
            forces[ids, link, :2] = total_mass[ids, None] * hz * impulse_velocity
            env.robot.apply_forces_and_torques_at_position(forces, None, None,
                                                          env.ids, True)

        rehearsal = []

        def step(frame=None):
            observation = env.get_observations()
            action = infer(observation)
            if experiment is not None and frame is not None:
                action = experiment.action(frame, onset, observation, action)
            if args.rehearsal and frame is not None and frame % 2 == 0:
                ids = ((onset >= 0) & (frame > onset) & (frame <= onset+round(1.5/control_dt))).nonzero().flatten()
                if len(ids):
                    rehearsal.append((ids, observation['policy'][ids].clone(), action[ids].clone(), frame-onset[ids]))
            _, _, done, _ = env.step(action)
            return done.to(torch.bool)

        # Verify the physical impulse on two lifted actors and an unforced neighbor.
        # The control-step hook reads net horizontal COM momentum after each substep.
        actor_ids = torch.tensor([0, 3, env.num_envs // 2], device=device)
        env.reset(env.ids, command=[0., 0., 0.])
        lifted = env.robot.get_root_transforms().clone()
        lifted[:, 2] += 2.
        env.robot.set_root_transforms(lifted, env.ids)
        env.view.update_articulations_kinematic()
        step()

        def com_xy():
            link_vel = env.robot.get_link_velocities()[actor_ids, :, :2]
            return (link_vel * mass[actor_ids, :, None]).sum(1) / total_mass[actor_ids, None]

        original = task._record_substep
        before = com_xy()
        deltas = []

        def audit_substep():
            original()
            deltas.append(com_xy() - before)

        task._record_substep = audit_substep
        test_impulse = torch.tensor([[0., 0.], [.525, 0.], [0., -1.25]], device=device)
        try:
            apply(actor_ids[1:], test_impulse[1:])
            for _ in range((len(weights) + cfg['decimation'] - 1) // cfg['decimation']):
                step()
        finally:
            del task._record_substep
        delta = torch.stack(deltas)
        expected = envelope[torch.arange(len(deltas), device=device).clamp(max=len(weights))].cumsum(0)
        error = float((delta - expected[:, None, None] * test_impulse[None]).abs().max())
        if len(deltas) < len(weights) or error > .02:
            raise RuntimeError(f'Impulse calibration failed: {delta.tolist()} vs {test_impulse.tolist()}')
        print(f'PUSH_CALIBRATION actor_mass_kg={float(total_mass[0]):.5f} max_error_m_s={error:.7f}', flush=True)

        env.reset(env.ids, command=[0., 0., 0.])
        commands = torch.tensor([COMMANDS[case] for case, _, _, _ in entries], device=device)
        stationary = (commands == 0).all(dim=-1)
        phase_grid = torch.tensor([(phase / phases + args.phase_offset) % 1
                                   for _, phase, _, _ in entries], device=device)
        nominal_strength = torch.tensor([level for _, _, _, level in entries], device=device)
        angle = torch.tensor([d * np.pi / 4 + np.deg2rad(args.angle_offset_deg)
                              for _, _, d, _ in entries], device=device, dtype=torch.float32)
        if experiment is not None:
            phase_grid, angle = experiment.schedule(phase_grid, angle)
        onset = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        first_fall = onset.clone()
        target_phase = None
        lead_frames = round(args.lead_seconds / control_dt)
        post_frames = round(5. / control_dt)
        max_start = lead_frames + round(3. / control_dt)
        actual_phase = torch.full_like(nominal_strength, float('nan'))
        local_left = torch.zeros((env.num_envs, 2), device=device)
        applied_xy = torch.zeros_like(local_left)
        frames = []
        poses = []
        selected = getattr(args, 'trial_ids', None)
        selection = torch.as_tensor(selected, device=device) if selected else None
        start_time = time.perf_counter()
        for t in range(max_start + post_frames):
            if t == round(1. / control_dt):
                env.commands.copy_(commands)
            if t == lead_frames:
                target_phase = torch.ceil(env.phase - phase_grid) + phase_grid
            if t >= lead_frames and t <= max_start:
                standing_time = t >= lead_frames + (phase_grid * 1.2 / control_dt).round()
                moving_phase = env.phase >= target_phase
                ids = env.ids[(onset < 0) & torch.where(stationary, standing_time, moving_phase)].long()
                if len(ids):
                    onset[ids] = t
                    actual_phase[ids] = env.phase[ids] % 1
                    x, y, z, w = env.root[ids, 3:].unbind(-1)
                    forward = torch.stack((2 * (z*w - x*y), 2 * (x*x + z*z) - 1), dim=-1)
                    forward = torch.nn.functional.normalize(forward, dim=-1)
                    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
                    local_left[ids] = left
                    applied_xy[ids] = (angle[ids, None].cos() * forward + angle[ids, None].sin() * left)
                    pushed = ids[nominal_strength[ids] > 0]
                    if len(pushed):
                        if experiment is not None:
                            experiment.begin(pushed, applied_xy[pushed])
                        apply(pushed, applied_xy[pushed] * nominal_strength[pushed, None])
            frames.append(torch.cat((env.logical_vel, env.root[:, :2], env.poses[:, env.pelvis_id, 2:3],
                                     env.gravity[:, 2:3], env.continuous_ref_commands), dim=1))
            if selection is not None:
                poses.append(env.poses[selection])
            done = step(t)
            first_fall = torch.where((first_fall < 0) & done, t, first_fall)
            if t % 30 == 29:
                print(f'PUSH_PROGRESS frame={t+1} onset={int((onset>=0).sum())}/{env.num_envs} '
                      f'elapsed={time.perf_counter()-start_time:.1f}s', flush=True)
            if t >= max_start and (onset >= 0).all() and t >= int(onset.max()) + post_frames:
                break
        if (onset < 0).any():
            raise RuntimeError(f'{int((onset<0).sum())} phase onsets did not trigger')
        if len(frames) <= int(onset.max()) + post_frames:
            frames.append(torch.cat((env.logical_vel, env.root[:, :2], env.poses[:, env.pelvis_id, 2:3],
                                     env.gravity[:, 2:3], env.continuous_ref_commands), dim=1))
            if selection is not None:
                poses.append(env.poses[selection])
        traces = torch.stack(frames).cpu().numpy()
        data = dict(frames=traces, onset=onset.cpu().numpy(), first_fall=first_fall.cpu().numpy(),
                    actual_phase=actual_phase.cpu().numpy(), left_xy=local_left.cpu().numpy(),
                    force_direction_xy=applied_xy.cpu().numpy(),
                    mass=total_mass.cpu().numpy(), commands=commands.cpu().numpy())
        if poses:
            data['clip_poses'] = torch.stack(poses).cpu().numpy()
        if args.rehearsal:
            if not rehearsal:
                raise RuntimeError('No teacher samples were collected')
            data['rehearsal'] = {key: torch.cat(values).cpu().numpy() for key, values in zip(
                ('trial_ids', 'observations', 'actions', 'frames_after_push'), zip(*rehearsal))}
        audit = dict(actor_ids=actor_ids.tolist(), requested_xy_m_s=test_impulse.tolist(),
                     after_substep_xy_m_s=delta.tolist(), max_error_m_s=error)
        return data, audit, control_dt, hz, link, post_frames

    data, audit, dt, hz, link, post_frames = collect()
    records = score(entries, group_size, data, dt, hz, post_frames, np)
    if profile_name != 'impulse':
        for row in records:
            row['peak_force_N'] *= max(weights)
    meta = dict(physics_hz=hz, control_hz=1/dt, applied_substeps=sum(value > 0 for value in weights),
                force_profile=profile_name, force_substep_fractions=weights,
                force_application_link=env.links[link], observation_after_push_seconds=post_frames*dt,
                dof_names=list(env.names),
                control_dt=dt, paired_group_size=group_size, gait_phase_samples=phases,
                button_equivalent_m_s=[PUSH_DELTA_SPEED_MIN, PUSH_DELTA_SPEED_MAX],
                calibration=audit, recovery_average_seconds=.5, recovery_hold_seconds=.5,
                trace_columns=['vf','vl','yaw_rate','x','y','pelvis_z','projected_gravity_z',
                               'planned_vf','planned_vl','planned_yaw_rate'],
                stationary_onset='1.2 seconds of evenly spaced standing times',
                moving_onset='evenly spaced fractions of the running reference gait phase after the lead time',
                fall_definition='Inspector termination: pelvis below 0.65 m, tilt > 60 deg or nonfinite joints',
                direction_definition='force direction relative to projected pelvis forward at impact')
    return meta, records, data


def score(entries, group_size, data, dt, hz, post_frames, np):
    traces = data['frames']
    records = []
    for i, (case, phase, direction, level) in enumerate(entries):
        control = i // group_size * group_size
        onset = int(data['onset'][i])
        control_onset = int(data['onset'][control])
        t = traces[onset+1:onset+1+post_frames, i]
        base = traces[control_onset+1:control_onset+1+post_frames, control]
        if len(t) != post_frames or len(base) != post_frames:
            raise RuntimeError('Incomplete after-impact trace')
        target = data['commands'][i]
        first_fall = int(data['first_fall'][i])
        precondition_fell = 0 <= first_fall < onset
        fell = onset <= first_fall < onset + post_frames
        seconds = None if precondition_fell or fell else recovery_confirmation(t[:, :3], target, dt)
        offset = (t[:, 3:5] - traces[onset, i, 3:5]) - (base[:, 3:5] - traces[control_onset, control, 3:5])
        mass = float(data['mass'][i])
        records.append(dict(case=case, phase_bin=phase, actual_phase=float(data['actual_phase'][i]),
                            direction=DIRECTIONS[direction] if direction >= 0 else 'none',
                            impulse_m_s=level, impulse_Ns=level*mass, peak_force_N=level*mass*hz,
                            onset_seconds=onset*dt, control_id=control, trial_id=i,
                            precondition_fell=precondition_fell, fell=fell, recovered=seconds is not None,
                            recovery_seconds=seconds,
                            fall_seconds=(first_fall-onset+1)*dt if fell else None,
                            max_lateral_deviation_m=float(np.max(np.abs(offset @ data['left_xy'][i]))),
                            post_min_pelvis_z_m=float(t[:, 5].min()),
                            post_max_projected_gravity_z=float(t[:, 6].max()),
                            planned_at_impact_error=float(np.linalg.norm(traces[onset, i, 7:10]-target))))
    return records


if __name__ == '__main__':
    main()
