"""Run unified acceptance sequences on independent characters in one Isaac scene."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from source_layout import resolve_source

IDLE_COMMAND = (0.0, 0.0, 0.0)
FULL_GROUP_ENDS = (8, 20, 28, 40)
WARMUP_SECONDS = 4.0
PARALLEL_ENVS = 4


def resolve_parallel_envs(checkpoint, output, override_path, override):
    """Read an explicit override or the current scratch run's acceptance plan."""
    if 'unified_eval_parallel_envs' in override:
        environments = override['unified_eval_parallel_envs']
    else:
        environments = 1
        if checkpoint and override_path:
            output = output.resolve()
            prefix, _, index = checkpoint.stem.partition('_')
            suite_name = {'unified': 'full', 'long_stand_recheck': 'long_stand_recheck'}.get(output.name)
            candidate = output.parent
            if (prefix == 'scratch' and index.isdecimal() and candidate.name == f'checkpoint_{index}' and
                    suite_name and override_path.resolve() == candidate / f'{suite_name}_config.json'):
                plan_path = candidate.parent / 'acceptance_plan.json'
                if plan_path.is_file():
                    settings = json.loads(plan_path.read_text(encoding='utf-8'))
                    environments = settings.get('unified_eval_parallel_envs', 1)
    if type(environments) is not int or environments not in (1, PARALLEL_ENVS):
        raise ValueError('unified_eval_parallel_envs must be 1 or 4')
    return environments


@dataclass
class Sequence:
    index: int
    first_segment: int
    segments: list
    warmup_steps: int
    requested: np.ndarray


def plan(cases, dt, *, diagnostic=False):
    ends = (len(cases),) if diagnostic else FULL_GROUP_ENDS
    if len(cases) != ends[-1] or any(cases[end - 1][0] not in ('stop', 'stop_after_walk') for end in ends[:-1]):
        raise ValueError('The parallel acceptance schedule or its stop boundaries have changed')
    sequences = []
    from evaluate_unified_walk_run import command

    first = 0
    for index, end in enumerate(ends):
        segments = cases[first:end]
        warmup_steps = round(WARMUP_SECONDS / dt) if index else 0
        requested = [IDLE_COMMAND] * warmup_steps
        for label, seconds, fixed in segments:
            requested.extend(command(label, step * dt, seconds, fixed) for step in range(round(seconds / dt)))
        sequences.append(Sequence(index, first, segments, warmup_steps, np.asarray(requested, dtype=np.float32)))
        first = end
    return sequences


def track(env, infer, sequences):
    """Run each row on its own articulation and retain pre-reset and post-reset telemetry."""
    import torch

    env.reset(env.ids)
    env.refresh()
    initial_root = env.root
    initial_plan = env.continuous_ref_commands.clone()
    frames = max(len(sequence.requested) for sequence in sequences)
    requests = np.zeros((frames, env.num_envs, 3), np.float32)
    for index, sequence in enumerate(sequences):
        requests[:len(sequence.requested), index] = sequence.requested
    commands = torch.as_tensor(requests, device=env.device).clone()
    scale = (env.cfg.get('max_speed', .4) / commands[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1.)
    commands[..., :2] *= scale
    commands[..., 2].clamp_(-env.cfg.get('max_yaw', .6), env.cfg.get('max_yaw', .6))
    rows = []
    for step, control in enumerate(commands):
        # Same continuous-command clamp and planner invalidation as set_command().
        env.commands.copy_(control)
        env._pending_command_delta = None
        observations = env.get_observations()
        action = infer(observations)
        planned = env.next_planned_command()
        controller = {}
        if env.cfg.get('unified_record_controller'):
            controller.update(observations=observations['policy'], actions=action, controller_root=env.root,
                              root_velocities=env.root_vel)
        _, _, done, extras = env.step(action)
        poses, joints, actual, phase = env.recorded_state
        forces = env.last_substep_support_forces[..., 2].reshape(-1, env.num_envs, 2).transpose(0, 1)
        metrics = extras['per_env_metrics']
        head = extras['idle_head']
        rows.append(dict(
            actual=actual, planned=planned, post_step_plan=env.continuous_ref_commands.clone(), commands=env.commands.clone(),
            root=env.root, pelvis=poses[:, env.pelvis_id, 3:], poses=poses, joints=joints, phase=phase,
            headings=env.reference_heading.clone(), feet=env.poses[:, env.foot_ids],
            foot_velocity=env.link_vel[:, env.foot_ids], chest_velocity=env.vel_b,
            force=forces, head=head['error'], idle_head_active=head['active'],
            idle_head_gain=head['gain'], idle_head_reward_rate=head['reward_rate'],
            done=done, falls=done & ~extras['time_outs'], imitation=torch.stack(tuple(metrics.values()), dim=-1),
            **controller,
        ))
        if (step + 1) % round(5 / env.dt) == 0:
            print('UNIFIED_PARALLEL_PROGRESS ' + json.dumps(dict(active_groups=len(sequences),
                  physics_steps=step + 1, target_physics_steps=frames)), flush=True)
    host = {key: torch.stack([row[key] for row in rows]).cpu() for key in rows[0]}
    host.update(initial_root=initial_root.cpu(), initial_plan=initial_plan.cpu(), requested=torch.from_numpy(requests))
    return host, tuple(metrics), frames


def score_group(env, trace, sequence, lane, metrics, foot_com):
    """Feed one articulation's metrics to the formal response and steady-state scorer."""
    import torch
    from anon_locomotion_env import quat_rotate_inverse
    from evaluate_unified_walk_run import linear_tracking_error, yaw_tracking_error
    from locomotion_acceptance import STATIONARY_SEGMENTS, quality_passed, score_segment
    from command_plan import limit_yaw_command
    from unified_walk_run_reference import heading_velocity

    limit = len(sequence.requested)
    d = {key: value[:limit, lane] for key, value in trace.items() if not key.startswith('initial_')}
    prior_root = torch.cat((trace['initial_root'][lane:lane + 1], d['root'][:-1]))
    geometry = heading_velocity((d['root'][:, :3] - prior_root[:, :3]) / env.dt, d['headings'])
    chest = torch.stack((-d['chest_velocity'][:, 1], d['chest_velocity'][:, 0]), dim=-1)
    x, y, z, w = d['pelvis'].unbind(-1)
    pelvis_yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    quat = d['feet'][:, :, None, 3:].expand(-1, -1, 8, -1)
    conjugate = torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)
    offset = quat_rotate_inverse(conjugate, env.foot_corners.cpu()[None].expand_as(quat[..., :3]))
    sole = d['feet'][:, :, None, :3] + offset
    height = sole[..., 2]
    bottom = height.amin(dim=-1)
    contact = bottom < .012
    near = (height < bottom[:, :, None] + .003) & contact[:, :, None]
    com_world = quat_rotate_inverse(conjugate, foot_com[None, :, None].expand_as(offset))
    velocity = d['foot_velocity'][:, :, None, :]
    point_velocity = velocity[..., :3] + torch.cross(velocity[..., 3:].expand_as(offset), offset - com_world, dim=-1)
    point_slip = (point_velocity[..., :2].norm(dim=-1) * near).sum((1, 2)) / near.sum((1, 2)).clamp_min(1)
    stable = near[1:] & near[:-1] & ~d['done'][1:, None, None]
    movement = (sole[1:] - sole[:-1])[..., :2].norm(dim=-1) / env.dt
    displaced_slip = (movement * stable).sum((1, 2)) / stable.sum((1, 2)).clamp_min(1)
    force_flight = (d['force'].abs() < 1.).all(dim=-1).float().mean(dim=-1)
    force_peak = d['force'].abs().sum(dim=-1).amax(dim=-1)
    flight = ~contact.any(dim=-1)

    warmup_falls = int(d['falls'][:sequence.warmup_steps].sum())
    items = []
    start = sequence.warmup_steps
    legacy_config = dict(env.cfg, unified_acceptance_protocol='legacy_v1')
    for seq_index, (label, seconds, fixed) in enumerate(sequence.segments):
        end = start + round(seconds / env.dt)
        tail_start = start + round(2 / env.dt)
        segment, tail = slice(start, end), slice(tail_start, end)
        initial_plan = d['post_step_plan'][start - 1] if start else trace['initial_plan'][lane]
        target = d['requested'][tail] if fixed is None else limit_yaw_command(torch.tensor(fixed, dtype=torch.float32), env.cfg)
        yaw_target = d['planned'][tail, 2] if fixed is None else target[2].expand(end - tail_start)
        variable = fixed is None
        errors = d['actual'][tail, :2] - target[..., :2]
        geometric_errors = geometry[tail] - target[..., :2]
        chest_errors = chest[tail] - target[..., :2]
        yaw_errors = (d['actual'][tail, 2] - yaw_target).numpy().astype(np.float64)
        yaws = np.unwrap(pelvis_yaw[segment].numpy().astype(np.float64))
        average = np.mean(d['imitation'][tail].numpy(), axis=0, dtype=np.float64)
        linear_limit = .2 if variable else max(.06, float(np.linalg.norm(fixed[:2])) * .22)
        peak = float(force_peak[tail].amax())
        if peak < 50.:
            raise RuntimeError(f'Group {sequence.index + 1} {label} has no credible support load ({peak:.3f} N)')
        item = dict(
            segment=sequence.first_segment + seq_index + 1,
            label=label, seconds=seconds, command=None if fixed is None else target.tolist(),
            requested_command=fixed, falls=int(d['falls'][segment].sum()),
            velocity_error=linear_tracking_error(errors.numpy(), variable, env.dt),
            velocity_limit=linear_limit, velocity_error_instantaneous_mean=float(np.linalg.norm(errors.numpy(), axis=-1).mean()),
            velocity_averaging_seconds=1. if variable else seconds - 2.,
            geometric_velocity_error=linear_tracking_error(geometric_errors.numpy(), variable, env.dt),
            chest_frame_velocity_error=linear_tracking_error(chest_errors.numpy(), variable, env.dt),
            mean_speed=float(d['actual'][tail, :2].norm(dim=-1).numpy().mean(dtype=np.float64)),
            yaw_error=yaw_tracking_error(yaw_errors, yaw_target.numpy()),
            yaw_error_instantaneous_abs_mean=float(np.abs(yaw_errors).mean()),
            pelvis_yaw_drift_rad=float(np.max(np.abs(yaws - yaws[0]))),
            displacement_m=float((d['root'][end - 1, :2] - prior_root[start, :2]).norm()),
            flight_fraction=float(flight[tail].numpy().mean()),
            force_flight_fraction=float(force_flight[tail].numpy().mean(dtype=np.float64)),
            peak_total_support_force_n=peak, flight_force_threshold_n=1., force_sample_hz=env.cfg['physics_hz'],
            stance_slip_proxy_m_s=float(point_slip[tail].numpy().mean(dtype=np.float64)),
            stance_slip_geometry_m_s=float(displaced_slip[tail_start:end - 1].numpy().mean(dtype=np.float64)),
            metrics=dict(zip(metrics, average.tolist(), strict=True)),
        )
        scored = score_segment(
            item, d['actual'][segment].numpy(), d['planned'][segment].numpy(),
            d['requested'][segment].numpy(), geometry[segment].numpy(), initial_plan.numpy(),
            env.cfg, env.dt, head_errors=d['head'][segment].numpy(),
        )
        stationary = label in STATIONARY_SEGMENTS
        legacy_passed = (quality_passed(item, legacy_config) and item['velocity_error'] < linear_limit and
                         item['geometric_velocity_error'] < linear_limit and item['yaw_error'] < (.1 if stationary else .2)
                         and (not stationary or item['mean_speed'] < .08))
        item.update(response_steady=scored, passed=scored['passed'], legacy_passed=legacy_passed)
        items.append(item)
        start = end
    return dict(index=sequence.index + 1, first_segment=sequence.first_segment + 1,
                last_segment=sequence.first_segment + len(sequence.segments),
                warmup_seconds=sequence.warmup_steps * env.dt, warmup_falls=warmup_falls,
                simulated_seconds=sum(row['seconds'] for row in items), segments=items), geometry


def evaluate(env, runner, output, checkpoint):
    import torch
    from evaluate_unified_walk_run import full_gate_segments, full_suite_settings, suite
    from locomotion_acceptance import CURRENT_PROTOCOL, add_light_acceptance, specification, summarize

    started = perf_counter()
    cfg = env.cfg
    protocol = cfg['unified_acceptance_protocol']
    if (env.num_envs != PARALLEL_ENVS or protocol != CURRENT_PROTOCOL or cfg.get('unified_eval_low_speed') or
            not (cfg.get('unified_idle_head_acceptance') and cfg.get('unified_substep_support_forces') and
                 cfg.get('unified_command_frame_metrics') and cfg.get('command_yaw_limit_by_speed'))):
        raise ValueError('Four-character acceptance requires the current unified walk/run physical metrics')
    if env.ground_contact_view.sensor_count != 2 * env.num_envs:
        raise RuntimeError('Each character must have its own pair of foot-contact sensors')

    env.max_episode_length = round(2000 / env.dt)
    targeted = cfg.get('unified_eval_cases')
    if cfg.get('unified_eval_initial_command') is not None:
        raise ValueError('Use one character for an explicit reference-state diagnostic')
    settings = full_suite_settings(cfg, protocol)
    cases = targeted or suite(**settings)
    sequences = plan(cases, env.dt, diagnostic=bool(targeted))

    with torch.inference_mode():
        foot_com = env.robot.get_coms()[0, env.foot_ids, :3].cpu()
        raw, metrics, frames = track(env, runner.get_inference_policy(), sequences)
        groups, rows, labels, starts = [], [], [], []
        rollout = {}
        fields = dict(poses='poses', joints='joints', commands='commands', velocities='actual',
                      phases='phase', dones='done', headings='headings', planned_commands='planned',
                      head_errors='head', idle_head_active='idle_head_active',
                      idle_head_gain='idle_head_gain', idle_head_reward_rate='idle_head_reward_rate')
        if cfg.get('unified_record_controller'):
            fields.update(observations='observations', actions='actions', root='controller_root',
                          root_velocities='root_velocities')
        for lane, sequence in enumerate(sequences):
            group, geometry = score_group(env, raw, sequence, lane, metrics, foot_com)
            rows.extend(group.pop('segments'))
            starts.append(len(labels))
            for label, seconds, _ in sequence.segments:
                labels.extend([label] * round(seconds / env.dt))
            group['first_output_frame'] = starts[-1]
            groups.append(group)
            measured = slice(sequence.warmup_steps, len(sequence.requested))
            for name, key in fields.items():
                rollout.setdefault(name, []).append(raw[key][measured, lane:lane + 1].numpy())
            rollout.setdefault('geometric_velocities', []).append(geometry[measured, None].numpy())
            rollout.setdefault('requested_commands', []).append(sequence.requested[measured])
    output.mkdir(parents=True, exist_ok=True)
    rollout = {key: np.concatenate(parts) for key, parts in rollout.items()}
    np.savez_compressed(output / 'rollout.npz', **rollout,
                        labels=labels, dt=env.dt, link_names=env.links, dof_names=env.names,
                        sequence_start_frames=np.asarray(starts, dtype=np.int64))
    gate_rows = full_gate_segments(rows, protocol, cfg, diagnostic=bool(targeted))
    assessment = summarize(gate_rows)
    total_falls = sum(row['falls'] for row in rows)
    complete = not targeted
    report = dict(
        checkpoint=str(checkpoint), env_config=cfg, acceptance_protocol=protocol,
        acceptance_layout='parallel_grouped', acceptance_environments=env.num_envs, sequence_groups=groups,
        physics_steps=frames, physics_simulated_seconds=frames * env.dt,
        total_seconds=sum(row['seconds'] for row in rows), total_falls=total_falls,
        warmup_falls=sum(group['warmup_falls'] for group in groups), segments=rows,
        evaluator_seconds=perf_counter() - started,
        passed_unified_gate=bool(complete and total_falls == 0 and assessment['passed']),
        complete_scope=complete, acceptance_mode='diagnostic' if targeted else 'full',
        diagnostic=bool(targeted), response_steady_summary=assessment, runtime_compensation_removed=True,
        flight_acceptance_method='Both physical foot support loads below 1 N', velocity_command_frame='Controller reference heading',
        protocol_specification=specification(True, protocol),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acceptance_protocol_sha256=hashlib.sha256(resolve_source('locomotion_acceptance.py').read_bytes()).hexdigest(),
        legacy_metric_gate_on_same_trajectory=bool(complete and all(row['legacy_passed'] for row in rows)),
    )
    if complete:
        report['full_stand_hold_seconds'] = settings['stand_seconds']
        report['full_stop_timing'] = settings['stop_timing']
        report['backward_diagonal_run_speed_m_s'] = settings['backward_diagonal_run_speed']
    add_light_acceptance(report, rollout['velocities'], env.dt)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        print('UNIFIED_SEGMENT ' + json.dumps(row), flush=True)
    print('UNIFIED_REPORT ' + json.dumps(dict(mode=report['acceptance_mode'], layout=report['acceptance_layout'],
          environments=env.num_envs, physics_simulated_seconds=report['physics_simulated_seconds'],
          passed=report['passed_unified_gate'], standard=assessment, light=report['light_acceptance_summary'],
          light_passed=report['passed_light_gate'],
          falls=report['total_falls'])), flush=True)
