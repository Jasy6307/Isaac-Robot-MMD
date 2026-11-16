"""Read-only policy inference probes; no learning or optimizer updates."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import source_layout


def diagnose(env, runner, output, checkpoint):
    import numpy as np
    import torch
    output.mkdir(parents=True, exist_ok=True)
    runner.alg.actor.eval()
    infer = runner.get_inference_policy()
    env.max_episode_length = 100000
    phases = torch.tensor([0., .25, .5, .75], device=env.device).repeat(4)
    headings = torch.tensor([-np.pi, -np.pi/2, 0., np.pi/2], device=env.device).repeat_interleave(4)
    cases = {
        'stand_right': [(2, 0), (10, -.6)],
        'stand_left': [(2, 0), (10, .6)],
        'left_to_right': [(2, 0), (6, .6), (6, -.6)],
        'right_to_left': [(2, 0), (6, -.6), (6, .6)],
    }
    if args.cases:
        cases = {name: cases[name] for name in args.cases}
    models = json.loads(args.models.read_text(encoding='utf-8'))
    report = dict(diagnostic=True, training=False, num_envs=env.num_envs,
                  initial_phases=phases.cpu().tolist(), initial_headings=headings.cpu().tolist(), models={})
    started = time.perf_counter()
    with torch.inference_mode():
        for name, model_path in models.items():
            state = torch.load(model_path, map_location=env.device, weights_only=False)
            runner.load(model_path, map_location=env.device)
            if hasattr(env, 'amp') and 'amp_state' in state:
                env.amp.load_state_dict(state['amp_state'])
            results = []
            for case, segments in cases.items():
                env.reset(env.ids, heading=headings, phase=phases, command=[0, 0, 0])
                env.refresh()
                history = {key: [] for key in ['poses', 'joints', 'qd', 'commands', 'ref_commands', 'phases',
                           'headings', 'root_velocity', 'done', 'actions', 'reference_q', 'reference_positions',
                           'reference_rotations', 'contacts', 'reward', 'time']}
                elapsed = 0.
                for seconds, yaw in segments:
                    for _ in range(round(seconds/env.dt)):
                        env.set_command([0, 0, yaw])
                        observation = env.get_observations()
                        action = infer(observation)
                        ref = env.sample_reference(env.phase)
                        reference = {k: ref[k].clone() for k in ['q', 'positions', 'rotations', 'contacts']}
                        _, reward, done, extras = env.step(action)
                        values = dict(poses=env.recorded_state[0], joints=env.recorded_state[1], qd=env.qd,
                                      commands=env.commands, ref_commands=env.continuous_ref_commands,
                                      phases=env.phase, headings=env.reference_heading, root_velocity=env.root_vel,
                                      done=done, actions=action, reference_q=reference['q'],
                                      reference_positions=reference['positions'], reference_rotations=reference['rotations'],
                                      contacts=reference['contacts'], reward=reward)
                        for key, value in values.items():
                            history[key].append(value.detach().cpu().numpy().copy())
                        history['time'].append(elapsed)
                        elapsed += env.dt
                arrays = {key: np.asarray(value) for key, value in history.items()}
                arrays['link_names'] = np.asarray(env.links)
                arrays['dof_names'] = np.asarray(env.names)
                np.savez_compressed(output/f'{name}_{case}.npz', **arrays)
                falls = arrays['done'].sum(0)
                first_falls = [float(arrays['time'][np.flatnonzero(arrays['done'][:, i])[0]])
                               if falls[i] else None for i in range(env.num_envs)]
                item = dict(case=case, falls_by_env=falls.tolist(), total_falls=int(falls.sum()),
                            first_fall_seconds=first_falls)
                results.append(item)
                print('PIVOT_PROBE '+json.dumps(dict(model=name, **item)), flush=True)
                report['models'][name] = results
                report['wall_seconds'] = time.perf_counter()-started
                (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PIVOT_PROBES_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['stand_right', 'stand_left', 'left_to_right', 'right_to_left'])
    args = parser.parse_args()
    models = json.loads(args.models.read_text(encoding='utf-8'))
    import evaluate_unified_walk_run
    evaluate_unified_walk_run.evaluate = diagnose
    sys.argv = [str(SOURCE/'train_anon_locomotion.py'), '--mode', 'eval', '--headless',
                '--num-envs', '16', '--checkpoint', next(iter(models.values())), '--output', str(args.output)]
    runpy.run_path(str(SOURCE/'train_anon_locomotion.py'), run_name='__main__')
