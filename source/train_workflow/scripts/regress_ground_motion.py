"""Read-only accepted-asset regression, with all regenerated files in a new run."""
from pathlib import Path
import argparse
import json
import sys
import shutil
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'tests/fixtures/grounding/accepted_runs.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rebuild-targets', action='store_true')
    parser.add_argument('--solve-smoke-frames', type=int, default=0)
    parser.add_argument('--smooth-smoke-frames', type=int, default=0)
    args = parser.parse_args()
    from source.train_workflow.utils.grounding import pipeline
    from source.train_workflow.utils.grounding.motion import read_motion, write_motion, sha256
    from source.train_workflow.utils.grounding.ik import solve_side
    from source.train_workflow.utils.grounding.smoothing import SmoothingSettings
    torch.set_num_threads(4)
    entries = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protected = {ROOT/e[k] for e in entries for k in ('reference', 'candidate', 'source', 'input')}
    protected.update(ROOT/e['prepared']/name for e in entries for name in ('targets.npz', 'source_sole.npz', 'left_solved.npz', 'right_solved.npz'))
    hashes = {str(p): sha256(p) for p in protected}
    pipeline.write_json(args.output_dir/'protected_before.json', hashes)
    reports = []
    for entry in entries:
        name = entry['name']
        print('REGRESSION', name, flush=True)
        for key in ('reference', 'candidate'):
            if sha256(ROOT/entry[key]) != entry[key+'_sha256']:
                raise ValueError(f'Accepted asset hash changed: {name}/{key}')
        robot = ROOT/'assets/g1_29dof_o6_hand_V3.usd'
        report = pipeline.audit_motion(ROOT/entry['reference'], ROOT/entry['candidate'], robot,
            geometry=entry['geometry'], targets=ROOT/entry['prepared']/'targets.npz')
        if not report['passed'] or report['source_contact']['source_flight_frames'] != entry['source_flight_frames']:
            raise ValueError(report)
        if abs(report['minimum_proxy_clearance_m']-entry['minimum_clearance']) > 1e-6:
            raise ValueError('Historical geometry regression differs')
        if args.rebuild_targets:
            run = args.output_dir/name
            pipeline.prepare(ROOT/entry['input'], ROOT/entry['source'], robot, run,
                frames=entry['frames'], source_leg_mode=entry['source_leg_mode'], character='anon')
            differences = {}
            for filename in ('source_sole.npz', 'targets.npz'):
                with np.load(run/filename) as actual, np.load(ROOT/entry['prepared']/filename) as expected:
                    differences[filename] = {}
                    for key in expected.files:
                        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-6)
                        differences[filename][key] = float(np.max(abs(actual[key]-expected[key])))
            # Replay finalization with frozen full-clip solver outputs. The bounded
            # solver smoke below runs the unchanged numerical IK on new targets.
            for side in ('left', 'right'):
                shutil.copyfile(ROOT/entry['prepared']/(side+'_solved.npz'), run/(side+'_solved.npz'))
            pipeline.finalize_ik(run, root_feedback=entry.get('root_feedback', False))
            actual, expected = read_motion(run/'fixed_ik.h5'), read_motion(ROOT/entry['reference'])
            np.testing.assert_array_equal(actual.delta, expected.delta)
            np.testing.assert_allclose(actual.root_pos, expected.root_pos, atol=1e-6, rtol=0)
            np.testing.assert_allclose(actual.root_rot, expected.root_rot, atol=1e-6, rtol=0)
            report['rebuilt_target_max_errors'] = differences
            report['full_clip_finalize_replayed'] = True
        reports.append(dict(name=name, **report))
        pipeline.write_json(args.output_dir/(name+'.json'), report)
    first = entries[0]
    if args.solve_smoke_frames:
        n = args.solve_smoke_frames
        path = args.output_dir/first['name']/'targets.npz' if args.rebuild_targets else ROOT/first['prepared']/'targets.npz'
        with np.load(path) as source:
            data = {k: source[k][:n] if not k.endswith('_cols') else source[k] for k in source.files}
        from source.train_workflow.utils.motion.joint_constraints import validate_joint_positions
        q = data['q'].copy()
        for side in ('left', 'right'):
            solved, cols = solve_side(data, side, robot, 30.)
            q[:, cols] = solved
        limits = validate_joint_positions(q, read_motion(ROOT/first['reference']).names, 30., robot)
        if not limits['passed']:
            raise ValueError(limits)
        np.savez_compressed(args.output_dir/'solve_smoke.npz', q=q)
    if args.smooth_smoke_frames:
        n = args.smooth_smoke_frames
        ref, seed = read_motion(ROOT/first['reference'], n), read_motion(ROOT/first['candidate'], n)
        write_motion(ref, args.output_dir/'smoke_reference.h5', ref.delta)
        # Copy roots from reference so a candidate can never introduce alignment drift.
        write_motion(ref, args.output_dir/'smoke_seed.h5', seed.delta)
        pipeline.smooth_motion(args.output_dir/'smoke_reference.h5', args.output_dir/'smoke_seed.h5', robot,
            args.output_dir/'smooth_smoke', geometry=first['geometry'],
            settings=SmoothingSettings(window_frames=24, stride=12, sweeps=2, iterations=2, restoration_quadratic_weight=0.))
    changed = [p for p, digest in hashes.items() if sha256(p) != digest]
    result = dict(passed=not changed, protected_assets_unchanged=not changed, changed=changed,
                  cases=reports, numerical_solver_smoke_frames=args.solve_smoke_frames,
                  temporal_smoothing_smoke_frames=args.smooth_smoke_frames,
                  mesh_validation='Existing accepted FCL reports retained; this run does not repeat full FCL scans')
    pipeline.write_json(args.output_dir/'summary.json', result)
    if changed:
        raise ValueError('Protected original assets changed')
    print('REGRESSION_PASS', len(reports), flush=True)


if __name__ == '__main__':
    main()
