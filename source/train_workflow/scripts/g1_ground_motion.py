"""Prepare/solve repaired IK, smooth fixed-root motion, or audit an existing H5."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == 'legacy-legs':
        legacy = argparse.ArgumentParser(add_help=False, description='Historical full/window/spike leg processing; accepted grounding pipeline unchanged.')
        legacy.add_argument('method', choices=('full', 'window', 'spikes'))
        if arguments[1:] in ([], ['--help'], ['-h']):
            legacy.print_help()
            return 0
        method, remaining = legacy.parse_known_args(arguments[1:])
        if method.method == 'full':
            from source.train_workflow.utils.motion.legacy_leg_full import main as legacy_main
        else:
            from source.train_workflow.utils.motion import legacy_leg_local
            legacy_main = (legacy_leg_local.smooth_h5_leg_window_main if method.method == 'window'
                           else legacy_leg_local.repair_h5_leg_spikes_main)
        return legacy_main(remaining)
    if arguments and arguments[0] == 'legacy-refine':
        from source.train_workflow.utils.grounding.legacy_refinement import main as legacy_main
        return legacy_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    stages.add_parser('legacy-legs', add_help=False, help='Historical leg processing: full/window/spikes.')
    stages.add_parser('legacy-refine', add_help=False, help='Compatibility Cartesian refinement; use --method fidelity/smooth.')
    prepare = stages.add_parser('prepare')
    prepare.add_argument('--input', required=True, type=Path)
    prepare.add_argument('--source', required=True, type=Path)
    prepare.add_argument('--robot', required=True, type=Path)
    prepare.add_argument('--output-dir', required=True, type=Path)
    prepare.add_argument('--frames', type=int)
    prepare.add_argument('--source-leg-mode', choices=('fk', 'ik'), required=True)
    prepare.add_argument('--character', choices=('anon',), required=True)
    prepare.add_argument('--mesh-path', default='/Character/Body')
    solve = stages.add_parser('solve')
    solve.add_argument('--run', required=True, type=Path)
    solve.add_argument('--side', choices=('left', 'right'), required=True)
    finalize = stages.add_parser('finalize-ik')
    finalize.add_argument('--run', required=True, type=Path)
    finalize.add_argument('--root-feedback', action='store_true')
    for stage in ('smooth', 'audit'):
        sub = stages.add_parser(stage)
        sub.add_argument('--reference', required=True, type=Path)
        sub.add_argument('--robot', required=True, type=Path)
        sub.add_argument('--geometry', choices=('girls', 'batch', 'batch-hips', 'batch-hips-palm'), required=True)
        sub.add_argument('--device', default='cpu')
        sub.add_argument('--targets', type=Path)
        if stage == 'audit':
            sub.add_argument('--candidate', required=True, type=Path)
            sub.add_argument('--report', required=True, type=Path)
        else:
            sub.add_argument('--seed', required=True, type=Path)
            sub.add_argument('--output-dir', required=True, type=Path)
            sub.add_argument('--start', type=int, default=0)
            sub.add_argument('--end', type=int)
            sub.add_argument('--window-frames', type=int, default=60)
            sub.add_argument('--stride', type=int, default=30)
            sub.add_argument('--sweeps', type=int, default=2)
            sub.add_argument('--iterations', type=int, default=10)
            sub.add_argument('--acceleration-weight', type=float, default=.002)
            sub.add_argument('--jerk-weight', type=float, default=.000003)
            sub.add_argument('--escape-center', type=int, action='append', default=[])
            sub.add_argument('--restoration-quadratic-weight', type=float, default=2e8)
    args = parser.parse_args(argv)
    from source.train_workflow.utils.grounding import pipeline
    from source.train_workflow.utils.grounding.smoothing import SmoothingSettings
    import torch
    torch.set_num_threads(4)
    if args.stage == 'prepare':
        pipeline.prepare(args.input, args.source, args.robot, args.output_dir, frames=args.frames,
                         source_leg_mode=args.source_leg_mode, character=args.character, mesh_path=args.mesh_path)
    elif args.stage == 'solve':
        pipeline.solve(args.run, args.side)
    elif args.stage == 'finalize-ik':
        pipeline.finalize_ik(args.run, root_feedback=args.root_feedback)
    elif args.stage == 'audit':
        if args.report.exists():
            raise FileExistsError(args.report)
        report = pipeline.audit_motion(args.reference, args.candidate, args.robot,
            geometry=args.geometry, device=args.device, targets=args.targets)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        pipeline.write_json(args.report, report)
        return int(not report['passed'])
    else:
        settings = SmoothingSettings(args.window_frames, args.stride, args.sweeps, args.iterations,
            args.acceleration_weight, args.jerk_weight, args.restoration_quadratic_weight)
        pipeline.smooth_motion(args.reference, args.seed, args.robot, args.output_dir, geometry=args.geometry,
            device=args.device, settings=settings, start=args.start, end=args.end,
            escape_centers=args.escape_center, targets=args.targets)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
