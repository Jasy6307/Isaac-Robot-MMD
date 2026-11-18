"""Grouped locomotion diagnostics; list and dry-run never start inference."""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'source/pmx_prototype'
CATALOG = Path(__file__).parent/'analyses/catalog.json'


def catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


def build_task(profile, workspace, legacy_argv=None):
    for path in (ROOT, SOURCE, SOURCE/'training', SOURCE/'motion', SOURCE/'diagnostics'):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    row = catalog()[profile]
    module = importlib.import_module('source.pmx_prototype.diagnostics.analyses.'+row['family'])
    return getattr(module, row['builder'])(workspace, legacy_argv)


def legacy_exports(profile):
    from source.pmx_prototype.diagnostics.workspace import Workspace
    task = build_task(profile, Workspace())
    return {k: v for k, v in vars(task).items() if callable(v) or k in ('ROOT', 'OUT', 'SOURCE', 'CONFIGS')}


def add_options(parser):
    parser.add_argument('--input-dir', '--run', '--report-folder', dest='input_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Report filename for limits/replay compatibility.')
    parser.add_argument('--configs', type=Path, help='JSON mapping of named sampling configurations.')
    parser.add_argument('--dry-run', action='store_true')


def main(argv=None, *, profile=None):
    parser = argparse.ArgumentParser(description=__doc__)
    rows = catalog()
    if profile is None:
        groups = parser.add_subparsers(dest='family', required=True)
        for family in sorted({r['family'] for r in rows.values()}):
            actions = groups.add_parser(family).add_subparsers(dest='action', required=True)
            for name, row in rows.items():
                if row['family'] != family:
                    continue
                sub = actions.add_parser(row['action'])
                sub.set_defaults(profile=name)
                add_options(sub)
    else:
        parser.set_defaults(profile=profile)
        add_options(parser)
    args = parser.parse_args(argv)
    row = rows[args.profile]
    input_dir = args.input_dir or ROOT/row['default_input']
    output_dir = args.output_dir or (args.output.parent if args.output else
        ROOT/'output/locomotion_diagnostics'/args.profile/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    plan = dict(profile=args.profile, family=row['family'], inference=row['inference'], training=False,
                input_dir=str(input_dir.resolve()), output_dir=str(output_dir.resolve()),
                implementation='diagnostics/analyses/'+row['family']+'.py')
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    # Never publish new diagnostics over historical evidence.
    output_dir.mkdir(parents=True, exist_ok=bool(args.output and not args.output_dir))
    from source.pmx_prototype.diagnostics.workspace import Workspace
    workspace = Workspace(input_dir, output_dir, inference=row['inference'],
        config_paths={k: Path(v) for k, v in json.loads(args.configs.read_text(encoding='utf-8')).items()} if args.configs else None)
    legacy_args = []
    if args.profile == 'audit_scratch_acceptance_limits':
        legacy_args = ['--report-folder', str(input_dir), '--output', str(output_dir/(args.output.name if args.output else 'acceptance_limits.json'))]
    task = build_task(args.profile, workspace, legacy_args)
    if args.profile == 'replay_acceptance_v2':
        result = task.replay(input_dir)
        workspace.write_json(output_dir/(args.output.name if args.output else 'replayed_acceptance.json'), result)
    else:
        task.main()
    workspace.write_json(output_dir/(args.output.name+'.invocation.json' if args.output else 'invocation.json'), plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
