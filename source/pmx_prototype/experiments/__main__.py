"""List historical experiment recipes or preview their preserved entry commands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CATALOG = Path(__file__).parent/'catalog.json'


def catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list')
    run = actions.add_parser('run')
    run.add_argument('recipe', choices=sorted(catalog()))
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--wait-seconds', type=int, help='Original parallelism-controller wait option.')
    args = parser.parse_args(argv)
    recipes = catalog()
    if args.action == 'list':
        for name, row in recipes.items():
            print(f"{name:36} {row['status']:18} {row['entry']}")
        return 0
    recipe = recipes[args.recipe]
    command = [sys.executable, '-u', str(ROOT/recipe['entry'])]
    if args.wait_seconds is not None:
        if args.recipe != 'compare_anon_parallelism':
            parser.error('--wait-seconds belongs to compare_anon_parallelism')
        command += ['--wait-seconds', str(args.wait_seconds)]
    if args.dry_run:
        print(json.dumps(dict(recipe=args.recipe, status=recipe['status'], command=command,
                              implementation=recipe['implementation']), ensure_ascii=False, indent=2))
        return 0
    if recipe['status'] == 'active-preserved':
        parser.error('This active experiment and its entry are preserved; this launcher will not start a duplicate')
    # Keep original filenames visible in the child command line. The historical
    # pause/limit controller verifies that exact process identity before stopping.
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
