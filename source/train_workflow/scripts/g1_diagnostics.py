"""Offline saved-telemetry diagnostics; no policy execution or training."""
import argparse
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

COMMANDS = {
    'jump': {
        'decompose': ('jump_reports', 'decompose_jump_tracking_main'),
        'phases': ('jump_reports', 'mark_jump_phases_main'),
        'reference-change': ('jump_reports', 'plot_jump_reference_change_main'),
        'compare': ('jump_comparison', 'compare_jump_training_main'),
        'summary': ('jump_comparison', 'summarize_multi_jump_main'),
    },
    'turn': {
        'analyze': ('turn_analysis', 'analyze_turn_main'),
        'compare': ('turn_analysis', 'compare_turn_smoothing_main'),
    },
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest='group', required=True)
    for name, commands in COMMANDS.items():
        actions = groups.add_parser(name).add_subparsers(dest='action', required=True)
        for action in commands:
            actions.add_parser(action, add_help=False, help='Use this subcommand with --help for its original arguments.')
    args, remaining = parser.parse_known_args(arguments)
    module, function = COMMANDS[args.group][args.action]
    return getattr(import_module('source.train_workflow.utils.'+module), function)(remaining)


if __name__ == '__main__':
    raise SystemExit(main())
