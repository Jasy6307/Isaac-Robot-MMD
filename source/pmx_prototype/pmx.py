"""List and run categorized PMX tools without importing simulation libraries."""
import ast
from pathlib import Path
import runpy
import sys

from source_layout import MODULE_LOCATIONS, SOURCE_DIR, resolve_source


def commands():
    result = {}
    paths = set(MODULE_LOCATIONS.values())
    paths.update(p.name for p in SOURCE_DIR.glob('*.py'))
    for category in ('training', 'diagnostics', 'tools'):
        paths.update(p.relative_to(SOURCE_DIR).as_posix() for p in (SOURCE_DIR/category).glob('*.py'))
    for relative in sorted(paths):
        if relative.startswith(('tests/', 'runtime/')):
            continue
        path = SOURCE_DIR / relative
        if path.suffix != '.py' or path.name in ('pmx.py', '__init__.py', 'workspace.py'):
            continue
        text = path.read_text(encoding='utf-8-sig')
        if ('ArgumentParser(' not in text and "__main__" not in text
                and not relative.startswith(('archive/experiments/', 'diagnostics/',
                                              'tools/', 'motion/build_'))):
            continue
        tree = ast.parse(text)
        result[Path(relative).stem] = (relative, (ast.get_docstring(tree) or '').split('\n')[0])
    return result


def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == 'experiments':
        from source.pmx_prototype.experiments.__main__ import main as experiments
        return experiments(arguments[1:])
    if arguments[:2] == ['diagnostics', 'locomotion']:
        from source.pmx_prototype.diagnostics.locomotion_tools import main as diagnostics
        return diagnostics(arguments[2:])
    available = commands()
    if not arguments or arguments[0] in ('list', '--list', '--help', '-h'):
        print('Usage: python source/pmx_prototype/pmx.py <tool> [arguments...]')
        print('       python source/pmx_prototype/pmx.py <category> <tool> [arguments...]\n')
        print('Grouped: experiments list | diagnostics locomotion --help\n')
        for name, (relative, description) in available.items():
            print(f'{name:36} {relative:58} {description}')
        return
    # Category prefixes are optional; validate them rather than silently ignoring typos.
    if arguments[0] in ('motion', 'training', 'diagnostics', 'tools', 'archive'):
        category = arguments.pop(0)
    else:
        category = None
    if not arguments:
        raise SystemExit('Missing tool name; use pmx.py list')
    name = Path(arguments.pop(0)).stem
    if name not in available:
        raise SystemExit(f'Unknown tool: {name}; use pmx.py list')
    relative = available[name][0]
    if category and not relative.startswith(category + '/'):
        raise SystemExit(f'{name} belongs to {relative}, not {category}')
    path = resolve_source(relative)
    sys.argv = [str(path), *arguments]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    raise SystemExit(main())
