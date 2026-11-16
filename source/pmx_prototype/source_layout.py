"""Project paths, categorized imports and source-name compatibility."""
import json
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in (SOURCE_DIR, *SOURCE_DIR.parents)
     if (parent / 'run_isaac61.bat').is_file() and (parent / 'source/pmx_prototype').is_dir()),
    SOURCE_DIR.parents[1],
)
MODULE_LOCATIONS = json.loads((SOURCE_DIR / 'module_locations.json').read_text(encoding='utf-8'))
MODULE_DIRS = ('runtime', 'training', 'motion', 'diagnostics', 'tools')

for category in MODULE_DIRS:
    directory = str(SOURCE_DIR / category)
    if directory not in sys.path:
        sys.path.append(directory)


def resolve_source(name):
    """Resolve an old flat source name or a canonical category-relative path."""
    return SOURCE_DIR / MODULE_LOCATIONS.get(str(name), str(name))


def config_path(value):
    """Resolve retired flat project config paths while preserving custom files."""
    path = Path(value)
    if path.is_file():
        return path
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        relative = absolute.relative_to(SOURCE_DIR).as_posix()
    except ValueError:
        return path
    target = resolve_source(relative)
    return target if target.is_file() else path
