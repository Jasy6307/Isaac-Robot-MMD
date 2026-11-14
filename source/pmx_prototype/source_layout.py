"""Project paths, categorized imports and source-name compatibility."""
import json
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parents[1]
MODULE_LOCATIONS = json.loads((SOURCE_DIR / 'module_locations.json').read_text(encoding='utf-8'))
MODULE_DIRS = ('runtime', 'training', 'motion', 'diagnostics', 'tools')

for category in MODULE_DIRS:
    directory = str(SOURCE_DIR / category)
    if directory not in sys.path:
        sys.path.append(directory)


def resolve_source(name):
    """Resolve an old flat source name or a canonical category-relative path."""
    return SOURCE_DIR / MODULE_LOCATIONS.get(str(name), str(name))
