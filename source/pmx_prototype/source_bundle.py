"""Copy PMX source snapshots together with their local Python dependencies."""
import ast
import json
from pathlib import Path
import shutil

from source_layout import MODULE_LOCATIONS, SOURCE_DIR


def source_files(names, source_dir=SOURCE_DIR):
    """Resolve local imports, compatibility implementations and archived JSON.

    This reads source without importing it: Kit, GPU and audio startup must not
    run while a training snapshot or release is being packaged.
    """
    source_dir = Path(source_dir)
    archived = {}
    catalog = source_dir / 'archive/catalog.json'
    if catalog.exists():
        for item in json.loads(catalog.read_text(encoding='utf-8'))['configs']:
            for key in (item['legacy_path'], item['source_path']):
                archived[key] = item['archive_path']
    pending = [*names, 'source_bundle.py', 'source_layout.py', 'module_locations.json', '__init__.py']
    resolved = set()
    while pending:
        requested = Path(pending.pop()).as_posix()
        relative = MODULE_LOCATIONS.get(requested, requested)
        if relative in resolved:
            continue
        path = (source_dir / relative).resolve()
        if not path.is_relative_to(source_dir.resolve()):
            raise ValueError(f'Source must be inside the PMX source directory: {relative}')
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.add(relative)
        if requested in archived:
            pending.extend((archived[requested], 'archive/catalog.json'))
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'resolve_source' and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.endswith(('.json','.py','.mdl'))):
                pending.append(node.args[0].value)
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                if module.startswith('source.pmx_prototype.'):
                    module = module.removeprefix('source.pmx_prototype.')
                candidate = module.replace('.', '/') + '.py'
                if (source_dir / MODULE_LOCATIONS.get(candidate, candidate)).is_file():
                    pending.append(candidate)
    return tuple(sorted(resolved))


def copy_sources(destination, names, source_dir=SOURCE_DIR):
    """Preserve source-relative paths so both old and new tool paths still work."""
    destination = Path(destination)
    files = source_files(names, source_dir)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(source_dir) / relative
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    return files
