"""Replace pygame 2.6.1's deprecated resource reader in the active environment.

Run using the Python interpreter whose pygame installation needs repair.
The original module and a repair manifest are saved in the project output.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
MARKER='# robot_isaac: stdlib pygame resources v1'
REPLACEMENT='''# robot_isaac: stdlib pygame resources v1
from importlib import import_module
from importlib.resources import files


def _resource_path(package, name):
    module = import_module(package)
    anchor = module if hasattr(module, "__path__") else module.__package__
    return files(anchor).joinpath(name)


def resource_exists(package, name):
    path = _resource_path(package, name)
    return path.is_file() or path.is_dir()


def resource_stream(package, name):
    return _resource_path(package, name).open("rb")
'''


def main():
    spec=importlib.util.find_spec('pygame')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('pygame is not installed in this Python environment')
    target=Path(next(iter(spec.submodule_search_locations)))/'pkgdata.py'
    original=target.read_bytes();text=original.decode('utf-8').replace('\r\n','\n')
    if MARKER in text:
        print('PYGAME_RESOURCE_REPAIR_ALREADY_APPLIED',target);return
    if importlib.metadata.version('pygame')!='2.6.1':
        raise RuntimeError('Repair is limited to the inspected pygame 2.6.1; review other versions first')
    start=text.index('try:\n    from pkg_resources import resource_stream, resource_exists')
    end=text.index('\n\ndef getResource',start)
    patched=text[:start]+REPLACEMENT+text[end:]
    compile(patched,str(target),'exec')
    assert 'from pkg_resources import' not in patched
    digest=hashlib.sha256(original).hexdigest()
    output=ROOT/'output/pygame_resources_repair'/digest[:16];output.mkdir(parents=True,exist_ok=True)
    backup=output/'pkgdata.py.original'
    if backup.exists():assert backup.read_bytes()==original
    else:backup.write_bytes(original)
    manifest=dict(interpreter=sys.executable,target=str(target),backup=str(backup),
        original_sha256=digest,patched_sha256=hashlib.sha256(patched.encode('utf-8')).hexdigest(),
        pygame_version='2.6.1',replacement='importlib.resources.files',status='prepared')
    record=output/'repair.json';record.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    target.write_bytes(patched.encode('utf-8'))
    manifest['status']='applied';record.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PYGAME_RESOURCE_REPAIR_APPLIED',target)
    print('BACKUP',backup)


if __name__=='__main__':main()
