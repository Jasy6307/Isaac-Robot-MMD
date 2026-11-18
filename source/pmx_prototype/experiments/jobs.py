"""Shared experiment I/O and serial child-job bookkeeping."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execute(root, output, stage, arguments, *, training=None, started=True, check=True):
    output = Path(output)
    status = dict(stage=stage)
    if started:
        status['started_unix'] = time.time()
    if training is not None:
        status['training'] = training
    write(output/'status.json', status)
    begin = time.perf_counter()
    with (output/(stage+'.log')).open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', *map(str, arguments)], cwd=root,
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        write(output/'active_process.json', dict(stage=stage, pid=child.pid))
        code = child.wait()
    elapsed = time.perf_counter()-begin
    if check and code:
        raise RuntimeError(f'{stage} failed ({code}); inspect {output/(stage+".log")}')
    return elapsed if check else (code, elapsed)
