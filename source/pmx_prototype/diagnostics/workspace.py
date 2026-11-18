"""Separate historical evidence from fresh diagnostic results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from source.paths import REPO_ROOT
from source.pmx_prototype.experiments import jobs


class Workspace:
    def __init__(self, input_dir=None, output_dir=None, *, inference=False, config_paths=None):
        self.input_dir = Path(input_dir).resolve() if input_dir else None
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.inference = inference
        self.original = None
        self.config_paths = config_paths
        self.owned = set()

    def directory(self, original):
        self.original = Path(original).resolve()
        if self.input_dir is None:
            self.input_dir = self.original
        return self.output_dir if self.inference and self.output_dir else self.input_dir

    def input_path(self, path):
        path = Path(path)
        if path.is_absolute():
            path = path.resolve()
        if self.output_dir and path.is_relative_to(self.output_dir):
            if path.exists():
                return path
            return self.input_dir/path.relative_to(self.output_dir)
        if self.original and self.input_dir and path.is_absolute() and path.is_relative_to(self.original):
            return self.input_dir/path.relative_to(self.original)
        return path

    def read_text(self, path, *args, **kwargs):
        return self.input_path(path).read_text(*args, **kwargs)

    def read_json(self, path):
        return json.loads(self.read_text(path, encoding='utf-8'))

    def output_path(self, path):
        if self.output_dir is None:
            raise ValueError('Choose a fresh --output-dir before writing results')
        path = Path(path).resolve()
        for base in (self.output_dir, self.input_dir, self.original):
            if base and path.is_relative_to(base):
                target = self.output_dir/path.relative_to(base)
                break
        else:
            raise ValueError(f'Result must be inside the selected workspace: {path}')
        if target.exists() and target not in self.owned:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.owned.add(target)
        return target

    def write_text(self, path, text, *args, **kwargs):
        return self.output_path(path).write_text(text, encoding='utf-8')

    def write_json(self, path, value):
        jobs.write(self.output_path(path), value)

    def execute(self, stage, arguments, *, started=True):
        if not self.inference:
            raise ValueError('Offline analysis cannot launch inference')
        output = self.output_dir
        self.write_json(output/'status.json', dict(stage=stage, training=False,
                        **(dict(started_unix=time.time()) if started else {})))
        started_at = time.perf_counter()
        # Existing evidence/configs may live in the input tree; new evaluator
        # destinations must stay in the fresh output tree.
        arguments = list(map(str, arguments))
        for index, value in enumerate(arguments):
            if index and arguments[index-1] == '--output':
                continue
            if Path(value).is_absolute():
                arguments[index] = str(self.input_path(value))
        environment = os.environ.copy()
        environment.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
        with self.output_path(output/(stage+'.log')).open('x', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', *map(str, arguments)], cwd=REPO_ROOT,
                stdout=log, stderr=subprocess.STDOUT, env=environment,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.write_json(output/'active_process.json', dict(stage=stage, pid=child.pid))
            if child.wait():
                raise RuntimeError(stage+' failed; inspect its log')
        return time.perf_counter()-started_at
