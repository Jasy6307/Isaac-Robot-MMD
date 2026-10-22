"""Own one isolated repair process; keep optimization off the Kit/UI thread."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[4]


class CollisionH5Job:
    def __init__(self):
        self.process = None
        self.log = None
        self.state = {"state": "idle", "progress": 0, "message": ""}
        self.directory = None
        self.next_poll = 0
        self.outputs = {}
        self.lookup_cache = {}
        atexit.register(self.close)

    def prepare(self, csv_path):
        if self.process is not None:
            raise RuntimeError("A collision export is already running")
        self.directory = ROOT / "output/collision-h5-jobs" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        source = Path(csv_path).resolve()
        base = source.with_name(source.stem + "_without_collision.h5")
        output = base
        version = 1
        while output.exists() or output.with_suffix(".collision.json").exists():
            output = base.with_name(f"{base.stem}_{version:03d}.h5")
            version += 1
        self.state = {"state": "recording", "progress": 0, "message": "Capturing CSV targets", "source_csv": str(source), "output": str(output), "log": str(self.directory / "worker.log")}
        return self.directory / "raw.h5"

    def start(self, raw_path, robot_usd):
        if self.process is not None or self.directory is None:
            raise RuntimeError("Export must be prepared before starting")
        self.status_path = self.directory / "status.json"
        self.log = (self.directory / "worker.log").open("w", encoding="utf-8")
        command = [sys.executable, "-u", str(ROOT / "source/train_workflow/scripts/record_h5_without_collision.py"), "--input", str(raw_path), "--output", self.state["output"], "--robot_usd", str(robot_usd), "--status", str(self.status_path)]
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            self.log.close()
            self.log = None
            raise
        self.state.update(state="running", message="Starting geometry repair")
        self.next_poll = 0

    def poll(self):
        if self.process is None or time.monotonic() < self.next_poll:
            return self.state
        self.next_poll = time.monotonic()+.25
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
            # Only the parent reports completion after the process exits.
            if payload.get("state") in ("done", "failed"):
                payload["state"] = "running"
            self.state.update(payload)
        except (OSError, ValueError):
            pass
        code = self.process.poll()
        if code is not None:
            self.process = None
            self.log.close()
            self.log = None
            if code == 0 and self._validated(Path(self.state["output"])):
                self.state.update(state="done", progress=1, message=Path(self.state["output"]).name)
                self.outputs[self.state["source_csv"]] = self.state["output"]
                self.lookup_cache.clear()
            else:
                self.state.update(state="failed", message=self.state.get("message") if self.state.get("stage") == "failed" else f"Worker exited {code}; see {self.state['log']}")
        return self.state

    @staticmethod
    def _validated(path):
        try:
            import h5py
            with h5py.File(path) as stream:
                return bool(stream.attrs.get("without_collision_validated", False))
        except (OSError, ValueError):
            return False

    def latest_output(self, csv_path):
        source = Path(csv_path).resolve()
        key = str(source)
        if key in self.outputs and Path(self.outputs[key]).exists():
            return self.outputs[key]
        cached = self.lookup_cache.get(key)
        if cached and time.monotonic() < cached[0]:
            return cached[1]
        prefix = source.stem+"_without_collision"
        try:
            candidates = [p for p in source.parent.iterdir() if p.name == prefix+".h5" or (p.name.startswith(prefix+"_") and p.suffix == ".h5" and p.stem[len(prefix)+1:].isdigit())]
            candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
            output = next((str(p) for p in candidates if self._validated(p)), None)
        except OSError:
            output = None
        self.lookup_cache[key] = (time.monotonic()+3, output)
        return output

    def fail(self, message):
        self.state.update(state="failed", message=str(message))

    def cancel(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.state["state"] in ("recording", "running"):
            self.state.update(state="cancelled", message="Export cancelled")

    def close(self):
        self.cancel()
