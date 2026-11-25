"""Small, dependency-free startup journals shared by the three inspector processes.

Milestones count completed tasks, not an estimated fraction of loading time.
Each process owns one file; readers never interfere with the control transport.
"""

import json
import os
from pathlib import Path
import time


STAGES = {
    'viewer': (
        ('engine', 'Starting viewport / Isaac Kit'),
        ('imports', 'Loading rendering and character modules'),
        ('body', 'Waiting for the policy and body simulation'),
        ('assets', 'Reading character USD, PMX and skeleton'),
        ('scene', 'Building materials, lights and camera'),
        ('warmup', 'Preparing viewport and inspector'),
        ('secondary', 'Waiting for hair / skirt physics'),
        ('first_frame', 'Solving the first interactive frame'),
        ('ready', 'Ready - keyboard control available'),
    ),
    'policy': (
        ('engine', 'Starting policy Isaac Kit'),
        ('imports', 'Loading PyTorch and policy modules'),
        ('config', 'Reading controller and reference configuration'),
        ('environment', 'Loading motion bank and building physical body'),
        ('network', 'Creating network and restoring checkpoint'),
        ('reset', 'Preparing initial observations and body pose'),
        ('ready', 'Policy ready'),
    ),
    'secondary': (
        ('engine', 'Starting hair / skirt Isaac Kit'),
        ('imports', 'Loading secondary physics modules'),
        ('input', 'Waiting for initial body pose'),
        ('assets', 'Reading PMX and preparing hair / skirt rig'),
        ('rig', 'Creating PhysX nodes, constraints and proxies'),
        ('attach', 'Preparing scene and attaching PhysX'),
        ('warmup', 'Warming up hair / skirt physics'),
        ('ready', 'Hair / skirt ready'),
    ),
}


_write_warnings = set()


def write_progress(path, data):
    """Best-effort diagnostics: a Windows reader may temporarily deny replacement.

    Keep the last complete snapshot and retry on the next progress update, without
    blocking the UI or allowing an optional journal to terminate a simulation.
    """
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    except OSError as error:
        if path not in _write_warnings:
            print(f'ANON_STARTUP_WARNING Progress file unavailable; continuing startup '
                  f'and retrying on the next update: {path} ({error})', flush=True)
            _write_warnings.add(path)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    _write_warnings.discard(path)
    return True


class StartupJournal:
    def __init__(self, output, component, *, started=None, clock=time.perf_counter):
        self.clock = clock
        self.started = clock() if started is None else started
        self.component = component
        self.path = Path(output) / f'startup_{component}.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []

    def mark(self, stage, detail=''):
        now = self.clock()
        if self.events and self.events[-1]['stage'] == stage:
            self.events[-1]['detail'] = detail
        else:
            if self.events:
                self.events[-1]['seconds'] = now - self.events[-1]['at']
            titles = dict(STAGES[self.component])
            self.events.append(dict(stage=stage, title=titles.get(stage, stage), detail=detail,
                                    at=now, elapsed_seconds=now-self.started))
        data = self.snapshot(now)
        write_progress(self.path, data)
        print(f"ANON_STARTUP {self.component} {stage} elapsed={now-self.started:.2f}s {detail}", flush=True)

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        current = self.events[-1] if self.events else {}
        stage = current.get('stage', '')
        if stage in ('ready', 'failed', 'cancelled'):
            now = current['at']
        stages = [key for key, _ in STAGES[self.component]]
        return dict(component=self.component, pid=os.getpid(), started=self.started,
                    stage=stage, title=current.get('title', ''), detail=current.get('detail', ''),
                    completed_tasks=stages.index(stage) if stage in stages else 0,
                    total_tasks=len(stages)-1, elapsed_seconds=now-self.started,
                    stage_seconds=now-current.get('at', now), events=self.events)


def read_journal(path, now=None):
    """A missing or replacing file is simply not available on this UI tick."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    now = time.perf_counter() if now is None else now
    if data['stage'] not in ('ready', 'failed', 'cancelled'):
        data['elapsed_seconds'] = now - data['started']
        data['stage_seconds'] = now - data['events'][-1]['at']
    return data


class StartupWindow:
    def __init__(self, journal, *, visible=True):
        self.journal = journal
        self.last_update = 0.
        self.window = None
        self.labels = {}
        self.total_label = None
        if visible:
            import omni.ui as ui
            self.window = ui.Window('Loading Policy Inspector', width=590, height=300)
            with self.window.frame:
                with ui.VStack(spacing=10):
                    self.total_label = ui.Label('Preparing interactive preview...', height=28)
                    ui.Label('Completed tasks below are milestones, not a time estimate.', height=22)
                    for component in STAGES:
                        self.labels[component] = ui.Label('', height=52, word_wrap=True)

    def update(self, *, force=False):
        now = time.perf_counter()
        if not force and now-self.last_update < .2:
            return
        self.last_update = now
        data = {'viewer': self.journal.snapshot(now)}
        output = self.journal.path.parent
        for component, path in (('policy', output/'startup_policy.json'),
                                ('secondary', output/'secondary_worker/startup_secondary.json')):
            value = read_journal(path, now)
            if value:
                data[component] = value
        for component, label in self.labels.items():
            value = data.get(component)
            label.text = (f'{component.title()}: not started / not required' if value is None else
                          f"{component.title()}  {value['completed_tasks']}/{value['total_tasks']} tasks"
                          f"  |  {value['elapsed_seconds']:.1f}s\n{value['title']}"
                          f"  ({value['stage_seconds']:.1f}s)  {value['detail']}")
        if self.total_label:
            self.total_label.text = f'Loading preview - elapsed {now-self.journal.started:.1f}s'
        write_progress(output/'startup_state.json', data)

    def close(self):
        self.update(force=True)
        if self.window:
            self.window.destroy()
            self.window = None
