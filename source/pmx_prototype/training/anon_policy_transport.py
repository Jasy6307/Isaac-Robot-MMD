"""Local authenticated memory transport; file transport retained for A/B checks."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
import os
from pathlib import Path
import threading
import uuid
import socket
from multiprocessing.connection import Listener, Client


def low_latency(connection):
    # State packets can span an MTU. Do not hold the final short packet for
    # Nagle/delayed ACK, since every control step waits for that exact state.
    handle=socket.socket(fileno=connection.fileno())
    try:handle.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    finally:handle.detach()
    return connection


class PolicyServer:
    def __init__(self, directory, transport='memory'):
        self.directory = Path(directory)
        self.transport = transport
        self.connection = None
        self.listener = None
        self.error = None
        self.worker_args = []
        if transport == 'memory':
            key = uuid.uuid4().hex
            self.listener = Listener(('127.0.0.1', 0), authkey=key.encode())
            self.worker_args = ['--address', json.dumps(self.listener.address), '--authkey', key]
            def accept():
                try:
                    self.connection = low_latency(self.listener.accept())
                except Exception as error:
                    self.error = error
            threading.Thread(target=accept, daemon=True).start()

    def send(self, data):
        if self.transport == 'memory':
            self.connection.send(data)
        elif data.get('stop'):
            (self.directory/'stop.request').touch()
        else:
            temporary = self.directory/f'request_{data["seq"]}.tmp'
            temporary.write_text(json.dumps(data), encoding='utf-8')
            os.replace(temporary, self.directory/f'request_{data["seq"]}.json')

    def receive(self, seq):
        if self.error:
            raise self.error
        if self.transport == 'memory':
            if self.connection is None or not self.connection.poll():
                return None
            state = self.connection.recv()
            if state['seq'] != seq:
                raise RuntimeError(f'Out-of-order policy state: {state["seq"]} != {seq}')
            return state
        import numpy as np
        path = self.directory/f'state_{seq}.npz'
        try:
            with np.load(path) as state:
                value = {key:state[key].copy() for key in state.files}
            path.unlink()
            return value
        except (FileNotFoundError, PermissionError):
            return None

    def close(self):
        if self.connection is not None:
            self.connection.close()
        if self.listener is not None:
            self.listener.close()


def connect_worker(address, authkey):
    return low_latency(Client(tuple(json.loads(address)), authkey=authkey.encode()))
