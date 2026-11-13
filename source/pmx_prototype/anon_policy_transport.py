"""Local authenticated memory transport; file transport retained for A/B checks."""
import json
import os
from pathlib import Path
import threading
import uuid
from multiprocessing.connection import Listener, Client


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
                    self.connection = self.listener.accept()
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
    return Client(tuple(json.loads(address)), authkey=authkey.encode())
