"""Exercise real local IPC, payload ownership, reset/stop and sequence guards."""
import tempfile
import threading
import time
import unittest
import numpy as np
from anon_policy_transport import PolicyServer, connect_worker


class PolicyTransportTests(unittest.TestCase):
    def receive(self, server, seq):
        deadline = time.perf_counter()+5
        while time.perf_counter()<deadline:
            state = server.receive(seq)
            if state is not None:return state
            time.sleep(.001)
        self.fail('IPC response timed out')

    def test_memory_commands_states_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            server = PolicyServer(directory)
            errors = []
            def worker():
                try:
                    with connect_worker(server.worker_args[1], server.worker_args[3]) as connection:
                        connection.send(dict(seq=-1, poses=np.zeros((35,7),dtype=np.float32)))
                        command = connection.recv()
                        assert command == dict(seq=0, command=[.4,0,.6], reset=True)
                        connection.send(dict(seq=0, poses=np.ones((35,7),dtype=np.float32),reset=True))
                        assert connection.recv()==dict(stop=True)
                except Exception as error:errors.append(error)
            thread = threading.Thread(target=worker,daemon=True)
            thread.start()
            try:
                np.testing.assert_array_equal(self.receive(server,-1)['poses'],np.zeros((35,7)))
                server.send(dict(seq=0,command=[.4,0,.6],reset=True))
                state=self.receive(server,0)
                self.assertTrue(state['reset'])
                np.testing.assert_array_equal(state['poses'],np.ones((35,7)))
                server.send(dict(stop=True))
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors,[])
            finally:server.close()


if __name__=='__main__':unittest.main()
