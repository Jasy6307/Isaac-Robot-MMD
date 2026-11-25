"""Startup journals and deferred worker launch without requiring Isaac Kit."""

from pathlib import Path
import json
import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_startup_progress import StartupJournal, StartupWindow, read_journal
from anon_secondary_stream import AnonSecondaryStream


class StartupTests(unittest.TestCase):
    def test_stage_durations_and_ready_time_survive_later_reads(self):
        with TemporaryDirectory() as directory:
            clock = [10.]
            journal = StartupJournal(directory, 'viewer', clock=lambda: clock[0])
            journal.mark('engine')
            clock[0] = 14.
            journal.mark('imports')
            clock[0] = 16.
            journal.mark('ready')
            self.assertEqual(journal.events[0]['seconds'], 4.)
            self.assertEqual(journal.events[1]['seconds'], 2.)
            clock[0] = 90.
            self.assertEqual(journal.snapshot()['elapsed_seconds'], 6.)
            data = read_journal(journal.path, 90.)
            self.assertEqual(data['elapsed_seconds'], 6.)
            self.assertEqual(data['completed_tasks'], data['total_tasks'])

    def test_warmup_details_do_not_restart_stage_timer(self):
        with TemporaryDirectory() as directory:
            clock = [10.]
            journal = StartupJournal(directory, 'secondary', clock=lambda: clock[0])
            journal.mark('warmup', '0/120')
            clock[0] = 12.
            journal.mark('warmup', '30/120')
            data = read_journal(journal.path, 15.)
            self.assertEqual(data['stage_seconds'], 5.)
            self.assertEqual(data['detail'], '30/120')
            self.assertEqual(len(data['events']), 1)

    def test_missing_or_partially_written_worker_file_is_tolerated(self):
        with TemporaryDirectory() as directory:
            journal = StartupJournal(directory, 'viewer')
            journal.mark('body')
            policy = Path(directory)/'startup_policy.json'
            self.assertIsNone(read_journal(policy))
            policy.write_text('{', encoding='utf-8')
            window = StartupWindow(journal, visible=False)
            window.update(force=True)
            self.assertTrue((Path(directory)/'startup_state.json').is_file())
            window.close()

    def test_deferred_secondary_does_not_send_initial_pose_until_initialize(self):
        with TemporaryDirectory() as directory:
            channel = SimpleNamespace(worker_args=[], connection=None, error=None)
            channel.send = lambda value: sent.append(value)
            sent = []
            args = SimpleNamespace(output=Path(directory), pmx=Path('character.pmx'),
                                   quality='cloth', physics_threads=16, skirt_edge_stride=3,
                                   secondary_profile=None, reference_hair_compute=False, legacy_compute=False)
            with patch('anon_secondary_stream.PolicyServer', return_value=channel), \
                 patch('anon_secondary_stream.subprocess.Popen') as process:
                stream = AnonSecondaryStream(args)
                process.assert_called_once()
                self.assertEqual(sent, [])
                channel.connection = object()
                with patch.object(stream, 'receive', return_value={'seq': -1}) as receive:
                    stream.initialize('initial pose', 'manifest', 'links')
                self.assertEqual(sent, [dict(poses='initial pose', manifest='manifest', links='links')])
                receive.assert_called_once()
                stream.log.close()

    def test_denied_replace_keeps_previous_snapshot_and_recovers_next_update(self):
        with TemporaryDirectory() as directory:
            journal = StartupJournal(directory, 'viewer')
            journal.mark('secondary')
            window = StartupWindow(journal, visible=False)
            window.update(force=True)
            path = Path(directory)/'startup_state.json'
            previous = path.read_bytes()
            journal.mark('ready')
            with patch('anon_startup_progress.os.replace', side_effect=PermissionError(5, 'Access denied')):
                for _ in range(3):window.update(force=True)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(path.with_suffix('.tmp.json').exists())
            window.update(force=True)
            self.assertEqual(json.loads(path.read_text())['viewer']['stage'], 'ready')
            window.close()

    def test_optional_journal_write_failure_does_not_stop_stage_or_cleanup(self):
        with TemporaryDirectory() as directory:
            journal = StartupJournal(directory, 'secondary')
            journal.mark('rig')
            previous = journal.path.read_bytes()
            with patch('anon_startup_progress.Path.write_text', side_effect=PermissionError(5, 'Access denied')):
                journal.mark('attach')
                window = StartupWindow(journal, visible=False)
                window.update(force=True)
                window.close()
            self.assertEqual(journal.path.read_bytes(), previous)
            self.assertEqual(journal.snapshot()['stage'], 'attach')
            journal.mark('ready')
            self.assertEqual(read_journal(journal.path)['stage'], 'ready')

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing regression')
    def test_windows_open_reader_denying_delete_cannot_abort_startup(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        with TemporaryDirectory() as directory:
            journal = StartupJournal(directory, 'viewer')
            journal.mark('secondary')
            window = StartupWindow(journal, visible=False)
            window.update(force=True)
            path = Path(directory)/'startup_state.json'
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            try:
                journal.mark('ready')
                with patch('builtins.print') as warning:
                    window.update(force=True)
                self.assertTrue(any('ANON_STARTUP_WARNING' in str(call) for call in warning.call_args_list))
                self.assertEqual(json.loads(path.read_text())['viewer']['stage'], 'secondary')
            finally:
                kernel.CloseHandle(handle)
            window.close()
            self.assertEqual(json.loads(path.read_text())['viewer']['stage'], 'ready')


if __name__ == '__main__':
    unittest.main()
