"""Selection must not start Isaac implicitly or silently omit new tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tests as runner


class RunnerTests(unittest.TestCase):
    def test_cpu_selection_excludes_simulation_and_local_assets(self):
        catalog = runner.load_catalog()
        chosen = runner.select_tests(catalog, ["cpu"])
        self.assertTrue(chosen)
        self.assertTrue(all(e["suite"] == "cpu" for e in chosen))
        self.assertTrue(runner.select_tests(catalog, ["isaac"]))
        self.assertTrue(all(e["area"] == "pmx" for e in
                            runner.select_tests(catalog, ["cpu"], "pmx")))

    def test_new_test_requires_explicit_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests/suites.json").write_text("[]")
            (root / "tests/test_new.py").touch()
            with self.assertRaisesRegex(ValueError, "Classify new tests"):
                runner.load_catalog(root)

    def test_duplicate_and_missing_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            entry = dict(path="tests/test_new.py", suite="cpu", area="infra")
            manifest = root / "tests/suites.json"
            manifest.write_text(json.dumps([entry]))
            with self.assertRaisesRegex(ValueError, "Invalid test path"):
                runner.load_catalog(root)
            (root / entry["path"]).touch()
            manifest.write_text(json.dumps([entry, entry]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                runner.load_catalog(root)

    def test_failure_and_timeout_are_not_reported_as_success(self):
        entry = dict(path="tests/test_test_runner.py", suite="cpu", area="infra")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                self.assertEqual(runner.run_entry(entry, Path(directory), 1)["status"], "failed")
            with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 1)):
                self.assertEqual(runner.run_entry(entry, Path(directory), 1)["status"], "timeout")

    def test_failed_run_writes_report_and_returns_nonzero(self):
        entry = dict(path="tests/test_example.py", suite="cpu", area="infra")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner, "load_catalog", return_value=[entry]), \
                 patch.object(runner, "run_entry", return_value=dict(
                     **entry, status="failed", seconds=0, log="example.log")):
                self.assertEqual(runner.main([]), 1)
            report = next(root.glob("output/tests/*/summary.json"))
            self.assertEqual(json.loads(report.read_text())["results"][0]["status"], "failed")

    def test_empty_discovery_is_not_success(self):
        entry = dict(path='tests/test_empty.py', suite='cpu', area='infra')
        def no_cases(*args, **kwargs):
            kwargs['stdout'].write('Ran 0 tests in 0.000s\n\nOK\n')
            return subprocess.CompletedProcess([], 0)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.subprocess, 'run', side_effect=no_cases):
            result = runner.run_entry(entry, Path(directory), 1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['case_count'], 0)


if __name__ == "__main__":
    unittest.main()
