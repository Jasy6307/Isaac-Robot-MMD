"""Run explicitly classified repository tests in isolated Python processes."""
from __future__ import annotations

import argparse
from datetime import datetime
import fnmatch
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TEST_DIRS = ("tests", "source/train_workflow/tests", "source/pmx_prototype/tests")
SUITES = ("cpu", "usd", "assets", "isaac")


def load_catalog(root=ROOT):
    entries = json.loads((root / "tests/suites.json").read_text(encoding="utf-8"))
    seen = set()
    for entry in entries:
        path = entry["path"]
        if path in seen:
            raise ValueError(f"Duplicate test: {path}")
        seen.add(path)
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f"Invalid test path: {path}")
        if entry["suite"] not in SUITES or entry["area"] not in ("g1", "pmx", "infra"):
            raise ValueError(f"Invalid classification: {path}")
    discovered = {p.relative_to(root).as_posix() for directory in TEST_DIRS
                  for p in (root / directory).glob("test_*.py")}
    missing = discovered - seen
    if missing:
        raise ValueError("Classify new tests in tests/suites.json: " + ", ".join(sorted(missing)))
    return entries


def select_tests(entries, suites, area=None, pattern="*"):
    return [e for e in entries if e["suite"] in suites
            and (area is None or e["area"] == area)
            and fnmatch.fnmatch(Path(e["path"]).stem, pattern)]


def run_entry(entry, folder, timeout):
    path = ROOT / entry["path"]
    if entry["suite"] == "isaac":
        command = [sys.executable, str(path), *entry["args"],
                   "--output_dir", str(folder / "isaac")]
    else:
        command = [sys.executable, "-B", "-m", "unittest", "discover",
                   "-s", str(path.parent), "-p", path.name, "-v"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if entry["suite"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    logfile = folder / (entry["area"] + "_" + path.stem + ".log")
    started = time.monotonic()
    code = None
    with logfile.open("w", encoding="utf-8") as output:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except OSError as error:
            output.write(str(error))
            status = "failed"
    case_count = None
    if entry['suite'] != 'isaac':
        match = re.search(r'^Ran (\d+) tests?', logfile.read_text(encoding='utf-8'), re.MULTILINE)
        case_count = int(match[1]) if match else 0
        if status == 'passed' and not case_count:
            status = 'failed'
            with logfile.open('a', encoding='utf-8') as output:
                output.write('\nNo unittest cases executed; check test discovery.\n')
    return dict(**entry, status=status, returncode=code, case_count=case_count,
                seconds=round(time.monotonic() - started, 2), log=str(logfile), command=command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=SUITES,
                        help="Repeat to combine layers; default: cpu. Isaac is always opt-in.")
    parser.add_argument("--area", choices=("g1", "pmx", "infra"))
    parser.add_argument("--match", default="*", help="Test filename stem glob, e.g. '*audio*'.")
    parser.add_argument("--list", action="store_true", help="Validate inventory and list without importing tests.")
    parser.add_argument("--timeout", type=float, default=900, help="Seconds per test file (default: 900).")
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    try:
        entries = select_tests(load_catalog(), args.suite or ["cpu"], args.area, args.match)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if not entries:
        parser.error("No tests matched")
    if args.list:
        for entry in entries:
            print(f"{entry['suite']:7} {entry['area']:5} {entry['path']}")
        print(f"{len(entries)} test files selected")
        return 0
    folder = ROOT / "output/tests" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}")
    folder.mkdir(parents=True)
    print(f"Python: {sys.executable}\nReports: {folder}", flush=True)
    results = []
    for entry in entries:
        print(f"RUN {entry['path']}", flush=True)
        result = run_entry(entry, folder, args.timeout)
        results.append(result)
        (folder / "summary.json").write_text(json.dumps(
            dict(python=sys.executable, results=results), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{result['status'].upper()} ({result['seconds']}s) {result['log']}", flush=True)
    failed = sum(r["status"] != "passed" for r in results)
    print(f"Files: {len(results)}; passed: {len(results)-failed}; failed/timed out: {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
