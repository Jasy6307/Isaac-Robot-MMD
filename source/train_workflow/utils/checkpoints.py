"""Resolve G1 checkpoints from experiment folders or a specific run."""

from pathlib import Path
import re

from source.train_workflow.task_ids import G1_TRAIN_EXPERIMENT_NAME
from source.train_workflow.utils.motion.resolve import resolve_training_log_root

_FORMER_EXPERIMENT_NAME = "g1_auto_jump_conditional"


def checkpoint_search_roots(experiment_name: str, dance_name: str | None) -> tuple[Path, ...]:
    """Search the current experiment and its former directory, in tie-break order."""
    names = [experiment_name]
    if experiment_name == G1_TRAIN_EXPERIMENT_NAME or experiment_name.startswith(f"{G1_TRAIN_EXPERIMENT_NAME}_"):
        suffix = experiment_name[len(G1_TRAIN_EXPERIMENT_NAME):]
        names.append(f"{_FORMER_EXPERIMENT_NAME}{suffix}")
    return tuple(Path(resolve_training_log_root(name, dance_name)) for name in names)


def latest_checkpoint(directory: str | Path, *, pattern: str | None = None) -> str:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pt"]
    if pattern is not None:
        matches = re.compile(pattern).match
        candidates = [p for p in candidates if matches(p.name)]
    if not candidates:
        filters = f" matching {pattern!r}" if pattern is not None else ""
        raise FileNotFoundError(f"No .pt checkpoints{filters} in directory: {root}")
    # Numbered training saves use iteration order, independent of copy timestamps.
    numbered = [(int(m.group(1)), p) for p in candidates
                if (m := re.fullmatch(r"model_(\d+)\.pt", p.name, re.IGNORECASE))]
    if len(numbered) == len(candidates):
        chosen = max(numbered, key=lambda item: (item[0], item[1].name))[1]
    else:
        chosen = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
    return str(chosen)


def _require_checkpoint_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def resolve_training_checkpoint(
    experiment_name: str,
    dance_name: str | None,
    *,
    load_run: str | None = None,
    checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    run_pattern: str = ".*",
    checkpoint_pattern: str = "model_.*.pt",
) -> str:
    """Resolve an explicit path, a named run, or the newest matching run across both roots."""
    if checkpoint and Path(checkpoint).name != checkpoint:
        return _require_checkpoint_file(Path(checkpoint))
    if checkpoint_dir:
        directory = Path(checkpoint_dir).expanduser()
        return _require_checkpoint_file(directory / checkpoint) if checkpoint else latest_checkpoint(directory)

    roots = checkpoint_search_roots(experiment_name, dance_name)
    exact_runs = [root / load_run for root in roots if load_run and (root / load_run).is_dir()]
    runs = exact_runs
    if not exact_runs:
        matches_run = re.compile(load_run or run_pattern).match
        runs = sorted(
            (run for root in roots if root.is_dir() for run in root.iterdir() if run.is_dir() and matches_run(run.name)),
            key=lambda run: run.name,
            reverse=True,
        )
    for run in runs:
        try:
            if exact_runs and checkpoint:
                return _require_checkpoint_file(run / checkpoint)
            return latest_checkpoint(run, pattern=None if exact_runs else checkpoint or checkpoint_pattern)
        except FileNotFoundError:
            continue

    locations = ", ".join(str(path) for path in (exact_runs or roots))
    selector = f" for run {load_run!r}" if load_run else ""
    model = f" matching {checkpoint!r}" if checkpoint else ""
    raise FileNotFoundError(f"No checkpoint{model}{selector} for dance {dance_name!r} under: {locations}")
