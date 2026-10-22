"""Select a checkpoint strictly inside a requested run directory."""

from pathlib import Path
import re


def latest_checkpoint(directory: str) -> str:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pt"]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints in directory: {root}")
    # Numbered training saves use iteration order, independent of copy timestamps.
    numbered = [(int(m.group(1)), p) for p in candidates
                if (m := re.fullmatch(r"model_(\d+)\.pt", p.name, re.IGNORECASE))]
    if len(numbered) == len(candidates):
        chosen = max(numbered, key=lambda item: (item[0], item[1].name))[1]
    else:
        chosen = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))
    return str(chosen)
