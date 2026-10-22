"""Optional empty-file annotations for training run directories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def run_note_filename(value: str) -> str:
    """Keep the note verbatim, but require a usable Windows file name."""
    if (
        not value
        or value in {".", ".."}
        or value.endswith((" ", "."))
        or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in value)
        or len(value.encode("utf-16-le")) // 2 > 255
    ):
        raise argparse.ArgumentTypeError("--run_note must be a valid single Windows file name")
    stem = value.split(".", 1)[0].rstrip(" ").upper()
    if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(
        r"(?:COM|LPT)[1-9¹²³]", stem
    ):
        raise argparse.ArgumentTypeError("--run_note cannot use a reserved Windows device name")
    if value.lower() in {"params", "git", "videos"} or re.match(
        r"(?:model_.*\.pt$|events\.out\.tfevents)", value, re.IGNORECASE
    ):
        raise argparse.ArgumentTypeError("--run_note conflicts with a training log artifact name")
    return value


def create_run_note(log_dir: str, note: str) -> Path:
    """Create an empty note without replacing any existing artifact."""
    filename = run_note_filename(note)
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("xb"):
        pass
    return path
