"""Canonical workspace paths shared across training and playback scripts."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SOURCE_DIR, ".."))

MEDIA_DIR = os.path.join(REPO_ROOT, "media")
DANCE_DIR = os.path.join(MEDIA_DIR, "dance")
POSE_DIR = os.path.join(MEDIA_DIR, "pose")


def resolve_dance_asset(value: str | os.PathLike, *, dance_dir=None) -> Path:
    """Resolve an asset, accepting legacy flat paths under ``media/dance``.

    Existing paths always win. Only missing files directly under the dance root
    are redirected; unrelated or misspelled nested paths are left alone. New
    output filenames use the longest matching dance folder name, so variants
    such as ``IRIS_OUT_collision_corrected_v2.h5`` stay with their source dance.
    """
    path = Path(str(value).replace("\\", os.sep)).expanduser().resolve()
    root = Path(dance_dir or DANCE_DIR).resolve()
    if path.exists() or path.parent != root or not root.is_dir():
        return path
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    matches = [p / path.name for p in folders if (p / path.name).is_file()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous legacy dance path {value}: {matches}")
    if matches:
        return matches[0]
    name = path.name.casefold()
    owners = [p for p in folders if any(
        name.startswith(p.name.casefold() + sep) for sep in (".", "_"))]
    if owners:
        return max(owners, key=lambda p: len(p.name)) / path.name
    return path


def iter_dance_assets(pattern: str, *, dance_dir=None):
    """List both legacy top-level assets and assets in per-dance folders."""
    root = Path(dance_dir or DANCE_DIR)
    return sorted(p for p in root.glob(f"**/{pattern}") if p.is_file())
