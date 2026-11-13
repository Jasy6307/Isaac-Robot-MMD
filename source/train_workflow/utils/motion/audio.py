"""Resolve music through explicit motion provenance, independent of dance suffixes."""
from __future__ import annotations

import json
import math
from pathlib import Path

from source.paths import DANCE_DIR, resolve_dance_asset

ROOT = Path(__file__).resolve().parents[4]


def _audio_from_record(record, base):
    value = record.get("audio")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = resolve_dance_asset(path)
    return str(path) if path.is_file() else None


def resolve_motion_audio(motion_path, *, recordings_dir=None):
    if not motion_path:
        return None
    motion = resolve_dance_asset(motion_path)
    sidecar = motion.with_suffix(".motion.json")
    if sidecar.is_file():
        try:
            found = _audio_from_record(json.loads(sidecar.read_text(encoding="utf-8")), sidecar.parent)
            if found:
                return found
        except (OSError, ValueError, TypeError):
            pass
    # Existing generated motions already have pipeline manifests. Match the
    # exact H5 path rather than stripping guessed dance/version suffixes.
    directory = Path(recordings_dir) if recordings_dir else ROOT / "media/recordings"
    for manifest in sorted(directory.glob("*.pipeline.json")):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if not any(record.get(key) and resolve_dance_asset(record[key]) == motion for key in ("h5", "raw_h5")):
                continue
            found = _audio_from_record(record, manifest.parent)
            if found:
                return found
        except (OSError, ValueError, TypeError):
            continue
    # Keep ordinary, unmodified dances working too.
    stem = motion.stem.removesuffix("_z_editted")
    for suffix in (".wav", ".ogg", ".mp3", ".flac", ".m4a"):
        candidate = motion.with_name(stem + suffix)
        if candidate.is_file():
            return str(candidate)
    # A dance folder also identifies the source of older corrected H5 files
    # that predate motion sidecars and pipeline manifests.
    if motion.parent.parent == Path(DANCE_DIR):
        for suffix in (".wav", ".ogg", ".mp3", ".flac", ".m4a"):
            candidate = motion.parent / (motion.parent.name + suffix)
            if candidate.is_file():
                return str(candidate)
    return None


def ffmpeg_audio_args(audio, *, start_seconds=0., playback_rate=1., volume=.25):
    """Audio input/filters following video input 0; align previews at 25% by default."""
    if (not math.isfinite(start_seconds) or start_seconds < 0 or
            not math.isfinite(playback_rate) or playback_rate <= 0 or
            not math.isfinite(volume) or volume < 0 or volume > 2):
        raise ValueError("Invalid audio start or playback rate")
    rate = playback_rate
    filters = []
    while rate < .5:
        filters.append("atempo=0.5")
        rate /= .5
    while rate > 2.:
        filters.append("atempo=2.0")
        rate /= 2.
    if abs(rate - 1.) > 1e-9:
        filters.append(f"atempo={rate}")
    filters.extend((f"volume={volume:.6g}", "apad"))
    return ["-ss", str(start_seconds), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
        "-c:a", "aac", "-b:a", "192k", "-af", ",".join(filters)]
