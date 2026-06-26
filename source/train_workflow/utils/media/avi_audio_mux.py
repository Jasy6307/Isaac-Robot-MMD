# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Mux a WAV audio track into an AVI file via ffmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


def _resolve_ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def mux_wav_into_avi(
    video_path: str,
    wav_path: str,
    *,
    replace_original: bool = True,
    output_path: str | None = None,
) -> str | None:
    """Mux ``wav_path`` into ``video_path`` and return the output AVI path.

    When ``replace_original`` is True (default), overwrites ``video_path`` on success.
    Requires ffmpeg on PATH or the ``imageio-ffmpeg`` package.
    """
    video_in = os.path.abspath(str(video_path))
    wav_in = os.path.abspath(str(wav_path))
    if not os.path.isfile(video_in):
        print(f"[WARN] AVI mux skipped: video not found: {video_in}")
        return None
    if not os.path.isfile(wav_in):
        print(f"[WARN] AVI mux skipped: WAV not found: {wav_in}")
        return video_in

    ffmpeg = _resolve_ffmpeg_exe()
    if ffmpeg is None:
        print(
            "[WARN] ffmpeg not found; AVI saved without audio. "
            "Install ffmpeg or: pip install imageio-ffmpeg"
        )
        return video_in

    if replace_original:
        out_final = video_in
        parent = os.path.dirname(video_in) or "."
        fd, tmp_out = tempfile.mkstemp(suffix=".avi", dir=parent)
        os.close(fd)
    else:
        if output_path is None:
            base, ext = os.path.splitext(video_in)
            out_final = f"{base}_with_audio{ext or '.avi'}"
        else:
            out_final = os.path.abspath(str(output_path))
        tmp_out = out_final
        parent = os.path.dirname(out_final)
        if parent:
            os.makedirs(parent, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_in,
        "-i",
        wav_in,
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        tmp_out,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            cmd_reencode = [
                ffmpeg,
                "-y",
                "-i",
                video_in,
                "-i",
                wav_in,
                "-c:v",
                "libxvid",
                "-c:a",
                "pcm_s16le",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                tmp_out,
            ]
            result = subprocess.run(cmd_reencode, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            print(f"[WARN] ffmpeg mux failed; AVI kept without audio. {err[:400]}")
            if replace_original and tmp_out != out_final and os.path.isfile(tmp_out):
                os.remove(tmp_out)
            return video_in
        if replace_original:
            os.replace(tmp_out, out_final)
        print(f"[INFO] AVI audio mux complete: {out_final}")
        return out_final
    except Exception as exc:
        print(f"[WARN] AVI audio mux error: {exc}")
        if replace_original and tmp_out != out_final and os.path.isfile(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        return video_in
