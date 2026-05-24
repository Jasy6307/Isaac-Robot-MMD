# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""将 16 位线性 PCM WAV 整体乘一个增益后另存, 目的是降低音量（不依赖 pycaw / 混音器）。

典型用法（在 conda 环境中）::

    python robot_mmd/train_workflow/reduce_wav_gain.py --gain 0.35
    python robot_mmd/train_workflow/reduce_wav_gain.py --gain 0.35 -o path/to/out.wav

默认处理 ``robot_mmd/media/you_are_important.wav``，输出到同目录 ``you_are_important_quiet.wav``。
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

_DEFAULT_IN = Path(__file__).resolve().parent.parent / "media" / "dance" / "deepbluetown.wav"


def _scale_pcm16_wav(in_path: Path, out_path: Path, gain: float) -> None:
    gain = float(gain)
    if gain < 0.0 or gain > 1.0:
        raise ValueError("gain 应在 0~1 之间（相对原电平）")
    with wave.open(str(in_path), "rb") as w:
        if w.getcomptype() != "NONE" or w.getsampwidth() != 2:
            raise ValueError("仅支持 16 位无压缩 PCM WAV")
        params = w.getparams()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) * gain
    x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setparams(params)
        w.writeframes(x.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description="降低 16-bit PCM WAV 电平并另存")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_DEFAULT_IN,
        help=f"输入 WAV（默认: {_DEFAULT_IN.name}）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 WAV（默认: 与输入同目录 you_are_important_quiet.wav）",
    )
    p.add_argument(
        "--gain",
        type=float,
        default=0.1,
        metavar="0~1",
        help="增益，例如 0.35 表示约为原响度的 35%%（默认 0.35）",
    )
    args = p.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output
    if out_path is None:
        out_path = in_path.parent / f"{in_path.stem}_quiet{in_path.suffix}"
    if not in_path.is_file():
        raise SystemExit(f"找不到文件: {in_path}")
    _scale_pcm16_wav(in_path, out_path, args.gain)
    print(f"[OK] 已写入: {out_path}")


if __name__ == "__main__":
    main()
