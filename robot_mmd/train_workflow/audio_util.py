# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Windows 下基于 winsound 的简单 WAV 异步播放与停止（非 Windows 上为 no-op）。"""

import os

try:
    import winsound
except ImportError:  # pragma: no cover - 非 Windows 环境兼容
    winsound = None


def play_wav_async(filepath: str) -> None:
    """异步播放 WAV，不阻塞主线程。"""
    if winsound is None:
        print("[WARN] 当前平台不支持 winsound，跳过音频播放")
        return
    if not os.path.isfile(filepath):
        print(f"[WARN] 音频文件不存在: {filepath}")
        return
    try:
        winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
        print(f"[INFO] 开始播放音频: {filepath}")
    except Exception as exc:
        print(f"[WARN] 音频播放失败: {exc}")


def stop_wav() -> None:
    """停止当前由 winsound 发起的异步播放。"""
    if winsound is None:
        return
    try:
        winsound.PlaySound(None, 0)
        print("[INFO] 已停止音频播放")
    except Exception as exc:
        print(f"[WARN] 停止音频失败: {exc}")
