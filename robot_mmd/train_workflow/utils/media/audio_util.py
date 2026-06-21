# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""伴音：优先用 pygame.mixer.music（可 pause / 跳转与动作帧对齐）；否则回退 winsound。"""

from __future__ import annotations

import os
from typing import Any

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

_pygame_checked = False
_use_pygame: bool = False
_pygame: Any = None
_current_wav_path: str | None = None
_pygame_sync_hint_printed = False
# 当前这段实际在用哪种方式播：只有 "pygame" 时才响应 pause/seek
_playback_backend: str | None = None
# pygame.mixer.music volume: 0.0–1.0 (default below is quieter than mixer default 1.0)
DEFAULT_VOLUME: float = 0.2
_volume: float = DEFAULT_VOLUME


def get_volume() -> float:
    """Current playback volume (0.0–1.0). Only affects pygame backend."""
    return float(_volume)


def set_volume(value: float) -> None:
    """Set playback volume (0.0–1.0). Applies immediately when pygame is active."""
    global _volume
    _volume = max(0.0, min(1.0, float(value)))
    _apply_volume()


def _apply_volume() -> None:
    if not _ensure_pygame() or _pygame is None:
        return
    try:
        _pygame.mixer.music.set_volume(_volume)
    except Exception:
        pass


def has_pygame_audio() -> bool:
    """是否具备 pygame 混音路径（不一定当前曲在用 pygame）。"""
    return _ensure_pygame()


def current_clip_uses_pygame_transport() -> bool:
    """当前正在播的这条是否走 pygame（可暂停/按帧对齐）。"""
    return _playback_backend == "pygame"


def warn_if_no_pygame_sync() -> None:
    """若配置里有伴音但仍无法用 pygame，打印一次性安装提示。"""
    global _pygame_sync_hint_printed
    if _pygame_sync_hint_printed:
        return
    if _ensure_pygame():
        return
    _pygame_sync_hint_printed = True
    print("[INFO] WAV pause/seek with motion requires pygame: pip install pygame")


def _ensure_pygame() -> bool:
    global _pygame_checked, _use_pygame, _pygame
    if _pygame_checked:
        return _use_pygame
    _pygame_checked = True
    try:
        import pygame

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.music.set_volume(_volume)
        _pygame = pygame
        _use_pygame = True
    except Exception:
        _use_pygame = False
        _pygame = None
    return _use_pygame


def play_wav_async(filepath: str, start_sec: float = 0.0) -> None:
    """播放 WAV。若已启用 pygame，可从 start_sec（秒）起播，用于与帧同步。"""
    global _current_wav_path, _playback_backend
    path = os.path.abspath(filepath)
    if not os.path.isfile(path):
        print(f"[WARN] 音频文件不存在: {path}")
        return

    _current_wav_path = path
    _playback_backend = None

    if _ensure_pygame() and _pygame is not None:
        try:
            _pygame.mixer.music.load(path)
            start_sec = max(0.0, float(start_sec))
            try:
                _pygame.mixer.music.play(start=start_sec)
            except TypeError:
                _pygame.mixer.music.play()
                if start_sec > 0.0:
                    _pygame.mixer.music.set_pos(start_sec)
            _apply_volume()
            _playback_backend = "pygame"
            print(f"[INFO] 开始播放音频(pygame): {path} @ {start_sec:.3f}s vol={_volume:.2f}")
            return
        except Exception as exc:
            print(f"[WARN] pygame 播放失败，回退 winsound: {exc}")

    if winsound is None:
        print("[WARN] 当前平台无 winsound/pygame，跳过音频")
        _current_wav_path = None
        return
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        _playback_backend = "winsound"
        print(f"[INFO] 开始播放音频(winsound，无暂停/帧同步): {path}")
    except Exception as exc:
        print(f"[WARN] 音频播放失败: {exc}")
        _current_wav_path = None
        _playback_backend = None


def stop_wav() -> None:
    """停止伴音并清空当前曲目状态。"""
    global _current_wav_path, _playback_backend
    b = _playback_backend
    _current_wav_path = None
    _playback_backend = None
    if b == "pygame" and _pygame is not None:
        try:
            _pygame.mixer.music.stop()
        except Exception:
            pass
        return
    if b == "winsound" and winsound is not None:
        try:
            winsound.PlaySound(None, 0)
        except Exception:
            pass
        return
    if _use_pygame and _pygame is not None:
        try:
            _pygame.mixer.music.stop()
        except Exception:
            pass


def set_audio_paused(paused: bool) -> None:
    """暂停 / 继续 pygame 音乐（与动作暂停一致）。非 pygame 无操作。"""
    if _playback_backend != "pygame" or _pygame is None or _current_wav_path is None:
        return
    try:
        if paused:
            _pygame.mixer.music.pause()
        else:
            _pygame.mixer.music.unpause()
    except Exception:
        pass


def sync_audio_to_motion_frame(frame: int, motion_hz: float, paused: bool) -> None:
    """将伴音对齐到与逻辑帧对应的时间：t = frame / motion_hz（秒）。motion_hz = VMD_FPS * play_speed。"""
    if _playback_backend != "pygame" or _pygame is None or _current_wav_path is None:
        return
    if motion_hz <= 1e-9:
        return
    t = max(0.0, float(frame) / float(motion_hz))
    try:
        _pygame.mixer.music.stop()
        try:
            _pygame.mixer.music.play(start=t)
        except TypeError:
            _pygame.mixer.music.play()
            if t > 0.0:
                _pygame.mixer.music.set_pos(t)
        _apply_volume()
        if paused:
            _pygame.mixer.music.pause()
    except Exception:
        pass
