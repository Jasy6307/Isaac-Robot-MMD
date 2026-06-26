# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture Isaac Sim viewport frames into an AVI file."""

from __future__ import annotations

import asyncio
import ctypes
import os
import re
import tempfile
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_CAPTURE_SPIN_UPDATES = 48
_CAPTURE_SPIN_UPDATES_POLICY = 4
_capture_fail_warned = False


class ViewportAviRecorder:
    """Write viewport RGB frames to an AVI via OpenCV."""

    def __init__(
        self,
        *,
        fps: float = 30.0,
        pre_render: Callable[[], None] | None = None,
        policy_safe_capture: bool = False,
    ) -> None:
        self._fps = max(1.0, float(fps))
        self._pre_render = pre_render
        self._policy_safe_capture = bool(policy_safe_capture)
        self._writer: Any = None
        self._output_path: str | None = None
        self._frame_count = 0
        self._capture_failures = 0
        self._last_rgb: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def output_path(self) -> str | None:
        return self._output_path

    def start(self, output_path: str) -> None:
        if self._writer is not None:
            raise RuntimeError("ViewportAviRecorder already active")
        out = os.path.abspath(str(output_path))
        if not out.lower().endswith(".avi"):
            out = f"{out}.avi"
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._output_path = out
        self._frame_count = 0
        self._capture_failures = 0
        self._last_rgb = None
        self._writer = None

    def capture_frame(self) -> bool:
        if self._output_path is None:
            return False
        rgb = _capture_viewport_rgb(
            pre_render=self._pre_render,
            policy_safe_capture=self._policy_safe_capture,
        )
        if rgb is None:
            self._capture_failures += 1
            _warn_capture_failure_once(self._capture_failures)
            return False
        return self._write_rgb(rgb, cache_last=True)

    def write_duplicate_frames(self, count: int) -> int:
        """Repeat the last captured RGB ``count`` times (fills skipped motion indices)."""
        n = max(0, int(count))
        if n <= 0 or self._last_rgb is None:
            return 0
        written = 0
        for _ in range(n):
            if self._write_rgb(self._last_rgb, cache_last=False):
                written += 1
        return written

    def stop(self) -> str | None:
        out = self._output_path
        writer = self._writer
        self._writer = None
        self._output_path = None
        count = self._frame_count
        self._frame_count = 0
        self._capture_failures = 0
        self._last_rgb = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if out and count > 0:
            return out
        if out and count == 0 and os.path.isfile(out):
            try:
                os.remove(out)
            except OSError:
                pass
        return None

    def _write_rgb(self, rgb: np.ndarray, *, cache_last: bool = True) -> bool:
        import cv2

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return False
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if h <= 0 or w <= 0:
            return False
        bgr = np.ascontiguousarray(rgb[..., :3][:, :, ::-1])
        if self._writer is None:
            # XVID tends to preserve fps in AVI headers better than MJPG on Windows.
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open AVI writer: {self._output_path}")
            self._writer = writer
        try:
            self._writer.write(bgr)
        except Exception:
            return False
        if cache_last:
            self._last_rgb = np.ascontiguousarray(rgb[..., :3].copy())
        self._frame_count += 1
        return True


def default_playback_avi_path(*, media_dir: str, motion_label: str) -> str:
    """Build ``media/recordings/<label>_<timestamp>.avi``."""
    recordings_dir = os.path.join(media_dir, "recordings")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\-.]+", "_", str(motion_label or "playback")).strip("_")
    if not safe:
        safe = "playback"
    return os.path.join(recordings_dir, f"{safe}_{stamp}.avi")


def _warn_capture_failure_once(failures: int) -> None:
    global _capture_fail_warned
    if failures != 1 or _capture_fail_warned:
        return
    _capture_fail_warned = True
    print(
        "[WARN] AVI viewport capture returned no image (async capture or viewport API). "
        "Trying alternate capture paths each frame."
    )


def _resolve_viewport() -> Any | None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except ImportError:
        return None
    viewport = get_active_viewport()
    if viewport is not None:
        return viewport
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        for name in ("Viewport", "Scene", "Isaac Sim"):
            viewport = get_viewport_from_window_name(name)
            if viewport is not None:
                return viewport
    except Exception:
        pass
    return None


def _capture_viewport_rgb(
    *,
    pre_render: Callable[[], None] | None = None,
    policy_safe_capture: bool = False,
) -> np.ndarray | None:
    viewport = _resolve_viewport()
    if viewport is None:
        return None
    rgb = _capture_viewport_rgb_buffer(
        viewport,
        pre_render=pre_render,
        policy_safe_capture=policy_safe_capture,
    )
    if rgb is not None:
        return rgb
    return _capture_viewport_rgb_file(
        viewport,
        pre_render=pre_render,
        policy_safe_capture=policy_safe_capture,
    )


def _capture_viewport_rgb_buffer(
    viewport: Any,
    *,
    pre_render: Callable[[], None] | None = None,
    policy_safe_capture: bool = False,
) -> np.ndarray | None:
    try:
        import omni.kit.app
        from omni.kit.viewport.utility import capture_viewport_to_buffer
    except ImportError:
        return None

    state: dict[str, Any] = {"rgb": None, "done": False, "error": None}

    def on_capture(*args: Any, **kwargs: Any) -> None:
        try:
            if len(args) >= 5:
                buffer, buffer_size, width, height = args[0], args[1], args[2], args[3]
            elif len(args) >= 4:
                buffer, width, height = args[0], args[1], args[2]
                buffer_size = int(width) * int(height) * 4
            else:
                return
            state["rgb"] = _pycapsule_rgba_to_rgb(
                buffer,
                int(buffer_size),
                int(width),
                int(height),
            )
        except Exception as exc:
            state["error"] = exc
        finally:
            state["done"] = True

    capture_helper = capture_viewport_to_buffer(viewport, on_capture)
    app = omni.kit.app.get_app()
    _spin_until_capture_done(
        app,
        state,
        capture_helper,
        pre_render=pre_render,
        policy_safe_capture=policy_safe_capture,
    )
    return state.get("rgb")


def _capture_viewport_rgb_file(
    viewport: Any,
    *,
    pre_render: Callable[[], None] | None = None,
    policy_safe_capture: bool = False,
) -> np.ndarray | None:
    try:
        import cv2
        import omni.kit.app
        from omni.kit.viewport.utility import capture_viewport_to_file
    except ImportError:
        return None

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        capture_helper = capture_viewport_to_file(viewport, tmp_path)
        app = omni.kit.app.get_app()
        state: dict[str, Any] = {"done": False}
        _spin_until_capture_done(
            app,
            state,
            capture_helper,
            file_path=tmp_path,
            pre_render=pre_render,
            policy_safe_capture=policy_safe_capture,
        )
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            return None
        bgr = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return np.ascontiguousarray(bgr[:, :, ::-1])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _spin_until_capture_done(
    app: Any,
    state: dict[str, Any],
    capture_helper: Any | None,
    *,
    file_path: str | None = None,
    pre_render: Callable[[], None] | None = None,
    policy_safe_capture: bool = False,
) -> None:
    max_spins = _CAPTURE_SPIN_UPDATES_POLICY if policy_safe_capture else _CAPTURE_SPIN_UPDATES
    wait_task: asyncio.Task | None = None
    if capture_helper is not None and hasattr(capture_helper, "wait_for_result"):

        async def _wait_for_helper() -> None:
            await capture_helper.wait_for_result(
                completion_frames=4 if policy_safe_capture else 30
            )

        try:
            wait_task = asyncio.ensure_future(_wait_for_helper())
        except Exception:
            wait_task = None

    if pre_render is not None:
        try:
            pre_render()
        except Exception:
            pass

    for _ in range(max_spins):
        app.update()
        if state.get("done"):
            return
        if file_path is not None and os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            state["done"] = True
            return
        if wait_task is not None and wait_task.done():
            if state.get("done"):
                return
            if file_path is not None and os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                state["done"] = True
                return
            break


def _pycapsule_rgba_to_rgb(
    buffer: Any,
    buffer_size: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    if buffer is None or buffer_size <= 0 or width <= 0 or height <= 0:
        return None
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
    if not ptr:
        return None
    expected = int(width) * int(height) * 4
    size = min(int(buffer_size), expected)
    raw = (ctypes.c_byte * size).from_address(ptr)
    arr = np.frombuffer(raw, dtype=np.uint8, count=size)
    if arr.size < expected:
        return None
    rgba = arr[:expected].reshape(int(height), int(width), 4)
    return np.ascontiguousarray(rgba[..., :3].copy())


def _buffer_to_rgb_numpy(
    buffer: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray | None:
    if buffer is None:
        return None
    if isinstance(buffer, np.ndarray):
        arr = buffer
    elif hasattr(buffer, "numpy"):
        try:
            arr = np.asarray(buffer.numpy())
        except Exception:
            arr = None
    elif isinstance(buffer, (bytes, bytearray, memoryview)):
        if width is None or height is None or width <= 0 or height <= 0:
            return None
        arr = np.frombuffer(buffer, dtype=np.uint8)
        n_px = int(width) * int(height)
        if arr.size < n_px * 3:
            return None
        arr = arr.reshape(int(height), int(width), -1)
    else:
        try:
            arr = np.asarray(buffer)
        except Exception:
            arr = None
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 1 and width is not None and height is not None:
        try:
            arr = arr.reshape(int(height), int(width), -1)
        except Exception:
            return None
    if arr.ndim != 3:
        return None
    rgb = np.ascontiguousarray(arr[..., :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb
