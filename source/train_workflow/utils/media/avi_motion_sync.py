# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Sync integer motion indices to fixed-fps viewport AVI captures."""

from __future__ import annotations

from source.train_workflow.utils.media.viewport_avi_recorder import ViewportAviRecorder


class AviMotionFrameSync:
    """Map motion frame indices to AVI frames with duplicate-fill for skipped indices."""

    def __init__(self, recorder: ViewportAviRecorder) -> None:
        self._recorder = recorder
        self._last: int | None = None

    def reset(self) -> None:
        self._last = None

    @property
    def last_motion_frame(self) -> int | None:
        return self._last

    def sync(self, motion_frame: int) -> None:
        mf = int(motion_frame)
        if self._last is None:
            if self._recorder.capture_frame():
                self._last = mf
            return
        if mf == self._last:
            return
        if mf > self._last:
            gap = mf - self._last - 1
            if gap > 0:
                self._recorder.write_duplicate_frames(gap)
            if self._recorder.capture_frame():
                self._last = mf
        elif self._recorder.capture_frame():
            self._last = mf

    def pad_to(self, max_frame: int) -> None:
        if self._last is None:
            return
        pad = int(max_frame) - int(self._last)
        if pad > 0:
            self._recorder.write_duplicate_frames(pad)
            self._last = int(max_frame)
