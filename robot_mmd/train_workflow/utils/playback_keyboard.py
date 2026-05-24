"""Dance hotkey listener with Shift modifier for HDF5 playback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import carb


class DanceKeyboardListener:
    """Subscribe to dance keys; Shift+<key> requests HDF5 when available."""

    def __init__(
        self,
        *,
        dance_keys: set[str],
        pose_cycle_key: str,
        on_dance_request: Callable[[str, bool], None],
    ) -> None:
        self._dance_keys = {str(k).upper()[:1] for k in dance_keys}
        self._pose_cycle_key = str(pose_cycle_key or "P").strip().upper()[:1]
        self._on_dance_request = on_dance_request
        self._input_iface = carb.input.acquire_input_interface()
        self._kb_dev: Any = None
        self._kb_sub: Any = None
        self._shift_keys = tuple(
            k
            for k in (
                getattr(carb.input.KeyboardInput, "LEFT_SHIFT", None),
                getattr(carb.input.KeyboardInput, "RIGHT_SHIFT", None),
            )
            if k is not None
        )

    def subscribe(self) -> bool:
        import omni

        app_window = omni.appwindow.get_default_app_window()
        self._kb_dev = app_window.get_keyboard() if app_window is not None else None
        if self._kb_dev is None:
            print("[WARN] 未获取到键盘设备，dance 快捷键不可用")
            return False
        self._kb_sub = self._input_iface.subscribe_to_keyboard_events(self._kb_dev, self._on_event)
        return True

    def unsubscribe(self) -> None:
        if self._kb_sub is not None and self._kb_dev is not None:
            self._input_iface.unsubscribe_to_keyboard_events(self._kb_dev, self._kb_sub)
        self._kb_sub = None
        self._kb_dev = None

    def _modifier_down(self, keys: tuple[Any, ...]) -> bool:
        if self._kb_dev is None:
            return False
        for key in keys:
            try:
                if float(self._input_iface.get_keyboard_value(self._kb_dev, key)) > 0.0:
                    return True
            except Exception:
                continue
        return False

    def _on_event(self, event, *args):  # type: ignore[no-untyped-def]
        key_name = str(getattr(getattr(event, "input", None), "name", "") or "").upper()
        if getattr(event, "type", None) != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if key_name not in self._dance_keys:
            return True
        prefer_hdf5 = self._modifier_down(self._shift_keys)
        if key_name == self._pose_cycle_key and not prefer_hdf5:
            return True
        self._on_dance_request(key_name, prefer_hdf5)
        return True
