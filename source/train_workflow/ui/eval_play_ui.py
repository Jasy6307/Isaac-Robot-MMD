# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal Play/Stop panel for G1 policy eval with optional audio."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

WINDOW_TITLE = "G1 Policy Eval"
_AUTO_OPEN = True

_play_cb: Callable[[], None] | None = None
_stop_cb: Callable[[], None] | None = None
_status_provider: Callable[[], dict[str, Any]] | None = None
_audio_enabled_provider: Callable[[], bool] | None = None
_audio_enabled_setter: Callable[[bool], None] | None = None

_window_ref: list[Any] = []
_title_label_ref: Any | None = None
_policy_progress_label_ref: Any | None = None
_progress_bar_ref: Any | None = None
_progress_bar_model_ref: Any | None = None
_audio_enabled_model_ref: Any | None = None
_audio_checkbox_ref: Any | None = None
_btn_play_ref: Any | None = None
_btn_stop_ref: Any | None = None
_refresh_started = False
_suppress_audio_enabled_sync = False


def set_play_callback(cb: Callable[[], None] | None) -> None:
    global _play_cb
    _play_cb = cb


def set_stop_callback(cb: Callable[[], None] | None) -> None:
    global _stop_cb
    _stop_cb = cb


def set_status_provider(provider: Callable[[], dict[str, Any]] | None) -> None:
    global _status_provider
    _status_provider = provider


def set_audio_enabled_provider(provider: Callable[[], bool] | None) -> None:
    global _audio_enabled_provider
    _audio_enabled_provider = provider


def set_audio_enabled_setter(setter: Callable[[bool], None] | None) -> None:
    global _audio_enabled_setter
    _audio_enabled_setter = setter


def _make_policy_progress_bar_model(ui: Any) -> Any:
    class PolicyProgressBarModel(ui.AbstractValueModel):
        def __init__(self) -> None:
            super().__init__()
            self._value = 0.0

        def set_value(self, value: float) -> None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            value = max(0.0, min(1.0, value))
            if value != self._value:
                self._value = value
                self._value_changed()

        def get_value_as_float(self) -> float:
            return self._value

        def get_value_as_string(self) -> str:
            return f"{self._value * 100.0:.2f}%"

    return PolicyProgressBarModel()


def _build_window(ui: Any) -> None:
    global _title_label_ref, _policy_progress_label_ref
    global _progress_bar_ref, _progress_bar_model_ref, _audio_enabled_model_ref
    global _audio_checkbox_ref, _btn_play_ref, _btn_stop_ref

    with ui.VStack(spacing=8, height=0):
        _title_label_ref = ui.Label("Policy eval idle", word_wrap=True)
        with ui.HStack(spacing=8, height=28):
            _btn_play_ref = ui.Button("Play", width=72, height=24)
            _btn_stop_ref = ui.Button("Stop", width=72, height=24)
            ui.Spacer()
        with ui.HStack(spacing=8, height=24):
            ui.Label("Play audio", width=72, height=22)
            audio_enabled_model = ui.SimpleBoolModel(True)
            _audio_enabled_model_ref = audio_enabled_model
            _audio_checkbox_ref = ui.CheckBox(model=audio_enabled_model, width=24, height=22)
            ui.Spacer()
        _policy_progress_label_ref = ui.Label("Policy: step 0 / 0")
        progress_bar_model = _make_policy_progress_bar_model(ui)
        _progress_bar_model_ref = progress_bar_model
        _progress_bar_ref = ui.ProgressBar(model=progress_bar_model, height=8)
        ui.Spacer(height=4)

    def _on_play() -> None:
        if _play_cb is not None:
            _play_cb()

    def _on_stop() -> None:
        if _stop_cb is not None:
            _stop_cb()

    def _on_audio_enabled_changed(model: Any) -> None:
        global _suppress_audio_enabled_sync
        if _suppress_audio_enabled_sync:
            return
        if _audio_enabled_setter is not None:
            _audio_enabled_setter(bool(model.get_value_as_bool()))

    _btn_play_ref.set_clicked_fn(_on_play)
    _btn_stop_ref.set_clicked_fn(_on_stop)
    audio_enabled_model.add_value_changed_fn(_on_audio_enabled_changed)


def schedule_eval_play_ui_refresh() -> None:
    global _refresh_started
    if _refresh_started:
        return
    _refresh_started = True
    asyncio.ensure_future(_refresh_loop())


async def _refresh_loop() -> None:
    import omni.kit.app

    global _suppress_audio_enabled_sync

    while True:
        await omni.kit.app.get_app().next_update_async()
        if _status_provider is None:
            continue
        if _title_label_ref is None:
            continue
        try:
            st = _status_provider() or {}
        except Exception:
            st = {}

        playing = bool(st.get("playing"))
        dance_title = str(st.get("dance_title") or "Policy eval")
        policy_step = int(st.get("policy_step") or 0)
        policy_total = max(0, int(st.get("policy_total") or 0))
        has_audio = bool(st.get("has_audio"))

        _title_label_ref.text = f"{dance_title} — {'Playing' if playing else 'Idle'}"
        _policy_progress_label_ref.text = f"Policy: step {policy_step} / {policy_total}"

        if policy_total > 0:
            ratio = max(0.0, min(1.0, float(policy_step) / float(policy_total)))
        else:
            ratio = 0.0
        if _progress_bar_model_ref is not None:
            _progress_bar_model_ref.set_value(ratio)

        if _audio_checkbox_ref is not None:
            _audio_checkbox_ref.enabled = has_audio
        if _audio_enabled_model_ref is not None and _audio_enabled_provider is not None:
            desired = bool(_audio_enabled_provider()) if has_audio else False
            current = bool(_audio_enabled_model_ref.get_value_as_bool())
            if current != desired:
                _suppress_audio_enabled_sync = True
                try:
                    _audio_enabled_model_ref.set_value(desired)
                finally:
                    _suppress_audio_enabled_sync = False

        if _btn_play_ref is not None:
            _btn_play_ref.enabled = not playing
        if _btn_stop_ref is not None:
            _btn_stop_ref.enabled = playing


def create_eval_play_ui(*, auto_open: bool = _AUTO_OPEN) -> bool | None:
    """Create the eval Play/Stop window and register it under Window menu."""
    try:
        import omni.ui as ui
        import omni.kit.app
        from omni.kit.menu.utils import add_menu_items, MenuItemDescription
    except ImportError:
        print("[WARN] omni.ui unavailable; eval Play UI skipped (headless mode?)")
        return None

    def _create_window() -> Any:
        from source.train_workflow.ui.mmd_config_ui import _schedule_content_console_tab_dock

        window = ui.Window(
            WINDOW_TITLE,
            width=360,
            height=175,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _build_window(ui)
        window.visible = True
        _window_ref.clear()
        _window_ref.append(window)
        _schedule_content_console_tab_dock(window, WINDOW_TITLE)
        return window

    def _on_menu_click() -> None:
        if _window_ref:
            _window_ref[0].visible = True
        else:
            _create_window()

    add_menu_items(
        [MenuItemDescription(name=WINDOW_TITLE, onclick_fn=_on_menu_click)],
        "Window",
    )

    async def _auto_open() -> None:
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        _create_window()

    if auto_open:
        asyncio.ensure_future(_auto_open())

    schedule_eval_play_ui_refresh()
    print(f"[INFO] Eval Play UI: Window menu -> {WINDOW_TITLE} (docks to Content/Console)")
    return True
