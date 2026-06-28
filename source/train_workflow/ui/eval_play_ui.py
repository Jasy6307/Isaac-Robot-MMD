# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal Play/Stop panel for G1 policy eval with optional audio."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from source.train_workflow.utils.media.audio_util import DEFAULT_VOLUME

WINDOW_TITLE = "G1 Policy Eval"
_AUTO_OPEN = True

_play_cb: Callable[[], None] | None = None
_stop_cb: Callable[[], None] | None = None
_status_provider: Callable[[], dict[str, Any]] | None = None
_audio_enabled_provider: Callable[[], bool] | None = None
_audio_enabled_setter: Callable[[bool], None] | None = None
_audio_volume_provider: Callable[[], float] | None = None
_audio_volume_setter: Callable[[float], None] | None = None
_record_avi_provider: Callable[[], bool] | None = None
_record_avi_setter: Callable[[bool], None] | None = None
_dance_entries_provider: Callable[[], list[tuple[str, str]]] | None = None
_dance_select_setter: Callable[[str], None] | None = None
_policy_entries_provider: Callable[[], list[tuple[str, str]]] | None = None
_policy_select_setter: Callable[[str], None] | None = None

_window_ref: list[Any] = []
_title_label_ref: Any | None = None
_policy_progress_label_ref: Any | None = None
_progress_bar_ref: Any | None = None
_progress_bar_model_ref: Any | None = None
_audio_enabled_model_ref: Any | None = None
_audio_checkbox_ref: Any | None = None
_audio_volume_model_ref: Any | None = None
_record_avi_model_ref: Any | None = None
_record_avi_checkbox_ref: Any | None = None
_btn_play_ref: Any | None = None
_btn_stop_ref: Any | None = None
_policy_path_label_ref: Any | None = None
_dance_combo_ref: Any | None = None
_policy_combo_ref: Any | None = None
_dance_combo_entries: list[tuple[str, str]] = []
_policy_combo_entries: list[tuple[str, str]] = []
_refresh_started = False
_suppress_audio_enabled_sync = False
_suppress_audio_volume_sync = False
_suppress_record_avi_sync = False
_suppress_dance_sync = False
_suppress_policy_sync = False


def _disabled_eval_btn_style(*names: str) -> dict[str, dict[str, int]]:
    style: dict[str, dict[str, int]] = {}
    for name in names:
        style[f"Button::{name}:disabled"] = {
            "background_color": 0xFF3A3A3A,
            "background_gradient_color": 0xFF3A3A3A,
            "border_color": 0xFF2C2C2C,
        }
        style[f"Button.Label::{name}:disabled"] = {"color": 0xFF8E8E8E}
    return style


_EVAL_BTN_STYLE = _disabled_eval_btn_style("eval_play", "eval_stop")


def _compact_error_for_ui(message: str, *, max_len: int = 100) -> str:
    """Compress multiline backend error into one short UI line."""
    text = " ".join(str(message or "").replace("\r", "\n").split())
    if not text:
        return "policy unavailable"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


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


def set_audio_volume_callbacks(
    provider: Callable[[], float] | None,
    setter: Callable[[float], None] | None,
) -> None:
    global _audio_volume_provider, _audio_volume_setter
    _audio_volume_provider = provider
    _audio_volume_setter = setter


def set_record_avi_callbacks(
    provider: Callable[[], bool] | None,
    setter: Callable[[bool], None] | None,
) -> None:
    global _record_avi_provider, _record_avi_setter
    _record_avi_provider = provider
    _record_avi_setter = setter


def set_dance_selector_callbacks(
    entries_provider: Callable[[], list[tuple[str, str]]] | None,
    setter: Callable[[str], None] | None,
) -> None:
    global _dance_entries_provider, _dance_select_setter
    _dance_entries_provider = entries_provider
    _dance_select_setter = setter


def set_policy_selector_callbacks(
    entries_provider: Callable[[], list[tuple[str, str]]] | None,
    setter: Callable[[str], None] | None,
) -> None:
    global _policy_entries_provider, _policy_select_setter
    _policy_entries_provider = entries_provider
    _policy_select_setter = setter


def _normalize_combo_entries(raw: object) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            key = str(item[0]).strip()
            label = str(item[1]).strip()
            if key and label:
                out.append((key, label))
        elif isinstance(item, str):
            val = str(item).strip()
            if val:
                out.append((val, val))
    return out


def _combo_selected_key(combo: Any, entries: list[tuple[str, str]]) -> str:
    if not entries:
        return ""
    if combo is None:
        return entries[0][0]
    try:
        idx = int(combo.model.get_item_value_model().as_int)
    except Exception:
        idx = 0
    idx = max(0, min(len(entries) - 1, idx))
    return entries[idx][0]


def _combo_set_selected_key(combo: Any, entries: list[tuple[str, str]], key: str) -> None:
    if combo is None or (not entries):
        return
    target = str(key or "").strip()
    if not target:
        return
    idx = next((i for i, (k, _lbl) in enumerate(entries) if k == target), None)
    if idx is None:
        return
    try:
        combo.model.get_item_value_model().set_value(int(idx))
    except Exception:
        pass


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
    global _audio_checkbox_ref, _btn_play_ref, _btn_stop_ref, _policy_path_label_ref
    global _dance_combo_ref, _policy_combo_ref, _dance_combo_entries, _policy_combo_entries
    global _record_avi_model_ref, _record_avi_checkbox_ref, _audio_volume_model_ref

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

    def _on_record_avi_changed(model: Any) -> None:
        global _suppress_record_avi_sync
        if _suppress_record_avi_sync:
            return
        if _record_avi_setter is not None:
            _record_avi_setter(bool(model.get_value_as_bool()))

    def _on_audio_volume_changed(model: Any) -> None:
        global _suppress_audio_volume_sync
        if _suppress_audio_volume_sync:
            return
        if _audio_volume_setter is not None:
            _audio_volume_setter(float(model.get_value_as_float()))

    def _on_dance_changed(_model: Any) -> None:
        global _suppress_dance_sync
        if _suppress_dance_sync:
            return
        if _dance_select_setter is None:
            return
        key = _combo_selected_key(dance_combo, _dance_combo_entries)
        if key:
            _dance_select_setter(key)

    def _on_policy_changed(_model: Any) -> None:
        global _suppress_policy_sync
        if _suppress_policy_sync:
            return
        if _policy_select_setter is None:
            return
        key = _combo_selected_key(policy_combo, _policy_combo_entries)
        if key:
            _policy_select_setter(key)

    title_label: Any | None = None
    policy_progress_label: Any | None = None
    progress_bar: Any | None = None
    progress_bar_model: Any | None = None
    audio_enabled_model: Any | None = None
    audio_checkbox: Any | None = None
    audio_volume_model: Any | None = None
    btn_play: Any | None = None
    btn_stop: Any | None = None
    policy_path_label: Any | None = None
    dance_combo: Any | None = None
    policy_combo: Any | None = None
    record_avi_model: Any | None = None
    record_avi_checkbox: Any | None = None

    _dance_combo_entries = (
        _normalize_combo_entries(_dance_entries_provider()) if _dance_entries_provider is not None else []
    )
    _policy_combo_entries = (
        _normalize_combo_entries(_policy_entries_provider()) if _policy_entries_provider is not None else []
    )
    dance_labels = [label for _key, label in _dance_combo_entries] or ["(none)"]
    policy_labels = [label for _key, label in _policy_combo_entries] or ["(none)"]

    with ui.VStack(spacing=8, height=0):
        title_label = ui.Label("Policy eval idle", word_wrap=True)
        with ui.HStack(spacing=8, height=28, style=_EVAL_BTN_STYLE):
            btn_play = ui.Button("Play", width=72, height=24, name="eval_play")
            btn_stop = ui.Button("Stop", width=72, height=24, name="eval_stop")
            if btn_play is not None:
                btn_play.set_clicked_fn(_on_play)
            if btn_stop is not None:
                btn_stop.set_clicked_fn(_on_stop)
            policy_path_label = ui.Label("(loading checkpoint)", word_wrap=True)
            ui.Spacer()
        with ui.HStack(spacing=8, height=26):
            ui.Label("Dance", width=72, height=22)
            dance_combo = ui.ComboBox(0, *dance_labels, width=260, height=24)
            ui.Spacer()
        with ui.HStack(spacing=8, height=26):
            ui.Label("Policy", width=72, height=22)
            policy_combo = ui.ComboBox(0, *policy_labels, width=260, height=24)
            ui.Spacer()
        with ui.HStack(spacing=8, height=24):
            ui.Label("Play audio", width=72, height=22)
            audio_enabled_model = ui.SimpleBoolModel(True)
            audio_checkbox = ui.CheckBox(model=audio_enabled_model, width=24, height=22)
            ui.Spacer()
        with ui.HStack(spacing=8, height=24):
            ui.Label("Audio volume", width=72, height=22)
            audio_volume_model = ui.SimpleFloatModel(float(DEFAULT_VOLUME))
            ui.FloatField(model=audio_volume_model, width=48, height=22)
            ui.Spacer(width=4)
            ui.FloatSlider(model=audio_volume_model, min=0.0, max=1.0, width=120, height=22)
            ui.Spacer()
        with ui.HStack(spacing=8, height=24):
            ui.Label("Record AVI", width=72, height=22)
            record_avi_model = ui.SimpleBoolModel(False)
            record_avi_checkbox = ui.CheckBox(model=record_avi_model, width=24, height=22)
            ui.Spacer()
        policy_progress_label = ui.Label("Policy: step 0 / 0")
        progress_bar_model = _make_policy_progress_bar_model(ui)
        progress_bar = ui.ProgressBar(model=progress_bar_model, height=8)
        ui.Spacer(height=4)

    if audio_enabled_model is not None:
        audio_enabled_model.add_value_changed_fn(_on_audio_enabled_changed)
    if audio_volume_model is not None:
        audio_volume_model.add_value_changed_fn(_on_audio_volume_changed)
    if record_avi_model is not None:
        record_avi_model.add_value_changed_fn(_on_record_avi_changed)
    if dance_combo is not None:
        dance_combo.model.get_item_value_model().add_value_changed_fn(_on_dance_changed)
    if policy_combo is not None:
        policy_combo.model.get_item_value_model().add_value_changed_fn(_on_policy_changed)

    _title_label_ref = title_label
    _policy_progress_label_ref = policy_progress_label
    _progress_bar_ref = progress_bar
    _progress_bar_model_ref = progress_bar_model
    _audio_enabled_model_ref = audio_enabled_model
    _audio_checkbox_ref = audio_checkbox
    _audio_volume_model_ref = audio_volume_model
    _btn_play_ref = btn_play
    _btn_stop_ref = btn_stop
    _policy_path_label_ref = policy_path_label
    _dance_combo_ref = dance_combo
    _policy_combo_ref = policy_combo
    _record_avi_model_ref = record_avi_model
    _record_avi_checkbox_ref = record_avi_checkbox


def schedule_eval_play_ui_refresh() -> None:
    global _refresh_started
    if _refresh_started:
        return
    _refresh_started = True
    asyncio.ensure_future(_refresh_loop())


async def _refresh_loop() -> None:
    import omni.kit.app

    global _suppress_audio_enabled_sync, _suppress_record_avi_sync, _suppress_audio_volume_sync
    global _suppress_dance_sync, _suppress_policy_sync

    while True:
        await omni.kit.app.get_app().next_update_async()
        if _status_provider is None:
            continue
        if _title_label_ref is None or _policy_progress_label_ref is None:
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
        record_avi_enabled = bool(st.get("record_avi_enabled"))
        policy_relpath = str(st.get("policy_relpath") or "").strip()
        dance_selected = str(st.get("dance_selected") or "").strip()
        policy_selected = str(st.get("policy_selected") or "").strip()
        play_enabled = bool(st.get("play_enabled", True))
        policy_error = str(st.get("policy_error") or "").strip()

        _title_label_ref.text = f"{dance_title} — {'Playing' if playing else 'Idle'}"
        _policy_progress_label_ref.text = f"Policy: step {policy_step} / {policy_total}"
        if _policy_path_label_ref is not None:
            if policy_error:
                _policy_path_label_ref.text = f"No runnable policy: {_compact_error_for_ui(policy_error)}"
            else:
                _policy_path_label_ref.text = policy_relpath if policy_relpath else "(unknown checkpoint)"
        if _dance_combo_ref is not None and dance_selected:
            _suppress_dance_sync = True
            try:
                _combo_set_selected_key(_dance_combo_ref, _dance_combo_entries, dance_selected)
            finally:
                _suppress_dance_sync = False
        if _policy_combo_ref is not None and policy_selected:
            _suppress_policy_sync = True
            try:
                _combo_set_selected_key(_policy_combo_ref, _policy_combo_entries, policy_selected)
            finally:
                _suppress_policy_sync = False

        if policy_total > 0:
            ratio = max(0.0, min(1.0, float(policy_step) / float(policy_total)))
        else:
            ratio = 0.0
        if _progress_bar_model_ref is not None:
            _progress_bar_model_ref.set_value(ratio)

        if _audio_checkbox_ref is not None:
            _audio_checkbox_ref.enabled = has_audio and (not record_avi_enabled)
        if _audio_enabled_model_ref is not None and _audio_enabled_provider is not None:
            desired = bool(_audio_enabled_provider()) if (has_audio and not record_avi_enabled) else False
            current = bool(_audio_enabled_model_ref.get_value_as_bool())
            if current != desired:
                _suppress_audio_enabled_sync = True
                try:
                    _audio_enabled_model_ref.set_value(desired)
                finally:
                    _suppress_audio_enabled_sync = False
        if _audio_volume_model_ref is not None and _audio_volume_provider is not None:
            desired_vol = float(_audio_volume_provider())
            current_vol = float(_audio_volume_model_ref.get_value_as_float())
            if abs(current_vol - desired_vol) > 1e-4:
                _suppress_audio_volume_sync = True
                try:
                    _audio_volume_model_ref.set_value(desired_vol)
                finally:
                    _suppress_audio_volume_sync = False

        if _record_avi_model_ref is not None and _record_avi_provider is not None:
            desired_rec = bool(_record_avi_provider())
            current_rec = bool(_record_avi_model_ref.get_value_as_bool())
            if current_rec != desired_rec:
                _suppress_record_avi_sync = True
                try:
                    _record_avi_model_ref.set_value(desired_rec)
                finally:
                    _suppress_record_avi_sync = False

        if _btn_play_ref is not None:
            _btn_play_ref.enabled = (not playing) and play_enabled
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
            height=320,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _build_window(ui)
        window.visible = True
        _window_ref.clear()
        _window_ref.append(window)
        _schedule_content_console_tab_dock(window, WINDOW_TITLE)
        schedule_eval_play_ui_refresh()
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

    print(f"[INFO] Eval Play UI: Window menu -> {WINDOW_TITLE} (docks to Content/Console)")
    return True
