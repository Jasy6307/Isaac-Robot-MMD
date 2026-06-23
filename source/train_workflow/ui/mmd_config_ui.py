"""
G1 MMD 播放与舞蹈选项窗口。

功能概览：
1) Window 菜单 ``G1 MMD config``：舞蹈文件、播放传输、Dance Option（PD / Z / IK 等）；
2) Root 与关节 RPY 映射见 ``G1 Joint RPY Mapping``（``jointRPY_maping_ui.create_joint_rpy_mapping_ui``）；
3) Retarget Tune（肩/腿基变换）见 ``retargeting_tune_ui``；
4) 实时关节角度显示；映射变更回调驱动仿真重算。
"""
import asyncio
from typing import Any, Callable

from source.train_workflow.ui import jointRPY_maping_ui as joint_rpy_mapping_ui
from source.train_workflow.ui.jointRPY_maping_ui import (
    _ELBOW_MMD_SUFFIX,
    _HINGE_DETAIL_ROW_JOINTS,
    _KNEE_JOINT_NAMES,
    _KNEE_MMD_SUFFIX,
    mmd_bone_to_romaji,
    wrap_long_hinge_text,
)
from source.train_workflow.utils.media.audio_util import DEFAULT_VOLUME

WINDOW_TITLE = "G1 MMD config"
_AUTO_OPEN = True

# 外部注入：用于在 UI 中显示“当前环境下的关节值（度制）”
# 返回值: dict[joint_name] = angle_deg；膝/肘可含 ``__knee_mmd`` / ``__elbow_mmd`` 分解说明字符串
_joint_value_provider: Callable[[], dict[str, Any]] | None = None

# 外部注入：当前播放片段（dance/pose）名称与帧；主循环每帧刷新
# 约定：返回值 dict，可含
#   ``playing`` (bool) / ``kind`` ("dance"|"pose"|"") / ``title`` (str 已排版短名) /
#   ``frame`` (int|None) / ``max_frame`` (int|None) —— 仅在 dance 时使用帧数
_playback_status_provider: Callable[[], dict[str, Any]] | None = None

# Transport: pause/resume/stop, seek to frame index (only while clip loaded in g1_vmd_0_replay)
_playback_toggle_cb: Callable[[], None] | None = None
_playback_seek_cb: Callable[[int], None] | None = None
_playback_stop_cb: Callable[[], None] | None = None
DanceUiEntry = tuple[str, str]  # (dance_key, combo_label)
_dance_entries_provider: Callable[[], list[DanceUiEntry]] | None = None
_dance_request_cb: Callable[[str, bool], None] | None = None
_dance_z_edited_status_provider: Callable[[str], bool] | None = None
_dance_z_edit_ui_status_provider: Callable[[str], str] | None = None
_dance_z_edit_request_cb: Callable[[str], None] | None = None
_dance_z_edit_delete_cb: Callable[[str], None] | None = None
_z_edit_busy_provider: Callable[[], bool] | None = None
_dance_record_h5_request_cb: Callable[[str], None] | None = None
_dance_h5_delete_cb: Callable[[str], None] | None = None
_h5_record_busy_provider: Callable[[], bool] | None = None
_dance_h5_exists_provider: Callable[[str], bool] | None = None
_dance_h5_deletable_provider: Callable[[str], bool] | None = None
_pd_drive_provider: Callable[[], bool] | None = None
_pd_drive_setter: Callable[[bool], None] | None = None
_z_offset_enable_provider: Callable[[], bool] | None = None
_z_offset_enable_setter: Callable[[bool], None] | None = None
_root_z_compress_provider: Callable[[], tuple[float, float]] | None = None
_root_z_compress_setter: Callable[[float, float], None] | None = None
_foot_ground_comp_provider: Callable[[], bool] | None = None
_foot_ground_comp_setter: Callable[[bool], None] | None = None
_root_quat_rpy_provider: Callable[
    [], tuple[tuple[float, float, float], tuple[int, int, int]]
] | None = None
_root_quat_rpy_setter: Callable[
    [tuple[float, float, float], tuple[int, int, int]], None
] | None = None
_root_rot_bone_name_provider: Callable[[], str] | None = None
_foot_ik_provider: Callable[[], dict[str, Any]] | None = None
_foot_ik_setter: Callable[[dict[str, Any]], None] | None = None
_audio_volume_provider: Callable[[], float] | None = None
_audio_volume_setter: Callable[[float], None] | None = None

# True while refresh assigns scrub IntField from sim; blocks value_changed -> seek storm
_scrub_sync_suppress_seek: bool = False
_root_quat_sync_suppress_set: bool = False
_pd_drive_sync_suppress_set: bool = False
_z_offset_sync_suppress_set: bool = False
_root_z_compress_sync_suppress_set: bool = False
_foot_ik_sync_suppress_set: bool = False
_foot_ground_comp_sync_suppress_set: bool = False
_audio_volume_sync_suppress_set: bool = False

# 映射表被用户修改后通知主循环（例如在非播放状态下按新映射重算当前姿势）
_mapping_changed_cb: Callable[[], None] | None = None


def set_joint_value_provider(provider: Callable[[], dict[str, Any]] | None) -> None:
    """设置关节值提供器，用于 UI 实时显示当前角度（deg）。"""
    global _joint_value_provider
    _joint_value_provider = provider


def set_playback_status_provider(provider: Callable[[], dict[str, Any]] | None) -> None:
    """设置播放状态提供器：片段名、dance 时当前帧/总帧。"""
    global _playback_status_provider
    _playback_status_provider = provider


def set_playback_transport_callbacks(
    toggle_pause: Callable[[], None] | None,
    seek_frame: Callable[[int], None] | None,
    stop_playback: Callable[[], None] | None = None,
) -> None:
    """Pause / resume / stop and seek-to-frame (clip-relative index)."""
    global _playback_toggle_cb, _playback_seek_cb, _playback_stop_cb
    _playback_toggle_cb = toggle_pause
    _playback_seek_cb = seek_frame
    _playback_stop_cb = stop_playback


def set_dance_play_callbacks(
    entries_provider: Callable[[], list[DanceUiEntry]] | None,
    on_dance_request: Callable[[str, bool], None] | None,
) -> None:
    """Set dance list provider and play request callback for UI controls.

    ``entries_provider`` returns ``(dance_key, display_label)`` pairs; combo shows labels,
    play callback receives the dance key.
    """
    global _dance_entries_provider, _dance_request_cb
    _dance_entries_provider = entries_provider
    _dance_request_cb = on_dance_request


def set_dance_z_edit_callbacks(
    z_editted_status: Callable[[str], bool] | None,
    on_generate: Callable[[str], None] | None,
    busy_provider: Callable[[], bool] | None = None,
    ui_status_provider: Callable[[str], str] | None = None,
    on_delete_request: Callable[[str], None] | None = None,
) -> None:
    """Z_editted sibling status + generate request for the dance file combo row."""
    global _dance_z_edited_status_provider, _dance_z_edit_ui_status_provider
    global _dance_z_edit_request_cb, _dance_z_edit_delete_cb, _z_edit_busy_provider
    _dance_z_edited_status_provider = z_editted_status
    _dance_z_edit_ui_status_provider = ui_status_provider
    _dance_z_edit_request_cb = on_generate
    _dance_z_edit_delete_cb = on_delete_request
    _z_edit_busy_provider = busy_provider


def set_dance_record_h5_callbacks(
    on_record_request: Callable[[str], None] | None,
    busy_provider: Callable[[], bool] | None = None,
    h5_exists_provider: Callable[[str], bool] | None = None,
    on_delete_request: Callable[[str], None] | None = None,
    h5_deletable_provider: Callable[[str], bool] | None = None,
) -> None:
    """Record HDF5 from Isaac CSV playback for the selected dance combo entry."""
    global _dance_record_h5_request_cb, _h5_record_busy_provider, _dance_h5_exists_provider
    global _dance_h5_delete_cb, _dance_h5_deletable_provider
    _dance_record_h5_request_cb = on_record_request
    _h5_record_busy_provider = busy_provider
    _dance_h5_exists_provider = h5_exists_provider
    _dance_h5_delete_cb = on_delete_request
    _dance_h5_deletable_provider = h5_deletable_provider


def set_pd_drive_callbacks(
    provider: Callable[[], bool] | None,
    setter: Callable[[bool], None] | None,
) -> None:
    """Set PD drive mode callbacks for the mapping UI checkbox."""
    global _pd_drive_provider, _pd_drive_setter
    _pd_drive_provider = provider
    _pd_drive_setter = setter


def set_z_offset_enable_callbacks(
    provider: Callable[[], bool] | None,
    setter: Callable[[bool], None] | None,
) -> None:
    """Set Z root-offset motion variant callbacks for the mapping UI checkbox."""
    global _z_offset_enable_provider, _z_offset_enable_setter
    _z_offset_enable_provider = provider
    _z_offset_enable_setter = setter


def set_root_z_compress_callbacks(
    provider: Callable[[], tuple[float, float]] | None,
    setter: Callable[[float, float], None] | None,
) -> None:
    """Set root-Z attenuation callbacks: (baseline_offset_m, outlier_scale[0..1])."""
    global _root_z_compress_provider, _root_z_compress_setter
    _root_z_compress_provider = provider
    _root_z_compress_setter = setter


def set_foot_ground_comp_callbacks(
    provider: Callable[[], bool] | None,
    setter: Callable[[bool], None] | None,
) -> None:
    """Runtime ankle pitch/roll ground fix (Mapping UI checkbox)."""
    global _foot_ground_comp_provider, _foot_ground_comp_setter
    _foot_ground_comp_provider = provider
    _foot_ground_comp_setter = setter


def set_root_quat_rpy_callbacks(
    provider: Callable[[], tuple[tuple[float, float, float], tuple[int, int, int]]] | None,
    setter: Callable[[tuple[float, float, float], tuple[int, int, int]], None] | None,
) -> None:
    """Root R/P/Y: scale + axis_idx (0=csv roll, 1=pitch, 2=yaw) per output row."""
    global _root_quat_rpy_provider, _root_quat_rpy_setter
    _root_quat_rpy_provider = provider
    _root_quat_rpy_setter = setter


def push_root_quat_rpy_ui(
    scale: tuple[float, float, float],
    axis_idx: tuple[int, int, int],
    *,
    notify: bool = True,
) -> None:
    """Apply Root R/P/Y scale and axis index from the joint RPY mapping window."""
    global _root_quat_sync_suppress_set
    if _root_quat_sync_suppress_set:
        return
    if _root_quat_rpy_setter is None:
        return
    try:
        _root_quat_rpy_setter(
            (float(scale[0]), float(scale[1]), float(scale[2])),
            (
                max(0, min(2, int(axis_idx[0]))),
                max(0, min(2, int(axis_idx[1]))),
                max(0, min(2, int(axis_idx[2]))),
            ),
        )
        if notify:
            _notify_mapping_changed()
    except Exception:
        pass


def set_root_quat_scale_callbacks(
    provider: Callable[[], tuple[float, float, float]] | None,
    setter: Callable[[tuple[float, float, float]], None] | None,
) -> None:
    """Legacy: scale only; axis_idx fixed to (0, 1, 2). Prefer ``set_root_quat_rpy_callbacks``."""
    if provider is None or setter is None:
        set_root_quat_rpy_callbacks(None, None)
        return

    def _prov() -> tuple[tuple[float, float, float], tuple[int, int, int]]:
        s = provider()
        return s, (0, 1, 2)

    def _set(scale: tuple[float, float, float], _axis: tuple[int, int, int]) -> None:
        setter(scale)

    set_root_quat_rpy_callbacks(_prov, _set)


def set_root_rot_bone_name_provider(provider: Callable[[], str] | None) -> None:
    """Active CSV bone name used for root rotation (display only)."""
    global _root_rot_bone_name_provider
    _root_rot_bone_name_provider = provider


def set_foot_ik_callbacks(
    provider: Callable[[], dict[str, Any]] | None,
    setter: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Set Foot IK config callbacks for mapping UI tuning controls."""
    global _foot_ik_provider, _foot_ik_setter
    _foot_ik_provider = provider
    _foot_ik_setter = setter


def set_audio_volume_callbacks(
    provider: Callable[[], float] | None,
    setter: Callable[[float], None] | None,
) -> None:
    """Set WAV playback volume callbacks for the mapping UI slider (0.0–1.0)."""
    global _audio_volume_provider, _audio_volume_setter
    _audio_volume_provider = provider
    _audio_volume_setter = setter


def set_mapping_changed_callback(cb: Callable[[], None] | None) -> None:
    """注册映射变更回调（欧拉轴/缩放/重置时调用），供仿真循环即时重算姿势。"""
    global _mapping_changed_cb
    _mapping_changed_cb = cb


def _notify_mapping_changed() -> None:
    if _mapping_changed_cb is not None:
        try:
            _mapping_changed_cb()
        except Exception:
            pass


# 腰两骨共轭状态标签在 jointRPY_maping_ui 模块内维护
# 由 create_mmd_config_ui / create_joint_rpy_mapping_ui / create_retarget_tune_ui 写入；刷新循环读取
_playback_title_ref: Any | None = None
_playback_transport_ref: dict[str, Any] | None = None
_retarget_tune_refs: dict[str, Any] | None = None
_mapping_ui_refresh_started: bool = False

# Dance Option two-column layout (equal left/right column width)
_DANCE_OPT_ROW_H = 28
_DANCE_OPT_COL_W = 350
_DANCE_OPT_COL_GAP = 5
_DANCE_OPT_LABEL_W = 120
_DANCE_OPT_FIELD_W = 50
_DANCE_OPT_SLIDER_W = 100


def _dance_option_checkbox_pair(
    ui: Any,
    left_label: str,
    left_model: Any,
    right_label: str,
    right_model: Any,
) -> None:
    with ui.HStack(height=_DANCE_OPT_ROW_H, spacing=_DANCE_OPT_COL_GAP):
        with ui.HStack(width=_DANCE_OPT_COL_W):
            ui.Label(left_label, width=_DANCE_OPT_LABEL_W)
            ui.CheckBox(model=left_model, width=24, height=22)
            ui.Spacer()
        with ui.HStack(width=_DANCE_OPT_COL_W):
            ui.Label(right_label, width=_DANCE_OPT_LABEL_W)
            ui.CheckBox(model=right_model, width=24, height=22)
            ui.Spacer()


def _dance_option_float_slider_pair(
    ui: Any,
    left_label: str,
    left_model: Any,
    left_slider_min: float,
    left_slider_max: float,
    right_label: str,
    right_model: Any,
    right_slider_min: float,
    right_slider_max: float,
) -> None:
    with ui.HStack(height=_DANCE_OPT_ROW_H, spacing=_DANCE_OPT_COL_GAP):
        with ui.HStack(width=_DANCE_OPT_COL_W):
            ui.Label(left_label, width=_DANCE_OPT_LABEL_W)
            ui.FloatField(model=left_model, width=_DANCE_OPT_FIELD_W)
            ui.Spacer(width=4)
            ui.FloatSlider(
                model=left_model,
                min=left_slider_min,
                max=left_slider_max,
                width=_DANCE_OPT_SLIDER_W,
            )
            ui.Spacer()
        with ui.HStack(width=_DANCE_OPT_COL_W):
            ui.Label(right_label, width=_DANCE_OPT_LABEL_W)
            ui.FloatField(model=right_model, width=_DANCE_OPT_FIELD_W)
            ui.Spacer(width=4)
            ui.FloatSlider(
                model=right_model,
                min=right_slider_min,
                max=right_slider_max,
                width=_DANCE_OPT_SLIDER_W,
            )
            ui.Spacer()


# Dance combo H5 / Z_editted status label colors (ARGB)
_H5_STATUS_COLOR_OK = 0xFF66FF66
_H5_STATUS_COLOR_MISSING = 0xFFFF6666
_Z_EDIT_STATUS_COLOR_OK = 0xFF66FF66
_Z_EDIT_STATUS_COLOR_MISSING = 0xFFFF6666
_Z_EDIT_STATUS_COLOR_IK = 0xFF88CCFF

# Disabled button look for dance-file action rows (omni.ui :disabled state)
def _disabled_dance_btn_style(*names: str) -> dict[str, dict[str, int]]:
    style: dict[str, dict[str, int]] = {}
    for name in names:
        style[f"Button::{name}:disabled"] = {
            "background_color": 0xFF3A3A3A,
            "background_gradient_color": 0xFF3A3A3A,
            "border_color": 0xFF2C2C2C,
        }
        style[f"Button.Label::{name}:disabled"] = {"color": 0xFF8E8E8E}
    return style


_DANCE_FILE_ROW_BTN_STYLE = _disabled_dance_btn_style(
    "gen_z_edited",
    "delete_z_edit",
    "delete_h5",
)


def _combo_selected_dance_key(
    combo: Any,
    entries_provider: Callable[[], list[DanceUiEntry]] | None,
) -> str | None:
    if combo is None or entries_provider is None:
        return None
    try:
        entries: list[DanceUiEntry] = []
        for item in entries_provider() or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                key = str(item[0]).strip()
                if key:
                    entries.append((key, str(item[1])))
        if not entries:
            return None
        idx = int(combo.model.get_item_value_model().as_int)
        idx = max(0, min(len(entries) - 1, idx))
        return entries[idx][0]
    except Exception:
        return None


def _update_h5_status_label(label: Any, selected_key: str | None) -> None:
    has_h5 = False
    if selected_key and _dance_h5_exists_provider is not None:
        try:
            has_h5 = bool(_dance_h5_exists_provider(selected_key))
        except Exception:
            has_h5 = False
    if has_h5:
        label.text = "H5 file available"
        label.style = {"color": _H5_STATUS_COLOR_OK}
    else:
        label.text = "No H5 file"
        label.style = {"color": _H5_STATUS_COLOR_MISSING}


def _resolve_z_edit_ui_status(selected_key: str | None) -> str:
    if selected_key and _dance_z_edit_ui_status_provider is not None:
        try:
            status = str(_dance_z_edit_ui_status_provider(selected_key) or "").strip().lower()
            if status in ("ik_control", "available", "missing"):
                return status
        except Exception:
            pass
    if selected_key and _dance_z_edited_status_provider is not None:
        try:
            if bool(_dance_z_edited_status_provider(selected_key)):
                return "available"
        except Exception:
            pass
    return "missing"


def _update_z_edit_status_label(label: Any, selected_key: str | None) -> None:
    status = _resolve_z_edit_ui_status(selected_key)
    if status == "ik_control":
        label.text = "IK control, don't need z edit"
        label.style = {"color": _Z_EDIT_STATUS_COLOR_IK}
    elif status == "available":
        label.text = "Z_editted file available"
        label.style = {"color": _Z_EDIT_STATUS_COLOR_OK}
    else:
        label.text = "No Z_editted file"
        label.style = {"color": _Z_EDIT_STATUS_COLOR_MISSING}


def _can_delete_h5(selected_key: str | None) -> bool:
    if not selected_key or _dance_h5_deletable_provider is None:
        return False
    try:
        return bool(_dance_h5_deletable_provider(selected_key))
    except Exception:
        return False


def _build_mapping_window(ui):
    """Playback controls and dance/IK options (Root + joint RPY mapping is a separate window)."""
    # ========== 整体布局：垂直堆叠 ==========
    with ui.VStack(spacing=4):
        scrub_model = ui.SimpleIntModel(0)

        def _on_pause_click():
            if _playback_toggle_cb is not None:
                _playback_toggle_cb()

        def _on_resume_click():
            if _playback_toggle_cb is not None:
                _playback_toggle_cb()

        def _on_stop_click():
            if _playback_stop_cb is not None:
                _playback_stop_cb()

        def _on_prev_frame_click():
            try:
                cur = int(scrub_model.get_value_as_int())
            except Exception:
                cur = 0
            scrub_model.set_value(cur - 1)

        def _on_next_frame_click():
            try:
                cur = int(scrub_model.get_value_as_int())
            except Exception:
                cur = 0
            scrub_model.set_value(cur + 1)

        def _on_scrub_changed(m):
            if _scrub_sync_suppress_seek:
                return
            if _playback_seek_cb is not None:
                try:
                    _playback_seek_cb(int(m.get_value_as_int()))
                except Exception:
                    pass

        pd_drive_model = ui.SimpleBoolModel(False)
        z_offset_enable_model = ui.SimpleBoolModel(False)
        root_z_baseline_offset_model = ui.SimpleFloatModel(0.76)
        root_z_outlier_scale_model = ui.SimpleFloatModel(0.6)
        foot_ground_comp_model = ui.SimpleBoolModel(True)
        audio_volume_model = ui.SimpleFloatModel(float(DEFAULT_VOLUME))
        foot_ik_enable_model = ui.SimpleBoolModel(False)
        foot_ik_reach_model = ui.SimpleFloatModel(1.0)
        foot_ik_leg_scale_model = ui.SimpleFloatModel(0.75)
        foot_ik_ankle_offset_x_model = ui.SimpleFloatModel(0.0)
        foot_ik_ankle_offset_y_model = ui.SimpleFloatModel(0.0)
        foot_ik_ankle_offset_z_model = ui.SimpleFloatModel(0.02)
        foot_ik_debug_every_model = ui.SimpleIntModel(0)
        foot_ik_reg_weight_model = ui.SimpleFloatModel(0.15)

        def _dance_entries() -> list[DanceUiEntry]:
            if _dance_entries_provider is None:
                return []
            try:
                out: list[DanceUiEntry] = []
                for item in _dance_entries_provider() or []:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        key = str(item[0]).strip()
                        label = str(item[1]).strip()
                        if key and label:
                            out.append((key, label))
                    elif isinstance(item, str) and str(item).strip():
                        k = str(item).strip()
                        out.append((k, k))
                return out
            except Exception:
                return []

        dance_entries = _dance_entries()
        dance_combo_labels = [lbl for _k, lbl in dance_entries] if dance_entries else ["(none)"]
        dance_combo = None

        def _selected_dance_key() -> str | None:
            entries = _dance_entries()
            if not entries:
                return None
            if dance_combo is None:
                return entries[0][0]
            try:
                idx = int(dance_combo.model.get_item_value_model().as_int)
            except Exception:
                idx = 0
            idx = max(0, min(len(entries) - 1, idx))
            return entries[idx][0]

        def _request_selected_dance(prefer_hdf5: bool) -> None:
            if _dance_request_cb is None:
                return
            selected_key = _selected_dance_key()
            if not selected_key:
                return
            try:
                _dance_request_cb(selected_key, bool(prefer_hdf5))
            except Exception:
                pass

        def _on_gen_z_editted_click() -> None:
            if _dance_z_edit_request_cb is None:
                return
            selected_key = _selected_dance_key()
            if not selected_key:
                return
            try:
                _dance_z_edit_request_cb(selected_key)
            except Exception:
                pass

        def _on_record_h5_click() -> None:
            if _dance_record_h5_request_cb is None:
                return
            selected_key = _selected_dance_key()
            if not selected_key:
                return
            try:
                _dance_record_h5_request_cb(selected_key)
            except Exception:
                pass

        def _on_delete_z_edit_click() -> None:
            if _dance_z_edit_delete_cb is None:
                return
            selected_key = _selected_dance_key()
            if not selected_key:
                return
            try:
                _dance_z_edit_delete_cb(selected_key)
            except Exception:
                pass

        def _on_delete_h5_click() -> None:
            if _dance_h5_delete_cb is None:
                return
            selected_key = _selected_dance_key()
            if not selected_key:
                return
            try:
                _dance_h5_delete_cb(selected_key)
            except Exception:
                pass

        btn_gen_z_editted = None
        btn_delete_z_edit = None
        z_edit_status_label = None
        btn_record_h5 = None
        btn_delete_h5 = None
        h5_status_label = None

        def _on_pd_drive_changed(m: Any) -> None:
            if _pd_drive_sync_suppress_set:
                return
            if _pd_drive_setter is None:
                return
            try:
                _pd_drive_setter(bool(m.get_value_as_bool()))
            except Exception:
                pass

        def _on_z_offset_enable_changed(m: Any) -> None:
            if _z_offset_sync_suppress_set:
                return
            if _z_offset_enable_setter is None:
                return
            try:
                _z_offset_enable_setter(bool(m.get_value_as_bool()))
            except Exception:
                pass

        def _push_root_z_compress() -> None:
            if _root_z_compress_sync_suppress_set:
                return
            if _root_z_compress_setter is None:
                return
            try:
                baseline_off = float(root_z_baseline_offset_model.get_value_as_float())
                outlier_scale = float(root_z_outlier_scale_model.get_value_as_float())
                outlier_scale = max(0.0, min(1.0, outlier_scale))
                _root_z_compress_setter(baseline_off, outlier_scale)
                _notify_mapping_changed()
            except Exception:
                pass

        def _on_foot_ground_comp_changed(m: Any) -> None:
            if _foot_ground_comp_sync_suppress_set:
                return
            if _foot_ground_comp_setter is None:
                return
            try:
                _foot_ground_comp_setter(bool(m.get_value_as_bool()))
            except Exception:
                pass

        def _on_audio_volume_changed(m: Any) -> None:
            if _audio_volume_sync_suppress_set:
                return
            if _audio_volume_setter is None:
                return
            try:
                _audio_volume_setter(float(m.get_value_as_float()))
            except Exception:
                pass

        def _push_foot_ik() -> None:
            if _foot_ik_sync_suppress_set:
                return
            if _foot_ik_setter is None:
                return
            try:
                payload = {
                    "enable": bool(foot_ik_enable_model.get_value_as_bool()),
                    "reach": float(foot_ik_reach_model.get_value_as_float()),
                    "leg_scale": float(foot_ik_leg_scale_model.get_value_as_float()),
                    "debug_every": max(0, int(foot_ik_debug_every_model.get_value_as_int())),
                    "ik_reg_weight": float(foot_ik_reg_weight_model.get_value_as_float()),
                    "ankle_offset": (
                        float(foot_ik_ankle_offset_x_model.get_value_as_float()),
                        float(foot_ik_ankle_offset_y_model.get_value_as_float()),
                        float(foot_ik_ankle_offset_z_model.get_value_as_float()),
                    ),
                }
                _foot_ik_setter(payload)
                _notify_mapping_changed()
            except Exception:
                pass

        pd_drive_model.add_value_changed_fn(_on_pd_drive_changed)
        z_offset_enable_model.add_value_changed_fn(_on_z_offset_enable_changed)
        root_z_baseline_offset_model.add_value_changed_fn(lambda _m: _push_root_z_compress())
        root_z_outlier_scale_model.add_value_changed_fn(lambda _m: _push_root_z_compress())
        foot_ground_comp_model.add_value_changed_fn(_on_foot_ground_comp_changed)
        audio_volume_model.add_value_changed_fn(_on_audio_volume_changed)
        foot_ik_enable_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_reach_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_leg_scale_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_x_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_y_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_z_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_debug_every_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_reg_weight_model.add_value_changed_fn(lambda _m: _push_foot_ik())

        with ui.VStack(spacing=4):
            with ui.HStack(height=28):
                ui.Label("Dance File", width=74, height=22)
                ui.Spacer(width=6)
                dance_combo = ui.ComboBox(0, *dance_combo_labels, width=200, height=24)
                ui.Button(
                    "Play VMD(CSV)",
                    width=100,
                    height=24,
                    clicked_fn=lambda: _request_selected_dance(False),
                )
                ui.Button("Play H5", width=62, height=24, clicked_fn=lambda: _request_selected_dance(True))
            with ui.HStack(height=28, style=_DANCE_FILE_ROW_BTN_STYLE):
                btn_gen_z_editted = ui.Button(
                    "Gen Z_edited",
                    width=96,
                    height=24,
                    name="gen_z_edited",
                    clicked_fn=_on_gen_z_editted_click,
                )
                btn_delete_z_edit = ui.Button(
                    "Delete Z_edit",
                    width=100,
                    height=24,
                    name="delete_z_edit",
                    clicked_fn=_on_delete_z_edit_click,
                )
                z_edit_status_label = ui.Label(
                    "No Z_editted file",
                    height=22,
                    style={"color": _Z_EDIT_STATUS_COLOR_MISSING},
                )
                ui.Spacer()
            with ui.HStack(height=28, style=_DANCE_FILE_ROW_BTN_STYLE):
                btn_record_h5 = ui.Button("Record H5", width=82, height=24, clicked_fn=_on_record_h5_click)
                btn_delete_h5 = ui.Button(
                    "Delete H5",
                    width=82,
                    height=24,
                    name="delete_h5",
                    clicked_fn=_on_delete_h5_click,
                )
                h5_status_label = ui.Label(
                    "No H5 file",
                    height=22,
                    style={"color": _H5_STATUS_COLOR_MISSING},
                )
                ui.Spacer()
            with ui.HStack(height=28):
                playback_title_label = ui.Label("Playback: (idle)", width=188, height=22)
                ui.IntField(model=scrub_model, width=64, height=22)
                max_frame_label = ui.Label("/ -", width=40, height=22)
                ui.Spacer(width=4)
                btn_prev = ui.Button("Prev", width=42, height=24, clicked_fn=_on_prev_frame_click)
                btn_next = ui.Button("Next", width=42, height=24, clicked_fn=_on_next_frame_click)
                btn_pause = ui.Button("Pause", width=56, height=24, clicked_fn=_on_pause_click)
                btn_resume = ui.Button("Resume", width=58, height=24, clicked_fn=_on_resume_click)
                btn_stop = ui.Button("Stop", width=48, height=24, clicked_fn=_on_stop_click)
            with ui.HStack(height=28):
                ui.Label("Audio volume", width=88, height=22)
                ui.FloatField(model=audio_volume_model, width=48, height=22)
                ui.Spacer(width=4)
                ui.FloatSlider(model=audio_volume_model, min=0.0, max=1.0, width=112, height=22)
                ui.Spacer()
        btn_prev.visible = False
        btn_next.visible = False
        btn_pause.visible = False
        btn_resume.visible = False
        btn_stop.visible = False
        scrub_model.add_value_changed_fn(_on_scrub_changed)

        # ========== 可滚动区域：Dance / IK 选项 ==========
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2):
                ui.Label(
                    "--- Dance Option ---",
                    height=22,
                    style={"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00},
                )
                _dance_option_checkbox_pair(
                    ui,
                    "PD Drive",
                    pd_drive_model,
                    "Z_offset_enable",
                    z_offset_enable_model,
                )
                _dance_option_checkbox_pair(
                    ui,
                    "Ankle auto comp",
                    foot_ground_comp_model,
                    "Robot Leg IK",
                    foot_ik_enable_model,
                )
                _dance_option_float_slider_pair(
                    ui,
                    "Root Z baseline",
                    root_z_baseline_offset_model,
                    0.2,
                    1.2,
                    "Root Z outlier scale",
                    root_z_outlier_scale_model,
                    0.0,
                    1.0,
                )
                _dance_option_float_slider_pair(
                    ui,
                    "IK reach",
                    foot_ik_reach_model,
                    0.6,
                    1.2,
                    "Leg scale",
                    foot_ik_leg_scale_model,
                    0.5,
                    1.2,
                )
                ui.Label(
                    "Ankle offset (root-local m). Orange sphere = red + offset.",
                    height=20,
                    style={"font_size": 12, "color": 0xFFAAAAAA},
                )
                with ui.HStack(height=_DANCE_OPT_ROW_H, spacing=_DANCE_OPT_COL_GAP):
                    with ui.HStack(width=_DANCE_OPT_COL_W):
                        ui.Label("Ankle offset", width=_DANCE_OPT_LABEL_W)
                        ui.Label("X", width=14)
                        ui.FloatField(model=foot_ik_ankle_offset_x_model, width=46)
                        ui.Label("Y", width=14)
                        ui.FloatField(model=foot_ik_ankle_offset_y_model, width=46)
                        ui.Spacer()
                    with ui.HStack(width=_DANCE_OPT_COL_W):
                        ui.Label("Z", width=14)
                        ui.FloatField(model=foot_ik_ankle_offset_z_model, width=46)
                        ui.Spacer()
                with ui.HStack(height=22, spacing=_DANCE_OPT_COL_GAP):
                    with ui.HStack(width=_DANCE_OPT_COL_W):
                        ui.Label("L IK target", width=_DANCE_OPT_LABEL_W)
                        l_ik_target_label = ui.Label("x: -  y: -  z: -")
                        ui.Spacer()
                    with ui.HStack(width=_DANCE_OPT_COL_W):
                        ui.Label("R IK target", width=_DANCE_OPT_LABEL_W)
                        r_ik_target_label = ui.Label("x: -  y: -  z: -")
                        ui.Spacer()
                with ui.CollapsableFrame("Foot IK Advanced", collapsed=True, height=0):
                    with ui.VStack(spacing=4):
                        with ui.HStack(height=_DANCE_OPT_ROW_H, spacing=_DANCE_OPT_COL_GAP):
                            with ui.HStack(width=_DANCE_OPT_COL_W):
                                ui.Label("Reg weight", width=_DANCE_OPT_LABEL_W)
                                ui.FloatField(model=foot_ik_reg_weight_model, width=_DANCE_OPT_FIELD_W)
                                ui.Spacer(width=4)
                                ui.FloatSlider(
                                    model=foot_ik_reg_weight_model,
                                    min=0.0,
                                    max=1.0,
                                    width=_DANCE_OPT_SLIDER_W,
                                )
                                ui.Spacer()
                            with ui.HStack(width=_DANCE_OPT_COL_W):
                                ui.Label("Debug every N", width=_DANCE_OPT_LABEL_W)
                                ui.IntField(model=foot_ik_debug_every_model, width=56)
                                ui.Spacer()
                ui.Spacer(height=4)

    transport_refs = {
        "scrub_model": scrub_model,
        "pd_drive_model": pd_drive_model,
        "audio_volume_model": audio_volume_model,
        "z_offset_enable_model": z_offset_enable_model,
        "root_z_baseline_offset_model": root_z_baseline_offset_model,
        "root_z_outlier_scale_model": root_z_outlier_scale_model,
        "foot_ground_comp_model": foot_ground_comp_model,
        "max_label": max_frame_label,
        "btn_prev": btn_prev,
        "btn_next": btn_next,
        "btn_pause": btn_pause,
        "btn_resume": btn_resume,
        "btn_stop": btn_stop,
        "dance_combo": dance_combo,
        "btn_gen_z_editted": btn_gen_z_editted,
        "btn_delete_z_edit": btn_delete_z_edit,
        "z_edit_status_label": z_edit_status_label,
        "btn_record_h5": btn_record_h5,
        "btn_delete_h5": btn_delete_h5,
        "h5_status_label": h5_status_label,
        "foot_ik_enable_model": foot_ik_enable_model,
        "foot_ik_reach_model": foot_ik_reach_model,
        "foot_ik_leg_scale_model": foot_ik_leg_scale_model,
        "foot_ik_ankle_offset_x_model": foot_ik_ankle_offset_x_model,
        "foot_ik_ankle_offset_y_model": foot_ik_ankle_offset_y_model,
        "foot_ik_ankle_offset_z_model": foot_ik_ankle_offset_z_model,
        "foot_ik_debug_every_model": foot_ik_debug_every_model,
        "foot_ik_reg_weight_model": foot_ik_reg_weight_model,
        "foot_ik_l_ik_target_label": l_ik_target_label,
        "foot_ik_r_ik_target_label": r_ik_target_label,
    }
    return playback_title_label, transport_refs


def schedule_mapping_ui_refresh_loop() -> None:
    """启动唯一的 UI 刷新协程（映射窗口与 Retarget Tune 窗口共用）。"""
    global _mapping_ui_refresh_started
    if _mapping_ui_refresh_started:
        return
    _mapping_ui_refresh_started = True
    asyncio.ensure_future(_mapping_ui_refresh_loop())


async def _mapping_ui_refresh_loop() -> None:
    import omni.kit.app

    global _scrub_sync_suppress_seek, _root_quat_sync_suppress_set, _pd_drive_sync_suppress_set
    global _z_offset_sync_suppress_set, _root_z_compress_sync_suppress_set, _foot_ik_sync_suppress_set
    global _foot_ground_comp_sync_suppress_set
    global _audio_volume_sync_suppress_set
    while True:
        await omni.kit.app.get_app().next_update_async()
        if (
            _playback_title_ref is None
            and _playback_transport_ref is None
            and joint_rpy_mapping_ui._joint_models_ref is None
            and joint_rpy_mapping_ui._joint_rpy_refs is None
            and _retarget_tune_refs is None
        ):
            continue
        if _joint_value_provider is None:
            continue
        st: dict[str, Any] = {}
        if _playback_status_provider is not None:
            try:
                st = _playback_status_provider() or {}
            except Exception:
                st = {}
        try:
            values = _joint_value_provider() or {}
        except Exception:
            values = {}
        if _playback_title_ref is not None:
            playing = bool(st.get("playing"))
            if not playing:
                _playback_title_ref.text = "Playback: (idle)  "
            else:
                tag = str(st.get("tag") or "-")
                _playback_title_ref.text = f"Playback: {tag}  "
        if _playback_transport_ref is not None:
            tr = _playback_transport_ref
            if _pd_drive_provider is not None:
                try:
                    pd_on = bool(_pd_drive_provider())
                except Exception:
                    pd_on = False
                _pd_drive_sync_suppress_set = True
                try:
                    if bool(tr["pd_drive_model"].get_value_as_bool()) != pd_on:
                        tr["pd_drive_model"].set_value(pd_on)
                except Exception:
                    pass
                finally:
                    _pd_drive_sync_suppress_set = False
            if _z_offset_enable_provider is not None:
                try:
                    z_on = bool(_z_offset_enable_provider())
                except Exception:
                    z_on = False
                _z_offset_sync_suppress_set = True
                try:
                    if bool(tr["z_offset_enable_model"].get_value_as_bool()) != z_on:
                        tr["z_offset_enable_model"].set_value(z_on)
                except Exception:
                    pass
                finally:
                    _z_offset_sync_suppress_set = False
            if _root_z_compress_provider is not None:
                try:
                    z_base_off, z_scale = _root_z_compress_provider()
                except Exception:
                    z_base_off, z_scale = 0.0, 1.0
                _root_z_compress_sync_suppress_set = True
                try:
                    cur_off = float(tr["root_z_baseline_offset_model"].get_value_as_float())
                    cur_scale = float(tr["root_z_outlier_scale_model"].get_value_as_float())
                    if abs(cur_off - float(z_base_off)) > 1e-5:
                        tr["root_z_baseline_offset_model"].set_value(float(z_base_off))
                    z_scale = max(0.0, min(1.0, float(z_scale)))
                    if abs(cur_scale - z_scale) > 1e-5:
                        tr["root_z_outlier_scale_model"].set_value(z_scale)
                except Exception:
                    pass
                finally:
                    _root_z_compress_sync_suppress_set = False
            if _foot_ground_comp_provider is not None:
                try:
                    fg_on = bool(_foot_ground_comp_provider())
                except Exception:
                    fg_on = True
                _foot_ground_comp_sync_suppress_set = True
                try:
                    if bool(tr["foot_ground_comp_model"].get_value_as_bool()) != fg_on:
                        tr["foot_ground_comp_model"].set_value(fg_on)
                except Exception:
                    pass
                finally:
                    _foot_ground_comp_sync_suppress_set = False
            if _audio_volume_provider is not None:
                try:
                    vol = float(_audio_volume_provider())
                except Exception:
                    vol = float(DEFAULT_VOLUME)
                _audio_volume_sync_suppress_set = True
                try:
                    cur = float(tr["audio_volume_model"].get_value_as_float())
                    if abs(cur - vol) > 1e-4:
                        tr["audio_volume_model"].set_value(vol)
                except Exception:
                    pass
                finally:
                    _audio_volume_sync_suppress_set = False
            combo = tr.get("dance_combo")
            btn_z = tr.get("btn_gen_z_editted")
            btn_del_z = tr.get("btn_delete_z_edit")
            z_lbl = tr.get("z_edit_status_label")
            selected_key = _combo_selected_dance_key(combo, _dance_entries_provider)
            if z_lbl is not None:
                _update_z_edit_status_label(z_lbl, selected_key)
            if combo is not None and btn_z is not None and _dance_entries_provider is not None:
                z_status = _resolve_z_edit_ui_status(selected_key)
                z_busy = False
                if _z_edit_busy_provider is not None:
                    try:
                        z_busy = bool(_z_edit_busy_provider())
                    except Exception:
                        z_busy = False
                can_gen = bool(selected_key) and z_status == "missing" and not z_busy
                btn_z.enabled = can_gen
                btn_z.text = "Generating..." if z_busy else "Gen Z_edited"
            if btn_del_z is not None:
                z_status = _resolve_z_edit_ui_status(selected_key)
                z_busy = False
                if _z_edit_busy_provider is not None:
                    try:
                        z_busy = bool(_z_edit_busy_provider())
                    except Exception:
                        z_busy = False
                btn_del_z.enabled = bool(selected_key) and z_status == "available" and not z_busy
            btn_rec = tr.get("btn_record_h5")
            btn_del_h5 = tr.get("btn_delete_h5")
            h5_lbl = tr.get("h5_status_label")
            if h5_lbl is not None:
                _update_h5_status_label(h5_lbl, selected_key)
            if btn_rec is not None:
                h5_busy = False
                if _h5_record_busy_provider is not None:
                    try:
                        h5_busy = bool(_h5_record_busy_provider())
                    except Exception:
                        h5_busy = False
                btn_rec.enabled = not h5_busy
                btn_rec.text = "Recording..." if h5_busy else "Record H5"
            if btn_del_h5 is not None:
                h5_busy = False
                if _h5_record_busy_provider is not None:
                    try:
                        h5_busy = bool(_h5_record_busy_provider())
                    except Exception:
                        h5_busy = False
                btn_del_h5.enabled = _can_delete_h5(selected_key) and not h5_busy
            playing = bool(st.get("playing"))
            paused = bool(st.get("playback_paused"))
            mx = st.get("max_frame")
            if mx is not None:
                tr["max_label"].text = f"/ {int(mx)}"
            else:
                tr["max_label"].text = "/ -"
            show_btns = playing
            tr["btn_prev"].visible = show_btns
            tr["btn_next"].visible = show_btns
            tr["btn_pause"].visible = show_btns and not paused
            tr["btn_resume"].visible = show_btns and paused
            tr["btn_stop"].visible = show_btns
            fr = st.get("frame")
            _scrub_sync_suppress_seek = True
            try:
                if playing and fr is not None and not paused:
                    try:
                        if int(tr["scrub_model"].get_value_as_int()) != int(fr):
                            tr["scrub_model"].set_value(int(fr))
                    except Exception:
                        tr["scrub_model"].set_value(int(fr))
                elif not playing:
                    try:
                        if int(tr["scrub_model"].get_value_as_int()) != 0:
                            tr["scrub_model"].set_value(0)
                    except Exception:
                        pass
            finally:
                _scrub_sync_suppress_seek = False
            if _foot_ik_provider is not None:
                try:
                    fk = _foot_ik_provider() or {}
                except Exception:
                    fk = {}
                _foot_ik_sync_suppress_set = True
                try:
                    if "foot_ik_enable_model" in tr:
                        tr["foot_ik_enable_model"].set_value(bool(fk.get("enable", False)))
                    if "foot_ik_reach_model" in tr:
                        tr["foot_ik_reach_model"].set_value(float(fk.get("reach", 1.0)))
                    if "foot_ik_leg_scale_model" in tr:
                        tr["foot_ik_leg_scale_model"].set_value(float(fk.get("leg_scale", 0.75)))
                    ao = tuple(fk.get("ankle_offset", (0.0, 0.0, 0.02)))
                    if len(ao) == 3:
                        if "foot_ik_ankle_offset_x_model" in tr:
                            tr["foot_ik_ankle_offset_x_model"].set_value(float(ao[0]))
                        if "foot_ik_ankle_offset_y_model" in tr:
                            tr["foot_ik_ankle_offset_y_model"].set_value(float(ao[1]))
                        if "foot_ik_ankle_offset_z_model" in tr:
                            tr["foot_ik_ankle_offset_z_model"].set_value(float(ao[2]))
                    if "foot_ik_debug_every_model" in tr:
                        tr["foot_ik_debug_every_model"].set_value(int(fk.get("debug_every", 0)))
                    if "foot_ik_reg_weight_model" in tr:
                        tr["foot_ik_reg_weight_model"].set_value(float(fk.get("ik_reg_weight", 0.15)))
                except Exception:
                    pass
                finally:
                    _foot_ik_sync_suppress_set = False

            for _lbl_key, _xk, _yk, _zk in [
                ("foot_ik_l_ik_target_label", "__ik_target_l_x", "__ik_target_l_y", "__ik_target_l_z"),
                ("foot_ik_r_ik_target_label", "__ik_target_r_x", "__ik_target_r_y", "__ik_target_r_z"),
            ]:
                _lbl = tr.get(_lbl_key)
                if _lbl is None:
                    continue
                _xv = values.get(_xk) if isinstance(values, dict) else None
                _yv = values.get(_yk) if isinstance(values, dict) else None
                _zv = values.get(_zk) if isinstance(values, dict) else None
                if _xv is None or _yv is None or _zv is None:
                    _lbl.text = "x: -  y: -  z: -"
                else:
                    try:
                        _lbl.text = f"x: {float(_xv):+.3f}  y: {float(_yv):+.3f}  z: {float(_zv):+.3f}"
                    except Exception:
                        _lbl.text = "x: -  y: -  z: -"

        rr = _retarget_tune_refs
        if rr is not None and isinstance(values, dict):
            sho_raw = rr.get("sho_raw_labels", {})
            if sho_raw:
                def _fmt_shoulder_abs(prefix: str) -> str:
                    if prefix == "L":
                        keys = (
                            "left_shoulder_pitch_joint",
                            "left_shoulder_roll_joint",
                            "left_shoulder_yaw_joint",
                        )
                    else:
                        keys = (
                            "right_shoulder_pitch_joint",
                            "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint",
                        )
                    p = values.get(keys[0])
                    r = values.get(keys[1])
                    y = values.get(keys[2])
                    if not isinstance(p, (int, float)):
                        return "abs: —"
                    if not isinstance(r, (int, float)):
                        return "abs: —"
                    if not isinstance(y, (int, float)):
                        return "abs: —"
                    return f"abs: P:{float(p):+.1f}° R:{float(r):+.1f}° Y:{float(y):+.1f}°"

                for pfx, key in [("L", "__sho_left_raw"), ("R", "__sho_right_raw")]:
                    lbl = sho_raw.get(pfx)
                    if lbl is not None:
                        txt = values.get(key)
                        delta_line = f"delta: {txt}" if isinstance(txt, str) else "delta: —"
                        lbl.text = f"{delta_line}\n{_fmt_shoulder_abs(pfx)}"
            leg_raw = rr.get("leg_raw_labels", {})
            if leg_raw:
                for pfx, hk, ak in [
                    ("L", "__leg_left_hip_raw", "__leg_left_ank_raw"),
                    ("R", "__leg_right_hip_raw", "__leg_right_ank_raw"),
                ]:
                    hip_lbl = leg_raw.get(f"{pfx}_hip")
                    ank_lbl = leg_raw.get(f"{pfx}_ank")
                    if hip_lbl is not None:
                        t = values.get(hk)
                        hip_lbl.text = f"hip: {t}" if isinstance(t, str) else "hip: —"
                    if ank_lbl is not None:
                        t = values.get(ak)
                        ank_lbl.text = f"ank: {t}" if isinstance(t, str) else "ank: —"

        jr = joint_rpy_mapping_ui._joint_rpy_refs
        if jr is not None:
            np_lbl = jr.get("now_playing_label")
            if np_lbl is not None:
                playing = bool(st.get("playing"))
                if not playing:
                    np_lbl.text = "Now playing: (idle)"
                else:
                    tag = str(st.get("tag") or "-")
                    np_lbl.text = f"Now playing: {tag}"
            fr_lbl = jr.get("now_playing_frame_label")
            mx_lbl = jr.get("now_playing_max_label")
            if fr_lbl is not None or mx_lbl is not None:
                playing = bool(st.get("playing"))
                fr = st.get("frame")
                mx = st.get("max_frame")
                if fr_lbl is not None:
                    if playing and fr is not None:
                        fr_lbl.text = str(int(fr))
                    else:
                        fr_lbl.text = "0"
                if mx_lbl is not None:
                    if mx is not None:
                        mx_lbl.text = f"/ {int(mx)}"
                    else:
                        mx_lbl.text = "/ -"
                pb_model = jr.get("playback_progress_bar_model")
                if pb_model is not None:
                    if playing and fr is not None and mx is not None and int(mx) > 0:
                        ratio = max(0.0, min(1.0, float(int(fr)) / float(int(mx))))
                    else:
                        ratio = 0.0
                    pb_model.set_value(ratio)
            if _root_quat_rpy_provider is not None:
                try:
                    (sr, sp, sy), (ir, ip, iy) = _root_quat_rpy_provider()
                except Exception:
                    sr, sp, sy = 1.0, 1.0, 1.0
                    ir, ip, iy = 0, 1, 2
                _root_quat_sync_suppress_set = True
                try:
                    if abs(float(jr["root_roll_scale_model"].get_value_as_float()) - float(sr)) > 1e-6:
                        jr["root_roll_scale_model"].set_value(float(sr))
                    if abs(float(jr["root_pitch_scale_model"].get_value_as_float()) - float(sp)) > 1e-6:
                        jr["root_pitch_scale_model"].set_value(float(sp))
                    if abs(float(jr["root_yaw_scale_model"].get_value_as_float()) - float(sy)) > 1e-6:
                        jr["root_yaw_scale_model"].set_value(float(sy))
                    if int(jr["root_roll_euler_model"].get_value_as_int()) != int(ir):
                        jr["root_roll_euler_model"].set_value(int(ir))
                    if int(jr["root_pitch_euler_model"].get_value_as_int()) != int(ip):
                        jr["root_pitch_euler_model"].set_value(int(ip))
                    if int(jr["root_yaw_euler_model"].get_value_as_int()) != int(iy):
                        jr["root_yaw_euler_model"].set_value(int(iy))
                except Exception:
                    pass
                finally:
                    _root_quat_sync_suppress_set = False
            if _root_rot_bone_name_provider is not None:
                try:
                    bone_nm = str(_root_rot_bone_name_provider() or "")
                except Exception:
                    bone_nm = ""
                lbl = jr.get("root_rot_bone_label")
                if lbl is not None:
                    romaji = mmd_bone_to_romaji(bone_nm) if bone_nm else "-"
                    lbl.text = f"Rot bone: {romaji}" if bone_nm else "Rot bone: (none)"
            for _rk, _ck in [
                ("root_R_value_label", "__root_rpy_deg_r"),
                ("root_P_value_label", "__root_rpy_deg_p"),
                ("root_Y_value_label", "__root_rpy_deg_y"),
            ]:
                _lw = jr.get(_rk)
                if _lw is None:
                    continue
                _vv = values.get(_ck) if isinstance(values, dict) else None
                if _vv is None:
                    _lw.text = "N/A"
                else:
                    try:
                        _lw.text = f"{float(_vv):.2f}deg"
                    except (TypeError, ValueError):
                        _lw.text = "N/A"

        jm = joint_rpy_mapping_ui._joint_models_ref
        if jm is not None:
            for jname, (_euler_model, _scale_model, value_label, _absorb_model) in jm.items():
                v = values.get(jname) if isinstance(values, dict) else None
                if v is None:
                    value_label.text = "N/A"
                elif jname in _HINGE_DETAIL_ROW_JOINTS and isinstance(values, dict):
                    if jname in _KNEE_JOINT_NAMES:
                        mmd = values.get(f"{jname}{_KNEE_MMD_SUFFIX}")
                    else:
                        mmd = values.get(f"{jname}{_ELBOW_MMD_SUFFIX}")
                    if isinstance(mmd, str) and mmd:
                        mmd_fmt = wrap_long_hinge_text(mmd)
                        value_label.text = f"{float(v):.1f}deg sim\n{mmd_fmt}"
                    else:
                        value_label.text = f"{float(v):.2f}deg"
                else:
                    value_label.text = f"{float(v):.2f}deg"


_PROPERTY_DOCK_TARGETS = ("Property", "IsaacLab")
_CONTENT_CONSOLE_DOCK_TARGETS = ("Content", "Console")
_DOCK_TAB_RETRIES = 120


def _schedule_tab_dock(
    window: Any,
    window_title: str,
    target_names: tuple[str, ...],
    *,
    show_window_name: str | None = None,
    group_label: str = "target",
) -> None:
    """Dock window into an existing Kit workspace tab group (sync deferred + async retry)."""
    try:
        import omni.ui as ui
    except ImportError:
        return

    reveal = show_window_name or (target_names[0] if target_names else None)
    if reveal:
        try:
            ui.Workspace.show_window(reveal, True)
        except Exception:
            pass
    if target_names:
        try:
            window.deferred_dock_in(target_names[0], ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        except Exception:
            pass
    asyncio.ensure_future(
        _dock_window_to_tab_group(window, window_title, target_names, group_label=group_label)
    )


def _schedule_property_tab_dock(window: Any, window_title: str) -> None:
    """Dock window into Property / IsaacLab tab group."""
    _schedule_tab_dock(
        window,
        window_title,
        _PROPERTY_DOCK_TARGETS,
        show_window_name="Property",
        group_label="Property/IsaacLab",
    )


def _schedule_content_console_tab_dock(window: Any, window_title: str) -> None:
    """Dock window into Content / Console tab group (bottom panel)."""
    _schedule_tab_dock(
        window,
        window_title,
        _CONTENT_CONSOLE_DOCK_TARGETS,
        show_window_name="Content",
        group_label="Content/Console",
    )


async def _dock_window_to_tab_group(
    window: Any,
    window_title: str,
    target_names: tuple[str, ...],
    *,
    group_label: str,
) -> None:
    """Retry docking until the window joins the requested tab group."""
    try:
        import omni.ui as ui
        import omni.kit.app
    except ImportError:
        return

    app = omni.kit.app.get_app()

    for _ in range(30):
        if ui.Workspace.get_window(window_title):
            break
        await app.next_update_async()

    for _ in range(_DOCK_TAB_RETRIES):
        target_handle = None
        target_name = None
        for name in target_names:
            handle = ui.Workspace.get_window(name)
            if handle is not None:
                target_handle = handle
                target_name = name
                break

        if target_handle is not None:
            try:
                window.dock_in(target_handle, ui.DockPosition.SAME, 1.0)
            except Exception:
                pass
            custom_handle = ui.Workspace.get_window(window_title)
            if custom_handle is not None:
                try:
                    custom_handle.dock_in(target_handle, ui.DockPosition.SAME, 1.0)
                    custom_handle.focus()
                except Exception:
                    pass

            await app.next_update_async()
            if window.docked or (custom_handle is not None and custom_handle.docked):
                print(f"[INFO] Docked '{window_title}' into '{target_name}' tab group.")
                return

        await app.next_update_async()

    print(
        f"[WARN] Failed to dock '{window_title}' into {group_label} tab group "
        "(window stays floating; drag it manually or open the target panel)."
    )


async def _dock_window_to_property_tab(window: Any, window_title: str) -> None:
    """Backward-compatible wrapper."""
    await _dock_window_to_tab_group(
        window,
        window_title,
        _PROPERTY_DOCK_TARGETS,
        group_label="Property/IsaacLab",
    )


def create_mmd_config_ui():
    """创建映射编辑窗口，注册到 Window 菜单。"""
    try:
        import omni.ui as ui
        import omni.kit.app
        from omni.kit.menu.utils import add_menu_items, MenuItemDescription
    except ImportError:
        print("[WARN] omni.ui 不可用，映射编辑 UI 已跳过（可能为 headless 模式）")
        return None

    _window_ref = []

    def _create_window():
        global _playback_title_ref, _playback_transport_ref
        window = ui.Window(
            WINDOW_TITLE,
            width=620,
            height=520,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _playback_title_ref, _playback_transport_ref = _build_mapping_window(ui)
        window.visible = True
        _window_ref.append(window)
        _schedule_content_console_tab_dock(window, WINDOW_TITLE)
        return window

    def _on_menu_click():
        if _window_ref:
            _window_ref[0].visible = True
        else:
            _create_window()

    add_menu_items(
        [MenuItemDescription(name=WINDOW_TITLE, onclick_fn=_on_menu_click)],
        "Window",
    )

    async def _auto_open():
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        _create_window()

    if _AUTO_OPEN:
        asyncio.ensure_future(_auto_open())

    schedule_mapping_ui_refresh_loop()

    print("[INFO] G1 MMD config: Window menu →", WINDOW_TITLE)
    return True


# Backward-compatible alias
create_mapping_ui = create_mmd_config_ui
