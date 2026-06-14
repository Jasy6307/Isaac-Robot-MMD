"""
G1 关节映射编辑窗口。

功能概览：
1) 在 Isaac Sim Window 菜单注册 ``G1 Joint Mapping``（欧拉主轴 / scale / 播放、Root R/P/Y、腰两骨共轭开关；Root 与关节映射同处可滚动滑块区）；
2) Retarget Tune（肩/腿基变换 Rz·Ry·Rx）见独立菜单项 ``G1 Retarget Tune``（``create_retarget_tune_ui``）；
3) 实时显示当前机器人关节角度；映射重置；映射变更回调驱动仿真重算。
"""
import asyncio
from typing import Any, Callable

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.utils.csv_motion_loader import (
    G1_JOINT_TO_MMD,
    get_hinge_swing_absorb,
    get_waist_upper_pair_quat_conjugate,
    reset_mapping_to_default,
    set_hinge_swing_absorb,
    toggle_waist_upper_pair_quat_conjugate,
    update_mapping_entry,
)
from robot_mmd.train_workflow.ui.retargeting_tune import (
    RETARGET_TUNE_WINDOW_TITLE,
    build_retarget_tune_window,
)
from robot_mmd.train_workflow.utils.audio_util import DEFAULT_VOLUME
from robot_mmd.train_workflow.utils.mmd_fk import default_foot_ik_viz_config

WINDOW_TITLE = "G1 Joint Mapping"
_AUTO_OPEN = True

# 外部注入：用于在 UI 中显示“当前环境下的关节值（度制）”
# 返回值: dict[joint_name] = angle_deg；膝/肘可含 ``__knee_mmd`` / ``__elbow_mmd`` 分解说明字符串
_joint_value_provider: Callable[[], dict[str, Any]] | None = None

# 外部注入：当前播放片段（dance/pose）名称与帧；主循环每帧刷新
# 约定：返回值 dict，可含
#   ``playing`` (bool) / ``kind`` ("dance"|"pose"|"") / ``title`` (str 已排版短名) /
#   ``frame`` (int|None) / ``max_frame`` (int|None) —— 仅在 dance 时使用帧数
_playback_status_provider: Callable[[], dict[str, Any]] | None = None

# Transport: pause/resume, seek to frame index (only while clip loaded in g1_mmd_playback)
_playback_toggle_cb: Callable[[], None] | None = None
_playback_seek_cb: Callable[[int], None] | None = None
DanceUiEntry = tuple[str, str]  # (dance_key, combo_label)
_dance_entries_provider: Callable[[], list[DanceUiEntry]] | None = None
_dance_request_cb: Callable[[str, bool], None] | None = None
_dance_z_edited_status_provider: Callable[[str], bool] | None = None
_dance_z_edit_request_cb: Callable[[str], None] | None = None
_z_edit_busy_provider: Callable[[], bool] | None = None
_pd_drive_provider: Callable[[], bool] | None = None
_pd_drive_setter: Callable[[bool], None] | None = None
_z_offset_enable_provider: Callable[[], bool] | None = None
_z_offset_enable_setter: Callable[[bool], None] | None = None
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
_foot_ik_viz_provider: Callable[[], dict[str, Any]] | None = None
_foot_ik_viz_setter: Callable[[dict[str, Any]], None] | None = None
_audio_volume_provider: Callable[[], float] | None = None
_audio_volume_setter: Callable[[float], None] | None = None

# True while refresh assigns scrub IntField from sim; blocks value_changed -> seek storm
_scrub_sync_suppress_seek: bool = False
_root_quat_sync_suppress_set: bool = False
_pd_drive_sync_suppress_set: bool = False
_z_offset_sync_suppress_set: bool = False
_foot_ik_sync_suppress_set: bool = False
_foot_ik_viz_sync_suppress_set: bool = False
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
) -> None:
    """Pause / resume and seek-to-frame (clip-relative index)."""
    global _playback_toggle_cb, _playback_seek_cb
    _playback_toggle_cb = toggle_pause
    _playback_seek_cb = seek_frame


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
) -> None:
    """Z_editted sibling status + generate request for the dance file combo row."""
    global _dance_z_edited_status_provider, _dance_z_edit_request_cb, _z_edit_busy_provider
    _dance_z_edited_status_provider = z_editted_status
    _dance_z_edit_request_cb = on_generate
    _z_edit_busy_provider = busy_provider


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


def set_foot_ik_viz_callbacks(
    provider: Callable[[], dict[str, Any]] | None,
    setter: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Set red-sphere MMD->Isaac axis map callbacks (independent of robot leg IK)."""
    global _foot_ik_viz_provider, _foot_ik_viz_setter
    _foot_ik_viz_provider = provider
    _foot_ik_viz_setter = setter


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


# 腰两骨共轭开关：两个 Label 文案由 _sync_waist_pair_conj_status_labels 更新
_waist_pair_conj_status_labels: list[Any] = [None, None]
# 由 create_mapping_ui / create_retarget_tune_ui 写入；刷新循环读取
_joint_models_ref: dict[str, tuple] | None = None
_playback_title_ref: Any | None = None
_playback_transport_ref: dict[str, Any] | None = None
_retarget_tune_refs: dict[str, Any] | None = None
_mapping_ui_refresh_started: bool = False
MMD_BONE_TO_ROMAJI: dict[str, str] = {
    "右ひざ": "R_KNE",
    "左ひざ": "L_KNE",
    "右足": "R_FOOT",
    "左足": "L_FOOT",
    "下半身": "LOWER_B",
    "右足首": "R_ANK",
    "左足首": "L_ANK",
    "右肩": "R_SHO",
    "右腕": "R_WRI",
    "左肩": "L_SHO",
    "左腕": "L_WRI",
    "右ひじ": "R_ELB",
    "左ひじ": "L_ELB",
    "右手首": "R_WRI",
    "左手首": "L_WRI",
    "右親指０": "R_TH0",
    "右親指１": "R_TH1",
    "右親指２": "R_TH2",
    "右親指先": "R_THX",
    "左親指０": "L_TH0",
    "左親指１": "L_TH1",
    "左親指２": "L_TH2",
    "左親指先": "L_THX",
    "右人指１": "R_ID1",
    "右人指２": "R_ID2",
    "右人指３": "R_ID3",
    "左人指１": "L_ID1",
    "左人指２": "L_ID2",
    "左人指３": "L_ID3",
    "右中指１": "R_MD1",
    "右中指２": "R_MD2",
    "右中指３": "R_MD3",
    "左中指１": "L_MD1",
    "左中指２": "L_MD2",
    "左中指３": "L_MD3",
    "右薬指１": "R_RG1",
    "右薬指２": "R_RG2",
    "右薬指３": "R_RG3",
    "左薬指１": "L_RG1",
    "左薬指２": "L_RG2",
    "左薬指３": "L_RG3",
    "右小指１": "R_PK1",
    "右小指２": "R_PK2",
    "右小指３": "R_PK3",
    "左小指１": "L_PK1",
    "左小指２": "L_PK2",
    "左小指３": "L_PK3",
    "上半身": "UPPER_B",
    "上半身2": "UPPER_B2",
    "首": "HEAD",
    "グルーブ": "GROOVE",
    "センター": "CENTER",
    "センター先": "CENTER_TIP",
    "センター親": "CENTER_P",
    "腰": "WAIST",
    "全ての親": "ALL_PARENT",
}


def mmd_bone_to_romaji(name: str) -> str:
    """MMD bone label for omni.ui (Kit CJK fonts often render as mojibake)."""
    return MMD_BONE_TO_ROMAJI.get(str(name or ""), str(name or ""))

# 膝/肘行第二行：MMD hinge/swing 与映射补偿（见 csv_motion_loader.*_hinge_mapping_ui_extra）
_KNEE_JOINT_NAMES = frozenset({"left_knee_joint", "right_knee_joint"})
_KNEE_MMD_SUFFIX = "__knee_mmd"
_ELBOW_JOINT_NAMES = frozenset({"left_elbow_joint", "right_elbow_joint"})
_ELBOW_MMD_SUFFIX = "__elbow_mmd"
_HINGE_DETAIL_ROW_JOINTS = _KNEE_JOINT_NAMES | _ELBOW_JOINT_NAMES


def _wrap_long_hinge_text(mmd_line: str) -> str:
    """Knee/Elbow 第二行 MMD 说明过长时在词边界附近断行。"""
    s = mmd_line.strip()
    if len(s) <= 44:
        return s
    mid = len(s) // 2
    brk = s.rfind(" ", 8, mid + 14)
    if brk <= 0:
        brk = mid
    return s[:brk] + "\n" + s[brk:].lstrip()


# Joint categories split by side for easier tuning
JOINT_CATEGORIES: dict[str, list[str]] = {
    "Upper Body (Left)": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint",
    ],
    "Hand (Left)": [
        "lh_thumb_cmc_yaw", "lh_thumb_cmc_pitch", "lh_thumb_ip",
        "lh_index_mcp_pitch", "lh_index_dip",
        "lh_middle_mcp_pitch", "lh_middle_dip",
        "lh_ring_mcp_pitch", "lh_ring_dip",
        "lh_pinky_mcp_pitch", "lh_pinky_dip",
    ],
    "Upper Body (Right)": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint",
    ],
    "Hand (Right)": [
        "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch", "rh_thumb_ip",
        "rh_index_mcp_pitch", "rh_index_dip",
        "rh_middle_mcp_pitch", "rh_middle_dip",
        "rh_ring_mcp_pitch", "rh_ring_dip",
        "rh_pinky_mcp_pitch", "rh_pinky_dip",
    ],
    "Lower Body (Left)": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ],
    "Lower Body (Right)": [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ],
    "Waist": [
        "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    ],
}

# Root R/P/Y remapping rows (same slider layout as joint mapping; not in G1_JOINT_TO_MMD).
ROOT_RPY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("root_Roll", "out·R", "root_roll"),
    ("root_Pitch", "out·P", "root_pitch"),
    ("root_Yaw", "out·Y", "root_yaw"),
)


def _sync_waist_pair_conj_status_labels() -> None:
    """Update waist [upper spine, upper2] quaternion-conjugate preset row labels."""

    c0, c1 = get_waist_upper_pair_quat_conjugate()
    if _waist_pair_conj_status_labels[0] is not None:
        _waist_pair_conj_status_labels[0].text = f"Upper: {'conj' if c0 else 'as-is'}"
    if _waist_pair_conj_status_labels[1] is not None:
        _waist_pair_conj_status_labels[1].text = f"Upper2: {'conj' if c1 else 'as-is'}"


def _build_mapping_window(ui):
    """构建映射编辑窗口内容。

    Layout (top to bottom): Upper Body, Lower Body, Waist. Left joints before right.
    """
    joint_models: dict[str, tuple] = {}

    def _bone_str(bones) -> str:
        """MMD bone names as English romaji labels for omni.ui."""
        if isinstance(bones, list):
            return " + ".join(mmd_bone_to_romaji(b) for b in bones)
        return mmd_bone_to_romaji(str(bones))

    def _on_euler_changed(joint_name: str, _model):
        _update_mapping_from_models(joint_name)

    def _on_scale_changed(joint_name: str, _model):
        _update_mapping_from_models(joint_name)

    def _update_mapping_from_models(joint_name: str) -> None:
        """读取 UI 模型并提交到运行时映射。"""
        try:
            euler_model = joint_models[joint_name][0]
            scale_model = joint_models[joint_name][1]
            euler_idx = max(0, min(2, int(euler_model.get_value_as_int())))
            scale = float(scale_model.get_value_as_float())
            update_mapping_entry(joint_name, euler_idx, scale)
            _notify_mapping_changed()
        except Exception:
            pass

    def _on_flip_scale(joint_name: str):
        """将缩放系数取反（正负号切换）。"""
        try:
            _euler_model, scale_model, _value, _abs = joint_models[joint_name]
            new_scale = -float(scale_model.get_value_as_float())
            scale_model.set_value(new_scale)
            _update_mapping_from_models(joint_name)
        except Exception:
            pass

    def _on_absorb_changed(joint_name: str, _model):
        try:
            _eu, _sc, _vl, absorb_model = joint_models[joint_name]
            if absorb_model is None:
                return
            set_hinge_swing_absorb(joint_name, float(absorb_model.get_value_as_float()))
            _notify_mapping_changed()
        except Exception:
            pass

    def _on_absorb_flip(joint_name: str):
        try:
            _eu, _sc, _vl, absorb_model = joint_models[joint_name]
            if absorb_model is None:
                return
            absorb_model.set_value(-float(absorb_model.get_value_as_float()))
            _on_absorb_changed(joint_name, absorb_model)
        except Exception:
            pass

    def _on_reset():
        reset_mapping_to_default()
        _notify_mapping_changed()
        for jname, (euler_model, scale_model, _value_label, absorb_model) in joint_models.items():
            base = G1_JOINT_TO_MMD.get(jname)
            if base:
                euler_model.set_value(base[1])
                scale_model.set_value(base[2])
            if absorb_model is not None:
                absorb_model.set_value(1.0)
        _ir, _ip, _iy = MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT
        root_roll_euler_model.set_value(int(_ir))
        root_pitch_euler_model.set_value(int(_ip))
        root_yaw_euler_model.set_value(int(_iy))
        _sr, _sp, _sy = MMD_ROOT_QUAT_RPY_SCALE_DEFAULT
        root_roll_scale_model.set_value(float(_sr))
        root_pitch_scale_model.set_value(float(_sp))
        root_yaw_scale_model.set_value(float(_sy))
        _sync_waist_pair_conj_status_labels()

    # ========== 整体布局：垂直堆叠 ==========
    with ui.VStack(spacing=4):
        scrub_model = ui.SimpleIntModel(0)
        _drr, _drp, _dry = MMD_ROOT_QUAT_RPY_SCALE_DEFAULT
        _dir, _dip, _diy = MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT
        root_roll_euler_model = ui.SimpleIntModel(int(_dir))
        root_pitch_euler_model = ui.SimpleIntModel(int(_dip))
        root_yaw_euler_model = ui.SimpleIntModel(int(_diy))
        root_roll_scale_model = ui.SimpleFloatModel(float(_drr))
        root_pitch_scale_model = ui.SimpleFloatModel(float(_drp))
        root_yaw_scale_model = ui.SimpleFloatModel(float(_dry))

        def _on_pause_click():
            if _playback_toggle_cb is not None:
                _playback_toggle_cb()

        def _on_resume_click():
            if _playback_toggle_cb is not None:
                _playback_toggle_cb()

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
        foot_ground_comp_model = ui.SimpleBoolModel(True)
        audio_volume_model = ui.SimpleFloatModel(float(DEFAULT_VOLUME))
        foot_ik_enable_model = ui.SimpleBoolModel(False)
        foot_ik_weight_model = ui.SimpleFloatModel(1.0)
        foot_ik_reach_model = ui.SimpleFloatModel(0.985)
        foot_ik_leg_scale_model = ui.SimpleFloatModel(1.0)
        foot_ik_ankle_offset_x_model = ui.SimpleFloatModel(0.0)
        foot_ik_ankle_offset_y_model = ui.SimpleFloatModel(0.0)
        foot_ik_ankle_offset_z_model = ui.SimpleFloatModel(0.02)
        foot_ik_debug_every_model = ui.SimpleIntModel(0)
        foot_ik_solver_model = ui.SimpleIntModel(0)
        foot_ik_solver_combo = None
        foot_ik_reg_weight_model = ui.SimpleFloatModel(0.15)
        _sphere_viz_defaults = default_foot_ik_viz_config()
        sphere_map_scale_model = ui.SimpleFloatModel(float(_sphere_viz_defaults.pos_scale))
        sphere_map_axis_idx_models = (
            ui.SimpleIntModel(int(_sphere_viz_defaults.axis_idx[0])),
            ui.SimpleIntModel(int(_sphere_viz_defaults.axis_idx[1])),
            ui.SimpleIntModel(int(_sphere_viz_defaults.axis_idx[2])),
        )
        sphere_map_axis_sign_models = (
            ui.SimpleFloatModel(float(_sphere_viz_defaults.axis_sign[0])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.axis_sign[1])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.axis_sign[2])),
        )
        sphere_map_left_ref_origin_models = (
            ui.SimpleFloatModel(float(_sphere_viz_defaults.left_ref_origin_m[0])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.left_ref_origin_m[1])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.left_ref_origin_m[2])),
        )
        sphere_map_right_ref_origin_models = (
            ui.SimpleFloatModel(float(_sphere_viz_defaults.right_ref_origin_m[0])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.right_ref_origin_m[1])),
            ui.SimpleFloatModel(float(_sphere_viz_defaults.right_ref_origin_m[2])),
        )

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

        btn_gen_z_editted = None

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

        def _push_root_quat_rpy() -> None:
            if _root_quat_sync_suppress_set:
                return
            if _root_quat_rpy_setter is None:
                return
            try:
                _root_quat_rpy_setter(
                    (
                        float(root_roll_scale_model.get_value_as_float()),
                        float(root_pitch_scale_model.get_value_as_float()),
                        float(root_yaw_scale_model.get_value_as_float()),
                    ),
                    (
                        max(0, min(2, int(root_roll_euler_model.get_value_as_int()))),
                        max(0, min(2, int(root_pitch_euler_model.get_value_as_int()))),
                        max(0, min(2, int(root_yaw_euler_model.get_value_as_int()))),
                    ),
                )
                _notify_mapping_changed()
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
                    "weight": float(foot_ik_weight_model.get_value_as_float()),
                    "reach": float(foot_ik_reach_model.get_value_as_float()),
                    "leg_scale": float(foot_ik_leg_scale_model.get_value_as_float()),
                    "debug_every": max(0, int(foot_ik_debug_every_model.get_value_as_int())),
                    "solver": "planar" if int(foot_ik_solver_model.get_value_as_int()) == 1 else "full",
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

        def _push_sphere_map() -> None:
            if _foot_ik_viz_sync_suppress_set:
                return
            if _foot_ik_viz_setter is None:
                return
            try:
                payload = {
                    "scale": float(sphere_map_scale_model.get_value_as_float()),
                    "axis_idx": tuple(
                        max(0, min(2, int(m.get_value_as_int()))) for m in sphere_map_axis_idx_models
                    ),
                    "axis_sign": tuple(
                        float(m.get_value_as_float()) for m in sphere_map_axis_sign_models
                    ),
                    "left_ref_origin": tuple(
                        float(m.get_value_as_float()) for m in sphere_map_left_ref_origin_models
                    ),
                    "right_ref_origin": tuple(
                        float(m.get_value_as_float()) for m in sphere_map_right_ref_origin_models
                    ),
                }
                _foot_ik_viz_setter(payload)
                _notify_mapping_changed()
            except Exception:
                pass

        pd_drive_model.add_value_changed_fn(_on_pd_drive_changed)
        z_offset_enable_model.add_value_changed_fn(_on_z_offset_enable_changed)
        foot_ground_comp_model.add_value_changed_fn(_on_foot_ground_comp_changed)
        audio_volume_model.add_value_changed_fn(_on_audio_volume_changed)
        foot_ik_enable_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_weight_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_reach_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_leg_scale_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_x_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_y_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_ankle_offset_z_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_debug_every_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_solver_model.add_value_changed_fn(lambda _m: _push_foot_ik())
        foot_ik_reg_weight_model.add_value_changed_fn(lambda _m: _push_foot_ik())

        sphere_map_scale_model.add_value_changed_fn(lambda _m: _push_sphere_map())
        for _m in sphere_map_axis_idx_models:
            _m.add_value_changed_fn(lambda _x: _push_sphere_map())
        for _m in sphere_map_axis_sign_models:
            _m.add_value_changed_fn(lambda _x: _push_sphere_map())
        for _m in sphere_map_left_ref_origin_models:
            _m.add_value_changed_fn(lambda _x: _push_sphere_map())
        for _m in sphere_map_right_ref_origin_models:
            _m.add_value_changed_fn(lambda _x: _push_sphere_map())

        with ui.VStack(spacing=4):
            with ui.HStack(height=28):
                ui.Label("Dance File", width=74, height=22)
                ui.Spacer(width=6)
                dance_combo = ui.ComboBox(0, *dance_combo_labels, width=200, height=24)
                btn_gen_z_editted = ui.Button(
                    "Gen Z_edited",
                    width=96,
                    height=24,
                    clicked_fn=_on_gen_z_editted_click,
                )
                ui.Button("Play CSV", width=72, height=24, clicked_fn=lambda: _request_selected_dance(False))
                ui.Button("Play H5", width=62, height=24, clicked_fn=lambda: _request_selected_dance(True))
            with ui.HStack(height=28):
                playback_title_label = ui.Label("Playback: (idle)", width=188, height=22)
                ui.IntField(model=scrub_model, width=64, height=22)
                max_frame_label = ui.Label("/ -", width=40, height=22)
                ui.Spacer(width=4)
                btn_prev = ui.Button("Prev", width=42, height=24, clicked_fn=_on_prev_frame_click)
                btn_next = ui.Button("Next", width=42, height=24, clicked_fn=_on_next_frame_click)
                btn_pause = ui.Button("Pause", width=56, height=24, clicked_fn=_on_pause_click)
                btn_resume = ui.Button("Resume", width=58, height=24, clicked_fn=_on_resume_click)
            with ui.HStack(height=28):
                ui.Label("Audio volume", width=88, height=22)
                ui.FloatField(model=audio_volume_model, width=48, height=22)
                ui.FloatSlider(model=audio_volume_model, min=0.0, max=1.0, width=160, height=22)
                ui.Spacer()
        btn_prev.visible = False
        btn_next.visible = False
        btn_pause.visible = False
        btn_resume.visible = False
        scrub_model.add_value_changed_fn(_on_scrub_changed)

        def _flip_root_axis(m: Any) -> None:
            try:
                m.set_value(-float(m.get_value_as_float()))
                _push_root_quat_rpy()
            except Exception:
                pass

        root_row_models: dict[str, tuple[Any, Any, Any]] = {
            "root_roll": (root_roll_euler_model, root_roll_scale_model, None),
            "root_pitch": (root_pitch_euler_model, root_pitch_scale_model, None),
            "root_yaw": (root_yaw_euler_model, root_yaw_scale_model, None),
        }

        # ========== 可滚动区域：Root + 关节映射（统一滑块行布局） ==========
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2):
                ui.Label(
                    "--- Dance Option ---",
                    height=22,
                    style={"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00},
                )
                with ui.HStack(height=28):
                    ui.Label("PD Drive", width=118)
                    ui.CheckBox(model=pd_drive_model, width=24, height=22)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Z_offset_enable", width=118)
                    ui.CheckBox(model=z_offset_enable_model, width=24, height=22)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Ankle ground comp", width=118)
                    ui.CheckBox(model=foot_ground_comp_model, width=24, height=22)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Robot Leg IK", width=118)
                    ui.CheckBox(model=foot_ik_enable_model, width=24, height=22)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("IK weight", width=118)
                    ui.FloatField(model=foot_ik_weight_model, width=56)
                    ui.FloatSlider(model=foot_ik_weight_model, min=0.0, max=1.0, width=140)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("IK reach", width=118)
                    ui.FloatField(model=foot_ik_reach_model, width=64)
                    ui.FloatSlider(model=foot_ik_reach_model, min=0.6, max=1.2, width=140)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Leg scale", width=118)
                    ui.FloatField(model=foot_ik_leg_scale_model, width=64)
                    ui.FloatSlider(model=foot_ik_leg_scale_model, min=0.5, max=1.2, width=140)
                    ui.Spacer()
                ui.Label(
                    "Ankle offset (root-local m). Orange sphere = red + offset.",
                    height=20,
                    style={"font_size": 12, "color": 0xFFAAAAAA},
                )
                with ui.HStack(height=28):
                    ui.Label("offset X", width=118)
                    ui.FloatField(model=foot_ik_ankle_offset_x_model, width=56)
                    ui.FloatSlider(
                        model=foot_ik_ankle_offset_x_model, min=-0.12, max=0.12, width=140
                    )
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("offset Y", width=118)
                    ui.FloatField(model=foot_ik_ankle_offset_y_model, width=56)
                    ui.FloatSlider(
                        model=foot_ik_ankle_offset_y_model, min=-0.12, max=0.12, width=140
                    )
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("offset Z", width=118)
                    ui.FloatField(model=foot_ik_ankle_offset_z_model, width=56)
                    ui.FloatSlider(
                        model=foot_ik_ankle_offset_z_model, min=-0.12, max=0.12, width=140
                    )
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("L IK target", width=118)
                    l_ik_target_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("R IK target", width=118)
                    r_ik_target_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.CollapsableFrame("Foot IK Advanced", collapsed=True, height=0):
                    with ui.VStack(spacing=4):
                        ui.Label(
                            "Planar solver is debug fallback only; full is recommended.",
                            height=20,
                            style={"font_size": 12, "color": 0xFFAAAAAA},
                        )
                        with ui.HStack(height=28):
                            ui.Label("IK solver", width=118)
                            foot_ik_solver_combo = ui.ComboBox(
                                0, "full", "planar (debug)", width=140, height=22
                            )
                            ui.Spacer()
                        with ui.HStack(height=28):
                            ui.Label("Reg weight", width=118)
                            ui.FloatField(model=foot_ik_reg_weight_model, width=56)
                            ui.FloatSlider(
                                model=foot_ik_reg_weight_model, min=0.0, max=1.0, width=140
                            )
                            ui.Spacer()
                        with ui.HStack(height=28):
                            ui.Label("Debug every N", width=118)
                            ui.IntField(model=foot_ik_debug_every_model, width=56)
                            ui.Spacer()

                def _on_foot_ik_solver_combo(_m=None) -> None:
                    if _foot_ik_sync_suppress_set or foot_ik_solver_combo is None:
                        return
                    idx = int(foot_ik_solver_combo.model.get_item_value_model().as_int)
                    foot_ik_solver_model.set_value(idx)
                    _push_foot_ik()

                if foot_ik_solver_combo is not None:
                    foot_ik_solver_combo.model.add_item_changed_fn(
                        lambda _m: _on_foot_ik_solver_combo()
                    )
                ui.Label(
                    "--- Red Sphere Map (sphere + IK target) ---",
                    height=22,
                    style={"font_size": 15, "font_style": "bold", "color": 0xFFFFFF00},
                )
                with ui.HStack(height=28):
                    ui.Label("L ref origin", width=118)
                    for _m in sphere_map_left_ref_origin_models:
                        ui.FloatField(model=_m, width=62)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("R ref origin", width=118)
                    for _m in sphere_map_right_ref_origin_models:
                        ui.FloatField(model=_m, width=62)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Sphere scale", width=118)
                    ui.FloatField(model=sphere_map_scale_model, width=64)
                    ui.FloatSlider(model=sphere_map_scale_model, min=0.0, max=2.0, width=140)
                    ui.Spacer()
                with ui.HStack(height=28):
                    ui.Label("Sphere idx", width=118)
                    for _m in sphere_map_axis_idx_models:
                        ui.IntField(model=_m, width=32)
                    ui.Spacer(width=10)
                    ui.Label("sign", width=36)
                    for _m in sphere_map_axis_sign_models:
                        ui.FloatField(model=_m, width=46)
                    ui.Spacer()
                ui.Label(
                    "Panel local z is NOT Isaac height; sphere z comes from rotated MMD Y.",
                    height=20,
                    style={"font_size": 12, "color": 0xFFAAAAAA},
                )
                with ui.HStack(height=22):
                    ui.Label("L panel local", width=118)
                    l_foot_local_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("L sphere", width=118)
                    l_foot_xyz_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("R panel local", width=118)
                    r_foot_local_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("R sphere", width=118)
                    r_foot_xyz_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("L toe sphere", width=118)
                    l_toe_xyz_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                with ui.HStack(height=22):
                    ui.Label("R toe sphere", width=118)
                    r_toe_xyz_label = ui.Label("x: -  y: -  z: -", width=300)
                    ui.Spacer()
                ui.Spacer(height=4)

                ui.Label(
                    "--- Root ---",
                    height=22,
                    style={"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00},
                )
                ui.Label(
                    "Root idx: 0=csv roll 1=pitch 2=yaw (per output row). Bone: LOWER_B when active.",
                    height=20,
                    style={"font_size": 12, "color": 0xFFAAAAAA},
                )
                root_row_value_labels: list[Any] = []
                for axis_label, csv_hint, row_key in ROOT_RPY_ROWS:
                    euler_model, scale_model, _ = root_row_models[row_key]
                    with ui.HStack(height=28):
                        ui.Label(axis_label, width=118)
                        ui.Label(csv_hint, width=102)
                        ui.IntField(model=euler_model, width=30)
                        euler_model.add_value_changed_fn(lambda _m: _push_root_quat_rpy())
                        ui.Spacer(width=8)
                        ui.FloatField(model=scale_model, width=50)
                        ui.Spacer(width=4)
                        ui.FloatSlider(model=scale_model, min=-3.0, max=3.0, width=112)
                        scale_model.add_value_changed_fn(lambda _m: _push_root_quat_rpy())
                        ui.Spacer(width=8)
                        ui.Button(
                            "Flip",
                            width=44,
                            height=22,
                            clicked_fn=lambda m=scale_model: _flip_root_axis(m),
                        )
                        ui.Spacer(width=8)
                        root_row_value_labels.append(ui.Label("N/A", width=72))
                root_rot_bone_label = ui.Label("Rot bone: (idle)", width=280, height=20)
                ui.Spacer(height=4)

                ui.Label(
                    "Joint idx: shoulder/hip/waist 0:P 1:R 2:Y, ankle 0:P 1:R",
                    height=20,
                    style={"font_size": 12, "color": 0xFFAAAAAA},
                )
                ui.Spacer(height=2)

                for category_name, joint_names in JOINT_CATEGORIES.items():
                    # 分类标题
                    ui.Label(
                        f"--- {category_name} ---",
                        height=22,
                        style={"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00},
                    )
                    if category_name == "Waist":
                        global _waist_pair_conj_status_labels

                        def _waist_conj_toggle(which: int) -> None:
                            toggle_waist_upper_pair_quat_conjugate(which)
                            _sync_waist_pair_conj_status_labels()
                            _notify_mapping_changed()

                        ui.Label(
                            "Waist quats: optionally conjugate (invert) each bone before q_upper*q_upper2. Toggle each.",
                            height=20,
                            style={"font_size": 12, "color": 0xFFAAAAAA},
                        )
                        with ui.HStack(height=26):
                            ui.Spacer(width=8)
                            lbl_ub = ui.Label("", width=108, height=22)
                            ui.Button(
                                "Toggle",
                                width=52,
                                height=22,
                                clicked_fn=lambda: _waist_conj_toggle(0),
                            )
                            ui.Spacer(width=8)
                            lbl_ub2 = ui.Label("", width=108, height=22)
                            ui.Button(
                                "Toggle",
                                width=52,
                                height=22,
                                clicked_fn=lambda: _waist_conj_toggle(1),
                            )
                            ui.Spacer()
                        _waist_pair_conj_status_labels[0] = lbl_ub
                        _waist_pair_conj_status_labels[1] = lbl_ub2
                        _sync_waist_pair_conj_status_labels()
                    for joint_name in joint_names:
                        if joint_name not in G1_JOINT_TO_MMD:
                            continue
                        bones, euler_idx, scale = G1_JOINT_TO_MMD[joint_name]
                        val_w = 204 if joint_name in _HINGE_DETAIL_ROW_JOINTS else 72

                        def _main_hstack_row() -> tuple:
                            # 列1：G1 关节名（去掉 _joint 和 _）
                            ui.Label(
                                joint_name.replace("_joint", ""),
                                width=118,
                            )
                            # 列2：对应的 MMD 骨骼名（R_/L_ 短罗马音）
                            ui.Label(_bone_str(bones), width=102)
                            # 列3：欧拉分量索引输入框（0/1/2）
                            euler_model = ui.SimpleIntModel(euler_idx)
                            ui.IntField(model=euler_model, width=30)
                            euler_model.add_value_changed_fn(
                                lambda m, j=joint_name: _on_euler_changed(j, m)
                            )
                            ui.Spacer(width=8)
                            scale_model = ui.SimpleFloatModel(scale)
                            ui.FloatField(model=scale_model, width=50)
                            ui.Spacer(width=4)
                            ui.FloatSlider(model=scale_model, min=-3.0, max=3.0, width=112)
                            scale_model.add_value_changed_fn(
                                lambda m, j=joint_name: _on_scale_changed(j, m)
                            )
                            ui.Spacer(width=8)
                            ui.Button(
                                "Flip",
                                width=44,
                                height=22,
                                clicked_fn=lambda j=joint_name: _on_flip_scale(j),
                            )
                            ui.Spacer(width=8)
                            value_label = ui.Label("N/A", width=val_w)
                            return euler_model, scale_model, value_label

                        if joint_name in _HINGE_DETAIL_ROW_JOINTS:
                            absorb_model = ui.SimpleFloatModel(get_hinge_swing_absorb(joint_name))
                            with ui.VStack(spacing=2):
                                with ui.HStack(height=28):
                                    euler_model, scale_model, value_label = _main_hstack_row()
                                if joint_name in _KNEE_JOINT_NAMES:
                                    with ui.HStack(height=22):
                                        ui.Spacer(width=218)
                                        ui.Label("swing abs", width=72)
                                        ui.FloatField(model=absorb_model, width=50)
                                        absorb_model.add_value_changed_fn(
                                            lambda m, j=joint_name: _on_absorb_changed(j, m)
                                        )
                                        ui.Button(
                                            "AbsFlip",
                                            width=52,
                                            height=22,
                                            clicked_fn=lambda j=joint_name: _on_absorb_flip(j),
                                        )
                                        ui.Spacer()
                                else:
                                    absorb_model = None
                            joint_models[joint_name] = (
                                euler_model,
                                scale_model,
                                value_label,
                                absorb_model,
                            )
                        else:
                            with ui.HStack(height=28):
                                euler_model, scale_model, value_label = _main_hstack_row()
                            joint_models[joint_name] = (euler_model, scale_model, value_label, None)
                    ui.Spacer(height=4)  # 分类之间的间隔

        ui.Spacer(height=8)

        # ========== 底部：居中的重置按钮，高度 20 ==========
        with ui.HStack(height=20):
            ui.Spacer()
            ui.Button("Reset to Default", clicked_fn=_on_reset, width=120, height=20)
            ui.Spacer()

    transport_refs = {
        "scrub_model": scrub_model,
        "pd_drive_model": pd_drive_model,
        "audio_volume_model": audio_volume_model,
        "z_offset_enable_model": z_offset_enable_model,
        "foot_ground_comp_model": foot_ground_comp_model,
        "max_label": max_frame_label,
        "btn_prev": btn_prev,
        "btn_next": btn_next,
        "btn_pause": btn_pause,
        "btn_resume": btn_resume,
        "root_roll_euler_model": root_roll_euler_model,
        "root_pitch_euler_model": root_pitch_euler_model,
        "root_yaw_euler_model": root_yaw_euler_model,
        "root_roll_scale_model": root_roll_scale_model,
        "root_pitch_scale_model": root_pitch_scale_model,
        "root_yaw_scale_model": root_yaw_scale_model,
        "root_R_value_label": root_row_value_labels[0],
        "root_P_value_label": root_row_value_labels[1],
        "root_Y_value_label": root_row_value_labels[2],
        "root_rot_bone_label": root_rot_bone_label,
        "dance_combo": dance_combo,
        "btn_gen_z_editted": btn_gen_z_editted,
        "foot_ik_enable_model": foot_ik_enable_model,
        "foot_ik_weight_model": foot_ik_weight_model,
        "foot_ik_reach_model": foot_ik_reach_model,
        "foot_ik_leg_scale_model": foot_ik_leg_scale_model,
        "foot_ik_ankle_offset_x_model": foot_ik_ankle_offset_x_model,
        "foot_ik_ankle_offset_y_model": foot_ik_ankle_offset_y_model,
        "foot_ik_ankle_offset_z_model": foot_ik_ankle_offset_z_model,
        "foot_ik_debug_every_model": foot_ik_debug_every_model,
        "foot_ik_solver_model": foot_ik_solver_model,
        "foot_ik_solver_combo": foot_ik_solver_combo,
        "foot_ik_reg_weight_model": foot_ik_reg_weight_model,
        "sphere_map_scale_model": sphere_map_scale_model,
        "sphere_map_axis_idx_models": sphere_map_axis_idx_models,
        "sphere_map_axis_sign_models": sphere_map_axis_sign_models,
        "sphere_map_left_ref_origin_models": sphere_map_left_ref_origin_models,
        "sphere_map_right_ref_origin_models": sphere_map_right_ref_origin_models,
        "foot_ik_l_local_label": l_foot_local_label,
        "foot_ik_r_local_label": r_foot_local_label,
        "foot_ik_l_xyz_label": l_foot_xyz_label,
        "foot_ik_r_xyz_label": r_foot_xyz_label,
        "foot_ik_l_ik_target_label": l_ik_target_label,
        "foot_ik_r_ik_target_label": r_ik_target_label,
        "toe_ik_l_xyz_label": l_toe_xyz_label,
        "toe_ik_r_xyz_label": r_toe_xyz_label,
    }
    return joint_models, playback_title_label, transport_refs


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
    global _z_offset_sync_suppress_set, _foot_ik_sync_suppress_set, _foot_ik_viz_sync_suppress_set
    global _foot_ground_comp_sync_suppress_set
    global _audio_volume_sync_suppress_set
    while True:
        await omni.kit.app.get_app().next_update_async()
        if _joint_models_ref is None and _retarget_tune_refs is None:
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
            if combo is not None and btn_z is not None and _dance_entries_provider is not None:
                selected_key: str | None = None
                try:
                    entries: list[DanceUiEntry] = []
                    for item in _dance_entries_provider() or []:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            key = str(item[0]).strip()
                            if key:
                                entries.append((key, str(item[1])))
                    if entries:
                        idx = int(combo.model.get_item_value_model().as_int)
                        idx = max(0, min(len(entries) - 1, idx))
                        selected_key = entries[idx][0]
                except Exception:
                    selected_key = None
                has_editted = False
                if selected_key and _dance_z_edited_status_provider is not None:
                    try:
                        has_editted = bool(_dance_z_edited_status_provider(selected_key))
                    except Exception:
                        has_editted = False
                z_busy = False
                if _z_edit_busy_provider is not None:
                    try:
                        z_busy = bool(_z_edit_busy_provider())
                    except Exception:
                        z_busy = False
                show_gen = bool(selected_key) and not has_editted
                btn_z.visible = show_gen
                btn_z.enabled = show_gen and not z_busy
                btn_z.text = "Generating..." if z_busy else "Gen Z_edited"
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
            if _root_quat_rpy_provider is not None:
                try:
                    (sr, sp, sy), (ir, ip, iy) = _root_quat_rpy_provider()
                except Exception:
                    sr, sp, sy = 1.0, 1.0, 1.0
                    ir, ip, iy = 0, 1, 2
                _root_quat_sync_suppress_set = True
                try:
                    if abs(float(tr["root_roll_scale_model"].get_value_as_float()) - float(sr)) > 1e-6:
                        tr["root_roll_scale_model"].set_value(float(sr))
                    if abs(float(tr["root_pitch_scale_model"].get_value_as_float()) - float(sp)) > 1e-6:
                        tr["root_pitch_scale_model"].set_value(float(sp))
                    if abs(float(tr["root_yaw_scale_model"].get_value_as_float()) - float(sy)) > 1e-6:
                        tr["root_yaw_scale_model"].set_value(float(sy))
                    if int(tr["root_roll_euler_model"].get_value_as_int()) != int(ir):
                        tr["root_roll_euler_model"].set_value(int(ir))
                    if int(tr["root_pitch_euler_model"].get_value_as_int()) != int(ip):
                        tr["root_pitch_euler_model"].set_value(int(ip))
                    if int(tr["root_yaw_euler_model"].get_value_as_int()) != int(iy):
                        tr["root_yaw_euler_model"].set_value(int(iy))
                except Exception:
                    pass
                finally:
                    _root_quat_sync_suppress_set = False
            if _foot_ik_provider is not None:
                try:
                    fk = _foot_ik_provider() or {}
                except Exception:
                    fk = {}
                _foot_ik_sync_suppress_set = True
                try:
                    if "foot_ik_enable_model" in tr:
                        tr["foot_ik_enable_model"].set_value(bool(fk.get("enable", False)))
                    if "foot_ik_weight_model" in tr:
                        tr["foot_ik_weight_model"].set_value(float(fk.get("weight", 1.0)))
                    if "foot_ik_reach_model" in tr:
                        tr["foot_ik_reach_model"].set_value(float(fk.get("reach", 0.985)))
                    if "foot_ik_leg_scale_model" in tr:
                        tr["foot_ik_leg_scale_model"].set_value(float(fk.get("leg_scale", 1.0)))
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
                    if "foot_ik_solver_model" in tr:
                        solver = str(fk.get("solver", "full")).strip().lower()
                        tr["foot_ik_solver_model"].set_value(1 if solver == "planar" else 0)
                    combo = tr.get("foot_ik_solver_combo")
                    if combo is not None:
                        solver = str(fk.get("solver", "full")).strip().lower()
                        combo.model.get_item_value_model().set_value(1 if solver == "planar" else 0)
                    if "foot_ik_reg_weight_model" in tr:
                        tr["foot_ik_reg_weight_model"].set_value(float(fk.get("ik_reg_weight", 0.15)))
                except Exception:
                    pass
                finally:
                    _foot_ik_sync_suppress_set = False
            if _foot_ik_viz_provider is not None:
                try:
                    sv = _foot_ik_viz_provider() or {}
                except Exception:
                    sv = {}
                _foot_ik_viz_sync_suppress_set = True
                try:
                    _sphere_defaults = default_foot_ik_viz_config()
                    if "sphere_map_scale_model" in tr:
                        tr["sphere_map_scale_model"].set_value(float(sv.get("scale", _sphere_defaults.pos_scale)))
                    sidx = tuple(sv.get("axis_idx", _sphere_defaults.axis_idx))
                    ssig = tuple(sv.get("axis_sign", _sphere_defaults.axis_sign))
                    lorig = tuple(sv.get("left_ref_origin", _sphere_defaults.left_ref_origin_m))
                    rorig = tuple(sv.get("right_ref_origin", _sphere_defaults.right_ref_origin_m))
                    for _i, _m in enumerate(tr.get("sphere_map_axis_idx_models", ())):
                        _m.set_value(int(sidx[_i]))
                    for _i, _m in enumerate(tr.get("sphere_map_axis_sign_models", ())):
                        _m.set_value(float(ssig[_i]))
                    for _i, _m in enumerate(tr.get("sphere_map_left_ref_origin_models", ())):
                        _m.set_value(float(lorig[_i]))
                    for _i, _m in enumerate(tr.get("sphere_map_right_ref_origin_models", ())):
                        _m.set_value(float(rorig[_i]))
                except Exception:
                    pass
                finally:
                    _foot_ik_viz_sync_suppress_set = False

            if _root_rot_bone_name_provider is not None:
                try:
                    bone_nm = str(_root_rot_bone_name_provider() or "")
                except Exception:
                    bone_nm = ""
                lbl = tr.get("root_rot_bone_label")
                if lbl is not None:
                    romaji = mmd_bone_to_romaji(bone_nm) if bone_nm else "-"
                    lbl.text = f"Rot bone: {romaji}" if bone_nm else "Rot bone: (none)"

            for _rk, _ck in [
                ("root_R_value_label", "__root_rpy_deg_r"),
                ("root_P_value_label", "__root_rpy_deg_p"),
                ("root_Y_value_label", "__root_rpy_deg_y"),
            ]:
                _lw = tr.get(_rk)
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
            for _lbl_key, _xk, _yk, _zk in [
                ("foot_ik_l_local_label", "__foot_ik_l_local_x", "__foot_ik_l_local_y", "__foot_ik_l_local_z"),
                ("foot_ik_r_local_label", "__foot_ik_r_local_x", "__foot_ik_r_local_y", "__foot_ik_r_local_z"),
                ("foot_ik_l_xyz_label", "__foot_ik_l_x", "__foot_ik_l_y", "__foot_ik_l_z"),
                ("foot_ik_r_xyz_label", "__foot_ik_r_x", "__foot_ik_r_y", "__foot_ik_r_z"),
                ("foot_ik_l_ik_target_label", "__ik_target_l_x", "__ik_target_l_y", "__ik_target_l_z"),
                ("foot_ik_r_ik_target_label", "__ik_target_r_x", "__ik_target_r_y", "__ik_target_r_z"),
                ("toe_ik_l_xyz_label", "__toe_ik_l_x", "__toe_ik_l_y", "__toe_ik_l_z"),
                ("toe_ik_r_xyz_label", "__toe_ik_r_x", "__toe_ik_r_y", "__toe_ik_r_z"),
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
                for pfx, key in [("L", "__sho_left_raw"), ("R", "__sho_right_raw")]:
                    lbl = sho_raw.get(pfx)
                    if lbl is not None:
                        txt = values.get(key)
                        lbl.text = f"raw: {txt}" if isinstance(txt, str) else "raw: —"
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

        jm = _joint_models_ref
        if jm is None:
            continue
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
                    mmd_fmt = _wrap_long_hinge_text(mmd)
                    value_label.text = f"{float(v):.1f}deg sim\n{mmd_fmt}"
                else:
                    value_label.text = f"{float(v):.2f}deg"
            else:
                value_label.text = f"{float(v):.2f}deg"


_PROPERTY_DOCK_TARGETS = ("Property", "IsaacLab")
_PROPERTY_DOCK_RETRIES = 120


def _schedule_property_tab_dock(window: Any, window_title: str) -> None:
    """Dock window into Property tab group (sync deferred + async retry)."""
    try:
        import omni.ui as ui
    except ImportError:
        return

    try:
        ui.Workspace.show_window("Property", True)
    except Exception:
        pass
    try:
        window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
    except Exception:
        pass
    asyncio.ensure_future(_dock_window_to_property_tab(window, window_title))


async def _dock_window_to_property_tab(window: Any, window_title: str) -> None:
    """Retry docking until the window joins the Property / IsaacLab tab group."""
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

    for _ in range(_PROPERTY_DOCK_RETRIES):
        target_handle = None
        target_name = None
        for name in _PROPERTY_DOCK_TARGETS:
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
            try:
                window.dock_in_window(target_name, ui.DockPosition.SAME, 1.0)
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
        f"[WARN] Failed to dock '{window_title}' into Property/IsaacLab tab group "
        "(window stays floating; drag it manually or check Property panel is visible)."
    )


def create_mapping_ui():
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
        global _joint_models_ref, _playback_title_ref, _playback_transport_ref
        window = ui.Window(
            WINDOW_TITLE,
            width=620,
            height=720,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _joint_models_ref, _playback_title_ref, _playback_transport_ref = _build_mapping_window(ui)
        window.visible = True
        _window_ref.append(window)
        _schedule_property_tab_dock(window, WINDOW_TITLE)
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

    print("[INFO] G1 Joint Mapping: Window menu →", WINDOW_TITLE)
    return True


def create_retarget_tune_ui():
    """注册独立窗口 «G1 Retarget Tune»（肩/腿 basis 微调）；与映射窗口共用刷新循环。"""
    try:
        import omni.ui as ui
        from omni.kit.menu.utils import add_menu_items, MenuItemDescription
    except ImportError:
        print("[WARN] omni.ui 不可用，Retarget Tune UI 已跳过（可能为 headless 模式）")
        return None

    _tune_window_ref = []

    def _create_tune_window():
        global _retarget_tune_refs
        window = ui.Window(
            RETARGET_TUNE_WINDOW_TITLE,
            width=460,
            height=540,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _retarget_tune_refs = build_retarget_tune_window(ui, _notify_mapping_changed)
        window.visible = True
        _tune_window_ref.append(window)
        _schedule_property_tab_dock(window, RETARGET_TUNE_WINDOW_TITLE)
        return window

    def _on_tune_menu_click():
        if _tune_window_ref:
            _tune_window_ref[0].visible = True
        else:
            _create_tune_window()

    add_menu_items(
        [MenuItemDescription(name=RETARGET_TUNE_WINDOW_TITLE, onclick_fn=_on_tune_menu_click)],
        "Window",
    )

    schedule_mapping_ui_refresh_loop()

    print("[INFO] G1 Retarget Tune: Window menu →", RETARGET_TUNE_WINDOW_TITLE)
    return True
