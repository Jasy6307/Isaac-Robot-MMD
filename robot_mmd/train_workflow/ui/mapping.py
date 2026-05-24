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

WINDOW_TITLE = "G1 Joint Mapping"
_AUTO_OPEN = False

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
_root_quat_rpy_provider: Callable[
    [], tuple[tuple[float, float, float], tuple[int, int, int]]
] | None = None
_root_quat_rpy_setter: Callable[
    [tuple[float, float, float], tuple[int, int, int]], None
] | None = None
_root_rot_bone_name_provider: Callable[[], str] | None = None

# True while refresh assigns scrub IntField from sim; blocks value_changed -> seek storm
_scrub_sync_suppress_seek: bool = False
_root_quat_sync_suppress_set: bool = False

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
    "上半身": "UPPER_B",
    "上半身2": "UPPER_B2",
    "首": "HEAD",
}

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
    "Upper Body (Right)": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint",
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
        """将 MMD 骨骼名转为罗马音显示（Isaac 中日文显示异常）"""
        def _to_romaji(s: str) -> str:
            return MMD_BONE_TO_ROMAJI.get(s, s)

        if isinstance(bones, list):
            return " + ".join(_to_romaji(b) for b in bones)
        return _to_romaji(str(bones))

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

        with ui.HStack(height=28):
            playback_title_label = ui.Label("Playback: (idle)", width=188, height=22)
            ui.IntField(model=scrub_model, width=64, height=22)
            max_frame_label = ui.Label("/ -", width=40, height=22)
            ui.Spacer(width=4)
            btn_prev = ui.Button("Prev", width=42, height=24, clicked_fn=_on_prev_frame_click)
            btn_next = ui.Button("Next", width=42, height=24, clicked_fn=_on_next_frame_click)
            btn_pause = ui.Button("Pause", width=56, height=24, clicked_fn=_on_pause_click)
            btn_resume = ui.Button("Resume", width=58, height=24, clicked_fn=_on_resume_click)
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

    global _scrub_sync_suppress_seek, _root_quat_sync_suppress_set
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

            if _root_rot_bone_name_provider is not None:
                try:
                    bone_nm = str(_root_rot_bone_name_provider() or "")
                except Exception:
                    bone_nm = ""
                lbl = tr.get("root_rot_bone_label")
                if lbl is not None:
                    romaji = MMD_BONE_TO_ROMAJI.get(bone_nm, bone_nm) if bone_nm else "-"
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
        window = ui.Window(WINDOW_TITLE, width=620, height=720)
        with window.frame:
            _joint_models_ref, _playback_title_ref, _playback_transport_ref = _build_mapping_window(ui)
        window.visible = True
        _window_ref.append(window)
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
        window = ui.Window(RETARGET_TUNE_WINDOW_TITLE, width=460, height=540)
        with window.frame:
            _retarget_tune_refs = build_retarget_tune_window(ui, _notify_mapping_changed)
        window.visible = True
        _tune_window_ref.append(window)
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
