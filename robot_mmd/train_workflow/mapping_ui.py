"""
G1 关节映射编辑窗口。

功能概览：
1) 在 Isaac Sim Window 菜单注册 `G1 Joint Mapping` 面板；
2) 在线调整关节映射的欧拉主轴索引与缩放系数；
3) 实时显示当前机器人关节角度，便于映射调试；
4) 保留映射重置能力，支持恢复默认配置。
"""
import asyncio
from typing import Any, Callable

from robot_mmd.train_workflow.csv_motion_loader import (
    G1_JOINT_TO_MMD,
    get_elbow_shoulder_axis_scale,
    get_elbow_shoulder_yaw_source_axis_index,
    get_hinge_swing_absorb,
    reset_mapping_to_default,
    set_elbow_shoulder_axis_scale,
    set_elbow_shoulder_yaw_source_axis_index,
    set_hinge_swing_absorb,
    update_mapping_entry,
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

# Transport: pause/resume, seek to frame index (only while clip loaded in run_stand)
_playback_toggle_cb: Callable[[], None] | None = None
_playback_seek_cb: Callable[[int], None] | None = None
_root_quat_scale_provider: Callable[[], tuple[float, float, float]] | None = None
_root_quat_scale_setter: Callable[[tuple[float, float, float]], None] | None = None

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


def set_root_quat_scale_callbacks(
    provider: Callable[[], tuple[float, float, float]] | None,
    setter: Callable[[tuple[float, float, float]], None] | None,
) -> None:
    """设置 root 姿态 R/P/Y scale 的读取与写入回调。"""
    global _root_quat_scale_provider, _root_quat_scale_setter
    _root_quat_scale_provider = provider
    _root_quat_scale_setter = setter


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


# MMD 骨骼日文 -> 短罗马音（R_/L_ 替代 migi_/hidari_，Isaac 中日文显示异常）
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
            _euler_model, scale_model, _value, _abs, _elbow_axis, _src_label = joint_models[joint_name]
            new_scale = -float(scale_model.get_value_as_float())
            scale_model.set_value(new_scale)
            _update_mapping_from_models(joint_name)
        except Exception:
            pass

    def _on_absorb_changed(joint_name: str, _model):
        try:
            _eu, _sc, _vl, absorb_model, _elbow_axis, _src_label = joint_models[joint_name]
            if absorb_model is None:
                return
            set_hinge_swing_absorb(joint_name, float(absorb_model.get_value_as_float()))
            _notify_mapping_changed()
        except Exception:
            pass

    def _on_absorb_flip(joint_name: str):
        try:
            _eu, _sc, _vl, absorb_model, _elbow_axis, _src_label = joint_models[joint_name]
            if absorb_model is None:
                return
            absorb_model.set_value(-float(absorb_model.get_value_as_float()))
            _on_absorb_changed(joint_name, absorb_model)
        except Exception:
            pass

    def _on_elbow_axis_changed(joint_name: str) -> None:
        try:
            _eu, _sc, _vl, _ab, elbow_yaw_model, _src_label = joint_models[joint_name]
            if elbow_yaw_model is None:
                return
            set_elbow_shoulder_axis_scale(
                joint_name,
                (
                    1.0,
                    1.0,
                    float(elbow_yaw_model.get_value_as_float()),
                ),
            )
            _notify_mapping_changed()
        except Exception:
            pass

    def _on_elbow_axis_flip(joint_name: str) -> None:
        try:
            _eu, _sc, _vl, _ab, elbow_yaw_model, _src_label = joint_models[joint_name]
            if elbow_yaw_model is None:
                return
            elbow_yaw_model.set_value(-float(elbow_yaw_model.get_value_as_float()))
            _on_elbow_axis_changed(joint_name)
        except Exception:
            pass

    def _on_elbow_source_toggle(joint_name: str) -> None:
        try:
            _eu, _sc, _vl, _ab, _eyaw, src_label = joint_models[joint_name]
            if src_label is None:
                return
            cur = get_elbow_shoulder_yaw_source_axis_index(joint_name)
            nxt = 0 if cur == 2 else 2
            set_elbow_shoulder_yaw_source_axis_index(joint_name, nxt)
            src_label.text = "Src: Roll" if nxt == 0 else "Src: Yaw"
            _notify_mapping_changed()
        except Exception:
            pass

    def _on_reset():
        reset_mapping_to_default()
        _notify_mapping_changed()
        for jname, (euler_model, scale_model, _value_label, absorb_model, elbow_yaw_model, src_label) in joint_models.items():
            base = G1_JOINT_TO_MMD.get(jname)
            if base:
                euler_model.set_value(base[1])
                scale_model.set_value(base[2])
            if absorb_model is not None:
                absorb_model.set_value(1.0)
            if elbow_yaw_model is not None:
                set_elbow_shoulder_axis_scale(jname, (1.0, 1.0, 1.0))
                elbow_yaw_model.set_value(1.0)
            if src_label is not None:
                set_elbow_shoulder_yaw_source_axis_index(jname, 2)
                src_label.text = "Src: Yaw"

    # ========== 整体布局：垂直堆叠 ==========
    with ui.VStack(spacing=4):
        scrub_model = ui.SimpleIntModel(0)
        root_roll_scale_model = ui.SimpleFloatModel(1.0)
        root_pitch_scale_model = ui.SimpleFloatModel(1.0)
        root_yaw_scale_model = ui.SimpleFloatModel(1.0)

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

        def _push_root_quat_scale() -> None:
            if _root_quat_sync_suppress_set:
                return
            if _root_quat_scale_setter is None:
                return
            try:
                _root_quat_scale_setter(
                    (
                        float(root_roll_scale_model.get_value_as_float()),
                        float(root_pitch_scale_model.get_value_as_float()),
                        float(root_yaw_scale_model.get_value_as_float()),
                    )
                )
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
        with ui.HStack(height=24):
            ui.Label("Root R/P/Y", width=70)
            ui.FloatField(model=root_roll_scale_model, width=44)
            ui.Button(
                "FlipR",
                width=46,
                height=22,
                clicked_fn=lambda: (
                    root_roll_scale_model.set_value(-float(root_roll_scale_model.get_value_as_float())),
                    _push_root_quat_scale(),
                ),
            )
            ui.FloatField(model=root_pitch_scale_model, width=44)
            ui.Button(
                "FlipP",
                width=46,
                height=22,
                clicked_fn=lambda: (
                    root_pitch_scale_model.set_value(-float(root_pitch_scale_model.get_value_as_float())),
                    _push_root_quat_scale(),
                ),
            )
            ui.FloatField(model=root_yaw_scale_model, width=44)
            ui.Button(
                "FlipY",
                width=46,
                height=22,
                clicked_fn=lambda: (
                    root_yaw_scale_model.set_value(-float(root_yaw_scale_model.get_value_as_float())),
                    _push_root_quat_scale(),
                ),
            )
            ui.Spacer()
        root_roll_scale_model.add_value_changed_fn(lambda _m: _push_root_quat_scale())
        root_pitch_scale_model.add_value_changed_fn(lambda _m: _push_root_quat_scale())
        root_yaw_scale_model.add_value_changed_fn(lambda _m: _push_root_quat_scale())
        ui.Spacer(height=2)
        ui.Label("Euler axis: 0=roll, 1=pitch, 2=yaw", height=20)
        ui.Spacer(height=2)

        # ========== 可滚动区域：按分类展示关节列表 ==========
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2):
                for category_name, joint_names in JOINT_CATEGORIES.items():
                    # 分类标题
                    ui.Label(
                        f"--- {category_name} ---",
                        height=22,
                        style={"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00},
                    )
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
                                if joint_name in _ELBOW_JOINT_NAMES:
                                    sx, sy, sz = get_elbow_shoulder_axis_scale(joint_name)
                                    src_idx = get_elbow_shoulder_yaw_source_axis_index(joint_name)
                                    myaw = ui.SimpleFloatModel(sz)
                                    with ui.HStack(height=22):
                                        ui.Spacer(width=218)
                                        ui.Label("sho yaw", width=72)
                                        ui.FloatField(model=myaw, width=56)
                                        ui.Button(
                                            "FlipY",
                                            width=48,
                                            height=22,
                                            clicked_fn=lambda j=joint_name: _on_elbow_axis_flip(j),
                                        )
                                        src_text = "Src: Roll" if src_idx == 0 else "Src: Yaw"
                                        src_label = ui.Label(src_text, width=66)
                                        ui.Button(
                                            "Swap",
                                            width=40,
                                            height=22,
                                            clicked_fn=lambda j=joint_name: _on_elbow_source_toggle(j),
                                        )
                                        ui.Spacer()
                                    myaw.add_value_changed_fn(lambda _m, j=joint_name: _on_elbow_axis_changed(j))
                                    elbow_axis_models = myaw
                                else:
                                    elbow_axis_models = None
                                    src_label = None
                            joint_models[joint_name] = (
                                euler_model,
                                scale_model,
                                value_label,
                                absorb_model,
                                elbow_axis_models,
                                src_label,
                            )
                        else:
                            with ui.HStack(height=28):
                                euler_model, scale_model, value_label = _main_hstack_row()
                            joint_models[joint_name] = (euler_model, scale_model, value_label, None, None, None)
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
        "root_roll_scale_model": root_roll_scale_model,
        "root_pitch_scale_model": root_pitch_scale_model,
        "root_yaw_scale_model": root_yaw_scale_model,
    }
    return joint_models, playback_title_label, transport_refs


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
    _joint_models_ref = None
    _playback_title_ref: Any | None = None
    _playback_transport_ref: dict[str, Any] | None = None
    _refresh_started = False

    def _create_window():
        window = ui.Window(WINDOW_TITLE, width=620, height=700)
        with window.frame:
            nonlocal _joint_models_ref, _playback_title_ref, _playback_transport_ref
            _joint_models_ref, _playback_title_ref, _playback_transport_ref = _build_mapping_window(
                ui
            )
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

    async def _refresh_loop():
        nonlocal _refresh_started
        if _refresh_started:
            return
        _refresh_started = True
        while True:
            await omni.kit.app.get_app().next_update_async()
            if _joint_models_ref is None:
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
                global _scrub_sync_suppress_seek
                global _root_quat_sync_suppress_set
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
                if _root_quat_scale_provider is not None:
                    try:
                        sr, sp, sy = _root_quat_scale_provider()
                    except Exception:
                        sr, sp, sy = 1.0, 1.0, 1.0
                    _root_quat_sync_suppress_set = True
                    try:
                        if abs(float(tr["root_roll_scale_model"].get_value_as_float()) - float(sr)) > 1e-6:
                            tr["root_roll_scale_model"].set_value(float(sr))
                        if abs(float(tr["root_pitch_scale_model"].get_value_as_float()) - float(sp)) > 1e-6:
                            tr["root_pitch_scale_model"].set_value(float(sp))
                        if abs(float(tr["root_yaw_scale_model"].get_value_as_float()) - float(sy)) > 1e-6:
                            tr["root_yaw_scale_model"].set_value(float(sy))
                    except Exception:
                        pass
                    finally:
                        _root_quat_sync_suppress_set = False
            for jname, (_euler_model, _scale_model, value_label, _absorb_model, _elbow_axis, _src_label) in _joint_models_ref.items():
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

    asyncio.ensure_future(_refresh_loop())

    print("[INFO] G1 Joint Mapping window will open automatically (or use Window menu)")
    return True
