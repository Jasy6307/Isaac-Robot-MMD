"""
G1 Root RPY and per-joint Euler-axis mapping window.

Adjusts root Roll/Pitch/Yaw remapping and each G1 joint's MMD Euler index / scale
(including Flip and knee/elbow swing absorb). Separate from playback / dance options.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from source.train_workflow.utils.retarget.joint_axis_map import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from source.train_workflow.utils.format.csv_loader import (
    G1_JOINT_TO_MMD,
    get_hinge_swing_absorb,
    get_waist_upper_pair_quat_conjugate,
    reset_mapping_to_default,
    set_hinge_swing_absorb,
    toggle_waist_upper_pair_quat_conjugate,
    update_mapping_entry,
)

JOINT_RPY_WINDOW_TITLE = "G1 Joint RPY Mapping"
_AUTO_OPEN = True

_joint_models_ref: dict[str, tuple] | None = None
_joint_rpy_refs: dict[str, Any] | None = None

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

_KNEE_JOINT_NAMES = frozenset({"left_knee_joint", "right_knee_joint"})
_KNEE_MMD_SUFFIX = "__knee_mmd"
_ELBOW_JOINT_NAMES = frozenset({"left_elbow_joint", "right_elbow_joint"})
_ELBOW_MMD_SUFFIX = "__elbow_mmd"
_HINGE_DETAIL_ROW_JOINTS = _KNEE_JOINT_NAMES | _ELBOW_JOINT_NAMES

JOINT_CATEGORIES: dict[str, list[str]] = {
    "Upper Body (Left)": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
    ],
    "Hand (Left)": [
        "lh_thumb_cmc_yaw",
        "lh_thumb_cmc_pitch",
        "lh_thumb_ip",
        "lh_index_mcp_pitch",
        "lh_index_dip",
        "lh_middle_mcp_pitch",
        "lh_middle_dip",
        "lh_ring_mcp_pitch",
        "lh_ring_dip",
        "lh_pinky_mcp_pitch",
        "lh_pinky_dip",
    ],
    "Upper Body (Right)": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
    ],
    "Hand (Right)": [
        "rh_thumb_cmc_yaw",
        "rh_thumb_cmc_pitch",
        "rh_thumb_ip",
        "rh_index_mcp_pitch",
        "rh_index_dip",
        "rh_middle_mcp_pitch",
        "rh_middle_dip",
        "rh_ring_mcp_pitch",
        "rh_ring_dip",
        "rh_pinky_mcp_pitch",
        "rh_pinky_dip",
    ],
    "Lower Body (Left)": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "Lower Body (Right)": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    "Waist": [
        "waist_pitch_joint",
        "waist_roll_joint",
        "waist_yaw_joint",
    ],
}

ROOT_RPY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("root_Roll", "out·R", "root_roll"),
    ("root_Pitch", "out·P", "root_pitch"),
    ("root_Yaw", "out·Y", "root_yaw"),
)

_SECTION_HEADER_STYLE = {"font_size": 17, "font_style": "bold", "color": 0xFFFFFF00}
_waist_pair_conj_status_labels: list[Any] = [None, None]


def mmd_bone_to_romaji(name: str) -> str:
    return MMD_BONE_TO_ROMAJI.get(str(name or ""), str(name or ""))


def wrap_long_hinge_text(mmd_line: str) -> str:
    s = mmd_line.strip()
    if len(s) <= 44:
        return s
    mid = len(s) // 2
    brk = s.rfind(" ", 8, mid + 14)
    if brk <= 0:
        brk = mid
    return s[:brk] + "\n" + s[brk:].lstrip()


def _sync_waist_pair_conj_status_labels() -> None:
    c0, c1 = get_waist_upper_pair_quat_conjugate()
    if _waist_pair_conj_status_labels[0] is not None:
        _waist_pair_conj_status_labels[0].text = f"Upper: {'conj' if c0 else 'as-is'}"
    if _waist_pair_conj_status_labels[1] is not None:
        _waist_pair_conj_status_labels[1].text = f"Upper2: {'conj' if c1 else 'as-is'}"


def build_joint_rpy_mapping_window(
    ui: Any,
    notify_mapping_changed: Callable[[], None],
) -> tuple[dict[str, tuple], dict[str, Any]]:
    """Build scrollable Root RPY + joint mapping UI. Returns (joint_models, rpy_refs)."""
    from source.train_workflow.ui.mmd_config_ui import push_root_quat_rpy_ui

    joint_models: dict[str, tuple] = {}

    def _build_bold_section_header(_collapsed: bool, title: str) -> None:
        with ui.HStack(height=22):
            ui.Label(f"--- {title} ---", style=_SECTION_HEADER_STYLE)
            ui.Spacer()

    def _bone_str(bones) -> str:
        if isinstance(bones, list):
            return " + ".join(mmd_bone_to_romaji(b) for b in bones)
        return mmd_bone_to_romaji(str(bones))

    def _on_euler_changed(joint_name: str, _model):
        _update_mapping_from_models(joint_name)

    def _on_scale_changed(joint_name: str, _model):
        _update_mapping_from_models(joint_name)

    def _update_mapping_from_models(joint_name: str) -> None:
        try:
            euler_model = joint_models[joint_name][0]
            scale_model = joint_models[joint_name][1]
            euler_idx = max(0, min(2, int(euler_model.get_value_as_int())))
            scale = float(scale_model.get_value_as_float())
            update_mapping_entry(joint_name, euler_idx, scale)
            notify_mapping_changed()
        except Exception:
            pass

    def _on_flip_scale(joint_name: str):
        try:
            _euler_model, scale_model, _value, _abs = joint_models[joint_name]
            scale_model.set_value(-float(scale_model.get_value_as_float()))
            _update_mapping_from_models(joint_name)
        except Exception:
            pass

    def _on_absorb_changed(joint_name: str, _model):
        try:
            _eu, _sc, _vl, absorb_model = joint_models[joint_name]
            if absorb_model is None:
                return
            set_hinge_swing_absorb(joint_name, float(absorb_model.get_value_as_float()))
            notify_mapping_changed()
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

    _drr, _drp, _dry = MMD_ROOT_QUAT_RPY_SCALE_DEFAULT
    _dir, _dip, _diy = MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT
    root_roll_euler_model = ui.SimpleIntModel(int(_dir))
    root_pitch_euler_model = ui.SimpleIntModel(int(_dip))
    root_yaw_euler_model = ui.SimpleIntModel(int(_diy))
    root_roll_scale_model = ui.SimpleFloatModel(float(_drr))
    root_pitch_scale_model = ui.SimpleFloatModel(float(_drp))
    root_yaw_scale_model = ui.SimpleFloatModel(float(_dry))

    def _push_root_quat_rpy() -> None:
        push_root_quat_rpy_ui(
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

    def _on_reset():
        reset_mapping_to_default()
        notify_mapping_changed()
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
        push_root_quat_rpy_ui(
            (float(_sr), float(_sp), float(_sy)),
            (int(_ir), int(_ip), int(_iy)),
            notify=False,
        )
        _sync_waist_pair_conj_status_labels()

    with ui.VStack(spacing=4):
        with ui.HStack(height=28):
            now_playing_label = ui.Label("Now playing: (idle)", height=22)
            ui.Spacer(width=8)
            now_playing_frame_label = ui.Label("0", width=36, height=22)
            now_playing_max_label = ui.Label("/ -", width=48, height=22)
            ui.Spacer()
        ui.Spacer(height=4)
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
                        ui.FloatSlider(model=scale_model, min=-3.0, max=3.0, width=78)
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
                    collapsed_default = category_name.startswith("Hand ")
                    with ui.CollapsableFrame(
                        category_name,
                        collapsed=collapsed_default,
                        height=0,
                        build_header_fn=_build_bold_section_header,
                    ):
                        with ui.VStack(spacing=2):
                            if category_name == "Waist":
                                global _waist_pair_conj_status_labels

                                def _waist_conj_toggle(which: int) -> None:
                                    toggle_waist_upper_pair_quat_conjugate(which)
                                    _sync_waist_pair_conj_status_labels()
                                    notify_mapping_changed()

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
                                    ui.Label(joint_name.replace("_joint", ""), width=118)
                                    ui.Label(_bone_str(bones), width=102)
                                    euler_model = ui.SimpleIntModel(euler_idx)
                                    ui.IntField(model=euler_model, width=30)
                                    euler_model.add_value_changed_fn(
                                        lambda m, j=joint_name: _on_euler_changed(j, m)
                                    )
                                    ui.Spacer(width=8)
                                    scale_model = ui.SimpleFloatModel(scale)
                                    ui.FloatField(model=scale_model, width=50)
                                    ui.Spacer(width=4)
                                    ui.FloatSlider(model=scale_model, min=-3.0, max=3.0, width=78)
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
                                    joint_models[joint_name] = (
                                        euler_model,
                                        scale_model,
                                        value_label,
                                        None,
                                    )
                    ui.Spacer(height=4)

        ui.Spacer(height=8)
        with ui.HStack(height=20):
            ui.Spacer()
            ui.Button("Reset to Default", clicked_fn=_on_reset, width=120, height=20)
            ui.Spacer()

    rpy_refs = {
        "now_playing_label": now_playing_label,
        "now_playing_frame_label": now_playing_frame_label,
        "now_playing_max_label": now_playing_max_label,
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
    return joint_models, rpy_refs


def create_joint_rpy_mapping_ui() -> bool | None:
    """Register «G1 Joint RPY Mapping» under Window menu; shares mapping refresh loop."""
    try:
        import omni.ui as ui
        import omni.kit.app
        from omni.kit.menu.utils import add_menu_items, MenuItemDescription
    except ImportError:
        print("[WARN] omni.ui unavailable; Joint RPY mapping UI skipped (headless mode?)")
        return None

    from source.train_workflow.ui.mmd_config_ui import (
        _notify_mapping_changed,
        _schedule_property_tab_dock,
        schedule_mapping_ui_refresh_loop,
    )

    _window_ref: list[Any] = []

    def _create_window():
        global _joint_models_ref, _joint_rpy_refs
        window = ui.Window(
            JOINT_RPY_WINDOW_TITLE,
            width=620,
            height=720,
            dock_preference=ui.DockPreference.MAIN,
        )
        with window.frame:
            _joint_models_ref, _joint_rpy_refs = build_joint_rpy_mapping_window(
                ui, _notify_mapping_changed
            )
        window.visible = True
        _window_ref.append(window)
        _schedule_property_tab_dock(window, JOINT_RPY_WINDOW_TITLE)
        return window

    def _on_menu_click():
        if _window_ref:
            _window_ref[0].visible = True
        else:
            _create_window()

    add_menu_items(
        [MenuItemDescription(name=JOINT_RPY_WINDOW_TITLE, onclick_fn=_on_menu_click)],
        "Window",
    )

    async def _auto_open():
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        _create_window()

    if _AUTO_OPEN:
        asyncio.ensure_future(_auto_open())

    schedule_mapping_ui_refresh_loop()
    print("[INFO] G1 Joint RPY Mapping: Window menu →", JOINT_RPY_WINDOW_TITLE)
    return True
