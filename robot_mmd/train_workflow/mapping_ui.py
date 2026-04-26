"""
G1 关节映射编辑窗口。

功能概览：
1) 在 Isaac Sim Window 菜单注册 `G1 Joint Mapping` 面板；
2) 在线调整关节映射的欧拉主轴索引与缩放系数；
3) 实时显示当前机器人关节角度，便于映射调试；
4) 保留映射重置能力，支持恢复默认配置。
"""
import asyncio
from typing import Callable

from robot_mmd.train_workflow.csv_motion_loader import (
    G1_JOINT_TO_MMD,
    update_mapping_entry,
    reset_mapping_to_default,
)

WINDOW_TITLE = "G1 Joint Mapping"
_AUTO_OPEN = False

# 外部注入：用于在 UI 中显示“当前环境下的关节值（度制）”
# 返回值建议是: dict[joint_name] = angle_deg
_joint_value_provider: Callable[[], dict[str, float]] | None = None

# 映射表被用户修改后通知主循环（例如在非播放状态下按新映射重算当前姿势）
_mapping_changed_cb: Callable[[], None] | None = None


def set_joint_value_provider(provider: Callable[[], dict[str, float]] | None) -> None:
    """设置关节值提供器，用于 UI 实时显示当前角度（deg）。"""
    global _joint_value_provider
    _joint_value_provider = provider


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
    "右ひざ": "R_KNEE",
    "左ひざ": "L_KNEE",
    "右足": "R_FOOT",
    "左足": "L_FOOT",
    "下半身": "LOWER_BODY",
    "右足首": "R_ANKLE",
    "左足首": "L_ANKLE",
    "右肩": "R_SHOU",
    "右腕": "R_WRIST",
    "左肩": "L_SHOU",
    "左腕": "L_WRIST",
    "右ひじ": "R_ELBOW",
    "左ひじ": "L_ELBOW",
    "右手首": "R_WRIST",
    "左手首": "L_WRIST",
    "上半身": "UPPER_BODY",
    "上半身2": "UPPER_BODY2",
    "首": "HEAD",
}

# Joint categories: Upper Body / Lower Body / Waist (display order, left then right)
JOINT_CATEGORIES: dict[str, list[str]] = {
    "Upper Body": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", 
        "left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint",
    ],
    "Lower Body": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
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
            _euler_model, scale_model, _ = joint_models[joint_name]
            new_scale = -float(scale_model.get_value_as_float())
            scale_model.set_value(new_scale)
            _update_mapping_from_models(joint_name)
        except Exception:
            pass

    def _on_reset():
        reset_mapping_to_default()
        _notify_mapping_changed()
        for jname, (euler_model, scale_model, _value_label) in joint_models.items():
            base = G1_JOINT_TO_MMD.get(jname)
            if base:
                euler_model.set_value(base[1])
                scale_model.set_value(base[2])

    # ========== 整体布局：垂直堆叠 ==========
    with ui.VStack(spacing=4):
        # 顶部说明：欧拉分量含义（0=roll, 1=pitch, 2=yaw）
        ui.Label("Euler: 0=roll, 1=pitch, 2=yaw", height=20)
        ui.Spacer(height=4)

        # ========== 可滚动区域：按分类展示关节列表 ==========
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2):
                for category_name, joint_names in JOINT_CATEGORIES.items():
                    # 分类标题
                    ui.Label(f"--- {category_name} ---", height=22)
                    for joint_name in joint_names:
                        if joint_name not in G1_JOINT_TO_MMD:
                            continue
                        bones, euler_idx, scale = G1_JOINT_TO_MMD[joint_name]
                        # 每行：水平布局，包含 [关节名 | MMD骨骼 | 欧拉索引 | 缩放系数]
                        with ui.HStack(height=28):
                            # 列1：G1 关节名（去掉 _joint 和 _）
                            ui.Label(
                                joint_name.replace("_joint", ""),
                                width=150,
                            )
                            # 列2：对应的 MMD 骨骼名（R_/L_ 短罗马音）
                            ui.Label(_bone_str(bones), width=130)
                            # 列3：欧拉分量索引输入框（0/1/2）
                            euler_model = ui.SimpleIntModel(euler_idx)
                            ui.IntField(model=euler_model, width=30)
                            euler_model.add_value_changed_fn(
                                lambda m, j=joint_name: _on_euler_changed(j, m)
                            )
                            ui.Spacer(width=8)  # 索引与系数输入框之间的间隔
                            # 列4：缩放系数输入框
                            scale_model = ui.SimpleFloatModel(scale)
                            ui.FloatField(model=scale_model, width=50)
                            scale_model.add_value_changed_fn(
                                lambda m, j=joint_name: _on_scale_changed(j, m)
                            )
                            ui.Spacer(width=8)  # 系数与 Flip 之间的间隔
                            # 列5：Flip — 缩放系数取反
                            ui.Button(
                                "Flip",
                                width=44,
                                height=22,
                                clicked_fn=lambda j=joint_name: _on_flip_scale(j),
                            )
                            ui.Spacer(width=8)  # Flip 与关节值之间的间隔
                            # 列6：当前关节值（deg，只读）
                            value_label = ui.Label("N/A", width=80)
                            joint_models[joint_name] = (euler_model, scale_model, value_label)
                    ui.Spacer(height=4)  # 分类之间的间隔

        ui.Spacer(height=8)

        # ========== 底部：居中的重置按钮，高度 20 ==========
        with ui.HStack(height=20):
            ui.Spacer()
            ui.Button("Reset to Default", clicked_fn=_on_reset, width=120, height=20)
            ui.Spacer()

    return joint_models


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
    _refresh_started = False

    def _create_window():
        window = ui.Window(WINDOW_TITLE, width=560, height=600)
        with window.frame:
            nonlocal _joint_models_ref
            _joint_models_ref = _build_mapping_window(ui)
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
            try:
                values = _joint_value_provider() or {}
            except Exception:
                values = {}
            for jname, (_euler_model, _scale_model, value_label) in _joint_models_ref.items():
                v = values.get(jname) if isinstance(values, dict) else None
                if v is None:
                    value_label.text = "N/A"
                else:
                    value_label.text = f"{float(v):.2f}deg"

    asyncio.ensure_future(_refresh_loop())

    print("[INFO] G1 Joint Mapping window will open automatically (or use Window menu)")
    return True
