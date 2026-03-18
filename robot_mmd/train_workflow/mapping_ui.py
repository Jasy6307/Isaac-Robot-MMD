"""
G1 关节映射编辑 UI - 在 Isaac Sim 中提供可编辑窗口，修改欧拉分量索引和缩放系数
"""
import asyncio

from robot_mmd.train_workflow.csv_motion_loader import (
    G1_JOINT_TO_MMD,
    update_mapping_entry,
    reset_mapping_to_default,
)

WINDOW_TITLE = "G1 Joint Mapping"


def _build_mapping_window(ui):
    """构建映射编辑窗口内容。

    布局结构（自上而下）：
    ┌─────────────────────────────────────────────────────────┐
    │ Euler: 0=roll, 1=pitch, 2=yaw                           │
    ├─────────────────────────────────────────────────────────┤
    │ [可滚动] 关节名 | MMD骨骼 | 欧拉索引 | 缩放系数           │
    │   right knee | 右ひざ   | [1]     | [1.0]               │
    │   left knee  | 左ひざ   | [1]     | [1.0]               │
    │   ...                                                   │
    ├─────────────────────────────────────────────────────────┤
    │                    [Reset to Default]                   │
    └─────────────────────────────────────────────────────────┘
    """
    joint_models: dict[str, tuple] = {}

    def _bone_str(bones) -> str:
        if isinstance(bones, list):
            return " + ".join(bones)
        return str(bones)

    def _on_euler_changed(joint_name: str, model):
        try:
            euler_idx = max(0, min(2, int(model.get_value_as_int())))
            scale_model = joint_models[joint_name][1]
            scale = float(scale_model.get_value_as_float())
            update_mapping_entry(joint_name, euler_idx, scale)
        except Exception:
            pass

    def _on_scale_changed(joint_name: str, model):
        try:
            scale = float(model.get_value_as_float())
            euler_model = joint_models[joint_name][0]
            euler_idx = max(0, min(2, int(euler_model.get_value_as_int())))
            update_mapping_entry(joint_name, euler_idx, scale)
        except Exception:
            pass

    def _on_reset():
        reset_mapping_to_default()
        for jname, (euler_model, scale_model) in joint_models.items():
            base = G1_JOINT_TO_MMD.get(jname)
            if base:
                euler_model.set_value(base[1])
                scale_model.set_value(base[2])

    # ========== 整体布局：垂直堆叠 ==========
    with ui.VStack(spacing=4):
        # 顶部说明：欧拉分量含义（0=roll, 1=pitch, 2=yaw）
        ui.Label("Euler: 0=roll, 1=pitch, 2=yaw", height=20)
        ui.Spacer(height=4)

        # ========== 可滚动区域：关节列表 ==========
        with ui.ScrollingFrame():
            with ui.VStack(spacing=2):
                for joint_name in sorted(G1_JOINT_TO_MMD.keys()):
                    bones, euler_idx, scale = G1_JOINT_TO_MMD[joint_name]
                    # 每行：水平布局，包含 [关节名 | MMD骨骼 | 欧拉索引 | 缩放系数]
                    with ui.HStack(height=28):
                        # 列1：G1 关节名（去掉 _joint 和 _）
                        ui.Label(
                            joint_name.replace("_joint", "").replace("_", " "),
                            width=180,
                        )
                        # 列2：对应的 MMD 骨骼名（可能为 "肩 + 腕" 组合）
                        ui.Label(_bone_str(bones), width=120)
                        # 列3：欧拉分量索引输入框（0/1/2）
                        euler_model = ui.SimpleIntModel(euler_idx)
                        ui.IntField(model=euler_model, width=50)
                        euler_model.add_value_changed_fn(
                            lambda m, j=joint_name: _on_euler_changed(j, m)
                        )
                        # 列4：缩放系数输入框
                        scale_model = ui.SimpleFloatModel(scale)
                        ui.FloatField(model=scale_model, width=80)
                        scale_model.add_value_changed_fn(
                            lambda m, j=joint_name: _on_scale_changed(j, m)
                        )
                        joint_models[joint_name] = (euler_model, scale_model)

        ui.Spacer(height=8)

        # ========== 底部：右对齐的重置按钮 ==========
        with ui.HStack():
            ui.Spacer()  # 左侧弹性空白，将按钮推到右侧
            ui.Button("Reset to Default", clicked_fn=_on_reset, width=120)


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
        window = ui.Window(WINDOW_TITLE, width=500, height=600)
        with window.frame:
            _build_mapping_window(ui)
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

    asyncio.ensure_future(_auto_open())

    print("[INFO] G1 Joint Mapping window will open automatically (or use Window menu)")
    return True
