"""
G1 肩 / 腿 Retarget Tune 独立窗口（MMD→G1 基变换前的 Rz·Ry·Rx 微调）。

与关节欧拉轴映射窗口分离；仍需同一 ``ui.mapping.set_joint_value_provider`` 提供 raw 调试字符串。
"""

from __future__ import annotations

from typing import Any, Callable

from robot_mmd.train_workflow.retarget_unitreeG1 import (
    get_leg_tune_axes_deg,
    get_tune_axes_deg as get_sho_tune_axes_deg,
    reset_leg_tune_axes,
    reset_tune_axes as reset_sho_tune_axes,
    set_leg_tune_axes_deg,
    set_tune_axes_deg as set_sho_tune_axes_deg,
)

RETARGET_TUNE_WINDOW_TITLE = "G1 Retarget Tune"


def build_retarget_tune_window(ui: Any, notify_mapping_changed: Callable[[], None]) -> dict[str, Any]:
    """构建 Retarget Tune 垂直布局；返回供刷新循环使用的 ``sho_raw_labels`` / ``leg_raw_labels`` 引用字典。"""
    _sho_tune_models: dict[str, Any] = {}
    _sho_raw_labels: dict[str, Any] = {}

    def _push_sho_tune(side: str) -> None:
        try:
            pfx = "L" if side == "left" else "R"
            rx = float(_sho_tune_models[f"{pfx}_rx"].get_value_as_float())
            ry = float(_sho_tune_models[f"{pfx}_ry"].get_value_as_float())
            rz = float(_sho_tune_models[f"{pfx}_rz"].get_value_as_float())
            set_sho_tune_axes_deg(side, rx, ry, rz)
            notify_mapping_changed()
        except Exception:
            pass

    def _flip_sho_tune(side: str, axis: str) -> None:
        try:
            pfx = "L" if side == "left" else "R"
            key = f"{pfx}_{axis}"
            m = _sho_tune_models[key]
            m.set_value(-float(m.get_value_as_float()))
            _push_sho_tune(side)
        except Exception:
            pass

    def _reset_sho_tune(side: str | None) -> None:
        try:
            reset_sho_tune_axes(side)
            for s, pfx in [("left", "L"), ("right", "R")]:
                if side is not None and s != side:
                    continue
                rx, ry, rz = get_sho_tune_axes_deg(s)
                _sho_tune_models[f"{pfx}_rx"].set_value(rx)
                _sho_tune_models[f"{pfx}_ry"].set_value(ry)
                _sho_tune_models[f"{pfx}_rz"].set_value(rz)
            notify_mapping_changed()
        except Exception:
            pass

    _leg_tune_models: dict[str, Any] = {}
    _leg_raw_labels: dict[str, Any] = {}

    def _push_leg_tune(side: str) -> None:
        try:
            pfx = "L" if side == "left" else "R"
            rx = float(_leg_tune_models[f"{pfx}_rx"].get_value_as_float())
            ry = float(_leg_tune_models[f"{pfx}_ry"].get_value_as_float())
            rz = float(_leg_tune_models[f"{pfx}_rz"].get_value_as_float())
            set_leg_tune_axes_deg(side, rx, ry, rz)
            notify_mapping_changed()
        except Exception:
            pass

    def _flip_leg_tune(side: str, axis: str) -> None:
        try:
            pfx = "L" if side == "left" else "R"
            key = f"{pfx}_{axis}"
            m = _leg_tune_models[key]
            m.set_value(-float(m.get_value_as_float()))
            _push_leg_tune(side)
        except Exception:
            pass

    def _reset_leg_tune(side: str | None) -> None:
        try:
            reset_leg_tune_axes(side)
            for s, pfx in [("left", "L"), ("right", "R")]:
                if side is not None and s != side:
                    continue
                rx, ry, rz = get_leg_tune_axes_deg(s)
                _leg_tune_models[f"{pfx}_rx"].set_value(rx)
                _leg_tune_models[f"{pfx}_ry"].set_value(ry)
                _leg_tune_models[f"{pfx}_rz"].set_value(rz)
            notify_mapping_changed()
        except Exception:
            pass

    with ui.VStack(spacing=4):
        ui.Label(
            "--- Shoulder Retarget Tune ---",
            height=20,
            style={"font_size": 15, "font_style": "bold", "color": 0xFF88FFAA},
        )
        ui.Label(
            "Tune = Rz(rz)*Ry(ry)*Rx(rx) extra rotation on basis. Start with 0; try ±90 if axis wrong.",
            height=18,
            style={"font_size": 12, "color": 0xFFAAAAAA},
        )

        for side, pfx, label in [("left", "L", "L-Sho Tune"), ("right", "R", "R-Sho Tune")]:
            init_rx, init_ry, init_rz = get_sho_tune_axes_deg(side)
            m_rx = ui.SimpleFloatModel(init_rx)
            m_ry = ui.SimpleFloatModel(init_ry)
            m_rz = ui.SimpleFloatModel(init_rz)
            _sho_tune_models[f"{pfx}_rx"] = m_rx
            _sho_tune_models[f"{pfx}_ry"] = m_ry
            _sho_tune_models[f"{pfx}_rz"] = m_rz

            with ui.VStack(spacing=2):
                with ui.HStack(height=24):
                    ui.Label(label, width=80)
                    ui.Label("Rx", width=18, style={"color": 0xFFFF8888})
                    ui.FloatField(model=m_rx, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_sho_tune(s, "rx"),
                    )
                    ui.Spacer(width=4)
                    ui.Label("Ry", width=18, style={"color": 0xFF88FF88})
                    ui.FloatField(model=m_ry, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_sho_tune(s, "ry"),
                    )
                    ui.Spacer(width=4)
                    ui.Label("Rz", width=18, style={"color": 0xFF8888FF})
                    ui.FloatField(model=m_rz, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_sho_tune(s, "rz"),
                    )
                    ui.Spacer(width=4)
                    ui.Button("Rst", width=32, height=22, clicked_fn=lambda s=side: _reset_sho_tune(s))
                with ui.HStack(height=18):
                    ui.Spacer(width=80)
                    raw_lbl = ui.Label("raw: —", width=380, style={"color": 0xFFCCCCCC, "font_size": 12})
                    _sho_raw_labels[pfx] = raw_lbl

            for m in (m_rx, m_ry, m_rz):
                m.add_value_changed_fn(lambda _m, s=side: _push_sho_tune(s))

        with ui.HStack(height=22):
            ui.Spacer()
            ui.Button("Reset Both Sides", width=110, height=20, clicked_fn=lambda: _reset_sho_tune(None))
            ui.Spacer()
        ui.Spacer(height=4)

        ui.Label(
            "--- Leg Retarget Tune ---",
            height=20,
            style={"font_size": 15, "font_style": "bold", "color": 0xFF88CCFF},
        )
        ui.Label(
            "Tune = Rz(rz)*Ry(ry)*Rx(rx) on leg basis (applies to hip+ankle). Start with 0; try ±90 first.",
            height=18,
            style={"font_size": 12, "color": 0xFFAAAAAA},
        )

        for side, pfx, label in [("left", "L", "L-Leg Tune"), ("right", "R", "R-Leg Tune")]:
            init_rx, init_ry, init_rz = get_leg_tune_axes_deg(side)
            m_rx = ui.SimpleFloatModel(init_rx)
            m_ry = ui.SimpleFloatModel(init_ry)
            m_rz = ui.SimpleFloatModel(init_rz)
            _leg_tune_models[f"{pfx}_rx"] = m_rx
            _leg_tune_models[f"{pfx}_ry"] = m_ry
            _leg_tune_models[f"{pfx}_rz"] = m_rz

            with ui.VStack(spacing=2):
                with ui.HStack(height=24):
                    ui.Label(label, width=80)
                    ui.Label("Rx", width=18, style={"color": 0xFFFF8888})
                    ui.FloatField(model=m_rx, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_leg_tune(s, "rx"),
                    )
                    ui.Spacer(width=4)
                    ui.Label("Ry", width=18, style={"color": 0xFF88FF88})
                    ui.FloatField(model=m_ry, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_leg_tune(s, "ry"),
                    )
                    ui.Spacer(width=4)
                    ui.Label("Rz", width=18, style={"color": 0xFF8888FF})
                    ui.FloatField(model=m_rz, width=52)
                    ui.Button(
                        "Flip",
                        width=36,
                        height=22,
                        clicked_fn=lambda s=side: _flip_leg_tune(s, "rz"),
                    )
                    ui.Spacer(width=4)
                    ui.Button("Rst", width=32, height=22, clicked_fn=lambda s=side: _reset_leg_tune(s))
                with ui.HStack(height=18):
                    ui.Spacer(width=80)
                    hip_lbl = ui.Label("hip: —", width=190, style={"color": 0xFFCCCCCC, "font_size": 12})
                    ank_lbl = ui.Label("ank: —", width=190, style={"color": 0xFFCCCCCC, "font_size": 12})
                    _leg_raw_labels[f"{pfx}_hip"] = hip_lbl
                    _leg_raw_labels[f"{pfx}_ank"] = ank_lbl

            for m in (m_rx, m_ry, m_rz):
                m.add_value_changed_fn(lambda _m, s=side: _push_leg_tune(s))

        with ui.HStack(height=22):
            ui.Spacer()
            ui.Button("Reset Both Sides", width=110, height=20, clicked_fn=lambda: _reset_leg_tune(None))
            ui.Spacer()

        ui.Spacer(height=4)
        ui.Label(
            "Joint angles / playback: «G1 Joint Mapping». Raw lines update when sim runs.",
            height=22,
            style={"font_size": 11, "color": 0xFF888888},
        )

    return {
        "sho_raw_labels": _sho_raw_labels,
        "leg_raw_labels": _leg_raw_labels,
    }
