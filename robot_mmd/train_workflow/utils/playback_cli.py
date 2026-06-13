"""CLI helpers for G1 MMD playback entry script."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def apply_app_window_kit_flags(ns: argparse.Namespace) -> None:
    """Merge Omniverse main-window carb settings into kit_args."""
    fragments: list[str] = []
    app_window_width = getattr(ns, "app_window_width", None)
    if app_window_width is None:
        app_window_width = int(getattr(ns, "width", 1920))
    app_window_height = getattr(ns, "app_window_height", None)
    if app_window_height is None:
        app_window_height = int(getattr(ns, "height", 1080))
    if app_window_width is not None:
        fragments.append(f"--/app/window/width={int(app_window_width)}")
        fragments.append(f"--/persistent/app/window/width={int(app_window_width)}")
    if app_window_height is not None:
        fragments.append(f"--/app/window/height={int(app_window_height)}")
        fragments.append(f"--/persistent/app/window/height={int(app_window_height)}")
    mode = getattr(ns, "app_window_mode", "normal") or "normal"
    if mode == "maximized":
        fragments.append("--/app/window/maximized=true")
        fragments.append("--/persistent/app/window/maximized=true")
        fragments.append("--/app/window/fullscreen=false")
        fragments.append("--/persistent/app/window/fullscreen=false")
    elif mode == "fullscreen":
        fragments.append("--/app/window/fullscreen=true")
        fragments.append("--/app/window/maximized=false")
        fragments.append("--/persistent/app/window/maximized=false")
    else:
        fragments.append("--/app/window/maximized=false")
        fragments.append("--/persistent/app/window/maximized=false")
        fragments.append("--/app/window/fullscreen=false")
        fragments.append("--/persistent/app/window/fullscreen=false")
    existing = str(getattr(ns, "kit_args", "") or "").strip()
    ns.kit_args = (existing + " " + " ".join(fragments)).strip()


def build_arg_parser(pose_dir: str) -> argparse.ArgumentParser:
    """Build command-line parser for playback."""
    parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
    parser.add_argument(
        "--pose_cycle_key",
        type=str,
        default="P",
        help=f"按该键按序播放姿势 CSV（目录固定为 {pose_dir}，默认键 P）",
    )
    parser.add_argument("--play_speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument(
        "--groove_pos_to_world",
        type=float,
        default=0.1,
        help="CSV 根骨平移 pos 映射到仿真米制时的缩放：默认 0.1（常见为分米→米）。若为厘米设 0.01，若已是米则 1.0",
    )
    parser.add_argument(
        "--mmd_center_to_root_offset_local",
        type=str,
        default="0,0,0.0",
        help=(
            "articulation root 局部系中「从 VMD/CSV 的センター指向骨盆(机械 root)」的向量(米)，"
            "逗号分隔 x,y,z；会按本帧目标根四元数旋到世界系后加到根平移。"
            "MMD 里センター常在 root 沿躯干向下约 0.2m，可试 0,0,0.2 或按需改轴。默认 0 表示不补偿。"
        ),
    )
    parser.add_argument("--sim_fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    parser.add_argument(
        "--pd_profile",
        type=str,
        choices=("deploy", "isaaclab"),
        default="deploy",
        help=(
            "Actuator PD: deploy=deploy legs/ankle + stiffer arms/waist for dance tracking (default); "
            "isaaclab=G1_29DOF locomanipulation defaults."
        ),
    )
    parser.add_argument(
        "--mmd_knee_hinge_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 MMD ひざ的非铰链 swing 分量并回父骨(足)，由 hip 三轴吸收；默认开启",
    )
    parser.add_argument(
        "--mmd_foot_ik_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 VMD 足IK目标驱动腿部 IK 覆盖（默认开启）",
    )
    parser.add_argument(
        "--mmd_foot_ik_scale",
        type=float,
        default=1.0,
        help="足IK位移缩放（默认 1.0）",
    )
    parser.add_argument(
        "--mmd_foot_ik_weight",
        type=float,
        default=1.0,
        help="FK/IK 混合权重，0=纯FK，1=纯IK（默认 1.0）",
    )
    parser.add_argument(
        "--mmd_foot_ik_max_reach_ratio",
        type=float,
        default=0.985,
        help="IK 最远可达比例（相对 thigh+shin，默认 0.985）",
    )
    parser.add_argument(
        "--mmd_foot_ik_axis_idx",
        type=str,
        default="0,2,1",
        help="MMD->foot target 轴索引 x,y,z（逗号分隔，每项 0/1/2）",
    )
    parser.add_argument(
        "--mmd_foot_ik_axis_sign",
        type=str,
        default="-1,-1,1",
        help="MMD->foot target 轴符号 x,y,z（建议 ±1，逗号分隔）",
    )
    parser.add_argument(
        "--mmd_foot_ik_axis_sign_pose",
        type=str,
        default="-1,1,1",
        help="静态 pose 时的轴符号 x,y,z（逗号分隔）",
    )
    parser.add_argument(
        "--mmd_foot_ik_left_ref_local",
        type=str,
        default="0.0,0.095,-0.42",
        help="左脚参考点（root local，米）x,y,z",
    )
    parser.add_argument(
        "--mmd_foot_ik_right_ref_local",
        type=str,
        default="0.0,-0.095,-0.42",
        help="右脚参考点（root local，米）x,y,z",
    )
    parser.add_argument(
        "--mmd_foot_ik_hip_offset_y",
        type=float,
        default=0.095,
        help="髋关节左右偏置（米）",
    )
    parser.add_argument(
        "--mmd_foot_ik_hip_offset_z",
        type=float,
        default=0.0,
        help="髋关节高度偏置（米）",
    )
    parser.add_argument(
        "--mmd_foot_ik_thigh_length",
        type=float,
        default=0.213,
        help="大腿长度（米）",
    )
    parser.add_argument(
        "--mmd_foot_ik_shin_length",
        type=float,
        default=0.213,
        help="小腿长度（米）",
    )
    parser.add_argument(
        "--mmd_foot_ik_hip_roll_gain",
        type=float,
        default=0.85,
        help="侧向 hip roll 增益",
    )
    parser.add_argument(
        "--mmd_foot_ik_debug_every",
        type=int,
        default=0,
        help="每 N 帧打印 IK debug；0=关闭",
    )
    for _flag, _default, _help in (
        ("--width", 1920, "视口/生成图像宽度（像素）；默认 1280"),
        ("--height", 1080, "视口/生成图像高度（像素）；默认 720"),
        (
            "--app_window_width",
            None,
            "Isaac 主窗口宽度（像素）；不传则沿用 Isaac 默认或上次持久化尺寸",
        ),
        (
            "--app_window_height",
            None,
            "Isaac 主窗口高度（像素）；不传则沿用 Isaac 默认或上次持久化尺寸",
        ),
    ):
        parser.add_argument(_flag, type=int, default=_default, help=_help)
    parser.add_argument(
        "--app_window_mode",
        type=str,
        choices=("normal", "maximized", "fullscreen"),
        default="normal",
        metavar="MODE",
        help=(
            "Isaac 主窗口形态（写入 carb）：normal 普通、maximized 最大化、fullscreen 全屏；默认 normal"
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_center_to_root_offset(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError("须恰好三个数")
    return float(parts[0]), float(parts[1]), float(parts[2])
