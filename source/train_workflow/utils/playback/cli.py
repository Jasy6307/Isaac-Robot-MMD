"""CLI helpers for G1 MMD playback entry script."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import AppLauncher

if TYPE_CHECKING:
    from source.train_workflow.utils.format.csv_loader import FootIkConfig

from source.train_workflow.utils.ik.geometry import (
    G1_FOOT_IK_HIP_OFFSET_Y_M,
    G1_FOOT_IK_HIP_OFFSET_Z_M,
    G1_FOOT_IK_SHIN_LENGTH_M,
    G1_FOOT_IK_THIGH_LENGTH_M,
)


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
    """Build command-line parser for interactive playback.

    Foot IK, sphere map, and ankle ground-comp tuning live in the Mapping UI;
    batch tools (e.g. edit_csv_root_z) keep their own CLI via
    ``add_mmd_foot_ik_solver_cli_args`` / ``add_mmd_sphere_map_cli_args``.
    """
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


def parse_triplet_int(text: str, *, clamp_0_2: bool = False) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError("须恰好三个整数")
    out = tuple(int(p) for p in parts)
    if clamp_0_2:
        return tuple(max(0, min(2, v)) for v in out)  # type: ignore[return-value]
    return out  # type: ignore[return-value]


def parse_triplet_float(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError("须恰好三个浮点数")
    return float(parts[0]), float(parts[1]), float(parts[2])


def add_mmd_sphere_map_cli_args(parser: argparse.ArgumentParser) -> None:
    """Red-sphere / foot-target coordinate map (also drives leg IK target)."""
    from source.train_workflow.utils.ik.mmd_fk import (
        FOOT_IK_VIZ_AXIS_IDX,
        FOOT_IK_VIZ_AXIS_SIGN,
        FOOT_IK_VIZ_AXIS_SIGN_POSE,
        FOOT_IK_VIZ_LEFT_REF_ORIGIN_M,
        FOOT_IK_VIZ_POS_SCALE,
        FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M,
        foot_ik_viz_triplet_cli,
    )

    parser.add_argument(
        "--mmd_sphere_map_scale",
        type=float,
        default=FOOT_IK_VIZ_POS_SCALE,
        help="Foot IK panel -> Isaac world extra scale (default 1.0)",
    )
    parser.add_argument(
        "--mmd_sphere_map_axis_idx",
        type=str,
        default=foot_ik_viz_triplet_cli(FOOT_IK_VIZ_AXIS_IDX),
        help="Sphere/IK axis index x,y,z (0/1/2 each, comma-separated)",
    )
    parser.add_argument(
        "--mmd_sphere_map_axis_sign",
        type=str,
        default=foot_ik_viz_triplet_cli(FOOT_IK_VIZ_AXIS_SIGN),
        help="Sphere/IK axis sign x,y,z for dance motion (comma-separated)",
    )
    parser.add_argument(
        "--mmd_sphere_map_axis_sign_pose",
        type=str,
        default=foot_ik_viz_triplet_cli(FOOT_IK_VIZ_AXIS_SIGN_POSE),
        help="Sphere/IK axis sign x,y,z for static pose (comma-separated)",
    )
    parser.add_argument(
        "--mmd_sphere_map_left_ref_origin",
        type=str,
        default=foot_ik_viz_triplet_cli(FOOT_IK_VIZ_LEFT_REF_ORIGIN_M),
        help="Left foot Isaac-world ref origin when panel offset is zero (m)",
    )
    parser.add_argument(
        "--mmd_sphere_map_right_ref_origin",
        type=str,
        default=foot_ik_viz_triplet_cli(FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M),
        help="Right foot Isaac-world ref origin when panel offset is zero (m)",
    )


def add_mmd_foot_ik_solver_cli_args(parser: argparse.ArgumentParser) -> None:
    """Leg IK knobs for batch tools (target coords come from sphere map)."""
    parser.add_argument(
        "--mmd-foot-ik-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="mmd_foot_ik_enable",
        help="Enable leg IK override from foot IK targets (default off for batch tools)",
    )
    parser.add_argument(
        "--mmd-foot-ik-max-reach-ratio",
        type=float,
        default=1.0,
        dest="mmd_foot_ik_max_reach_ratio",
        help="Max reach ratio vs thigh+shin (default 1.0)",
    )
    parser.add_argument(
        "--mmd-foot-ik-hip-offset-y",
        type=float,
        default=G1_FOOT_IK_HIP_OFFSET_Y_M,
        dest="mmd_foot_ik_hip_offset_y",
        help="Hip lateral offset (m)",
    )
    parser.add_argument(
        "--mmd-foot-ik-hip-offset-z",
        type=float,
        default=G1_FOOT_IK_HIP_OFFSET_Z_M,
        dest="mmd_foot_ik_hip_offset_z",
        help="Hip height offset (m)",
    )
    parser.add_argument(
        "--mmd-foot-ik-thigh-length",
        type=float,
        default=G1_FOOT_IK_THIGH_LENGTH_M,
        dest="mmd_foot_ik_thigh_length",
        help="Thigh length (m)",
    )
    parser.add_argument(
        "--mmd-foot-ik-shin-length",
        type=float,
        default=G1_FOOT_IK_SHIN_LENGTH_M,
        dest="mmd_foot_ik_shin_length",
        help="Shin length (m)",
    )
    parser.add_argument(
        "--mmd-foot-ik-hip-roll-gain",
        type=float,
        default=0.85,
        dest="mmd_foot_ik_hip_roll_gain",
        help="Hip roll gain for lateral reach",
    )
    parser.add_argument(
        "--mmd-foot-ik-debug-every",
        type=int,
        default=0,
        dest="mmd_foot_ik_debug_every",
        help="Print IK debug every N frames; 0=off",
    )
    parser.add_argument(
        "--mmd-foot-ik-ik-max-iters",
        type=int,
        default=12,
        dest="mmd_foot_ik_ik_max_iters",
        help="Full IK max iterations (default 12)",
    )
    parser.add_argument(
        "--mmd-foot-ik-ik-pos-tol",
        type=float,
        default=1e-3,
        dest="mmd_foot_ik_ik_pos_tol",
        help="Full IK position tolerance in meters (default 0.001)",
    )
    parser.add_argument(
        "--mmd-foot-ik-ik-reg-weight",
        type=float,
        default=0.15,
        dest="mmd_foot_ik_ik_reg_weight",
        help="Full IK null-space FK regularization base weight",
    )
    parser.add_argument(
        "--mmd-foot-ik-ik-reg-hip-yaw",
        type=float,
        default=0.8,
        dest="mmd_foot_ik_ik_reg_hip_yaw",
        help="Extra FK regularization weight on hip_yaw (multiplier)",
    )
    parser.add_argument(
        "--mmd-foot-ik-ik-reg-ankle-roll",
        type=float,
        default=0.8,
        dest="mmd_foot_ik_ik_reg_ankle_roll",
        help="Extra FK regularization weight on ankle_roll (multiplier)",
    )


def default_playback_foot_ik_config(*, groove_pos_to_world: float = 0.1) -> FootIkConfig:
    """Defaults for interactive playback; tune live in Mapping UI."""
    from source.train_workflow.utils.format.csv_loader import FootIkConfig

    return FootIkConfig(
        enable=True,
        groove_pos_to_world=float(groove_pos_to_world),
    )


def foot_ik_viz_config_from_namespace(ns: argparse.Namespace):
    from source.train_workflow.utils.ik.mmd_fk import FootIkVizConfig, default_foot_ik_viz_config, foot_ik_viz_triplet_cli

    defaults = default_foot_ik_viz_config()
    axis_idx = parse_triplet_int(
        getattr(ns, "mmd_sphere_map_axis_idx", foot_ik_viz_triplet_cli(defaults.axis_idx)),
        clamp_0_2=True,
    )
    axis_sign = parse_triplet_float(
        getattr(ns, "mmd_sphere_map_axis_sign", foot_ik_viz_triplet_cli(defaults.axis_sign))
    )
    axis_sign_pose = parse_triplet_float(
        getattr(ns, "mmd_sphere_map_axis_sign_pose", foot_ik_viz_triplet_cli(defaults.axis_sign_pose))
    )
    left_ref = parse_triplet_float(
        getattr(ns, "mmd_sphere_map_left_ref_origin", foot_ik_viz_triplet_cli(defaults.left_ref_origin_m))
    )
    right_ref = parse_triplet_float(
        getattr(ns, "mmd_sphere_map_right_ref_origin", foot_ik_viz_triplet_cli(defaults.right_ref_origin_m))
    )
    return FootIkVizConfig(
        axis_idx=axis_idx,
        axis_sign=axis_sign,
        axis_sign_pose=axis_sign_pose,
        pos_scale=float(getattr(ns, "mmd_sphere_map_scale", defaults.pos_scale)),
        left_ref_origin_m=left_ref,
        right_ref_origin_m=right_ref,
    )


def foot_ik_config_from_namespace(ns: argparse.Namespace, *, groove_pos_to_world: float) -> FootIkConfig:
    from source.train_workflow.utils.format.csv_loader import FootIkConfig

    return FootIkConfig(
        enable=bool(getattr(ns, "mmd_foot_ik_enable", False)),
        groove_pos_to_world=float(groove_pos_to_world),
        max_reach_ratio=float(getattr(ns, "mmd_foot_ik_max_reach_ratio", 1.0)),
        leg_target_scale=float(getattr(ns, "mmd_foot_ik_leg_scale", 0.75)),
        hip_offset_y=float(getattr(ns, "mmd_foot_ik_hip_offset_y", G1_FOOT_IK_HIP_OFFSET_Y_M)),
        hip_offset_z=float(getattr(ns, "mmd_foot_ik_hip_offset_z", G1_FOOT_IK_HIP_OFFSET_Z_M)),
        thigh_length=float(getattr(ns, "mmd_foot_ik_thigh_length", G1_FOOT_IK_THIGH_LENGTH_M)),
        shin_length=float(getattr(ns, "mmd_foot_ik_shin_length", G1_FOOT_IK_SHIN_LENGTH_M)),
        hip_roll_gain=float(getattr(ns, "mmd_foot_ik_hip_roll_gain", 0.85)),
        debug_every_n_frames=max(0, int(getattr(ns, "mmd_foot_ik_debug_every", 0))),
        ik_max_iters=max(1, int(getattr(ns, "mmd_foot_ik_ik_max_iters", 20))),
        ik_pos_tol_m=float(getattr(ns, "mmd_foot_ik_ik_pos_tol", 1e-3)),
        ik_reg_weight=float(getattr(ns, "mmd_foot_ik_ik_reg_weight", 0.15)),
        ik_reg_hip_yaw=float(getattr(ns, "mmd_foot_ik_ik_reg_hip_yaw", 0.8)),
        ik_reg_ankle_roll=float(getattr(ns, "mmd_foot_ik_ik_reg_ankle_roll", 0.8)),
    )


def foot_ankle_ground_comp_config_from_namespace(ns: argparse.Namespace):
    from source.train_workflow.utils.ik.ankle_ground import (
        foot_ankle_ground_comp_config_from_namespace as _from_ns,
    )

    return _from_ns(ns)
