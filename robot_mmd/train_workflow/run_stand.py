# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 站立任务动作回放主脚本。

功能概览：
1) 舞蹈由 dances_config.yaml（模块内 DANCES_CONFIG_PATH）登记：键、CSV、可选音频；读 pose 目录 P 键循环；
2) 支持关节映射 UI，实时显示当前关节角度；
3) 有 audio 的 dance 播 WAV，与动作同一「逻辑帧时间轴」；安装 pygame 后支持 UI 暂停/帧跳转时伴音同步（无则仍用 winsound，无暂停对准）；
4) 在重置和切换动作时维护控制参考姿态，避免姿态回弹。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 仅给类型检查器用；运行脚本时不会执行，故不会级联 import omni.*
    from isaaclab.assets import Articulation

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDIA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../media"))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

POSE_DIR = os.path.join(_MEDIA_DIR, "pose")
DANCES_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "dances_config.yaml")

# 映射 UI：最近一帧插值后的 MMD 骨骼数据（用于膝铰链分解行显示）
_MMD_UI_LAST_INTERP_FRAME_DATA: dict[str, dict] | None = None


def _apply_app_window_kit_flags(ns: argparse.Namespace) -> None:
    """将 Omniverse 主窗口（系统壳层）的 carb 设置并入 kit_args。"""
    fragments: list[str] = []
    app_window_width = getattr(ns, "app_window_width", None)
    app_window_height = getattr(ns, "app_window_height", None)
    if app_window_width is not None:
        fragments.append(f"--/app/window/width={int(app_window_width)}")
        fragments.append(f"--/persistent/app/window/width={int(app_window_width)}")
    if app_window_height is not None:
        fragments.append(f"--/app/window/height={int(app_window_height)}")
        fragments.append(f"--/persistent/app/window/height={int(app_window_height)}")
    if getattr(ns, "app_window_maximized", False):
        fragments.append("--/app/window/maximized=true")
        fragments.append("--/persistent/app/window/maximized=true")
    if getattr(ns, "app_window_fullscreen", False):
        fragments.append("--/app/window/fullscreen=true")
    if getattr(ns, "app_window_no_decorations", False):
        fragments.append("--/app/window/noDecorations=true")
        fragments.append("--/persistent/app/window/noDecorations=true")
    if not fragments:
        return
    existing = str(getattr(ns, "kit_args", "") or "").strip()
    ns.kit_args = (existing + " " + " ".join(fragments)).strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
    parser.add_argument(
        "--pose_cycle_key",
        type=str,
        default="P",
        help=f"按该键按序播放姿势 CSV（目录固定为 {POSE_DIR}，默认键 P）",
    )
    parser.add_argument(
        "--motion_playback",
        action="store_true",
        default=True,
        help="动作回放模式：不固定根链接、禁用重力、增加阻尼",
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
        default="0,0,0",
        help=(
            "articulation root 局部系中「从 VMD/CSV 的センター指向骨盆(机械 root)」的向量(米)，"
            "逗号分隔 x,y,z；会按本帧目标根四元数旋到世界系后加到根平移。"
            "MMD 里センター常在 root 沿躯干向下约 0.2m，可试 0,0,0.2 或按需改轴。默认 0 表示不补偿。"
        ),
    )
    parser.add_argument("--sim_fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    parser.add_argument(
        "--export_isaac_csv",
        type=str,
        default="targets.csv",
        help="某段动作播放结束后，将每帧写入 Isaac 的根位姿(wxyz)与关节角(弧度)导出为该路径的 CSV；空则关闭",
    )
    parser.add_argument(
        "--mmd_knee_hinge_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 MMD ひざ的非铰链 swing 分量并回父骨(足)，由 hip 三轴吸收；默认开启",
    )
    # 视口渲染分辨率（与操作系统窗口外框不同；见 --app_window_*）
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="视口/生成图像宽度（像素）；默认 1280",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="视口/生成图像高度（像素）；默认 720",
    )
    # 操作系统主窗口：统一走 carb --/app/window/*，避免被其它默认值覆盖
    parser.add_argument(
        "--app_window_width",
        type=int,
        default=None,
        help="Isaac 主窗口宽度（像素）；不传则沿用 Isaac 默认或上次持久化尺寸",
    )
    parser.add_argument(
        "--app_window_height",
        type=int,
        default=None,
        help="Isaac 主窗口高度（像素）；不传则沿用 Isaac 默认或上次持久化尺寸",
    )
    parser.add_argument(
        "--app_window_maximized",
        action="store_true",
        help="启动时最大化主窗口（标题栏仍在，除非另开无边框）",
    )
    parser.add_argument(
        "--app_window_fullscreen",
        action="store_true",
        help="启动时全屏主窗口",
    )
    parser.add_argument(
        "--app_window_no_decorations",
        action="store_true",
        help="无边框窗口（无原生标题栏/系统装饰；按需与 maximized 组合）",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_arg_parser()
args_cli = parser.parse_args()
args_cli.device = "cpu"

try:
    _co = [p.strip() for p in str(args_cli.mmd_center_to_root_offset_local or "").split(",")]
    if len(_co) != 3:
        raise ValueError("须恰好三个数")
    args_cli.mmd_center_to_root_offset_local_xyz = (
        float(_co[0]),
        float(_co[1]),
        float(_co[2]),
    )
except Exception as exc:
    raise SystemExit(
        f"--mmd_center_to_root_offset_local 需为 x,y,z 三个浮点数（逗号分隔），例如 0,0,0.2: {exc}"
    ) from exc

_apply_app_window_kit_flags(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robot_mmd.my_task  # noqa: F401
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from robot_mmd.train_workflow.csv_motion_loader import (
    build_joint_positions_from_frame,
    elbow_hinge_mapping_ui_extra,
    get_bone_frame_lists,
    get_frame_indices,
    interpolate_bone,
    knee_hinge_mapping_ui_extra,
    load_csv_motion,
)
from robot_mmd.train_workflow.mapping_ui import (
    create_mapping_ui,
    set_joint_value_provider,
    set_mapping_changed_callback,
    set_playback_status_provider,
    set_playback_transport_callbacks,
    set_root_quat_scale_callbacks,
)
from robot_mmd.train_workflow.trans_util import (
    coerce_quat,
    mmd_root_offset_quat_to_world,
    quat_from_euler_xyz,
    quat_mul,
    quat_normalize,
    quat_to_euler_xyz,
    root_quat_from_state_row,
    rotate_vec_by_quat_wxyz,
)
import audio_util

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


def _robot_root_row_clone(env: Any) -> Any | None:
    """取 env 内机器人 root_state_w 第一行 CPU 副本；不可用则 None。"""
    rs = getattr(env.unwrapped.scene["robot"].data, "root_state_w", None)
    if torch.is_tensor(rs) and rs.shape[1] >= 7:
        return rs[0].detach().cpu().clone()
    return None


def _format_playback_log_label(label: str) -> str:
    """将内部动作标签格式化为播放日志短名，如 dance [Y : deepbluetown]。"""
    m = re.match(r"^dance\[([^\]]+)\]\s+(.+)$", label)
    if m:
        key, path = m.group(1), m.group(2).strip()
        base = os.path.splitext(os.path.basename(path))[0]
        return f"dance [{key} : {base}]"
    m = re.match(r"^pose\[([^\]]+)\]\s+(.+)$", label)
    if m:
        idx, path = m.group(1), m.group(2).strip()
        base = os.path.splitext(os.path.basename(path))[0]
        return f"pose [{idx} : {base}]"
    return label


def _load_motion(filepath: str) -> tuple | None:
    """加载 CSV 动作，返回 (frames, frame_list, bone_frame_lists, all_bones) 或 None"""
    if not os.path.isfile(filepath):
        return None
    frames = load_csv_motion(filepath)
    frame_list = get_frame_indices(frames)
    all_bones = set()
    for f in frames.values():
        all_bones.update(f.keys())
    bone_frame_lists = get_bone_frame_lists(frames, frame_list, all_bones)
    return (frames, frame_list, bone_frame_lists, all_bones)


def _load_pose_motion_dir(pose_dir: str) -> list[tuple[str, str, tuple]]:
    """读取目录下全部 CSV，返回 [(文件名, 全路径, motion_data)]。"""
    csv_files = _list_csv_files(pose_dir, "pose")
    out: list[tuple[str, str, tuple]] = []
    for name in csv_files:
        fullpath = os.path.join(pose_dir, name)
        data = _load_motion(fullpath)
        if data is None:
            print(f"[WARN] 无法加载 CSV: {fullpath}")
            continue
        out.append((name, fullpath, data))
        print(f"[INFO] 已加载 pose: {name}，共 {len(data[1])} 帧")
    if not out:
        print(f"[WARN] pose 目录没有可用 CSV: {pose_dir}")
    return out


def _list_csv_files(dir_path: str, label: str) -> list[str]:
    """列出目录中的 CSV 文件（按文件名排序）。"""
    if not os.path.isdir(dir_path):
        print(f"[WARN] {label} 目录不存在: {dir_path}")
        return []
    csv_files = sorted(
        f for f in os.listdir(dir_path) if f.lower().endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))
    )
    if not csv_files:
        print(f"[WARN] {label} 目录没有可用 CSV: {dir_path}")
    return csv_files


def _resolve_path_under_media(relative: str) -> str:
    """将配置中的相对路径解析到 ``robot_mmd/media/`` 下。"""
    rel = (relative or "").strip().replace("\\", "/")
    if not rel:
        return ""
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(_MEDIA_DIR, rel))


def _load_dances_from_yaml(
    config_path: str,
) -> tuple[dict[str, tuple[str, tuple]], dict[str, str]]:
    """从 YAML 加载舞蹈：``dance_motion_by_key`` 与仅含「有有效音频」条目的 ``dance_wav_by_key``。"""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "读取舞蹈配置需要 PyYAML: pip install pyyaml "
            f"(见 {os.path.join(_SCRIPT_DIR, 'dances_requirements.txt')})"
        ) from e

    raw = (config_path or "").strip()
    if not raw:
        print("[WARN] 舞蹈配置文件路径为空，无 dance 键")
        return {}, {}
    if os.path.isfile(raw):
        path = os.path.normpath(os.path.abspath(raw))
    else:
        p1 = os.path.join(_SCRIPT_DIR, raw)
        p2 = os.path.abspath(raw)
        if os.path.isfile(p1):
            path = os.path.normpath(p1)
        elif os.path.isfile(p2):
            path = os.path.normpath(p2)
        else:
            print(f"[WARN] 未找到舞蹈配置 YAML: {raw}，无 dance 键")
            return {}, {}

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not doc:
        print(f"[WARN] 舞蹈配置为空: {path}")
        return {}, {}
    items = doc.get("dances")
    if not isinstance(items, list):
        print(f"[WARN] 舞蹈配置缺少 ``dances`` 列表: {path}")
        return {}, {}

    motion_by_key: dict[str, tuple[str, tuple]] = {}
    wav_by_key: dict[str, str] = {}
    for i, ent in enumerate(items):
        if not isinstance(ent, dict):
            print(f"[WARN] dances[{i}] 非映射，已跳过")
            continue
        raw_key = ent.get("key")
        if raw_key is None or str(raw_key).strip() == "":
            print(f"[WARN] dances[{i}] 无 key，已跳过")
            continue
        key = str(raw_key).strip().upper()[:1]
        if key in motion_by_key:
            print(f"[WARN] 舞蹈键重复 [{key}]，后项已忽略: {ent.get('id', i)}")
            continue
        motion_rel = ent.get("motion")
        if not motion_rel or not str(motion_rel).strip():
            print(f"[WARN] dances[{i}] 无 motion，已跳过 key={key}")
            continue
        csv_p = _resolve_path_under_media(str(motion_rel).strip())
        if not os.path.isfile(csv_p):
            print(f"[WARN] 未找到 dance 键 [{key}] 的 CSV: {csv_p}")
            continue
        data = _load_motion(csv_p)
        if data is None:
            print(f"[WARN] 无法加载 dance 键 [{key}]: {csv_p}")
            continue
        label = ent.get("id") or ent.get("label")
        brief = f" [{label}]" if label else ""
        print(f"[INFO] 已绑定 dance 键 [{key}] -> {os.path.basename(csv_p)}（{len(data[1])} 帧）{brief}")
        motion_by_key[key] = (os.path.basename(csv_p), data)
        raw_audio = ent.get("audio", None)
        if raw_audio is None or str(raw_audio).strip() == "":
            continue
        ap = _resolve_path_under_media(str(raw_audio).strip())
        if os.path.isfile(ap):
            wav_by_key[key] = ap
        else:
            print(f"[WARN] 舞蹈 [{key}] 的音频不存在，将不播伴音: {ap}")
    return motion_by_key, wav_by_key


def _compute_action_for_frame(
    frame: int,
    current_frames: Any,
    current_bone_frame_lists: dict[str, list[int]],
    current_all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    knee_hinge_projection: bool = True,
) -> tuple[Any | None, dict[str, dict]]:
    """根据帧号插值得到动作（不平滑，每帧即用目标动作）。

    返回 (action_delta 或 None, 本帧插值后的 MMD 骨骼 dict)，供映射 UI 显示膝部分解。
    """
    frame_data: dict[str, dict] = {}
    for bone in current_all_bones:
        d = interpolate_bone(frame, bone, current_frames, current_bone_frame_lists.get(bone))
        if d is not None:
            frame_data[bone] = d

    if not frame_data or joint_names is None or default_joint_pos is None:
        return None, frame_data

    target_pos = build_joint_positions_from_frame(
        frame_data,
        joint_names,
        default_joint_pos,
        knee_hinge_projection=knee_hinge_projection,
    )
    target_action = (target_pos - default_joint_pos) / action_scale
    return target_action.copy(), frame_data


def _build_joint_pos_deg_cache(joint_names: list[str], joint_pos_cmd: Any) -> dict[str, float]:
    """将关节弧度数组转为 UI 使用的角度缓存。"""
    return {j: float(deg) for j, deg in zip(joint_names, joint_pos_cmd * (180.0 / math.pi))}


def _apply_joint_state_instant(env: Any, joint_pos_cmd: Any, joint_ids: Any) -> bool:
    """将关节状态直接写入仿真（瞬间到位）。成功返回 True。"""
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]
    joint_pos_tensor = torch.tensor(joint_pos_cmd, dtype=torch.float32, device=device).unsqueeze(0)
    joint_pos_tensor = joint_pos_tensor.repeat(num_envs, 1)
    joint_vel_tensor = torch.zeros_like(joint_pos_tensor)

    try:
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor, joint_ids=joint_ids)
        return True
    except TypeError:
        return False


def _apply_root_pos_instant(env: Any, root_pos_xyz: tuple[float, float, float], root_quat_wxyz: Any = None) -> bool:
    """将机器人根位姿直接写入仿真（含根线/角速度清零）。成功返回 True。

    fix_root_link=False 的浮动基座下，若只改位置、保留上一时刻根速度，物理子步会积分出漂移。
    因此优先写完整 root_state，并把线速度/角速度置 0，避免“每帧跟新 center 仍慢慢偏走”。
    """
    robot: Articulation = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]

    root_state = robot.data.root_state_w
    fallback_wxyz = root_state[0, 3:7].detach().cpu().tolist()
    qwxyz = quat_normalize(coerce_quat(root_quat_wxyz, fallback_wxyz))

    root_pose = torch.tensor(
        [root_pos_xyz[0], root_pos_xyz[1], root_pos_xyz[2], qwxyz[0], qwxyz[1], qwxyz[2], qwxyz[3]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    root_pose = root_pose.repeat(num_envs, 1)

    # 优先写 13 维 root_state：位置+姿态+（强制）线/角速度为 0
    state = robot.data.root_state_w.clone()
    state[:, 0:3] = root_pose[:, 0:3]
    state[:, 3:7] = root_pose[:, 3:7]
    if state[:, 3:7].abs().sum() < 1e-6:
        state[:, 3:7] = 0.0
        state[:, 6] = 1.0
    state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(state)
    return True


def _motion_is_static_pose(frames: Any) -> bool:
    """单时间帧 CSV（姿势表）；多帧为舞蹈/动作片段。"""
    return len(get_frame_indices(frames)) <= 1


def _get_csv_root_quat(frame: int, frames: Any, bone_frame_lists: dict[str, list[int]]) -> list[float] | None:
    """从 CSV 当前帧提取根朝向四元数（wxyz）。"""
    # 优先使用“有连续关键帧”的根骨骼，避免误选只在 0 帧存在的常量骨骼（例如某些数据里的センター）。
    # 先尝试下半身（骨盆语义）参与根朝向，再回退到传统根骨候选。
    candidates = ("下半身", "グルーブ", "センター親", "腰", "センター")

    for require_dynamic in (True, False):
        for bone in candidates:
            keyframes = bone_frame_lists.get(bone) or []
            if require_dynamic and len(keyframes) <= 1:
                continue
            d = interpolate_bone(frame, bone, frames, keyframes)
            if d is None:
                continue
            quat_wxyz = d.get("quat_wxyz")
            if quat_wxyz is None or len(quat_wxyz) != 4:
                continue
            try:
                return quat_normalize([float(v) for v in quat_wxyz])
            except Exception:
                continue
    return None


def _interpolate_mmd_root_translation_bone(
    frame: int,
    frames: Any,
    bone_frame_lists: dict[str, list[int]],
) -> tuple[str | None, dict | None]:
    """根在 MMD 中的平移轨迹。默认 グルーブ 优先；若 センター 关键帧远多于 グルーブ（数据把位移写在 センター、グルーブ 仅占位 0），则改优先 センター。

    否则会出现：O 的 you_are_important（グルーブ 满轨迹）根跟着动，Y 的 deepbluetown（グルーブ 常静、センター 在动）根几乎不动、像「锁住身体」。"""
    g_list = bone_frame_lists.get("グルーブ") or []
    c_list = bone_frame_lists.get("センター") or []
    if c_list and len(c_list) > len(g_list):
        order: tuple[str, ...] = ("センター", "グルーブ")
    else:
        order = ("グルーブ", "センター")
    for bone in order:
        d = interpolate_bone(frame, bone, frames, bone_frame_lists.get(bone))
        if d is not None and "pos" in d:
            return bone, d
    return None, None


@dataclass
class MotionRootTrackState:
    """动作切换时缓存的仿真根位姿锚点：平移为重置/切换后 root 位置，朝向用于与 CSV 根四元数复合。"""

    root_origin_pos: tuple[float, float, float] | None = None
    root_quat_wxyz: list[float] | None = None


def _compute_targets_for_motion_frame(
    frame: int,
    frames: Any,
    bone_frame_lists: dict[str, list[int]],
    all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    groove_pos_to_world: float,
    robot: Any,
    state: MotionRootTrackState,
    root_snapshot_row: Any | None = None,
    knee_hinge_projection: bool = True,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[Any, tuple[float, float, float] | None, list[float] | None, Any, str | None]:
    """计算本帧关节目标、根位姿、动作向量；最后一项为用于平移的 MMD 根骨名（无则 None）。"""
    global _MMD_UI_LAST_INTERP_FRAME_DATA
    result, interp_fd = _compute_action_for_frame(
        frame,
        frames,
        bone_frame_lists,
        all_bones,
        joint_names,
        default_joint_pos,
        action_scale,
        knee_hinge_projection=knee_hinge_projection,
    )
    _MMD_UI_LAST_INTERP_FRAME_DATA = interp_fd
    if result is not None:
        joint_pos_cmd = default_joint_pos + action_scale * result
    else:
        joint_pos_cmd = default_joint_pos.copy()

    target_root_pos: tuple[float, float, float] | None = None
    target_root_quat_wxyz: list[float] | None = None
    mmd_root_trans_bone: str | None = None

    root_bone_name, root_mmd = _interpolate_mmd_root_translation_bone(frame, frames, bone_frame_lists)
    if root_mmd is not None and "pos" in root_mmd:
        mmd_root_trans_bone = root_bone_name
        try:
            gx, gy, gz = root_mmd["pos"]
            mmd_pos = (float(gx), float(gy), float(gz))

            if state.root_origin_pos is None:
                if root_snapshot_row is not None:
                    state.root_origin_pos = (
                        float(root_snapshot_row[0].item()),
                        float(root_snapshot_row[1].item()),
                        float(root_snapshot_row[2].item()),
                    )
                    state.root_quat_wxyz = root_quat_from_state_row(root_snapshot_row)
                else:
                    root_state = getattr(robot.data, "root_state_w", None)
                    if torch.is_tensor(root_state) and root_state.shape[1] >= 7:
                        state.root_origin_pos = (
                            float(root_state[0, 0].item()),
                            float(root_state[0, 1].item()),
                            float(root_state[0, 2].item()),
                        )
                        state.root_quat_wxyz = root_quat_from_state_row(root_state[0])

            is_pose = _motion_is_static_pose(frames)

            if state.root_origin_pos is not None:
                s = float(groove_pos_to_world)
                dx = mmd_pos[0] * s
                dy = mmd_pos[1] * s
                dz = mmd_pos[2] * s
                ox, oy, oz = (
                    state.root_origin_pos[0],
                    state.root_origin_pos[1],
                    state.root_origin_pos[2],
                )
                # 姿势：+ 号组合；多帧舞蹈：- 号 + 轴交换（MMD→Isaac）。
                if is_pose:
                    target_root_pos = (ox - dx, oy + dz, oz + dy)
                else:
                    target_root_pos = (ox - dx, oy - dz, oz - dy)
                target_root_quat_wxyz = list(state.root_quat_wxyz) if state.root_quat_wxyz else None
                csv_root_quat_wxyz = _get_csv_root_quat(frame, frames, bone_frame_lists)
                if csv_root_quat_wxyz is not None and state.root_quat_wxyz is not None:
                    q_w = mmd_root_offset_quat_to_world(csv_root_quat_wxyz)
                    sx, sy, sz = root_quat_rpy_scale
                    if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9 or abs(sz - 1.0) > 1e-9:
                        rr, rp, ry = quat_to_euler_xyz(q_w)
                        q_w = quat_from_euler_xyz(rr * float(sx), rp * float(sy), ry * float(sz))
                    target_root_quat_wxyz = quat_normalize(quat_mul(q_w, state.root_quat_wxyz))
                off_l = mmd_center_to_root_offset_local_xyz
                if (
                    target_root_pos is not None
                    and target_root_quat_wxyz is not None
                    and (abs(off_l[0]) > 1e-12 or abs(off_l[1]) > 1e-12 or abs(off_l[2]) > 1e-12)
                ):
                    dv = rotate_vec_by_quat_wxyz(target_root_quat_wxyz, off_l)
                    target_root_pos = (
                        target_root_pos[0] + dv[0],
                        target_root_pos[1] + dv[1],
                        target_root_pos[2] + dv[2],
                    )
        except Exception:
            pass

    return joint_pos_cmd, target_root_pos, target_root_quat_wxyz, result, mmd_root_trans_bone


def _write_isaac_applied_motion_csv(
    out_path: str,
    motion_data: tuple,
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
    groove_pos_to_world: float,
    root_snapshot_row: Any,
    env: Any,
    knee_hinge_projection: bool = True,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """按与实时播放相同的规则逐帧计算并写出 CSV（不逐步进仿真）。"""
    frames, frame_list, bone_frame_lists, all_bones = motion_data
    max_frame = frame_list[-1]
    robot = env.unwrapped.scene["robot"]
    state = MotionRootTrackState()

    last_rp = (
        float(root_snapshot_row[0].item()),
        float(root_snapshot_row[1].item()),
        float(root_snapshot_row[2].item()),
    )
    last_rq = root_quat_from_state_row(root_snapshot_row)

    out_abs = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "frame",
            "root_x",
            "root_y",
            "root_z",
            "root_qw",
            "root_qx",
            "root_qy",
            "root_qz",
        ] + list(joint_names)
        w.writerow(header)

        for frame in range(max_frame + 1):
            joint_cmd, trp, trq, _r, _b = _compute_targets_for_motion_frame(
                frame,
                frames,
                bone_frame_lists,
                all_bones,
                joint_names,
                default_joint_pos,
                action_scale,
                groove_pos_to_world,
                robot,
                state,
                root_snapshot_row=root_snapshot_row,
                knee_hinge_projection=knee_hinge_projection,
                mmd_center_to_root_offset_local_xyz=mmd_center_to_root_offset_local_xyz,
                root_quat_rpy_scale=root_quat_rpy_scale,
            )
            if trp is not None:
                last_rp = trp
            if trq is not None:
                last_rq = trq
            row = [frame, last_rp[0], last_rp[1], last_rp[2], last_rq[0], last_rq[1], last_rq[2], last_rq[3]]
            row.extend(float(x) for x in joint_cmd.tolist())
            w.writerow(row)

    print(f"[INFO] 已导出 Isaac 逐帧目标 CSV: {out_path}（共 {max_frame + 1} 行）")


def main():
    """零动作运行 G1 站立环境。"""
    pose_cycle_key = (args_cli.pose_cycle_key or "P").strip().upper()[:1]
    pose_motions = _load_pose_motion_dir(POSE_DIR)
    dance_motion_by_key, dance_wav_by_key = _load_dances_from_yaml(DANCES_CONFIG_PATH)

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.motion_playback:
        from robot_mmd.my_task.g1_stand_env_cfg import G1_TPOSE_INIT_STATE
        env_cfg.scene.robot.init_state = G1_TPOSE_INIT_STATE
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = False
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        env_cfg.scene.robot.spawn.rigid_props.linear_damping = 10.0
        env_cfg.scene.robot.spawn.rigid_props.angular_damping = 10.0
        print("[INFO] 已启用动作回放模式")

    # UI 用：缓存当前关节角度（度制），由动作回放每步更新
    joint_pos_deg_cache: dict[str, float] = {}

    if args_cli.sim_fps > 0:
        control_dt = 1.0 / args_cli.sim_fps
        env_cfg.sim.dt = control_dt / 2
        env_cfg.decimation = 2
        env_cfg.sim.render_interval = env_cfg.decimation
        print(f"[INFO] 仿真控制: {args_cli.sim_fps} FPS")

    ox, oy, oz = args_cli.mmd_center_to_root_offset_local_xyz
    if abs(ox) > 1e-12 or abs(oy) > 1e-12 or abs(oz) > 1e-12:
        print(f"[INFO] センター→root 局部偏移(米): ({ox}, {oy}, {oz})")

    env = gym.make(TASK_ID, cfg=env_cfg)

    print(f"[INFO] 观测: {env.observation_space}, 动作: {env.action_space}")
    dance_hint = ", ".join(f"{k}=dance" for k in dance_motion_by_key.keys()) or "无 dance 键"
    print(f"[INFO] L=重置, {pose_cycle_key}=按序播放 pose, {dance_hint}")
    if dance_wav_by_key:
        audio_util.warn_if_no_pygame_sync()

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    reset_requested = False
    pending_cycle_play = False
    pending_dance_key: str | None = None

    def _on_reset():
        """键盘回调：请求在主循环中执行 reset。"""
        nonlocal reset_requested
        reset_requested = True

    def _request_cycle_play():
        """键盘回调：请求切换到下一个 pose。"""
        nonlocal pending_cycle_play
        pending_cycle_play = True

    def _request_dance_play(key: str):
        """键盘回调：请求播放指定 dance。"""
        nonlocal pending_dance_key
        pending_dance_key = key

    keyboard.add_callback("L", _on_reset)
    keyboard.add_callback(pose_cycle_key, _request_cycle_play)
    for dkey in dance_motion_by_key.keys():
        if dkey == pose_cycle_key:
            print(f"[WARN] dance 键 [{dkey}] 与 pose 循环键冲突，已跳过该 dance 键")
            continue
        keyboard.add_callback(dkey, lambda k=dkey: _request_dance_play(k))

    initial_root_snapshot_row: Any = None
    env.reset()
    keyboard.reset()
    initial_root_snapshot_row = _robot_root_row_clone(env)

    current_motion = None  # (frames, frame_list, bone_frame_lists, all_bones)
    current_motion_label = ""
    current_pose_idx = -1
    play_start_time = 0.0
    is_playing = False
    last_printed_frame = -1
    action_scale = env_cfg.actions.joint_pos.scale
    joint_names: list[str] = []
    joint_ids: Any = None
    default_joint_pos: Any = None
    initial_default_joint_pos: Any = None
    instant_mode_warned = False
    root_track_warned = False
    csv_root_track_warned = False
    motion_track: MotionRootTrackState | None = None
    playback_default_joint_pos: Any = None
    mapping_reapply_requested = False
    last_csv_motion_frame: int | None = None
    playback_paused = False
    pause_hold_frame = 0
    pending_playback_toggle = False
    pending_seek_frame: int | None = None
    motion_has_wav = False  # 当前段是否为带伴音的 dance（供暂停/帧跳转同步 audio）
    root_quat_rpy_scale = [-1.0, -1.0, -1.0]  # root 姿态翻转调试：按 roll/pitch/yaw 逐轴乘 scale

    def _on_mapping_ui_changed():
        nonlocal mapping_reapply_requested
        mapping_reapply_requested = True

    def _playback_status_for_ui() -> dict[str, Any]:
        if not is_playing or not current_motion or not (current_motion_label or "").strip():
            return {"playing": False}
        tag = _format_playback_log_label(current_motion_label)
        _frames, frame_list = current_motion[0], current_motion[1]  # type: ignore[index]
        max_f = int(frame_list[-1])
        fr = int(last_csv_motion_frame) if last_csv_motion_frame is not None else 0
        if current_motion_label.startswith("dance["):
            return {
                "playing": True,
                "kind": "dance",
                "tag": tag,
                "frame": fr,
                "max_frame": max_f,
                "playback_paused": playback_paused,
            }
        if current_motion_label.startswith("pose["):
            return {"playing": True, "kind": "pose", "tag": tag, "playback_paused": playback_paused, "frame": fr, "max_frame": max_f}
        return {"playing": True, "kind": "", "tag": tag, "playback_paused": playback_paused, "frame": fr, "max_frame": max_f}

    def _ui_toggle_pause() -> None:
        nonlocal pending_playback_toggle
        if is_playing and current_motion:
            pending_playback_toggle = True

    def _ui_seek_frame(idx: int) -> None:
        nonlocal pending_seek_frame
        if not is_playing or not current_motion:
            return
        idx_i = int(idx)
        if last_csv_motion_frame is not None and idx_i == int(last_csv_motion_frame):
            return
        pending_seek_frame = idx_i

    def _get_root_quat_scale_for_ui() -> tuple[float, float, float]:
        return (float(root_quat_rpy_scale[0]), float(root_quat_rpy_scale[1]), float(root_quat_rpy_scale[2]))

    def _set_root_quat_scale_from_ui(v: tuple[float, float, float]) -> None:
        root_quat_rpy_scale[0] = float(v[0])
        root_quat_rpy_scale[1] = float(v[1])
        root_quat_rpy_scale[2] = float(v[2])

    set_joint_value_provider(lambda: joint_pos_deg_cache)
    set_playback_status_provider(_playback_status_for_ui)
    set_playback_transport_callbacks(_ui_toggle_pause, _ui_seek_frame)
    set_root_quat_scale_callbacks(_get_root_quat_scale_for_ui, _set_root_quat_scale_from_ui)
    set_mapping_changed_callback(_on_mapping_ui_changed)
    create_mapping_ui()

    def _ensure_joint_info():
        """惰性读取关节元数据，仅在首次需要时初始化。"""
        nonlocal joint_names, joint_ids, default_joint_pos, initial_default_joint_pos
        if not joint_names:
            action_term = env.unwrapped.action_manager.get_term("joint_pos")
            joint_names = action_term._joint_names
            joint_ids = action_term._joint_ids
            default_joint_pos = (
                env.unwrapped.scene["robot"]
                .data.default_joint_pos[0, action_term._joint_ids]
                .cpu()
                .numpy()
            )
            initial_default_joint_pos = default_joint_pos.copy()
            _update_joint_pos_cache(default_joint_pos)

    def _update_joint_pos_cache(joint_pos_cmd: Any) -> None:
        """将关节值写入 UI 缓存（度制）。"""
        joint_pos_deg_cache.clear()
        joint_pos_deg_cache.update(_build_joint_pos_deg_cache(joint_names, joint_pos_cmd))
        fd = _MMD_UI_LAST_INTERP_FRAME_DATA
        if fd:
            pe = bool(args_cli.mmd_knee_hinge_projection)
            joint_pos_deg_cache.update(knee_hinge_mapping_ui_extra(fd, projection_enabled=pe))
            joint_pos_deg_cache.update(elbow_hinge_mapping_ui_extra(fd, projection_enabled=pe))

    def _set_control_reference_pose(new_default_joint_pos: Any) -> bool:
        """更新控制器参考姿态，避免 zero action 把关节拉回旧默认姿态。"""
        nonlocal default_joint_pos
        if new_default_joint_pos is None or not joint_names:
            return False
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        new_default = torch.tensor(
            new_default_joint_pos, dtype=torch.float32, device=env.unwrapped.device
        )

        # IsaacLab JointPositionAction 使用 _offset 作为 zero_action 的参考姿态
        offset_updated = False
        try:
            offset_ref = getattr(action_term, "_offset", None)
            if torch.is_tensor(offset_ref):
                if offset_ref.ndim == 1 and offset_ref.shape[0] == new_default.shape[0]:
                    offset_ref.copy_(new_default)
                    offset_updated = True
                elif offset_ref.ndim == 2 and offset_ref.shape[1] == new_default.shape[0]:
                    offset_ref.copy_(new_default.unsqueeze(0).repeat(offset_ref.shape[0], 1))
                    offset_updated = True
        except Exception as exc:
            print(f"[WARN] 更新 joint_pos._offset 失败: {exc}")

        # 同步 robot 默认关节位，确保相对量观测与控制参考一致
        try:
            robot = env.unwrapped.scene["robot"]
            if hasattr(robot.data, "default_joint_pos"):
                robot_default = robot.data.default_joint_pos
                if torch.is_tensor(robot_default) and robot_default.ndim == 2:
                    robot_default[:, joint_ids] = new_default.unsqueeze(0).repeat(robot_default.shape[0], 1)
        except Exception as exc:
            print(f"[WARN] 同步 robot.default_joint_pos 失败: {exc}")

        if not offset_updated:
            print("[WARN] 未能更新 joint_pos 控制参考，zero_action 可能会回弹到旧姿态")
            return False
        default_joint_pos = new_default.detach().cpu().numpy().copy()
        return True

    def _switch_to_motion(data, label: str):
        """切换当前播放动作，并重置播放状态。"""
        nonlocal current_motion, current_motion_label, play_start_time, is_playing, last_printed_frame
        nonlocal motion_track, playback_default_joint_pos
        nonlocal last_csv_motion_frame, mapping_reapply_requested
        nonlocal playback_paused, pause_hold_frame, pending_playback_toggle, pending_seek_frame
        if data is None:
            return
        last_csv_motion_frame = None
        mapping_reapply_requested = False
        playback_paused = False
        pause_hold_frame = 0
        pending_playback_toggle = False
        pending_seek_frame = None
        current_motion = data
        current_motion_label = label
        play_start_time = time.perf_counter()
        is_playing = True
        last_printed_frame = -1
        motion_track = MotionRootTrackState()
        playback_default_joint_pos = None
        _ensure_joint_info()
        if default_joint_pos is not None:
            playback_default_joint_pos = default_joint_pos.copy()
        print(f"[INFO] 开始播放 {label}")

    def _reset_to_initial_pose(sync_ui_cache: bool = False) -> None:
        """将控制参考和机器人姿态恢复到初始默认位。"""
        _ensure_joint_info()
        if initial_default_joint_pos is None:
            return
        _set_control_reference_pose(initial_default_joint_pos)
        if joint_ids is not None:
            _apply_joint_state_instant(env, initial_default_joint_pos, joint_ids)
        if sync_ui_cache and default_joint_pos is not None and joint_names:
            _update_joint_pos_cache(default_joint_pos)

    def _prepare_motion_switch() -> None:
        """动作切换前：停音频，并把控制参考恢复为初始默认关节（与 CSV 角度解算一致）。

        不强制把身体关节写回初始位，避免“先回 T 再开始新动作”的明显跳变；新片段仍用
        initial_default_joint_pos 参与 build_joint_positions，因此解算与原先一致。
        """
        nonlocal motion_has_wav
        motion_has_wav = False
        audio_util.stop_wav()
        _ensure_joint_info()
        if initial_default_joint_pos is not None:
            _set_control_reference_pose(initial_default_joint_pos)

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                audio_util.stop_wav()
                motion_has_wav = False
                env.reset()
                keyboard.reset()
                is_playing = False
                last_csv_motion_frame = None
                mapping_reapply_requested = False
                playback_paused = False
                pause_hold_frame = 0
                pending_playback_toggle = False
                pending_seek_frame = None
                global _MMD_UI_LAST_INTERP_FRAME_DATA
                _MMD_UI_LAST_INTERP_FRAME_DATA = None
                initial_root_snapshot_row = _robot_root_row_clone(env)
                _reset_to_initial_pose(sync_ui_cache=True)
                print("[INFO] 环境已重置")

            if pending_cycle_play:
                pending_cycle_play = False
                if not pose_motions:
                    print(f"[WARN] pose 目录无可播放 CSV: {POSE_DIR}")
                else:
                    _prepare_motion_switch()
                    current_pose_idx = (current_pose_idx + 1) % len(pose_motions)
                    name, _, data = pose_motions[current_pose_idx]
                    _switch_to_motion(data, f"pose[{current_pose_idx + 1}/{len(pose_motions)}] {name}")

            if pending_dance_key is not None:
                dkey = pending_dance_key
                pending_dance_key = None
                entry = dance_motion_by_key.get(dkey)
                if entry is None:
                    print(f"[WARN] dance 键 [{dkey}] 未绑定文件")
                else:
                    _prepare_motion_switch()
                    name, data = entry
                    _switch_to_motion(data, f"dance[{dkey}] {name}")
                    wav = dance_wav_by_key.get(dkey)
                    if wav and str(wav).strip():
                        if os.path.isfile(wav):
                            motion_has_wav = True
                            audio_util.play_wav_async(wav)
                        else:
                            print(f"[WARN] 音频文件不存在: {wav}")

            if is_playing and current_motion:
                frames, frame_list, bone_frame_lists, all_bones = current_motion  # type: ignore
                robot = env.unwrapped.scene["robot"]
                max_frame = int(frame_list[-1])
                play_hz = VMD_FPS * args_cli.play_speed
                paused_before_seek = playback_paused
                did_seek_audio = False
                sf_applied = 0

                if pending_seek_frame is not None:
                    sf = max(0, min(int(pending_seek_frame), max_frame))
                    pending_seek_frame = None
                    if sf != pause_hold_frame:
                        sf_applied = sf
                        did_seek_audio = True
                        pause_hold_frame = sf
                        play_start_time = time.perf_counter() - sf / play_hz

                if did_seek_audio and motion_has_wav:
                    audio_util.sync_audio_to_motion_frame(sf_applied, play_hz, paused_before_seek)

                if playback_paused:
                    frame = min(pause_hold_frame, max_frame)
                else:
                    elapsed_sec = max(0.0, time.perf_counter() - play_start_time)
                    frame = min(int(elapsed_sec * play_hz), max_frame)
                    pause_hold_frame = frame

                did_toggle_audio = False
                if pending_playback_toggle:
                    pending_playback_toggle = False
                    did_toggle_audio = True
                    if playback_paused:
                        playback_paused = False
                        play_start_time = time.perf_counter() - frame / play_hz
                    else:
                        playback_paused = True
                        pause_hold_frame = frame

                if did_toggle_audio and motion_has_wav:
                    audio_util.set_audio_paused(playback_paused)

                last_csv_motion_frame = frame

                joint_pos_cmd, target_root_pos, target_root_quat_wxyz, result, mmd_root_trans_bone = (
                    _compute_targets_for_motion_frame(
                        frame,
                        frames,
                        bone_frame_lists,
                        all_bones,
                        joint_names,
                        default_joint_pos,
                        action_scale,
                        args_cli.groove_pos_to_world,
                        robot,
                        motion_track,
                        root_snapshot_row=initial_root_snapshot_row,
                        knee_hinge_projection=args_cli.mmd_knee_hinge_projection,
                        mmd_center_to_root_offset_local_xyz=args_cli.mmd_center_to_root_offset_local_xyz,
                        root_quat_rpy_scale=(
                            root_quat_rpy_scale[0],
                            root_quat_rpy_scale[1],
                            root_quat_rpy_scale[2],
                        ),
                    )
                )

                if target_root_pos is not None and target_root_quat_wxyz is not None:
                    if _get_csv_root_quat(frame, frames, bone_frame_lists) is None and not csv_root_track_warned:
                        print("[WARN] 当前 CSV 未找到可用根旋转骨骼，root 朝向将保持动作起始值")
                        csv_root_track_warned = True
                    applied_root = _apply_root_pos_instant(env, target_root_pos, target_root_quat_wxyz)
                    if not applied_root and not root_track_warned:
                        print(
                            "[WARN] 当前环境不支持直接写 root 位姿，已跳过根位姿同步"
                            f"（平移骨: {mmd_root_trans_bone}）"
                        )
                        root_track_warned = True

                # 每 10 帧一行：帧进度 + 仿真根位姿（在写根之后读取，与画面一致）。
                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    tag = _format_playback_log_label(current_motion_label)
                    root_suffix = ""
                    root_state_now = getattr(robot.data, "root_state_w", None)
                    if torch.is_tensor(root_state_now) and root_state_now.shape[1] >= 7:
                        px = float(root_state_now[0, 0].item())
                        py = float(root_state_now[0, 1].item())
                        pz = float(root_state_now[0, 2].item())
                        qw, qx, qy, qz = root_quat_from_state_row(root_state_now[0])
                        root_suffix = (
                            f" [pos=({px:.4f}, {py:.4f}, {pz:.4f}) "
                            f"quat_wxyz=({qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f})]"
                        )
                    print(f"[播放] {tag} [帧:{frame}/{max_frame}]{root_suffix}")

                last_frame_joint_pos_cmd = None
                if result is not None:
                    try:
                        last_frame_joint_pos_cmd = joint_pos_cmd
                        _update_joint_pos_cache(joint_pos_cmd)
                    except Exception:
                        pass

                    # 直接写关节以减少跟踪误差；同时仍通过 actions 下发，避免 step 时 PD 把关节拉回旧 offset
                    if joint_pos_cmd is not None:
                        applied = _apply_joint_state_instant(env, joint_pos_cmd, joint_ids)
                        if not applied and not instant_mode_warned:
                            print("[WARN] 当前环境不支持直接写关节状态，自动回退为驱动模式")
                            instant_mode_warned = True

                    actions = torch.tensor(
                        result, dtype=torch.float32, device=env.unwrapped.device
                    ).unsqueeze(0)
                else:
                    actions = zero_action
                if frame >= max_frame:
                    # 播放结束时把控制参考更新到最后一帧，使后续 zero_action 维持末姿态，不会被拉回初始
                    if last_frame_joint_pos_cmd is not None:
                        _set_control_reference_pose(last_frame_joint_pos_cmd)
                    export_path = (args_cli.export_isaac_csv or "").strip()
                    if (
                        export_path
                        and initial_root_snapshot_row is not None
                        and playback_default_joint_pos is not None
                    ):
                        _write_isaac_applied_motion_csv(
                            export_path,
                            current_motion,
                            joint_names,
                            playback_default_joint_pos,
                            action_scale,
                            args_cli.groove_pos_to_world,
                            initial_root_snapshot_row,
                            env,
                            knee_hinge_projection=args_cli.mmd_knee_hinge_projection,
                            mmd_center_to_root_offset_local_xyz=args_cli.mmd_center_to_root_offset_local_xyz,
                            root_quat_rpy_scale=(
                                root_quat_rpy_scale[0],
                                root_quat_rpy_scale[1],
                                root_quat_rpy_scale[2],
                            ),
                        )
                    is_playing = False
                    playback_paused = False
                    motion_has_wav = False
                    audio_util.stop_wav()
                    print(f"[INFO] 播放结束: {current_motion_label}")
            else:
                actions = zero_action

            # 非播放时主循环不再解算 CSV；映射 UI 改动后需按「最后停留帧」重算并写回关节
            if (
                mapping_reapply_requested
                and not is_playing
                and current_motion is not None
                and last_csv_motion_frame is not None
            ):
                mapping_reapply_requested = False
                _ensure_joint_info()
                # 播放结束后 default_joint_pos 已被设为末帧绝对值；解算 CSV 须用本段动作开始时的基准
                base_default = playback_default_joint_pos
                if base_default is None:
                    base_default = initial_default_joint_pos
                if base_default is None:
                    base_default = default_joint_pos
                frames, frame_list, bone_frame_lists, all_bones = current_motion  # type: ignore
                f_hi = frame_list[-1]
                f_apply = max(0, min(int(last_csv_motion_frame), f_hi))
                robot = env.unwrapped.scene["robot"]
                mt = MotionRootTrackState()
                jp_cmd, tr_pos, tr_quat, res, _mb = _compute_targets_for_motion_frame(
                    f_apply,
                    frames,
                    bone_frame_lists,
                    all_bones,
                    joint_names,
                    base_default,
                    action_scale,
                    args_cli.groove_pos_to_world,
                    robot,
                    mt,
                    root_snapshot_row=initial_root_snapshot_row,
                    knee_hinge_projection=args_cli.mmd_knee_hinge_projection,
                    mmd_center_to_root_offset_local_xyz=args_cli.mmd_center_to_root_offset_local_xyz,
                    root_quat_rpy_scale=(
                        root_quat_rpy_scale[0],
                        root_quat_rpy_scale[1],
                        root_quat_rpy_scale[2],
                    ),
                )
                if tr_pos is not None and tr_quat is not None:
                    _apply_root_pos_instant(env, tr_pos, tr_quat)
                if res is not None and jp_cmd is not None:
                    _set_control_reference_pose(jp_cmd)
                    try:
                        _update_joint_pos_cache(jp_cmd)
                    except Exception:
                        pass
                    _apply_joint_state_instant(env, jp_cmd, joint_ids)
                    actions = zero_action
                else:
                    actions = zero_action

            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
