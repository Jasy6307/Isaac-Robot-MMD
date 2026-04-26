# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 站立任务动作回放主脚本。

功能概览：
1) 读取 pose/dance 目录下的 CSV 动作并按键触发播放；
2) 支持关节映射 UI，实时显示当前关节角度；
3) 支持 O 键触发音频并与动作回放共用真实时间基准；
4) 在重置和切换动作时维护控制参考姿态，避免姿态回弹。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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

DEFAULT_POSE_DIR = os.path.join(_MEDIA_DIR, "pose")
DEFAULT_DANCE_DIR = os.path.join(_MEDIA_DIR, "dance")
# 若存在，单帧姿势的 MMD 平移 = (本帧 グルーブ/センター) - (参考文件中的同骨)，再乘 groove 比例
DEFAULT_POSE_MMD_BASELINE_VPD = os.path.join(DEFAULT_POSE_DIR, "mmd_baseline.vpd")
DEFAULT_POSE_MMD_BASELINE_CSV = os.path.join(DEFAULT_POSE_DIR, "mmd_baseline.csv")
_POSE_MMD_NO_BASELINE_WARNED = False


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
    parser.add_argument(
        "--pose_cycle_key",
        type=str,
        default="P",
        help=f"按该键按序播放姿势 CSV（目录固定为 {DEFAULT_POSE_DIR}，默认键 P）",
    )
    parser.add_argument(
        "--dance_keys",
        type=str,
        default="I,O,U",
        help=f"舞蹈触发键（逗号分隔），CSV 目录固定为 {DEFAULT_DANCE_DIR}，按文件名排序绑定",
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
        help="グルーブ/CSV 平移差分到仿真米：默认 0.1（分米→米）。若为厘米设 0.01，若已是米则 1.0",
    )
    parser.add_argument(
        "--pose_mmd_baseline",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="单帧姿势：MMD「センター/グルーブ」在参考姿态下的数值；本帧 mmd 平移先减该点再乘 groove_pos_to_world。与舞蹈用首帧锚点同理",
    )
    parser.add_argument(
        "--pose_mmd_baseline_vpd",
        type=str,
        default="",
        help="从该 VPD/CSV 读平移作参考（CSV 为最先帧 グルーブ 或 センター)；未设时若存在 pose/mmd_baseline.vpd 或 mmd_baseline.csv 则自动用",
    )
    parser.add_argument("--sim_fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    parser.add_argument(
        "--dance_audio_wav",
        type=str,
        default=os.path.join(_MEDIA_DIR, "you_are_important_quiet.wav"),
        help="按 O 键触发 dance 时同步播放的 WAV 音频路径",
    )
    parser.add_argument(
        "--export_isaac_csv",
        type=str,
        default="targets.csv",
        help="某段动作播放结束后，将每帧写入 Isaac 的根位姿(wxyz)与关节角(弧度)导出为该路径的 CSV；空则关闭",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_arg_parser()
args_cli = parser.parse_args()
args_cli.device = "cpu"

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
    get_bone_frame_lists,
    get_frame_indices,
    interpolate_bone,
    load_csv_motion,
)
from robot_mmd.train_workflow.mapping_ui import (
    create_mapping_ui,
    set_joint_value_provider,
    set_mapping_changed_callback,
)
from robot_mmd.train_workflow.trans_util import (
    coerce_quat,
    mmd_root_offset_quat_to_world,
    quat_inv,
    quat_mul,
    quat_normalize,
    root_quat_from_state_row,
)
import audio_util

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


def _mmd_root_translation_from_csv_baseline(path: str) -> tuple[float, float, float] | None:
    """与 _interpolate_mmd_root_translation_bone 一致：有グルーブ用其，否则用センター。取该 CSV 最前的帧。"""
    try:
        frames = load_csv_motion(path)
    except (OSError, ValueError) as e:
        print(f"[WARN] 无法读取参考姿势 CSV: {path}: {e}")
        return None
    if not frames:
        return None
    fi = min(frames.keys())
    fd = frames[fi]
    for bname in ("グルーブ", "センター"):
        row = fd.get(bname)
        if row is not None and "pos" in row:
            p = row["pos"]
            return (float(p[0]), float(p[1]), float(p[2]))
    print(f"[WARN] 参考 CSV 中无「グルーブ/センター」平移: {path}")
    return None


def _mmd_root_translation_from_vpd_baseline(path: str) -> tuple[float, float, float] | None:
    from robot_mmd.train_workflow.vmd_2_csv import read_vpd_pose

    try:
        bones = read_vpd_pose(path)
    except OSError as e:
        print(f"[WARN] 无法读取 VPD: {path}: {e}")
        return None
    by_name = {b.get("bone"): b for b in bones}
    for bname in ("グルーブ", "センター"):
        b = by_name.get(bname)
        if b is not None:
            p = b["position"]
            return (float(p[0]), float(p[1]), float(p[2]))
    print(f"[WARN] VPD 中无「グルーブ/センター」: {path}")
    return None


def resolve_pose_mmd_baseline(args: Any) -> tuple[float, float, float] | None:
    """单帧 MMD 平移参考点：与当前姿势 CSV 同模型、同坐标约定（如站立 VPD/CSV 的首帧 センター）。"""
    if getattr(args, "pose_mmd_baseline", None) is not None:
        t = args.pose_mmd_baseline
        c = (float(t[0]), float(t[1]), float(t[2]))
        print(f"[INFO] 单帧姿势 MMD 平移参考（--pose_mmd_baseline）: ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})")
        return c
    p_user = (getattr(args, "pose_mmd_baseline_vpd", None) or "").strip()
    for p in (p_user, DEFAULT_POSE_MMD_BASELINE_VPD, DEFAULT_POSE_MMD_BASELINE_CSV):
        if not p or not os.path.isfile(os.path.abspath(p)):
            continue
        ap = os.path.abspath(p)
        if ap.lower().endswith(".csv"):
            c = _mmd_root_translation_from_csv_baseline(ap)
        else:
            c = _mmd_root_translation_from_vpd_baseline(ap)
        if c is not None:
            print(
                f"[INFO] 单帧姿势 MMD 平移参考: ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}) 来自 {ap}"
            )
            return c
    return None


def _robot_root_row_clone(env: Any) -> Any | None:
    """取 env 内机器人 root_state_w 第一行 CPU 副本；不可用则 None。"""
    rs = getattr(env.unwrapped.scene["robot"].data, "root_state_w", None)
    if torch.is_tensor(rs) and rs.shape[1] >= 7:
        return rs[0].detach().cpu().clone()
    return None


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


def _load_dance_key_mapping(dance_dir: str, dance_keys: list[str]) -> dict[str, tuple[str, tuple]]:
    """读取 dance 目录并按键位绑定，返回 key -> (文件名, motion_data)。"""
    csv_files = _list_csv_files(dance_dir, "dance")
    if not csv_files:
        return {}

    mapping: dict[str, tuple[str, tuple]] = {}
    for key, name in zip(dance_keys, csv_files):
        fullpath = os.path.join(dance_dir, name)
        data = _load_motion(fullpath)
        if data is None:
            print(f"[WARN] 无法加载 dance CSV: {fullpath}")
            continue
        mapping[key] = (name, data)
        print(f"[INFO] 已绑定 dance 键 [{key}] -> {name}（{len(data[1])} 帧）")

    if len(csv_files) > len(dance_keys):
        print(
            f"[WARN] dance 文件数量({len(csv_files)})超过按键数量({len(dance_keys)})，"
            "超出部分未绑定"
        )
    return mapping


def _compute_action_for_frame(
    frame: int,
    current_frames: Any,
    current_bone_frame_lists: dict[str, list[int]],
    current_all_bones: set[str],
    joint_names: list[str],
    default_joint_pos: Any,
    action_scale: float,
) -> Any:
    """根据帧号插值得到动作（不平滑，每帧即用目标动作）。"""
    frame_data = {}
    for bone in current_all_bones:
        d = interpolate_bone(frame, bone, current_frames, current_bone_frame_lists.get(bone))
        if d is not None:
            frame_data[bone] = d

    if not frame_data or joint_names is None or default_joint_pos is None:
        return None

    target_pos = build_joint_positions_from_frame(frame_data, joint_names, default_joint_pos)
    target_action = (target_pos - default_joint_pos) / action_scale
    return target_action.copy()


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
    candidates = ("グルーブ", "センター親", "腰", "センター")

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
    """根在 MMD 中的平移轨迹：有「グルーブ」用其（舞台位移），否则用「センター」（无グルーブ 的 pose 常见）。"""
    for bone in ("グルーブ", "センター"):
        d = interpolate_bone(frame, bone, frames, bone_frame_lists.get(bone))
        if d is not None and "pos" in d:
            return bone, d
    return None, None


@dataclass
class MotionRootTrackState:
    """根轨迹：仿真根在重置时采样；多帧片段另存首帧 MMD 平移/根旋转锚点。"""

    groove_origin_pos: tuple[float, float, float] | None = None
    csv_root_origin_quat_wxyz: list[float] | None = None
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
    pose_mmd_baseline: tuple[float, float, float] | None = None,
) -> tuple[Any, tuple[float, float, float] | None, list[float] | None, Any, str | None]:
    """计算本帧关节目标、根位姿、动作向量；最后一项为用于平移的 MMD 根骨名（无则 None）。"""
    global _POSE_MMD_NO_BASELINE_WARNED
    result = _compute_action_for_frame(
        frame, frames, bone_frame_lists, all_bones, joint_names, default_joint_pos, action_scale
    )
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
            if is_pose:
                # 单帧：须相对「同 PMX 的参考姿势」的 グルーブ/センター（--pose_mmd_baseline*），否则 (0,0,0) 会把绑定位移当世界位移。
                # 与多帧舞蹈用「首帧 mmd_pos」作 groove_ref 同理，只是参考来自外置站立 VPD/CSV。
                if pose_mmd_baseline is not None:
                    groove_ref = pose_mmd_baseline
                else:
                    groove_ref = (0.0, 0.0, 0.0)
                    if not _POSE_MMD_NO_BASELINE_WARNED:
                        print(
                            "[WARN] 单帧姿势未设置 MMD 平移参考（--pose_mmd_baseline / --pose_mmd_baseline_vpd，"
                            f"或放置 {os.path.basename(DEFAULT_POSE_MMD_BASELINE_VPD)} / "
                            f"{os.path.basename(DEFAULT_POSE_MMD_BASELINE_CSV)} 于 pose 目录）："
                            "将相对 MMD(0,0,0) 计算，易与模型绑定数值叠加导致根位置错误。"
                        )
                        _POSE_MMD_NO_BASELINE_WARNED = True
            else:
                if state.groove_origin_pos is None:
                    state.groove_origin_pos = mmd_pos
                groove_ref = state.groove_origin_pos

            if state.root_origin_pos is not None:
                s = float(groove_pos_to_world)
                dx = (mmd_pos[0] - groove_ref[0]) * s
                dy = (mmd_pos[1] - groove_ref[1]) * s
                dz = (mmd_pos[2] - groove_ref[2]) * s
                ox, oy, oz = (
                    state.root_origin_pos[0],
                    state.root_origin_pos[1],
                    state.root_origin_pos[2],
                )
                # 姿势：MMD 原点 + 加号（世界 X 用 -dx，与舞蹈首帧差分约定一致，避免仅沿 X 反向）。舞蹈：首帧锚点 + 减号平移；旋转 q0*inv(q_w)。
                if is_pose:
                    target_root_pos = (ox - dx, oy + dz, oz + dy)
                else:
                    target_root_pos = (ox - dx, oy - dz, oz - dy)
                target_root_quat_wxyz = list(state.root_quat_wxyz) if state.root_quat_wxyz else None
                csv_root_quat_wxyz = _get_csv_root_quat(frame, frames, bone_frame_lists)
                if csv_root_quat_wxyz is not None and state.root_quat_wxyz is not None:
                    q_w = mmd_root_offset_quat_to_world(csv_root_quat_wxyz)
                    if is_pose:
                        target_root_quat_wxyz = quat_normalize(quat_mul(q_w, state.root_quat_wxyz))
                    else:
                        if state.csv_root_origin_quat_wxyz is None:
                            state.csv_root_origin_quat_wxyz = list(q_w)
                        q0 = state.csv_root_origin_quat_wxyz
                        d = quat_normalize(quat_mul(q0, quat_inv(q_w)))
                        target_root_quat_wxyz = quat_normalize(quat_mul(d, state.root_quat_wxyz))
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
    pose_mmd_baseline: tuple[float, float, float] | None = None,
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
                pose_mmd_baseline=pose_mmd_baseline,
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
    pose_mmd_baseline = resolve_pose_mmd_baseline(args_cli)
    pose_cycle_key = (args_cli.pose_cycle_key or "P").strip().upper()[:1]
    dance_keys = [k.strip().upper()[:1] for k in args_cli.dance_keys.split(",") if k.strip()]
    pose_motions = _load_pose_motion_dir(DEFAULT_POSE_DIR)
    dance_motion_by_key = _load_dance_key_mapping(DEFAULT_DANCE_DIR, dance_keys)

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

    env = gym.make(TASK_ID, cfg=env_cfg)

    print(f"[INFO] 观测: {env.observation_space}, 动作: {env.action_space}")
    dance_hint = ", ".join(f"{k}=dance" for k in dance_motion_by_key.keys()) or "无 dance 键"
    print(f"[INFO] L=重置, {pose_cycle_key}=按序播放 pose, {dance_hint}")

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
    last_printed_root_frame = -1
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

    def _on_mapping_ui_changed():
        nonlocal mapping_reapply_requested
        mapping_reapply_requested = True

    set_joint_value_provider(lambda: joint_pos_deg_cache)
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
        nonlocal last_printed_root_frame
        nonlocal motion_track, playback_default_joint_pos
        nonlocal last_csv_motion_frame, mapping_reapply_requested
        if data is None:
            return
        last_csv_motion_frame = None
        mapping_reapply_requested = False
        current_motion = data
        current_motion_label = label
        play_start_time = time.perf_counter()
        is_playing = True
        last_printed_frame = -1
        last_printed_root_frame = -1
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
                env.reset()
                keyboard.reset()
                is_playing = False
                last_csv_motion_frame = None
                mapping_reapply_requested = False
                initial_root_snapshot_row = _robot_root_row_clone(env)
                _reset_to_initial_pose(sync_ui_cache=True)
                print("[INFO] 环境已重置")

            if pending_cycle_play:
                pending_cycle_play = False
                if not pose_motions:
                    print(f"[WARN] pose 目录无可播放 CSV: {DEFAULT_POSE_DIR}")
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
                    if dkey == "O":
                        audio_util.play_wav_async(args_cli.dance_audio_wav)

            if is_playing and current_motion:
                frames, frame_list, bone_frame_lists, all_bones = current_motion  # type: ignore
                robot = env.unwrapped.scene["robot"]
                elapsed_sec = max(0.0, time.perf_counter() - play_start_time)
                frame = int(elapsed_sec * VMD_FPS * args_cli.play_speed)
                max_frame = frame_list[-1]
                frame = min(frame, max_frame)
                last_csv_motion_frame = frame

                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    print(f"[播放] {current_motion_label} 帧 {frame}/{max_frame}")

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
                        pose_mmd_baseline=pose_mmd_baseline,
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

                # 与 [播放] 日志保持同频：每 10 帧输出一次 root 位姿，避免刷屏。
                if frame // 10 != last_printed_root_frame:
                    root_state_now = getattr(robot.data, "root_state_w", None)
                    if torch.is_tensor(root_state_now) and root_state_now.shape[1] >= 7:
                        px = float(root_state_now[0, 0].item())
                        py = float(root_state_now[0, 1].item())
                        pz = float(root_state_now[0, 2].item())
                        qw, qx, qy, qz = root_quat_from_state_row(root_state_now[0])
                        print(
                            f"[ROOT] {current_motion_label} 帧 {frame}: "
                            f"pos=({px:.4f}, {py:.4f}, {pz:.4f}) "
                            f"quat_wxyz=({qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f})"
                        )
                        last_printed_root_frame = frame // 10

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
                            pose_mmd_baseline=pose_mmd_baseline,
                        )
                    is_playing = False
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
                    pose_mmd_baseline=pose_mmd_baseline,
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
