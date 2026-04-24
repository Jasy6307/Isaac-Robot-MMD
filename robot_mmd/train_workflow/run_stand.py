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

import argparse
import math
import os
import sys
import time
from typing import Any
try:
    import winsound
except ImportError:  # pragma: no cover - 非 Windows 环境兼容
    winsound = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDIA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../media"))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

DEFAULT_POSE_DIR = os.path.join(_MEDIA_DIR, "pose")
DEFAULT_DANCE_DIR = os.path.join(_MEDIA_DIR, "dance")


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="宇树 G1 站立 - 零动作运行。")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量（默认 1）")
    parser.add_argument("--disable_fabric", action="store_true", help="禁用 fabric，使用 USD I/O")
    parser.add_argument("--pose_dir", type=str, default=DEFAULT_POSE_DIR, help="按序播放的姿势 CSV 目录")
    parser.add_argument(
        "--pose_cycle_key",
        type=str,
        default="P",
        help="按该键按序播放 pose_dir 下 CSV（默认 P）",
    )
    parser.add_argument("--dance_dir", type=str, default=DEFAULT_DANCE_DIR, help="按键触发的舞蹈 CSV 目录")
    parser.add_argument(
        "--dance_keys",
        type=str,
        default="I,O,U",
        help="舞蹈触发键列表（逗号分隔），按文件名排序依次绑定，如 I,O,U",
    )

    parser.add_argument(
        "--motion_playback",
        action="store_true",
        default=True,
        help="动作回放模式：不固定根链接、禁用重力、增加阻尼",
    )
    parser.add_argument("--play_speed", type=float, default=1.0, help="播放速度倍率")
    parser.add_argument("--smooth_alpha", type=float, default=1.0, help="动作平滑系数 0~1")
    parser.add_argument(
        "--groove_pos_to_world",
        type=float,
        default=0.01,
        help="グルーブ/CSV 位置单位到仿真世界米：默认 0.01（即 CSV 为厘米时 厘米→米），若已是米则设 1.0",
    )
    parser.add_argument("--sim_fps", type=int, default=0, help="仿真控制频率 FPS（0 使用默认）")
    parser.add_argument(
        "--dance_audio_wav",
        type=str,
        default=os.path.join(_MEDIA_DIR, "you_are_important.wav"),
        help="按 O 键触发 dance 时同步播放的 WAV 音频路径",
    )
    parser.add_argument(
        "--instant_joint_set",
        default=True,
        action="store_true",
        help="直接写入关节状态（瞬间到位），不通过关节驱动器跟踪",
    )
    parser.add_argument("--mapping_ui", action="store_true", default=True, help="开启 G1 关节映射编辑窗口")
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
from robot_mmd.train_workflow.mapping_ui import create_mapping_ui, set_joint_value_provider

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


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
    smooth_alpha: float,
    smoothed_action: Any,
) -> tuple[Any, Any]:
    """根据帧号插值得到动作，并做平滑。返回 (action_tensor, new_smoothed_action)"""
    frame_data = {}
    for bone in current_all_bones:
        d = interpolate_bone(frame, bone, current_frames, current_bone_frame_lists.get(bone))
        if d is not None:
            frame_data[bone] = d

    if not frame_data or joint_names is None or default_joint_pos is None:
        return None, smoothed_action

    target_pos = build_joint_positions_from_frame(frame_data, joint_names, default_joint_pos)
    target_action = (target_pos - default_joint_pos) / action_scale

    if smoothed_action is None:
        smoothed_action = target_action.copy()
    else:
        smoothed_action = smooth_alpha * target_action + (1.0 - smooth_alpha) * smoothed_action
    return smoothed_action, smoothed_action


def _build_joint_pos_deg_cache(joint_names: list[str], joint_pos_cmd: Any) -> dict[str, float]:
    """将关节弧度数组转为 UI 使用的角度缓存。"""
    return {j: float(deg) for j, deg in zip(joint_names, joint_pos_cmd * (180.0 / math.pi))}

def _play_wav_async(filepath: str) -> None:
    """异步播放 WAV，不阻塞仿真循环。"""
    if winsound is None:
        print("[WARN] 当前平台不支持 winsound，跳过音频播放")
        return
    if not os.path.isfile(filepath):
        print(f"[WARN] 音频文件不存在: {filepath}")
        return
    try:
        winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
        print(f"[INFO] 开始播放音频: {filepath}")
    except Exception as exc:
        print(f"[WARN] 音频播放失败: {exc}")


def _stop_wav() -> None:
    """停止当前异步音频播放。"""
    if winsound is None:
        return
    try:
        winsound.PlaySound(None, 0)
        print("[INFO] 已停止音频播放")
    except Exception as exc:
        print(f"[WARN] 停止音频失败: {exc}")


def _apply_joint_state_instant(env: Any, joint_pos_cmd: Any, joint_ids: Any) -> bool:
    """将关节状态直接写入仿真（瞬间到位）。成功返回 True。"""
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]
    joint_pos_tensor = torch.tensor(joint_pos_cmd, dtype=torch.float32, device=device).unsqueeze(0)
    joint_pos_tensor = joint_pos_tensor.repeat(num_envs, 1)
    joint_vel_tensor = torch.zeros_like(joint_pos_tensor)

    try:
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor, joint_ids=joint_ids)
        return True
    except TypeError:
        # 兼容不同 IsaacLab 版本的函数签名
        try:
            robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _apply_root_pos_instant(env: Any, root_pos_xyz: tuple[float, float, float], root_quat_xyzw: Any = None) -> bool:
    """将机器人根位置直接写入仿真（位置即时刷新）。成功返回 True。"""
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = robot.data.joint_pos.shape[0]

    # 优先沿用当前根姿态的四元数，避免每帧更新位置时引入旋转抖动。
    if root_quat_xyzw is None:
        try:
            root_state = getattr(robot.data, "root_state_w", None)
            if torch.is_tensor(root_state) and root_state.shape[1] >= 7:
                root_quat_xyzw = root_state[0, 3:7].detach().cpu().tolist()
        except Exception:
            root_quat_xyzw = None
    if root_quat_xyzw is None:
        root_quat_xyzw = [0.0, 0.0, 0.0, 1.0]

    root_pose = torch.tensor(
        [root_pos_xyz[0], root_pos_xyz[1], root_pos_xyz[2], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2], root_quat_xyzw[3]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    root_pose = root_pose.repeat(num_envs, 1)

    if hasattr(robot, "write_root_pose_to_sim"):
        try:
            robot.write_root_pose_to_sim(root_pose)
            return True
        except Exception:
            pass

    if hasattr(robot, "write_root_state_to_sim"):
        try:
            root_state = getattr(robot.data, "root_state_w", None)
            if torch.is_tensor(root_state) and root_state.shape[1] >= 13:
                state = root_state.clone()
            else:
                state = torch.zeros((num_envs, 13), dtype=torch.float32, device=device)
                state[:, 3:7] = root_pose[:, 3:7]
            state[:, 0:3] = root_pose[:, 0:3]
            if state[:, 3:7].abs().sum() < 1e-6:
                state[:, 6] = 1.0
            robot.write_root_state_to_sim(state)
            return True
        except Exception:
            pass

    return False


def main():
    """零动作运行 G1 站立环境。"""
    pose_cycle_key = (args_cli.pose_cycle_key or "P").strip().upper()[:1]
    dance_keys = [k.strip().upper()[:1] for k in args_cli.dance_keys.split(",") if k.strip()]
    pose_motions = _load_pose_motion_dir(args_cli.pose_dir)
    dance_motion_by_key = _load_dance_key_mapping(args_cli.dance_dir, dance_keys)

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
        # env_cfg.scene.robot.spawn.rigid_props.linear_damping = 2.0
        # env_cfg.scene.robot.spawn.rigid_props.angular_damping = 2.0

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
    if args_cli.mapping_ui:
        # UI 通过 provider 获取“当前关节值（deg）”
        set_joint_value_provider(lambda: joint_pos_deg_cache)
        create_mapping_ui()

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

    env.reset()
    keyboard.reset()

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
    smoothed_action = None
    instant_mode_warned = False
    root_track_warned = False
    motion_groove_origin_pos: tuple[float, float, float] | None = None
    motion_root_origin_pos: tuple[float, float, float] | None = None
    motion_root_quat_xyzw: list[float] | None = None

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
        nonlocal smoothed_action, motion_groove_origin_pos, motion_root_origin_pos, motion_root_quat_xyzw
        if data is None:
            return
        current_motion = data
        current_motion_label = label
        play_start_time = time.perf_counter()
        is_playing = True
        last_printed_frame = -1
        smoothed_action = None
        motion_groove_origin_pos = None
        motion_root_origin_pos = None
        motion_root_quat_xyzw = None
        _ensure_joint_info()
        print(f"[INFO] 开始播放 {label}")

    def _reset_to_initial_pose(sync_ui_cache: bool = False) -> None:
        """将控制参考和机器人姿态恢复到初始默认位。"""
        _ensure_joint_info()
        if initial_default_joint_pos is None:
            return
        _set_control_reference_pose(initial_default_joint_pos)
        if args_cli.instant_joint_set and joint_ids is not None:
            _apply_joint_state_instant(env, initial_default_joint_pos, joint_ids)
        if sync_ui_cache and default_joint_pos is not None and joint_names:
            _update_joint_pos_cache(default_joint_pos)

    def _prepare_motion_switch() -> None:
        """动作切换前的公共准备：停音频并回到初始姿态。"""
        _stop_wav()
        _reset_to_initial_pose(sync_ui_cache=False)

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                _stop_wav()
                env.reset()
                keyboard.reset()
                is_playing = False
                smoothed_action = None
                _reset_to_initial_pose(sync_ui_cache=True)
                print("[INFO] 环境已重置")

            if pending_cycle_play:
                pending_cycle_play = False
                if not pose_motions:
                    print(f"[WARN] pose 目录无可播放 CSV: {args_cli.pose_dir}")
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
                        _play_wav_async(args_cli.dance_audio_wav)

            if is_playing and current_motion:
                frames, frame_list, bone_frame_lists, all_bones = current_motion  # type: ignore
                elapsed_sec = max(0.0, time.perf_counter() - play_start_time)
                frame = int(elapsed_sec * VMD_FPS * args_cli.play_speed)
                max_frame = frame_list[-1]
                frame = min(frame, max_frame)

                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    print(f"[播放] {current_motion_label} 帧 {frame}/{max_frame}")

                result, smoothed_action = _compute_action_for_frame(
                    frame,
                    frames,
                    bone_frame_lists,
                    all_bones,
                    joint_names,
                    default_joint_pos,
                    action_scale,
                    args_cli.smooth_alpha,
                    smoothed_action,
                )

                # 同步根位置：将「グルーブ」视作 torso 中心平移轨迹，按每帧增量刷新到机器人根位置。
                groove = interpolate_bone(frame, "グルーブ", frames, bone_frame_lists.get("グルーブ"))
                if groove is not None and "pos" in groove:
                    try:
                        gx, gy, gz = groove["pos"]
                        groove_pos = (float(gx), float(gy), float(gz))
                        robot = env.unwrapped.scene["robot"]

                        if motion_root_origin_pos is None:
                            root_state = getattr(robot.data, "root_state_w", None)
                            if torch.is_tensor(root_state) and root_state.shape[1] >= 7:
                                motion_root_origin_pos = (
                                    float(root_state[0, 0].item()),
                                    float(root_state[0, 1].item()),
                                    float(root_state[0, 2].item()),
                                )
                                motion_root_quat_xyzw = [
                                    float(root_state[0, 3].item()),
                                    float(root_state[0, 4].item()),
                                    float(root_state[0, 5].item()),
                                    float(root_state[0, 6].item()),
                                ]

                        if motion_groove_origin_pos is None:
                            motion_groove_origin_pos = groove_pos

                        if motion_root_origin_pos is not None and motion_groove_origin_pos is not None:
                            # グルーブ pos 为 MMD 导出的厘米单位；将差分换算为米后再叠加到 root。
                            s = float(args_cli.groove_pos_to_world)
                            # MMD 常用 Y-up；位移映射到仿真世界：X->X, Z->Y, Y->Z
                            dx = (groove_pos[0] - motion_groove_origin_pos[0]) * s
                            dy = (groove_pos[1] - motion_groove_origin_pos[1]) * s
                            dz = (groove_pos[2] - motion_groove_origin_pos[2]) * s
                            target_root_pos = (
                                motion_root_origin_pos[0] + dx,
                                motion_root_origin_pos[1] + dz,
                                motion_root_origin_pos[2] + dy,
                            )
                            applied_root = _apply_root_pos_instant(env, target_root_pos, motion_root_quat_xyzw)
                            if not applied_root and not root_track_warned:
                                print("[WARN] 当前环境不支持直接写 root 位姿，已跳过グルーブ位置同步")
                                root_track_warned = True
                    except Exception:
                        pass

                last_frame_joint_pos_cmd = None
                if result is not None:
                    joint_pos_cmd = None
                    try:
                        joint_pos_cmd = default_joint_pos + action_scale * result  # radians
                        last_frame_joint_pos_cmd = joint_pos_cmd
                        _update_joint_pos_cache(joint_pos_cmd)
                    except Exception:
                        pass

                    # 保持 instant write 以减少关节跟踪误差，但控制目标仍通过 actions 下发，
                    # 避免同步 step 阶段 PD 控制器把关节拉回 offset（即初始姿态）
                    if args_cli.instant_joint_set and joint_pos_cmd is not None:
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
                    is_playing = False
                    smoothed_action = None
                    print(f"[INFO] 播放结束: {current_motion_label}")
            else:
                actions = zero_action

            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
