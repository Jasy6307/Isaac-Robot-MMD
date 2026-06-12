# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 MMD 动作回放主入口（Isaac Sim）。

功能概览：
1) 舞蹈由 dances_config.yaml 登记：键、motion(.csv/.h5/.hdf5)、可选音频；pose 目录 P 键循环；
2) 支持关节映射 UI，实时显示当前关节角度；
3) 有 audio 的 dance 播 WAV，与动作同一「逻辑帧时间轴」；
4) 在重置和切换动作时维护控制参考姿态，避免姿态回弹。
5) 映射 UI 顶部 ``PD Drive`` 复选框：勾选=全身关节 PD，不勾选=关节瞬移写入。

启动：``python robot_mmd/train_workflow/run_g1_mmd_playback.py``
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDIA_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../media"))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

from isaaclab.app import AppLauncher

from robot_mmd.train_workflow.utils.playback_cli import (
    apply_app_window_kit_flags,
    build_arg_parser,
    parse_center_to_root_offset,
)

POSE_DIR = os.path.join(_MEDIA_DIR, "pose")
DANCES_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "dances_config.yaml")

parser = build_arg_parser(POSE_DIR)
args_cli = parser.parse_args()
args_cli.device = "cpu"

try:
    args_cli.mmd_center_to_root_offset_local_xyz = parse_center_to_root_offset(
        args_cli.mmd_center_to_root_offset_local
    )
except Exception as exc:
    raise SystemExit(
        f"--mmd_center_to_root_offset_local 需为 x,y,z 三个浮点数（逗号分隔），例如 0,0,0.2: {exc}"
    ) from exc

apply_app_window_kit_flags(args_cli)

# Hybrid GPU (AMD iGPU + NVIDIA dGPU): avoid sporadic Kit deadlocks during viewport init.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robot_mmd.my_task  # noqa: F401
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg

from robot_mmd.my_task.g1_29dof_o6_cfg import G1_29DOF_O6_CFG
from robot_mmd.train_workflow.g1_deploy_actuator_cfg import (
    apply_robot_pd_profile,
    log_pd_profile_summary,
)
from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.ui.mapping import (
    create_mapping_ui,
    create_retarget_tune_ui,
    set_dance_play_callbacks,
    set_joint_value_provider,
    set_mapping_changed_callback,
    set_pd_drive_callbacks,
    set_playback_status_provider,
    set_playback_transport_callbacks,
    set_root_rot_bone_name_provider,
    set_root_quat_rpy_callbacks,
)
from robot_mmd.train_workflow.utils import audio_util
from robot_mmd.train_workflow.utils.csv_motion_loader import (
    elbow_hinge_mapping_ui_extra,
    knee_hinge_mapping_ui_extra,
    retarget_leg_debug_ui_extra,
    shoulder_retarget_debug_ui_extra,
)
from robot_mmd.train_workflow.utils.motion_loader import (
    MotionBundle,
    build_dance_hand_hdf5_motion_by_key,
    build_dance_hand_motion_by_key,
    build_dance_hdf5_motion_by_key,
    format_playback_log_label,
    load_dances_from_yaml,
    load_pose_motion_dir,
)
from robot_mmd.train_workflow.utils.playback_keyboard import DanceKeyboardListener
from robot_mmd.train_workflow.utils.playback_targets import (
    MotionRootTrackState,
    PlaybackUiDebugState,
    build_joint_pos_deg_cache,
    compute_targets_for_hdf5_frame,
    compute_targets_for_motion_frame,
)
from robot_mmd.train_workflow.utils.sim_robot import (
    apply_joint_state_instant,
    apply_root_pos_instant,
    robot_root_row_clone,
)
from robot_mmd.train_workflow.utils.trans_util import root_quat_from_state_row

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


def main():
    """零动作运行 G1 站立环境。"""
    use_root_teleport = True
    pose_cycle_key = (args_cli.pose_cycle_key or "P").strip().upper()[:1]
    pose_motions = load_pose_motion_dir(POSE_DIR)
    dance_motion_by_key, dance_wav_by_key = load_dances_from_yaml(
        DANCES_CONFIG_PATH,
        media_dir=_MEDIA_DIR,
        script_dir=_SCRIPT_DIR,
    )
    dance_hdf5_motion_by_key = build_dance_hdf5_motion_by_key(dance_motion_by_key)
    dance_hand_motion_by_key = build_dance_hand_motion_by_key(dance_motion_by_key)
    dance_hand_hdf5_motion_by_key = build_dance_hand_hdf5_motion_by_key(dance_hand_motion_by_key)

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    from robot_mmd.my_task.g1_stand_env_cfg import G1_TPOSE_INIT_STATE

    env_cfg.scene.robot.init_state = G1_TPOSE_INIT_STATE
    env_cfg.scene.robot = G1_29DOF_O6_CFG.replace(
        prim_path=env_cfg.scene.robot.prim_path,
        init_state=env_cfg.scene.robot.init_state,
    )
    env_cfg.scene.robot = apply_robot_pd_profile(
        env_cfg.scene.robot, args_cli.pd_profile, o6_hands=True
    )
    log_pd_profile_summary(args_cli.pd_profile, o6_hands=True)
    robot_spawn = env_cfg.scene.robot.spawn
    if robot_spawn is not None:
        if robot_spawn.articulation_props is not None:
            robot_spawn.articulation_props.fix_root_link = False
        if robot_spawn.rigid_props is not None:
            robot_spawn.rigid_props.disable_gravity = True
            robot_spawn.rigid_props.linear_damping = 10.0
            robot_spawn.rigid_props.angular_damping = 10.0

    joint_pos_deg_cache: dict[str, float] = {}
    ui_debug = PlaybackUiDebugState()

    if args_cli.sim_fps > 0:
        control_dt = 1.0 / args_cli.sim_fps
        env_cfg.sim.dt = control_dt / 2
        # env_cfg.sim.dt = control_dt
        env_cfg.decimation = 2
        env_cfg.sim.render_interval = env_cfg.decimation
        print(f"[INFO] 仿真控制: {args_cli.sim_fps} FPS")

    ox, oy, oz = args_cli.mmd_center_to_root_offset_local_xyz
    if abs(ox) > 1e-12 or abs(oy) > 1e-12 or abs(oz) > 1e-12:
        print(f"[INFO] センター→root 局部偏移(米): ({ox}, {oy}, {oz})")

    env = gym.make(TASK_ID, cfg=env_cfg)

    print(f"[INFO] 观测: {env.observation_space}, 动作: {env.action_space}")
    dance_hint = ", ".join(f"{k}=dance" for k in dance_motion_by_key.keys()) or "无 dance 键"
    h5_hint = ", ".join(f"Shift+{k}=H5" for k in dance_hdf5_motion_by_key.keys()) or "无 H5 快捷键"
    print(f"[INFO] L=重置, {pose_cycle_key}=按序播放 pose, {dance_hint}, {h5_hint}")
    print(
        "[INFO] PD Drive mode is controlled by Mapping UI checkbox (top bar)."
    )
    if dance_wav_by_key:
        audio_util.warn_if_no_pygame_sync()

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    reset_requested = False
    pending_cycle_play = False
    pending_dance_key: str | None = None
    pending_dance_prefer_hdf5 = False
    pending_dance_prefer_hand = False

    def _on_reset():
        nonlocal reset_requested
        reset_requested = True

    def _request_cycle_play():
        nonlocal pending_cycle_play
        pending_cycle_play = True

    def _request_dance_play(key: str, *, prefer_hdf5: bool = False):
        nonlocal pending_dance_key, pending_dance_prefer_hdf5, pending_dance_prefer_hand
        key_raw = str(key)
        prefer_hand = key_raw.endswith("#HAND")
        pending_dance_key = key_raw.replace("#HAND", "").upper()[:1]
        pending_dance_prefer_hdf5 = bool(prefer_hdf5)
        pending_dance_prefer_hand = bool(prefer_hand)

    def _dance_entries_for_ui() -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for key in dance_motion_by_key.keys():
            filename, _data = dance_motion_by_key[key]
            entries.append((str(key), f"[{key}] {filename}"))
        return entries

    def _on_dance_request_from_ui(key: str, prefer_hdf5: bool) -> None:
        _request_dance_play(str(key), prefer_hdf5=bool(prefer_hdf5))

    keyboard.add_callback("L", _on_reset)
    keyboard.add_callback(pose_cycle_key, _request_cycle_play)

    dance_listener = DanceKeyboardListener(
        dance_keys=set(dance_motion_by_key.keys()),
        pose_cycle_key=pose_cycle_key,
        on_dance_request=lambda key, prefer_h5: _request_dance_play(key, prefer_hdf5=prefer_h5),
    )
    dance_listener.subscribe()

    initial_root_snapshot_row: Any = None
    env.reset()
    keyboard.reset()
    initial_root_snapshot_row = robot_root_row_clone(env)

    current_motion: MotionBundle | None = None
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
    motion_has_wav = False
    root_quat_rpy_scale = list(MMD_ROOT_QUAT_RPY_SCALE_DEFAULT)
    root_quat_rpy_axis_idx = list(MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT)
    pd_hold_joint_pos_cmd: Any = None
    pd_drive_enabled_ui = False

    def _on_mapping_ui_changed():
        nonlocal mapping_reapply_requested
        mapping_reapply_requested = True

    def _get_pd_drive_for_ui() -> bool:
        return bool(pd_drive_enabled_ui)

    def _set_pd_drive_from_ui(enabled: bool) -> None:
        nonlocal pd_drive_enabled_ui, pd_hold_joint_pos_cmd
        prev = bool(pd_drive_enabled_ui)
        pd_drive_enabled_ui = bool(enabled)
        if prev != pd_drive_enabled_ui:
            pd_hold_joint_pos_cmd = None
            mode = "PD drive" if pd_drive_enabled_ui else "instant teleport"
            print(f"[INFO] Joint control mode -> {mode}")

    def _playback_status_for_ui() -> dict[str, Any]:
        if not is_playing or not current_motion or not (current_motion_label or "").strip():
            return {"playing": False}
        tag = format_playback_log_label(current_motion_label)
        frame_list = current_motion["frame_list"]
        max_f = int(frame_list[-1])
        fr = int(last_csv_motion_frame) if last_csv_motion_frame is not None else 0
        out: dict[str, Any] = {
            "playing": True,
            "tag": tag,
            "frame": fr,
            "max_frame": max_f,
            "playback_paused": playback_paused,
        }
        if current_motion_label.startswith("dance["):
            out["kind"] = "dance"
        elif current_motion_label.startswith("pose["):
            out["kind"] = "pose"
        else:
            out["kind"] = ""
        return out

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

    def _get_root_quat_rpy_for_ui() -> tuple[tuple[float, float, float], tuple[int, int, int]]:
        return (
            (float(root_quat_rpy_scale[0]), float(root_quat_rpy_scale[1]), float(root_quat_rpy_scale[2])),
            (
                int(root_quat_rpy_axis_idx[0]),
                int(root_quat_rpy_axis_idx[1]),
                int(root_quat_rpy_axis_idx[2]),
            ),
        )

    def _set_root_quat_rpy_from_ui(
        scale: tuple[float, float, float],
        axis_idx: tuple[int, int, int],
    ) -> None:
        root_quat_rpy_scale[0] = float(scale[0])
        root_quat_rpy_scale[1] = float(scale[1])
        root_quat_rpy_scale[2] = float(scale[2])
        root_quat_rpy_axis_idx[0] = max(0, min(2, int(axis_idx[0])))
        root_quat_rpy_axis_idx[1] = max(0, min(2, int(axis_idx[1])))
        root_quat_rpy_axis_idx[2] = max(0, min(2, int(axis_idx[2])))

    set_joint_value_provider(lambda: joint_pos_deg_cache)
    set_playback_status_provider(_playback_status_for_ui)
    set_playback_transport_callbacks(_ui_toggle_pause, _ui_seek_frame)
    set_dance_play_callbacks(_dance_entries_for_ui, _on_dance_request_from_ui)
    set_pd_drive_callbacks(_get_pd_drive_for_ui, _set_pd_drive_from_ui)
    set_root_quat_rpy_callbacks(_get_root_quat_rpy_for_ui, _set_root_quat_rpy_from_ui)
    set_root_rot_bone_name_provider(lambda: str(ui_debug.root_rot_bone_name or ""))
    set_mapping_changed_callback(_on_mapping_ui_changed)
    create_mapping_ui()
    create_retarget_tune_ui()

    def _ensure_joint_info():
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
        joint_pos_deg_cache.clear()
        joint_pos_deg_cache.update(build_joint_pos_deg_cache(joint_names, joint_pos_cmd))
        fd = ui_debug.last_interp_frame_data
        if fd:
            pe = bool(args_cli.mmd_knee_hinge_projection)
            joint_pos_deg_cache.update(knee_hinge_mapping_ui_extra(fd, projection_enabled=pe))
            joint_pos_deg_cache.update(elbow_hinge_mapping_ui_extra(fd, projection_enabled=pe))
            joint_pos_deg_cache.update(shoulder_retarget_debug_ui_extra(fd))
            joint_pos_deg_cache.update(retarget_leg_debug_ui_extra(fd))
        dr, dp, dy = ui_debug.root_rpy_euler_scaled_deg
        if dr is not None:
            joint_pos_deg_cache["__root_rpy_deg_r"] = float(dr)
        if dp is not None:
            joint_pos_deg_cache["__root_rpy_deg_p"] = float(dp)
        if dy is not None:
            joint_pos_deg_cache["__root_rpy_deg_y"] = float(dy)
        if ui_debug.root_rot_bone_name:
            joint_pos_deg_cache["__root_rot_bone__"] = str(ui_debug.root_rot_bone_name)

    def _set_control_reference_pose(new_default_joint_pos: Any) -> bool:
        nonlocal default_joint_pos
        if new_default_joint_pos is None or not joint_names:
            return False
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        new_default = torch.tensor(
            new_default_joint_pos, dtype=torch.float32, device=env.unwrapped.device
        )

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

    def _apply_joint_pd_target(joint_pos_cmd: Any) -> bool:
        """Set absolute joint PD targets via action offset; use with zero_action."""
        if joint_pos_cmd is None or not joint_names:
            return False
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        target = torch.tensor(joint_pos_cmd, dtype=torch.float32, device=env.unwrapped.device)
        try:
            offset_ref = getattr(action_term, "_offset", None)
            if torch.is_tensor(offset_ref):
                if offset_ref.ndim == 1 and offset_ref.shape[0] == target.shape[0]:
                    offset_ref.copy_(target)
                    return True
                if offset_ref.ndim == 2 and offset_ref.shape[1] == target.shape[0]:
                    offset_ref.copy_(target.unsqueeze(0).repeat(offset_ref.shape[0], 1))
                    return True
        except Exception as exc:
            print(f"[WARN] 更新 joint_pos PD 目标失败: {exc}")
        return False

    def _switch_to_motion(data, label: str):
        nonlocal current_motion, current_motion_label, play_start_time, is_playing, last_printed_frame
        nonlocal motion_track, playback_default_joint_pos
        nonlocal last_csv_motion_frame, mapping_reapply_requested
        nonlocal playback_paused, pause_hold_frame, pending_playback_toggle, pending_seek_frame
        nonlocal pd_hold_joint_pos_cmd
        if data is None:
            return
        pd_hold_joint_pos_cmd = None
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
        _ensure_joint_info()
        if initial_default_joint_pos is None:
            return
        _set_control_reference_pose(initial_default_joint_pos)
        if (not pd_drive_enabled_ui) and joint_ids is not None:
            apply_joint_state_instant(env, initial_default_joint_pos, joint_ids)
        if sync_ui_cache and default_joint_pos is not None and joint_names:
            _update_joint_pos_cache(default_joint_pos)

    def _prepare_motion_switch() -> None:
        nonlocal motion_has_wav
        motion_has_wav = False
        audio_util.stop_wav()
        _ensure_joint_info()
        if initial_default_joint_pos is not None:
            _set_control_reference_pose(initial_default_joint_pos)

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    def _call_compute_targets(
        frame_idx: int,
        *,
        motion_bundle: MotionBundle,
        joint_default: Any,
        motion_track_state: MotionRootTrackState,
    ) -> tuple[Any, tuple[float, float, float] | None, list[float] | None, Any, str | None, bool | None]:
        robot_inner = env.unwrapped.scene["robot"]
        kind = str(motion_bundle.get("kind", ""))
        enable_hand = bool(motion_bundle.get("has_hand_data", False))
        if kind == "hdf5":
            return compute_targets_for_hdf5_frame(
                frame_idx,
                motion_bundle["hdf5"],
                joint_names,
                joint_default,
                action_scale,
                motion_track_state,
                robot_inner,
                ui_debug,
                root_snapshot_row=initial_root_snapshot_row,
            )
        return compute_targets_for_motion_frame(
            frame_idx,
            motion_bundle["frames"],
            motion_bundle["bone_frame_lists"],
            motion_bundle["all_bones"],
            joint_names,
            joint_default,
            action_scale,
            args_cli.groove_pos_to_world,
            robot_inner,
            motion_track_state,
            ui_debug,
            root_snapshot_row=initial_root_snapshot_row,
            knee_hinge_projection=args_cli.mmd_knee_hinge_projection,
            mmd_center_to_root_offset_local_xyz=args_cli.mmd_center_to_root_offset_local_xyz,
            root_quat_rpy_scale=tuple(root_quat_rpy_scale),
            root_quat_rpy_axis_idx=tuple(root_quat_rpy_axis_idx),
            enable_hand=enable_hand,
        )

    while simulation_app.is_running():
        with torch.inference_mode():
            if reset_requested:
                reset_requested = False
                audio_util.stop_wav()
                motion_has_wav = False
                pd_hold_joint_pos_cmd = None
                env.reset()
                keyboard.reset()
                is_playing = False
                last_csv_motion_frame = None
                mapping_reapply_requested = False
                playback_paused = False
                pause_hold_frame = 0
                pending_playback_toggle = False
                pending_seek_frame = None
                ui_debug.reset()
                initial_root_snapshot_row = robot_root_row_clone(env)
                _reset_to_initial_pose(sync_ui_cache=True)
                print("[INFO] 环境已重置")

            if pending_cycle_play:
                pending_cycle_play = False
                if not pose_motions:
                    print(f"[WARN] pose 目录无可播放 motion 文件: {POSE_DIR}")
                else:
                    _prepare_motion_switch()
                    current_pose_idx = (current_pose_idx + 1) % len(pose_motions)
                    name, _, data = pose_motions[current_pose_idx]
                    _switch_to_motion(data, f"pose[{current_pose_idx + 1}/{len(pose_motions)}] {name}")

            if pending_dance_key is not None:
                dkey = pending_dance_key
                pending_dance_key = None
                prefer_hdf5 = pending_dance_prefer_hdf5
                pending_dance_prefer_hdf5 = False
                prefer_hand = pending_dance_prefer_hand
                pending_dance_prefer_hand = False
                entry = dance_motion_by_key.get(dkey)
                if prefer_hand:
                    if prefer_hdf5:
                        hand_h5_entry = dance_hand_hdf5_motion_by_key.get(dkey)
                        if hand_h5_entry is not None:
                            entry = hand_h5_entry
                        else:
                            print(f"[WARN] dance 键 [{dkey}] 未找到 _hand H5，已取消播放")
                            entry = None
                    else:
                        hand_entry = dance_hand_motion_by_key.get(dkey)
                        if hand_entry is not None:
                            entry = hand_entry
                        else:
                            print(f"[WARN] dance 键 [{dkey}] 未找到 _hand CSV，已取消播放")
                            entry = None
                if prefer_hdf5:
                    h5_entry = dance_hdf5_motion_by_key.get(dkey)
                    if h5_entry is not None and not prefer_hand:
                        entry = h5_entry
                    elif entry is not None and not prefer_hand:
                        print(f"[WARN] dance 键 [{dkey}] 未找到对应 H5，回退为默认 motion")
                if entry is None:
                    print(f"[WARN] dance 键 [{dkey}] 未绑定文件")
                else:
                    _prepare_motion_switch()
                    name, data = entry
                    mode = str(data.get("kind", "unknown")).upper()
                    _switch_to_motion(data, f"dance[{dkey}] {name}")
                    print(f"[INFO] 播放模式: {mode} ({name})")
                    wav = dance_wav_by_key.get(dkey)
                    if wav and str(wav).strip():
                        if os.path.isfile(wav):
                            motion_has_wav = True
                            audio_util.play_wav_async(wav)
                        else:
                            print(f"[WARN] 音频文件不存在: {wav}")

            if is_playing and current_motion:
                frame_list = current_motion["frame_list"]
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

                (
                    joint_pos_cmd,
                    target_root_pos,
                    target_root_quat_wxyz,
                    result,
                    mmd_root_trans_bone,
                    csv_root_rotation_lookup,
                ) = _call_compute_targets(
                    frame,
                    motion_bundle=current_motion,
                    joint_default=default_joint_pos,
                    motion_track_state=motion_track,
                )

                if use_root_teleport and target_root_pos is not None and target_root_quat_wxyz is not None:
                    if csv_root_rotation_lookup is False and not csv_root_track_warned:
                        print("[WARN] 当前 motion 未找到可用根旋转轨迹，root 朝向将保持动作起始值")
                        csv_root_track_warned = True
                    applied_root = apply_root_pos_instant(env, target_root_pos, target_root_quat_wxyz)
                    if not applied_root and not root_track_warned:
                        print(
                            "[WARN] 当前环境不支持直接写 root 位姿，已跳过根位姿同步"
                            f"（平移骨: {mmd_root_trans_bone}）"
                        )
                        root_track_warned = True

                if frame // 10 != last_printed_frame:
                    last_printed_frame = frame // 10
                    tag = format_playback_log_label(current_motion_label)
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

                    if (not pd_drive_enabled_ui) and joint_pos_cmd is not None:
                        applied = apply_joint_state_instant(env, joint_pos_cmd, joint_ids)
                        if not applied and not instant_mode_warned:
                            print("[WARN] 当前环境不支持直接写关节状态，自动回退为驱动模式")
                            instant_mode_warned = True

                    if pd_drive_enabled_ui and joint_pos_cmd is not None:
                        _apply_joint_pd_target(joint_pos_cmd)
                        actions = zero_action
                    else:
                        actions = torch.tensor(
                            result, dtype=torch.float32, device=env.unwrapped.device
                        ).unsqueeze(0)
                else:
                    actions = zero_action
                if frame >= max_frame:
                    if (not pd_drive_enabled_ui) and last_frame_joint_pos_cmd is not None:
                        _set_control_reference_pose(last_frame_joint_pos_cmd)
                    elif pd_drive_enabled_ui and joint_pos_cmd is not None:
                        pd_hold_joint_pos_cmd = (
                            joint_pos_cmd.copy()
                            if hasattr(joint_pos_cmd, "copy")
                            else list(joint_pos_cmd)
                        )
                        _apply_joint_pd_target(pd_hold_joint_pos_cmd)
                        actions = zero_action
                    is_playing = False
                    playback_paused = False
                    motion_has_wav = False
                    audio_util.stop_wav()
                    print(f"[INFO] 播放结束: {current_motion_label}")
            else:
                if pd_drive_enabled_ui and pd_hold_joint_pos_cmd is not None:
                    _apply_joint_pd_target(pd_hold_joint_pos_cmd)
                    actions = zero_action
                else:
                    actions = zero_action

            if (
                mapping_reapply_requested
                and not is_playing
                and current_motion is not None
                and last_csv_motion_frame is not None
            ):
                mapping_reapply_requested = False
                _ensure_joint_info()
                base_default = playback_default_joint_pos
                if base_default is None:
                    base_default = initial_default_joint_pos
                if base_default is None:
                    base_default = default_joint_pos
                frame_hi = current_motion["frame_list"][-1]
                f_hi = int(frame_hi)
                f_apply = max(0, min(int(last_csv_motion_frame), f_hi))
                mt = MotionRootTrackState()
                jp_cmd, tr_pos, tr_quat, res, _mb, _csv_lookup = _call_compute_targets(
                    f_apply,
                    motion_bundle=current_motion,
                    joint_default=base_default,
                    motion_track_state=mt,
                )
                if use_root_teleport and tr_pos is not None and tr_quat is not None:
                    apply_root_pos_instant(env, tr_pos, tr_quat)
                if res is not None and jp_cmd is not None:
                    if pd_drive_enabled_ui:
                        try:
                            _update_joint_pos_cache(jp_cmd)
                        except Exception:
                            pass
                        _apply_joint_pd_target(jp_cmd)
                        actions = zero_action
                    else:
                        _set_control_reference_pose(jp_cmd)
                        try:
                            _update_joint_pos_cache(jp_cmd)
                        except Exception:
                            pass
                        apply_joint_state_instant(env, jp_cmd, joint_ids)
                        actions = zero_action
                else:
                    actions = zero_action

            env.step(actions)

    dance_listener.unsubscribe()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
