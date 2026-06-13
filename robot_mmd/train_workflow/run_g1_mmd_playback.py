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
6) 映射 UI ``Z_offset_enable``：勾选时自动播放同目录 ``*_z_editted.*`` sibling（无则回退原版并 WARN）。
7) 启动时扫描 ``media/dance/*.vmd``，自动生成缺失的 CSV/H5 并登记到 ``dances_config.yaml``（可无快捷键，UI 可选）。

启动：``python robot_mmd/train_workflow/run_g1_mmd_playback.py``
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
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
from robot_mmd.train_workflow.utils.mmd_fk import FootIkVizConfig
from robot_mmd.train_workflow.utils.csv_motion_loader import FootIkConfig

POSE_DIR = os.path.join(_MEDIA_DIR, "pose")
DANCE_DIR = os.path.join(_MEDIA_DIR, "dance")
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

from robot_mmd.train_workflow.utils.dance_asset_sync import sync_dance_assets_from_vmd

sync_dance_assets_from_vmd(
    dance_dir=DANCE_DIR,
    dances_config_path=DANCES_CONFIG_PATH,
    media_dir=_MEDIA_DIR,
    groove_pos_to_world=float(args_cli.groove_pos_to_world),
    mmd_center_to_root_offset_local_xyz=args_cli.mmd_center_to_root_offset_local_xyz,
    knee_hinge_projection=bool(args_cli.mmd_knee_hinge_projection),
)

apply_app_window_kit_flags(args_cli)

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
    set_dance_z_edit_callbacks,
    set_audio_volume_callbacks,
    set_foot_ik_callbacks,
    set_foot_ik_viz_callbacks,
    set_joint_value_provider,
    set_mapping_changed_callback,
    set_pd_drive_callbacks,
    set_playback_status_provider,
    set_playback_transport_callbacks,
    set_root_rot_bone_name_provider,
    set_root_quat_rpy_callbacks,
    set_z_offset_enable_callbacks,
)
from robot_mmd.train_workflow.utils import audio_util
from robot_mmd.train_workflow.utils.csv_motion_loader import (
    FootIkState,
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
    has_z_editted_sibling,
    load_dances_from_yaml,
    load_pose_motion_dir,
    resolve_playback_motion_entry,
)
from robot_mmd.train_workflow.utils.playback_keyboard import DanceKeyboardListener
from robot_mmd.train_workflow.utils.root_z_edit import RootZEditConfig, generate_z_editted_motion
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
from robot_mmd.train_workflow.utils.trans_util import (
    quat_inv,
    root_quat_from_state_row,
    rotate_vec_by_quat_wxyz,
)

TASK_ID = "Isaac-G1-Stand-v0"
VMD_FPS = 30


def _parse_triplet_float(text: Any, name: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} 需为 x,y,z 三个数字（逗号分隔）")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_triplet_int_clamped(text: Any, name: str, lo: int = 0, hi: int = 2) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} 需为 x,y,z 三个整数（逗号分隔）")
    vals = [int(v) for v in parts]
    return (
        max(lo, min(hi, vals[0])),
        max(lo, min(hi, vals[1])),
        max(lo, min(hi, vals[2])),
    )


class _FootIkTargetViz:
    """Render MMD foot IK world target spheres in USD scene."""

    def __init__(self) -> None:
        self._ready = False
        self._stage = None
        self._xform_cache: dict[str, Any] = {}

    def _ensure(self) -> bool:
        if self._ready:
            return True
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom
        except Exception:
            return False
        ctx = omni.usd.get_context()
        stage = ctx.get_stage() if ctx is not None else None
        if stage is None:
            return False
        root = "/World/Debug/FootIkTargets"
        if not stage.GetPrimAtPath(root):
            stage.DefinePrim(root, "Xform")
        for side in ("Left", "Right"):
            p = f"{root}/{side}FootIK"
            if not stage.GetPrimAtPath(p):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(p))
                sph.GetRadiusAttr().Set(0.025)
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            prim = stage.GetPrimAtPath(p)
            self._xform_cache[p] = UsdGeom.Xformable(prim)
            tp = f"{root}/{side}ToeIK"
            if not stage.GetPrimAtPath(tp):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(tp))
                sph.GetRadiusAttr().Set(0.018)
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.3, 0.3)])
            prim_t = stage.GetPrimAtPath(tp)
            self._xform_cache[tp] = UsdGeom.Xformable(prim_t)
        self._stage = stage
        self._ready = True
        return True

    def _set_visible(self, path: str, visible: bool) -> None:
        from pxr import UsdGeom

        stage = self._stage
        if stage is None:
            return
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible() if visible else imageable.MakeInvisible()

    def _set_translate(self, path: str, pos_xyz: tuple[float, float, float]) -> None:
        from pxr import Gf

        xformable = self._xform_cache.get(path)
        if xformable is None:
            return
        ops = xformable.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
        else:
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
            )

    def update(
        self,
        *,
        left_world: tuple[float, float, float] | None,
        right_world: tuple[float, float, float] | None,
        left_toe_world: tuple[float, float, float] | None,
        right_toe_world: tuple[float, float, float] | None,
    ) -> None:
        if not self._ensure():
            return
        lp = "/World/Debug/FootIkTargets/LeftFootIK"
        rp = "/World/Debug/FootIkTargets/RightFootIK"
        ltp = "/World/Debug/FootIkTargets/LeftToeIK"
        rtp = "/World/Debug/FootIkTargets/RightToeIK"
        if left_world is None:
            self._set_visible(lp, False)
        else:
            self._set_translate(lp, left_world)
            self._set_visible(lp, True)
        if right_world is None:
            self._set_visible(rp, False)
        else:
            self._set_translate(rp, right_world)
            self._set_visible(rp, True)
        if left_toe_world is None:
            self._set_visible(ltp, False)
        else:
            self._set_translate(ltp, left_toe_world)
            self._set_visible(ltp, True)
        if right_toe_world is None:
            self._set_visible(rtp, False)
        else:
            self._set_translate(rtp, right_toe_world)
            self._set_visible(rtp, True)


def _try_get_body_names(robot: Any) -> list[str]:
    for holder in (getattr(robot, "data", None), robot):
        if holder is None:
            continue
        for name in ("body_names", "link_names"):
            v = getattr(holder, name, None)
            if isinstance(v, (list, tuple)) and v:
                return [str(x) for x in v]
    return []


def _extract_body_pose_wxyz(robot: Any, body_idx: int) -> tuple[tuple[float, float, float], list[float]] | None:
    data = getattr(robot, "data", None)
    if data is None:
        return None
    state = None
    for field in ("body_state_w", "body_link_state_w", "link_state_w"):
        v = getattr(data, field, None)
        if torch.is_tensor(v):
            state = v
            break
    if state is not None:
        row = state[0, body_idx] if state.ndim == 3 else state[body_idx]
        return (
            (float(row[0].item()), float(row[1].item()), float(row[2].item())),
            [float(row[3].item()), float(row[4].item()), float(row[5].item()), float(row[6].item())],
        )
    return None


def _calibrate_foot_ik_refs_from_robot(
    robot: Any,
    foot_ik_cfg: FootIkConfig,
    *,
    heel_offset_in_ankle_local: tuple[float, float, float] = (-0.00, 0.0, 0.0),
    toe_offset_in_ankle_local: tuple[float, float, float] = (0.12, 0.0, 0.0),
) -> bool:
    body_names = _try_get_body_names(robot)
    if not body_names:
        return False
    try:
        li = int(body_names.index("left_ankle_roll_link"))
        ri = int(body_names.index("right_ankle_roll_link"))
    except ValueError:
        return False
    root_state = getattr(robot.data, "root_state_w", None)
    if (not torch.is_tensor(root_state)) or root_state.shape[1] < 7:
        return False
    root_pos = (
        float(root_state[0, 0].item()),
        float(root_state[0, 1].item()),
        float(root_state[0, 2].item()),
    )
    root_quat = root_quat_from_state_row(root_state[0])
    root_q_inv = quat_inv(list(root_quat))

    heel_vals: list[tuple[float, float, float]] = []
    toe_vals: list[tuple[float, float, float]] = []
    for bi in (li, ri):
        pose = _extract_body_pose_wxyz(robot, bi)
        if pose is None:
            return False
        pos_w, quat_w = pose
        heel_dv = rotate_vec_by_quat_wxyz(quat_w, heel_offset_in_ankle_local)
        heel_w = (
            float(pos_w[0] + heel_dv[0]),
            float(pos_w[1] + heel_dv[1]),
            float(pos_w[2] + heel_dv[2]),
        )
        rel = (
            float(heel_w[0] - root_pos[0]),
            float(heel_w[1] - root_pos[1]),
            float(heel_w[2] - root_pos[2]),
        )
        loc = rotate_vec_by_quat_wxyz(root_q_inv, rel)
        heel_vals.append((float(loc[0]), float(loc[1]), float(loc[2])))

        toe_dv = rotate_vec_by_quat_wxyz(quat_w, toe_offset_in_ankle_local)
        toe_w = (
            float(pos_w[0] + toe_dv[0]),
            float(pos_w[1] + toe_dv[1]),
            float(pos_w[2] + toe_dv[2]),
        )
        toe_rel = (
            float(toe_w[0] - root_pos[0]),
            float(toe_w[1] - root_pos[1]),
            float(toe_w[2] - root_pos[2]),
        )
        toe_loc = rotate_vec_by_quat_wxyz(root_q_inv, toe_rel)
        toe_vals.append((float(toe_loc[0]), float(toe_loc[1]), float(toe_loc[2])))
    if len(heel_vals) != 2 or len(toe_vals) != 2:
        return False
    foot_ik_cfg.left_foot_ref_local = heel_vals[0]
    foot_ik_cfg.right_foot_ref_local = heel_vals[1]
    foot_ik_cfg.left_toe_ref_local = toe_vals[0]
    foot_ik_cfg.right_toe_ref_local = toe_vals[1]
    return True


def main():
    """零动作运行 G1 站立环境。"""
    use_root_teleport = True
    foot_axis_idx = _parse_triplet_int_clamped(args_cli.mmd_foot_ik_axis_idx, "--mmd_foot_ik_axis_idx")
    foot_axis_sign = _parse_triplet_float(args_cli.mmd_foot_ik_axis_sign, "--mmd_foot_ik_axis_sign")
    foot_axis_sign_pose = _parse_triplet_float(
        args_cli.mmd_foot_ik_axis_sign_pose, "--mmd_foot_ik_axis_sign_pose"
    )
    left_ref_local = _parse_triplet_float(args_cli.mmd_foot_ik_left_ref_local, "--mmd_foot_ik_left_ref_local")
    right_ref_local = _parse_triplet_float(
        args_cli.mmd_foot_ik_right_ref_local, "--mmd_foot_ik_right_ref_local"
    )
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
    dance_path_by_key = {
        k: str(data.get("path", ""))
        for k, (_name, data) in dance_motion_by_key.items()
        if str(data.get("path", "")).strip()
    }
    hotkey_dance_keys = {k for k in dance_motion_by_key.keys() if not str(k).startswith("ui:")}

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
    dance_hint = ", ".join(f"{k}=dance" for k in hotkey_dance_keys) or "无 dance 键"
    h5_hint = ", ".join(f"Shift+{k}=H5" for k in dance_hdf5_motion_by_key.keys() if not str(k).startswith("ui:")) or "无 H5 快捷键"
    print(f"[INFO] L=重置, {pose_cycle_key}=按序播放 pose, {dance_hint}, {h5_hint}")
    print(
        "[INFO] PD Drive / Z_offset_enable are controlled by Mapping UI checkboxes (top bar)."
    )
    if dance_wav_by_key:
        audio_util.warn_if_no_pygame_sync()

    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    reset_requested = False
    pending_cycle_play = False
    pending_dance_key: str | None = None
    pending_dance_prefer_hdf5 = False
    pending_dance_prefer_hand = False
    pending_z_edit_key: str | None = None
    z_edit_busy = False

    def _on_reset():
        nonlocal reset_requested
        reset_requested = True

    def _request_cycle_play():
        nonlocal pending_cycle_play
        pending_cycle_play = True

    def _dance_lookup_key(key: str) -> str:
        raw = str(key).replace("#HAND", "").strip()
        if raw.startswith("ui:"):
            return raw
        return raw.upper()[:1]

    def _request_dance_play(key: str, *, prefer_hdf5: bool = False):
        nonlocal pending_dance_key, pending_dance_prefer_hdf5, pending_dance_prefer_hand
        key_raw = str(key)
        prefer_hand = key_raw.endswith("#HAND")
        pending_dance_key = _dance_lookup_key(key_raw)
        pending_dance_prefer_hdf5 = bool(prefer_hdf5)
        pending_dance_prefer_hand = bool(prefer_hand)

    def _dance_entries_for_ui() -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for key in dance_motion_by_key.keys():
            filename, _data = dance_motion_by_key[key]
            if str(key).startswith("ui:"):
                entries.append((str(key), filename))
            else:
                entries.append((str(key), f"[{key}] {filename}"))
        return entries

    def _on_dance_request_from_ui(key: str, prefer_hdf5: bool) -> None:
        nonlocal pending_dance_key, pending_dance_prefer_hdf5, pending_dance_prefer_hand
        pending_dance_key = str(key)
        pending_dance_prefer_hdf5 = bool(prefer_hdf5)
        pending_dance_prefer_hand = False

    def _dance_has_z_editted(key: str) -> bool:
        path = dance_path_by_key.get(_dance_lookup_key(key), "")
        if not path:
            return True
        return has_z_editted_sibling(path)

    def _on_z_edit_request_from_ui(key: str) -> None:
        nonlocal pending_z_edit_key
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        if z_edit_busy:
            print("[WARN] Z_editted generation already in progress")
            return
        path = dance_path_by_key.get(dkey, "")
        if not path:
            print(f"[WARN] dance [{dkey}] has no motion path")
            return
        if has_z_editted_sibling(path):
            print(f"[INFO] Z_editted sibling already exists for dance [{dkey}]")
            return
        pending_z_edit_key = dkey
        print(f"[INFO] Queued Z_editted generation for dance [{dkey}]")

    def _z_edit_busy_for_ui() -> bool:
        return bool(z_edit_busy)

    keyboard.add_callback("L", _on_reset)
    keyboard.add_callback(pose_cycle_key, _request_cycle_play)

    dance_listener = DanceKeyboardListener(
        dance_keys=hotkey_dance_keys,
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
    foot_ik_cfg = FootIkConfig(
        enable=bool(args_cli.mmd_foot_ik_enable),
        groove_pos_to_world=float(args_cli.groove_pos_to_world),
        pos_scale=float(args_cli.mmd_foot_ik_scale),
        weight=float(args_cli.mmd_foot_ik_weight),
        max_reach_ratio=float(args_cli.mmd_foot_ik_max_reach_ratio),
        mmd_axis_idx=tuple(foot_axis_idx),
        mmd_axis_sign=tuple(float(v) for v in foot_axis_sign),
        mmd_axis_sign_static_pose=tuple(float(v) for v in foot_axis_sign_pose),
        left_foot_ref_local=tuple(float(v) for v in left_ref_local),
        right_foot_ref_local=tuple(float(v) for v in right_ref_local),
        hip_offset_y=float(args_cli.mmd_foot_ik_hip_offset_y),
        hip_offset_z=float(args_cli.mmd_foot_ik_hip_offset_z),
        thigh_length=float(args_cli.mmd_foot_ik_thigh_length),
        shin_length=float(args_cli.mmd_foot_ik_shin_length),
        hip_roll_gain=float(args_cli.mmd_foot_ik_hip_roll_gain),
        debug_every_n_frames=max(0, int(args_cli.mmd_foot_ik_debug_every)),
    )
    try:
        robot_for_calib = env.unwrapped.scene["robot"]
        if _calibrate_foot_ik_refs_from_robot(robot_for_calib, foot_ik_cfg):
            lx, ly, lz = foot_ik_cfg.left_foot_ref_local
            rx, ry, rz = foot_ik_cfg.right_foot_ref_local
            txl, tyl, tzl = foot_ik_cfg.left_toe_ref_local
            txr, tyr, tzr = foot_ik_cfg.right_toe_ref_local
            print(
                "[INFO] Foot IK heel refs calibrated: "
                f"L=({lx:.3f},{ly:.3f},{lz:.3f}) R=({rx:.3f},{ry:.3f},{rz:.3f})"
            )
            print(
                "[INFO] Foot IK toe refs calibrated: "
                f"L=({txl:.3f},{tyl:.3f},{tzl:.3f}) R=({txr:.3f},{tyr:.3f},{tzr:.3f})"
            )
        else:
            print("[WARN] Foot IK calibration skipped; using configured heel/toe refs")
    except Exception as exc:
        print(f"[WARN] Foot IK heel calibration failed: {exc}")
    print(
        "[INFO] Foot IK: %s (scale=%.3f weight=%.3f reach=%.3f)"
        % (
            "on" if foot_ik_cfg.enable else "off",
            float(foot_ik_cfg.pos_scale),
            float(foot_ik_cfg.weight),
            float(foot_ik_cfg.max_reach_ratio),
        )
    )
    print(
        "[INFO] Foot IK axis idx=%s sign=%s pose_sign=%s debug_every=%d"
        % (
            str(tuple(int(v) for v in foot_ik_cfg.mmd_axis_idx)),
            str(tuple(float(v) for v in foot_ik_cfg.mmd_axis_sign)),
            str(tuple(float(v) for v in foot_ik_cfg.mmd_axis_sign_static_pose)),
            int(foot_ik_cfg.debug_every_n_frames),
        )
    )
    foot_ik_state = FootIkState()
    foot_ik_viz_cfg = FootIkVizConfig()
    foot_ik_viz = _FootIkTargetViz()
    pd_hold_joint_pos_cmd: Any = None
    pd_drive_enabled_ui = False
    z_offset_enabled_ui = False

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

    def _get_z_offset_enable_for_ui() -> bool:
        return bool(z_offset_enabled_ui)

    def _set_z_offset_enable_from_ui(enabled: bool) -> None:
        nonlocal z_offset_enabled_ui
        prev = bool(z_offset_enabled_ui)
        z_offset_enabled_ui = bool(enabled)
        if prev != z_offset_enabled_ui:
            mode = "on (*_z_editted sibling)" if z_offset_enabled_ui else "off (original motion)"
            print(f"[INFO] Z_offset_enable -> {mode}")

    def _get_audio_volume_for_ui() -> float:
        return audio_util.get_volume()

    def _set_audio_volume_from_ui(volume: float) -> None:
        audio_util.set_volume(volume)

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

    def _get_foot_ik_for_ui() -> dict[str, Any]:
        return {
            "enable": bool(foot_ik_cfg.enable),
            "scale": float(foot_ik_cfg.pos_scale),
            "weight": float(foot_ik_cfg.weight),
            "reach": float(foot_ik_cfg.max_reach_ratio),
            "axis_idx": tuple(int(v) for v in foot_ik_cfg.mmd_axis_idx),
            "axis_sign": tuple(float(v) for v in foot_ik_cfg.mmd_axis_sign),
            "left_ref": tuple(float(v) for v in foot_ik_cfg.left_foot_ref_local),
            "right_ref": tuple(float(v) for v in foot_ik_cfg.right_foot_ref_local),
            "debug_every": int(foot_ik_cfg.debug_every_n_frames),
        }

    def _set_foot_ik_from_ui(payload: dict[str, Any]) -> None:
        nonlocal foot_ik_cfg
        try:
            if "enable" in payload:
                foot_ik_cfg.enable = bool(payload.get("enable"))
            if "scale" in payload:
                foot_ik_cfg.pos_scale = float(payload.get("scale", foot_ik_cfg.pos_scale))
            if "weight" in payload:
                foot_ik_cfg.weight = float(payload.get("weight", foot_ik_cfg.weight))
            if "reach" in payload:
                foot_ik_cfg.max_reach_ratio = float(payload.get("reach", foot_ik_cfg.max_reach_ratio))
            if "axis_idx" in payload:
                ai = tuple(payload.get("axis_idx", foot_ik_cfg.mmd_axis_idx))
                if len(ai) == 3:
                    foot_ik_cfg.mmd_axis_idx = (
                        max(0, min(2, int(ai[0]))),
                        max(0, min(2, int(ai[1]))),
                        max(0, min(2, int(ai[2]))),
                    )
            if "axis_sign" in payload:
                sg = tuple(payload.get("axis_sign", foot_ik_cfg.mmd_axis_sign))
                if len(sg) == 3:
                    foot_ik_cfg.mmd_axis_sign = (float(sg[0]), float(sg[1]), float(sg[2]))
            if "left_ref" in payload:
                lv = tuple(payload.get("left_ref", foot_ik_cfg.left_foot_ref_local))
                if len(lv) == 3:
                    foot_ik_cfg.left_foot_ref_local = (float(lv[0]), float(lv[1]), float(lv[2]))
            if "right_ref" in payload:
                rv = tuple(payload.get("right_ref", foot_ik_cfg.right_foot_ref_local))
                if len(rv) == 3:
                    foot_ik_cfg.right_foot_ref_local = (float(rv[0]), float(rv[1]), float(rv[2]))
            if "debug_every" in payload:
                foot_ik_cfg.debug_every_n_frames = max(0, int(payload.get("debug_every", 0)))
        except Exception:
            return
        print(
            "[INFO] Foot IK UI update: enable=%s scale=%.3f weight=%.3f reach=%.3f axis_idx=%s sign=%s"
            % (
                "on" if foot_ik_cfg.enable else "off",
                float(foot_ik_cfg.pos_scale),
                float(foot_ik_cfg.weight),
                float(foot_ik_cfg.max_reach_ratio),
                str(tuple(int(v) for v in foot_ik_cfg.mmd_axis_idx)),
                str(tuple(float(v) for v in foot_ik_cfg.mmd_axis_sign)),
            )
        )

    def _get_foot_ik_viz_for_ui() -> dict[str, Any]:
        return {
            "scale": float(foot_ik_viz_cfg.pos_scale),
            "weight": float(foot_ik_viz_cfg.weight),
            "axis_idx": tuple(int(v) for v in foot_ik_viz_cfg.axis_idx),
            "axis_sign": tuple(float(v) for v in foot_ik_viz_cfg.axis_sign),
            "axis_sign_pose": tuple(float(v) for v in foot_ik_viz_cfg.axis_sign_pose),
            "left_ref_origin": tuple(float(v) for v in foot_ik_viz_cfg.left_ref_origin_m),
            "right_ref_origin": tuple(float(v) for v in foot_ik_viz_cfg.right_ref_origin_m),
        }

    def _set_foot_ik_viz_from_ui(payload: dict[str, Any]) -> None:
        nonlocal foot_ik_viz_cfg
        try:
            if "scale" in payload:
                foot_ik_viz_cfg.pos_scale = float(payload.get("scale", foot_ik_viz_cfg.pos_scale))
            if "weight" in payload:
                foot_ik_viz_cfg.weight = max(0.0, min(1.0, float(payload.get("weight", foot_ik_viz_cfg.weight))))
            if "axis_idx" in payload:
                ai = tuple(payload.get("axis_idx", foot_ik_viz_cfg.axis_idx))
                if len(ai) == 3:
                    foot_ik_viz_cfg.axis_idx = (
                        max(0, min(2, int(ai[0]))),
                        max(0, min(2, int(ai[1]))),
                        max(0, min(2, int(ai[2]))),
                    )
            if "axis_sign" in payload:
                sg = tuple(payload.get("axis_sign", foot_ik_viz_cfg.axis_sign))
                if len(sg) == 3:
                    foot_ik_viz_cfg.axis_sign = (float(sg[0]), float(sg[1]), float(sg[2]))
            if "axis_sign_pose" in payload:
                sp = tuple(payload.get("axis_sign_pose", foot_ik_viz_cfg.axis_sign_pose))
                if len(sp) == 3:
                    foot_ik_viz_cfg.axis_sign_pose = (float(sp[0]), float(sp[1]), float(sp[2]))
            if "left_ref_origin" in payload:
                lv = tuple(payload.get("left_ref_origin", foot_ik_viz_cfg.left_ref_origin_m))
                if len(lv) == 3:
                    foot_ik_viz_cfg.left_ref_origin_m = (float(lv[0]), float(lv[1]), float(lv[2]))
            if "right_ref_origin" in payload:
                rv = tuple(payload.get("right_ref_origin", foot_ik_viz_cfg.right_ref_origin_m))
                if len(rv) == 3:
                    foot_ik_viz_cfg.right_ref_origin_m = (float(rv[0]), float(rv[1]), float(rv[2]))
        except Exception:
            return
        print(
            "[INFO] Sphere map UI: scale=%.3f weight=%.3f idx=%s sign=%s pose_sign=%s "
            "Lorig=%s Rorig=%s"
            % (
                float(foot_ik_viz_cfg.pos_scale),
                float(foot_ik_viz_cfg.weight),
                str(tuple(int(v) for v in foot_ik_viz_cfg.axis_idx)),
                str(tuple(float(v) for v in foot_ik_viz_cfg.axis_sign)),
                str(tuple(float(v) for v in foot_ik_viz_cfg.axis_sign_pose)),
                str(tuple(float(v) for v in foot_ik_viz_cfg.left_ref_origin_m)),
                str(tuple(float(v) for v in foot_ik_viz_cfg.right_ref_origin_m)),
            )
        )

    set_joint_value_provider(lambda: joint_pos_deg_cache)
    set_playback_status_provider(_playback_status_for_ui)
    set_playback_transport_callbacks(_ui_toggle_pause, _ui_seek_frame)
    set_dance_play_callbacks(_dance_entries_for_ui, _on_dance_request_from_ui)
    set_dance_z_edit_callbacks(_dance_has_z_editted, _on_z_edit_request_from_ui, _z_edit_busy_for_ui)
    set_pd_drive_callbacks(_get_pd_drive_for_ui, _set_pd_drive_from_ui)
    set_z_offset_enable_callbacks(_get_z_offset_enable_for_ui, _set_z_offset_enable_from_ui)
    set_audio_volume_callbacks(_get_audio_volume_for_ui, _set_audio_volume_from_ui)
    set_root_quat_rpy_callbacks(_get_root_quat_rpy_for_ui, _set_root_quat_rpy_from_ui)
    set_foot_ik_callbacks(_get_foot_ik_for_ui, _set_foot_ik_from_ui)
    set_foot_ik_viz_callbacks(_get_foot_ik_viz_for_ui, _set_foot_ik_viz_from_ui)
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
        # Foot IK diagnostics: MMD local / FK world / Isaac red sphere (meters).
        def _cache_xyz(prefix: str, pos: tuple[float, float, float] | None) -> None:
            if pos is None:
                return
            joint_pos_deg_cache[f"{prefix}_x"] = float(pos[0])
            joint_pos_deg_cache[f"{prefix}_y"] = float(pos[1])
            joint_pos_deg_cache[f"{prefix}_z"] = float(pos[2])

        _cache_xyz("__foot_ik_l_local", foot_ik_state.last_left_foot_mmd_local_m)
        _cache_xyz("__foot_ik_r_local", foot_ik_state.last_right_foot_mmd_local_m)
        _cache_xyz("__foot_ik_l", foot_ik_state.last_left_foot_mmd_viz_world)
        _cache_xyz("__foot_ik_r", foot_ik_state.last_right_foot_mmd_viz_world)
        _cache_xyz("__toe_ik_l", foot_ik_state.last_left_toe_mmd_viz_world)
        _cache_xyz("__toe_ik_r", foot_ik_state.last_right_toe_mmd_viz_world)

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
        foot_ik_state.reset()
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
        foot_cfg = FootIkConfig(**vars(foot_ik_cfg))
        foot_cfg.is_static_pose = bool(len(motion_bundle["frame_list"]) <= 1)
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
            foot_ik_cfg=foot_cfg,
            foot_ik_state=foot_ik_state,
            foot_ik_viz_cfg=foot_ik_viz_cfg,
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
                pending_z_edit_key = None
                ui_debug.reset()
                initial_root_snapshot_row = robot_root_row_clone(env)
                _reset_to_initial_pose(sync_ui_cache=True)
                print("[INFO] 环境已重置")

            if pending_z_edit_key is not None and not z_edit_busy:
                dkey = pending_z_edit_key
                pending_z_edit_key = None
                motion_path = dance_path_by_key.get(dkey, "")
                if motion_path and not has_z_editted_sibling(motion_path):
                    z_edit_busy = True
                    if is_playing:
                        audio_util.stop_wav()
                        motion_has_wav = False
                        is_playing = False
                        current_motion = None
                        current_motion_label = ""
                        last_csv_motion_frame = None
                        playback_paused = False
                    try:
                        _ensure_joint_info()
                        print(
                            f"[INFO] Generating *_z_editted for dance [{dkey}] "
                            f"({os.path.basename(motion_path)}) ..."
                        )
                        generate_z_editted_motion(
                            env,
                            env_cfg,
                            motion_path,
                            config=RootZEditConfig(
                                groove_pos_to_world=float(args_cli.groove_pos_to_world),
                                mmd_center_to_root_offset_local_xyz=tuple(
                                    args_cli.mmd_center_to_root_offset_local_xyz
                                ),
                                root_quat_rpy_scale=tuple(root_quat_rpy_scale),
                                root_quat_rpy_axis_idx=tuple(root_quat_rpy_axis_idx),
                                knee_hinge_projection=bool(args_cli.mmd_knee_hinge_projection),
                            ),
                        )
                        initial_root_snapshot_row = robot_root_row_clone(env)
                        _reset_to_initial_pose(sync_ui_cache=True)
                    except Exception as exc:
                        print(f"[ERROR] Z_editted generation failed for [{dkey}]: {exc}")
                    finally:
                        z_edit_busy = False

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
                    entry, _used_z_editted = resolve_playback_motion_entry(
                        entry,
                        prefer_hdf5=prefer_hdf5,
                        z_offset_enabled=z_offset_enabled_ui,
                    )
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
                foot_ik_viz.update(
                    left_world=foot_ik_state.last_left_foot_mmd_viz_world,
                    right_world=foot_ik_state.last_right_foot_mmd_viz_world,
                    left_toe_world=foot_ik_state.last_left_toe_mmd_viz_world,
                    right_toe_world=foot_ik_state.last_right_toe_mmd_viz_world,
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
