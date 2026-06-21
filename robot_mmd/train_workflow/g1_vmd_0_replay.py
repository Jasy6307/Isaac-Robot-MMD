# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 MMD 动作回放主入口（Isaac Sim）。

功能概览：
1) 舞蹈由 dances_config.yaml 登记（请从 dances_config.example.yaml 复制）；键、motion(.csv/.h5/.hdf5)、可选音频；pose 目录 P 键循环；
2) 支持关节映射 UI，实时显示当前关节角度；
3) 有 audio 的 dance 播 WAV，与动作同一「逻辑帧时间轴」；
4) 在重置和切换动作时维护控制参考姿态，避免姿态回弹。
5) 映射 UI 顶部 ``PD Drive`` 复选框：勾选=全身关节 PD，不勾选=关节瞬移写入。
6) 映射 UI ``Z_offset_enable``：勾选时自动播放同目录 ``*_z_editted.*`` sibling（无则回退原版并 WARN）。
7) 启动时扫描 ``media/dance/*.vmd``，自动生成缺失的 CSV/H5 并登记到 ``dances_config.yaml``（可无快捷键，UI 可选）。

启动：``python robot_mmd/train_workflow/g1_vmd_0_replay.py``
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

from robot_mmd.train_workflow.utils.playback.cli import (
    apply_app_window_kit_flags,
    build_arg_parser,
    default_playback_foot_ik_config,
    parse_center_to_root_offset,
)

PLAYBACK_LOG_FRAME_STRIDE = 50

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

from robot_mmd.train_workflow.utils.motion.sync import sync_dance_assets_from_vmd

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

from robot_mmd.my_task.robots.g1_29dof_o6_cfg import G1_29DOF_O6_CFG
from robot_mmd.my_task.robots.actuator_pd import (
    apply_robot_pd_profile,
    log_pd_profile_summary,
)
from robot_mmd.train_workflow.utils.retarget.joint_axis_map import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.ui.jointRPY_maping_ui import create_joint_rpy_mapping_ui
from robot_mmd.train_workflow.ui.mmd_config_ui import (
    create_mmd_config_ui,
    set_dance_play_callbacks,
    set_dance_record_h5_callbacks,
    set_dance_z_edit_callbacks,
    set_audio_volume_callbacks,
    set_foot_ik_callbacks,
    set_foot_ground_comp_callbacks,
    set_joint_value_provider,
    set_mapping_changed_callback,
    set_pd_drive_callbacks,
    set_playback_status_provider,
    set_playback_transport_callbacks,
    set_root_rot_bone_name_provider,
    set_root_quat_rpy_callbacks,
    set_root_z_compress_callbacks,
    set_z_offset_enable_callbacks,
)
from robot_mmd.train_workflow.ui.retargeting_tune_ui import create_retarget_tune_ui
from robot_mmd.train_workflow.utils.media import audio_util
from robot_mmd.train_workflow.utils.format.csv_loader import (
    FootIkConfig,
    FootIkState,
    elbow_hinge_mapping_ui_extra,
    knee_hinge_mapping_ui_extra,
    retarget_leg_debug_ui_extra,
    shoulder_retarget_debug_ui_extra,
)
from robot_mmd.train_workflow.utils.motion.loader import (
    MotionBundle,
    build_dance_hand_hdf5_motion_by_key,
    build_dance_hand_motion_by_key,
    build_dance_hdf5_motion_by_key,
    delete_h5_siblings,
    delete_z_editted_siblings,
    format_playback_log_label,
    has_deletable_h5_sibling,
    has_z_editted_sibling,
    load_dances_from_yaml,
    load_motion,
    load_pose_motion_dir,
    resolve_playback_motion_entry,
)
from robot_mmd.train_workflow.utils.ik.ankle_ground import (
    FootAnkleGroundCompConfig,
    FootAnkleGroundCompState,
    apply_ankle_ground_comp_to_joint_cmd,
)
from robot_mmd.train_workflow.utils.ik.mmd_fk import (
    default_foot_ik_viz_config,
    motion_has_embedded_foot_ik,
)
from robot_mmd.train_workflow.utils.playback.recorder import PlaybackH5Recorder
from robot_mmd.train_workflow.utils.playback.root_z import (
    RootZEditConfig,
    generate_z_editted_motion,
    read_ankle_roll_link_world_positions,
    resolve_ankle_roll_link_body_indices,
)
from robot_mmd.train_workflow.utils.playback.targets import (
    MotionRootTrackState,
    PlaybackUiDebugState,
    RootZCompressConfig,
    build_joint_pos_deg_cache,
    compute_targets_for_hdf5_frame,
    compute_targets_for_motion_frame,
)
from robot_mmd.train_workflow.utils.playback.sim_robot import (
    apply_joint_state_instant,
    apply_root_pos_instant,
    robot_root_row_clone,
)
from robot_mmd.train_workflow.utils.math.trans_util import (
    dist3,
    quat_angular_error_deg,
    quat_wxyz_to_euler_xyz_deg,
    root_quat_from_state_row,
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
    """Render MMD foot IK targets (red) and ankle_roll_link poses (green) in USD."""

    _FOOT_IK_COLOR_OK = (1.0, 0.0, 0.0)
    _FOOT_IK_COLOR_CLAMP = (1.0, 1.0, 0.0)

    _LEFT_FOOT_IK = "/World/Debug/FootIkTargets/LeftFootIK"
    _RIGHT_FOOT_IK = "/World/Debug/FootIkTargets/RightFootIK"
    _LEFT_TOE_IK = "/World/Debug/FootIkTargets/LeftToeIK"
    _RIGHT_TOE_IK = "/World/Debug/FootIkTargets/RightToeIK"
    _LEFT_ANKLE_LINK = "/World/Debug/FootIkTargets/LeftAnkleRollLink"
    _RIGHT_ANKLE_LINK = "/World/Debug/FootIkTargets/RightAnkleRollLink"
    _LEFT_IK_PRED = "/World/Debug/FootIkTargets/LeftIkPred"
    _RIGHT_IK_PRED = "/World/Debug/FootIkTargets/RightIkPred"
    _LEFT_IK_TARGET = "/World/Debug/FootIkTargets/LeftIkTarget"
    _RIGHT_IK_TARGET = "/World/Debug/FootIkTargets/RightIkTarget"
    _ROOT_TARGET = "/World/Debug/FootIkTargets/RootTarget"
    _ROOT_SIM = "/World/Debug/FootIkTargets/RootSim"

    _TRANSLATE_EPS_M = 1e-5

    def __init__(self) -> None:
        self._ready = False
        self._stage = None
        self._xform_cache: dict[str, Any] = {}
        self._last_translate: dict[str, tuple[float, float, float]] = {}
        self._last_visible: dict[str, bool] = {}
        self._last_color: dict[str, tuple[float, float, float]] = {}

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
                prim = sph.GetPrim()
                pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                    "displayColor",
                    Sdf.ValueTypeNames.Color3fArray,
                    UsdGeom.Tokens.constant,
                )
                pv.Set([Gf.Vec3f(1.0, 0.0, 0.0)])
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
            ap = f"{root}/{side}AnkleRollLink"
            if not stage.GetPrimAtPath(ap):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(ap))
                sph.GetRadiusAttr().Set(0.022)
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
            prim_a = stage.GetPrimAtPath(ap)
            self._xform_cache[ap] = UsdGeom.Xformable(prim_a)
            pp = f"{root}/{side}IkPred"
            if not stage.GetPrimAtPath(pp):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(pp))
                sph.GetRadiusAttr().Set(0.02)
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set([Gf.Vec3f(0.75, 0.0, 0.85)])
            prim_p = stage.GetPrimAtPath(pp)
            self._xform_cache[pp] = UsdGeom.Xformable(prim_p)
            ip = f"{root}/{side}IkTarget"
            if not stage.GetPrimAtPath(ip):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(ip))
                sph.GetRadiusAttr().Set(0.023)
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.55, 0.0)])
            prim_i = stage.GetPrimAtPath(ip)
            self._xform_cache[ip] = UsdGeom.Xformable(prim_i)
        for label, path, radius, color in (
            ("RootTarget", self._ROOT_TARGET, 0.03, (0.1, 0.35, 1.0)),
            ("RootSim", self._ROOT_SIM, 0.028, (0.2, 0.85, 1.0)),
        ):
            del label
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
                sph.GetRadiusAttr().Set(float(radius))
                xform = UsdGeom.Xformable(sph.GetPrim())
                xform.AddTranslateOp()
                color_api = UsdGeom.Gprim(sph.GetPrim())
                color_api.CreateDisplayColorAttr().Set(
                    [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
                )
            prim_r = stage.GetPrimAtPath(path)
            self._xform_cache[path] = UsdGeom.Xformable(prim_r)
        self._stage = stage
        self._ready = True
        return True

    def _set_visible(self, path: str, visible: bool) -> None:
        from pxr import UsdGeom

        if self._last_visible.get(path) == visible:
            return
        self._last_visible[path] = visible
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

        key = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
        prev = self._last_translate.get(path)
        eps = self._TRANSLATE_EPS_M
        if prev is not None:
            if (
                abs(prev[0] - key[0]) <= eps
                and abs(prev[1] - key[1]) <= eps
                and abs(prev[2] - key[2]) <= eps
            ):
                return
        self._last_translate[path] = key
        xformable = self._xform_cache.get(path)
        if xformable is None:
            return
        ops = xformable.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(key[0], key[1], key[2]))
        else:
            xformable.AddTranslateOp().Set(Gf.Vec3d(key[0], key[1], key[2]))

    def _set_display_color(self, path: str, rgb: tuple[float, float, float]) -> None:
        from pxr import Gf, Sdf, UsdGeom

        key = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        if self._last_color.get(path) == key:
            return
        self._last_color[path] = key
        stage = self._stage
        if stage is None:
            return
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        color = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        gprim = UsdGeom.Gprim(prim)
        attr = gprim.GetDisplayColorAttr()
        if attr and attr.IsValid():
            attr.Set([color])
            return
        pv_api = UsdGeom.PrimvarsAPI(prim)
        pv = pv_api.GetPrimvar("displayColor")
        if pv and pv.IsDefined():
            pv.Set([color])
            return
        pv = pv_api.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.constant,
        )
        pv.Set([color])

    def update(
        self,
        *,
        left_world: tuple[float, float, float] | None,
        right_world: tuple[float, float, float] | None,
        left_toe_world: tuple[float, float, float] | None,
        right_toe_world: tuple[float, float, float] | None,
        left_reach_clamped: bool = False,
        right_reach_clamped: bool = False,
    ) -> None:
        if not self._ensure():
            return
        for path, pos, reach_clamped in (
            (self._LEFT_FOOT_IK, left_world, left_reach_clamped),
            (self._RIGHT_FOOT_IK, right_world, right_reach_clamped),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_display_color(
                    path,
                    self._FOOT_IK_COLOR_CLAMP if reach_clamped else self._FOOT_IK_COLOR_OK,
                )
                self._set_visible(path, True)
        for path, pos in (
            (self._LEFT_TOE_IK, left_toe_world),
            (self._RIGHT_TOE_IK, right_toe_world),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_visible(path, True)

    def update_ankle_links(
        self,
        *,
        left_ankle_world: tuple[float, float, float] | None,
        right_ankle_world: tuple[float, float, float] | None,
    ) -> None:
        """Show simulated ``ankle_roll_link`` world origins (green spheres)."""
        if not self._ensure():
            return
        for path, pos in (
            (self._LEFT_ANKLE_LINK, left_ankle_world),
            (self._RIGHT_ANKLE_LINK, right_ankle_world),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_visible(path, True)

    def update_ik_pred(
        self,
        *,
        left_pred_world: tuple[float, float, float] | None,
        right_pred_world: tuple[float, float, float] | None,
    ) -> None:
        """Show full IK FK prediction at ankle_roll_link (purple spheres)."""
        if not self._ensure():
            return
        for path, pos in (
            (self._LEFT_IK_PRED, left_pred_world),
            (self._RIGHT_IK_PRED, right_pred_world),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_visible(path, True)

    def update_ik_target(
        self,
        *,
        left_target_world: tuple[float, float, float] | None,
        right_target_world: tuple[float, float, float] | None,
    ) -> None:
        """Show IK chase target after ankle offset (orange spheres)."""
        if not self._ensure():
            return
        for path, pos in (
            (self._LEFT_IK_TARGET, left_target_world),
            (self._RIGHT_IK_TARGET, right_target_world),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_visible(path, True)

    def update_root_debug(
        self,
        *,
        target_root_world: tuple[float, float, float] | None,
        sim_root_world: tuple[float, float, float] | None,
    ) -> None:
        """Show commanded root target (blue) and simulated root (cyan)."""
        if not self._ensure():
            return
        for path, pos in (
            (self._ROOT_TARGET, target_root_world),
            (self._ROOT_SIM, sim_root_world),
        ):
            if pos is None:
                self._set_visible(path, False)
            else:
                self._set_translate(path, pos)
                self._set_visible(path, True)

    def clear(self) -> None:
        """Hide all debug spheres (idle / stop / reset)."""
        self.update(
            left_world=None,
            right_world=None,
            left_toe_world=None,
            right_toe_world=None,
        )
        self.update_ankle_links(left_ankle_world=None, right_ankle_world=None)
        self.update_ik_pred(left_pred_world=None, right_pred_world=None)
        self.update_ik_target(left_target_world=None, right_target_world=None)
        self.update_root_debug(target_root_world=None, sim_root_world=None)


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
    dance_path_by_key = {
        k: str(data.get("path", ""))
        for k, (_name, data) in dance_motion_by_key.items()
        if str(data.get("path", "")).strip()
    }
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    from robot_mmd.my_task.g1_replay_env_cfg import G1_TPOSE_INIT_STATE

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
    print(f"[INFO] L=重置, {pose_cycle_key}=按序播放 pose；选舞请用 Mapping UI 下拉框")
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

    def _dance_combo_label(filename: str, key: str) -> str:
        name = str(filename)
        if name.lower().endswith(".csv"):
            name = name[:-4]
        if str(key).startswith("ui:"):
            return name
        return f"[{key}] {name}"

    def _dance_entries_for_ui() -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for key in dance_motion_by_key.keys():
            filename, _data = dance_motion_by_key[key]
            entries.append((str(key), _dance_combo_label(filename, key)))
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

    def _dance_z_edit_ui_status(key: str) -> str:
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        entry = dance_motion_by_key.get(dkey)
        if entry is not None:
            _name, data = entry
            if str(data.get("kind", "")) == "csv":
                frames = data.get("frames")
                if motion_has_embedded_foot_ik(frames):
                    return "ik_control"
        path = dance_path_by_key.get(dkey, "")
        if path and has_z_editted_sibling(path):
            return "available"
        return "missing"

    def _on_z_edit_request_from_ui(key: str) -> None:
        nonlocal pending_z_edit_key
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        if _dance_z_edit_ui_status(key) == "ik_control":
            print(f"[INFO] dance [{dkey}] has foot IK data; Z_editted is not needed")
            return
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

    def _h5_record_busy_for_ui() -> bool:
        return bool(h5_record_busy)

    def _dance_has_h5(key: str) -> bool:
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        return dkey in dance_hdf5_motion_by_key

    def _dance_h5_deletable(key: str) -> bool:
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        path = dance_path_by_key.get(dkey, "")
        return has_deletable_h5_sibling(path)

    def _on_delete_z_edit_from_ui(key: str) -> None:
        if z_edit_busy or is_playing or h5_record_busy:
            print("[WARN] Cannot delete Z_editted while playback or generation is active")
            return
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        if _dance_z_edit_ui_status(key) != "available":
            print(f"[INFO] dance [{dkey}] has no deletable Z_editted sibling")
            return
        path = dance_path_by_key.get(dkey, "")
        if not path:
            print(f"[WARN] dance [{dkey}] has no motion path")
            return
        deleted = delete_z_editted_siblings(path)
        if deleted:
            print(
                f"[INFO] Deleted Z_editted sibling(s) for dance [{dkey}]: "
                + ", ".join(os.path.basename(p) for p in deleted)
            )
        else:
            print(f"[WARN] No Z_editted sibling deleted for dance [{dkey}]")

    def _on_delete_h5_from_ui(key: str) -> None:
        nonlocal dance_hdf5_motion_by_key, dance_hand_hdf5_motion_by_key
        if h5_record_busy or is_playing or z_edit_busy:
            print("[WARN] Cannot delete H5 while playback or recording is active")
            return
        dkey = _dance_lookup_key(key) if not str(key).startswith("ui:") else str(key)
        path = dance_path_by_key.get(dkey, "")
        if not has_deletable_h5_sibling(path):
            print(f"[INFO] dance [{dkey}] has no deletable H5 sibling")
            return
        deleted = delete_h5_siblings(path)
        dance_hdf5_motion_by_key = build_dance_hdf5_motion_by_key(dance_motion_by_key)
        dance_hand_hdf5_motion_by_key = build_dance_hand_hdf5_motion_by_key(dance_hand_motion_by_key)
        if deleted:
            print(
                f"[INFO] Deleted H5 sibling(s) for dance [{dkey}]: "
                + ", ".join(os.path.basename(p) for p in deleted)
            )
        else:
            print(f"[WARN] No H5 sibling deleted for dance [{dkey}]")

    def _on_record_h5_request_from_ui(key: str) -> None:
        nonlocal pending_record_h5_key
        if h5_record_busy or is_playing:
            print("[WARN] Cannot start H5 recording while playback is active")
            return
        pending_record_h5_key = str(key)

    keyboard.add_callback("L", _on_reset)
    keyboard.add_callback(pose_cycle_key, _request_cycle_play)

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
    pending_playback_stop = False
    pending_seek_frame: int | None = None
    pending_record_h5_key: str | None = None
    h5_record_busy = False
    h5_record_active = False
    h5_recorder: PlaybackH5Recorder | None = None
    h5_record_frame_cursor = 0
    h5_record_dance_key: str | None = None
    motion_has_wav = False
    root_quat_rpy_scale = list(MMD_ROOT_QUAT_RPY_SCALE_DEFAULT)
    root_quat_rpy_axis_idx = list(MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT)
    foot_ik_cfg = default_playback_foot_ik_config(
        groove_pos_to_world=float(args_cli.groove_pos_to_world),
    )
    foot_ankle_ground_comp_cfg = FootAnkleGroundCompConfig()
    foot_ik_viz_cfg = default_foot_ik_viz_config()
    print(
        "[INFO] Leg IK: %s (reach=%.3f debug_every=%d)"
        % (
            "on" if foot_ik_cfg.enable else "off",
            float(foot_ik_cfg.max_reach_ratio),
            int(foot_ik_cfg.debug_every_n_frames),
        )
    )
    print(
        "[INFO] Leg IK DLS: iters=%d pos_tol=%.4fm reg=%.3f hy=%.2f ar=%.2f pass_through_ankle=%s"
        % (
            int(foot_ik_cfg.ik_max_iters),
            float(foot_ik_cfg.ik_pos_tol_m),
            float(foot_ik_cfg.ik_reg_weight),
            float(foot_ik_cfg.ik_reg_hip_yaw),
            float(foot_ik_cfg.ik_reg_ankle_roll),
            "on" if foot_ik_cfg.ik_pass_through_ankle else "off",
        )
    )
    print(
        "[INFO] Leg IK geom: hip_y=%.4f hip_z=%.4f thigh=%.4f shin=%.4f max_reach=%.4fm"
        % (
            float(foot_ik_cfg.hip_offset_y),
            float(foot_ik_cfg.hip_offset_z),
            float(foot_ik_cfg.thigh_length),
            float(foot_ik_cfg.shin_length),
            float(foot_ik_cfg.thigh_length + foot_ik_cfg.shin_length)
            * float(foot_ik_cfg.max_reach_ratio),
        )
    )
    print(
        "[INFO] Sphere map: scale=%.3f idx=%s sign=%s Lorig=%s Rorig=%s"
        % (
            float(foot_ik_viz_cfg.pos_scale),
            str(tuple(int(v) for v in foot_ik_viz_cfg.axis_idx)),
            str(tuple(float(v) for v in foot_ik_viz_cfg.axis_sign)),
            str(tuple(float(v) for v in foot_ik_viz_cfg.left_ref_origin_m)),
            str(tuple(float(v) for v in foot_ik_viz_cfg.right_ref_origin_m)),
        )
    )
    print(
        "[INFO] Ankle ground comp: %s ground_z=%.3f clearance=%.1fmm max_pitch=%.0fdeg max_roll=%.0fdeg"
        % (
            "on" if foot_ankle_ground_comp_cfg.enable else "off",
            float(foot_ankle_ground_comp_cfg.ground_z),
            float(foot_ankle_ground_comp_cfg.clearance_m) * 1000.0,
            float(foot_ankle_ground_comp_cfg.max_pitch_delta_deg),
            float(foot_ankle_ground_comp_cfg.max_roll_delta_deg),
        )
    )
    foot_ik_state = FootIkState()
    foot_ankle_ground_comp_state = FootAnkleGroundCompState()
    foot_ik_viz = _FootIkTargetViz()
    ankle_link_body_indices: dict[str, int] | None = None
    last_playback_target_root: tuple[float, float, float] | None = None
    last_playback_target_root_quat: list[float] | None = None
    last_frame_joint_pos_cmd: Any = None
    print(
        "[INFO] Debug spheres: red=MMD foot, orange=IK target (offset), "
        "yellow=reach clamp, green=ankle sim, purple=IK FK pred, "
        "blue=root cmd, cyan=root sim"
    )
    pd_hold_joint_pos_cmd: Any = None
    pd_drive_enabled_ui = False
    z_offset_enabled_ui = False
    root_z_baseline_offset_m_ui = 0.76
    root_z_outlier_scale_ui = 0.6
    foot_ankle_ground_comp_enabled_ui = bool(foot_ankle_ground_comp_cfg.enable)

    def _apply_ankle_ground_comp_to_cmd(
        joint_pos_cmd: Any,
        root_pos: tuple[float, float, float] | None,
        root_quat_wxyz: list[float] | None,
        joint_default: Any | None = None,
    ) -> None:
        if not foot_ankle_ground_comp_enabled_ui or joint_pos_cmd is None:
            return
        jd = joint_default if joint_default is not None else default_joint_pos
        if jd is None or not joint_names:
            return
        foot_ankle_ground_comp_cfg.enable = bool(foot_ankle_ground_comp_enabled_ui)
        apply_ankle_ground_comp_to_joint_cmd(
            joint_pos_cmd,
            joint_names,
            jd,
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
            cfg=foot_ankle_ground_comp_cfg,
            state=foot_ankle_ground_comp_state,
        )

    def _update_ankle_link_debug_viz(
        target_root_pos: tuple[float, float, float] | None = None,
        target_root_quat_wxyz: list[float] | None = None,
    ) -> None:
        nonlocal ankle_link_body_indices
        left_ankle: tuple[float, float, float] | None = None
        right_ankle: tuple[float, float, float] | None = None
        try:
            robot_inner = env.unwrapped.scene["robot"]
            if ankle_link_body_indices is None:
                ankle_link_body_indices = resolve_ankle_roll_link_body_indices(robot_inner)
            left_ankle, right_ankle = read_ankle_roll_link_world_positions(
                robot_inner,
                body_indices=ankle_link_body_indices,
            )
            foot_ik_viz.update_ankle_links(
                left_ankle_world=left_ankle,
                right_ankle_world=right_ankle,
            )
            foot_ik_viz.update_ik_pred(
                left_pred_world=foot_ik_state.last_left_ik_pred_world,
                right_pred_world=foot_ik_state.last_right_ik_pred_world,
            )
            foot_ik_viz.update_ik_target(
                left_target_world=foot_ik_state.last_left_ik_target_world,
                right_target_world=foot_ik_state.last_right_ik_target_world,
            )
        except Exception:
            foot_ik_viz.update_ankle_links(left_ankle_world=None, right_ankle_world=None)
            foot_ik_viz.update_ik_pred(left_pred_world=None, right_pred_world=None)
            foot_ik_viz.update_ik_target(left_target_world=None, right_target_world=None)
        root_for_dbg = target_root_pos
        if root_for_dbg is None:
            root_for_dbg = foot_ik_state.last_target_root_world
        quat_for_dbg = target_root_quat_wxyz
        if quat_for_dbg is None:
            quat_for_dbg = foot_ik_state.last_target_root_quat_wxyz
        _update_foot_root_debug_metrics(
            target_root_pos=root_for_dbg,
            target_root_quat_wxyz=quat_for_dbg,
            left_ankle_world=left_ankle,
            right_ankle_world=right_ankle,
        )

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

    def _get_root_z_compress_for_ui() -> tuple[float, float]:
        return float(root_z_baseline_offset_m_ui), float(root_z_outlier_scale_ui)

    def _set_root_z_compress_from_ui(baseline_offset_m: float, outlier_scale: float) -> None:
        nonlocal root_z_baseline_offset_m_ui, root_z_outlier_scale_ui, mapping_reapply_requested
        new_off = float(baseline_offset_m)
        new_scale = max(0.0, min(1.0, float(outlier_scale)))
        changed = (
            abs(float(root_z_baseline_offset_m_ui) - new_off) > 1e-6
            or abs(float(root_z_outlier_scale_ui) - new_scale) > 1e-6
        )
        root_z_baseline_offset_m_ui = new_off
        root_z_outlier_scale_ui = new_scale
        if changed:
            mapping_reapply_requested = True
            print(
                "[INFO] Root Z compress -> baseline_off=%.3fm outlier_scale=%.3f"
                % (root_z_baseline_offset_m_ui, root_z_outlier_scale_ui)
            )

    def _get_foot_ground_comp_for_ui() -> bool:
        return bool(foot_ankle_ground_comp_enabled_ui)

    def _set_foot_ground_comp_from_ui(enabled: bool) -> None:
        nonlocal foot_ankle_ground_comp_enabled_ui
        prev = bool(foot_ankle_ground_comp_enabled_ui)
        foot_ankle_ground_comp_enabled_ui = bool(enabled)
        if prev != foot_ankle_ground_comp_enabled_ui:
            print(
                "[INFO] Ankle ground comp -> %s (clearance=%.1fmm max_pitch=%.0fdeg max_roll=%.0fdeg)"
                % (
                    "on" if foot_ankle_ground_comp_enabled_ui else "off",
                    float(foot_ankle_ground_comp_cfg.clearance_m) * 1000.0,
                    float(foot_ankle_ground_comp_cfg.max_pitch_delta_deg),
                    float(foot_ankle_ground_comp_cfg.max_roll_delta_deg),
                )
            )

    def _get_audio_volume_for_ui() -> float:
        return audio_util.get_volume()

    def _set_audio_volume_from_ui(volume: float) -> None:
        audio_util.set_volume(volume)

    def _playback_status_for_ui() -> dict[str, Any]:
        if h5_record_active and is_playing and current_motion and (current_motion_label or "").strip():
            tag = format_playback_log_label(current_motion_label)
            frame_list = current_motion["frame_list"]
            max_f = int(frame_list[-1])
            fr = int(h5_record_frame_cursor)
            if fr > max_f:
                fr = max_f
            return {
                "playing": True,
                "tag": f"{tag} [Record H5]",
                "frame": fr,
                "max_frame": max_f,
                "playback_paused": False,
                "kind": "dance",
            }
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

    def _ui_stop_playback() -> None:
        nonlocal pending_playback_stop
        if is_playing and current_motion:
            pending_playback_stop = True

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
        off = foot_ik_cfg.ankle_target_offset_local
        return {
            "enable": bool(foot_ik_cfg.enable),
            "reach": float(foot_ik_cfg.max_reach_ratio),
            "leg_scale": float(foot_ik_cfg.leg_target_scale),
            "debug_every": int(foot_ik_cfg.debug_every_n_frames),
            "ik_reg_weight": float(foot_ik_cfg.ik_reg_weight),
            "ik_reg_hip_yaw": float(foot_ik_cfg.ik_reg_hip_yaw),
            "ik_reg_ankle_roll": float(foot_ik_cfg.ik_reg_ankle_roll),
            "ankle_offset": (float(off[0]), float(off[1]), float(off[2])),
        }

    def _set_foot_ik_from_ui(payload: dict[str, Any]) -> None:
        nonlocal foot_ik_cfg
        try:
            if "enable" in payload:
                foot_ik_cfg.enable = bool(payload.get("enable"))
            if "reach" in payload:
                foot_ik_cfg.max_reach_ratio = float(payload.get("reach", foot_ik_cfg.max_reach_ratio))
            if "leg_scale" in payload:
                foot_ik_cfg.leg_target_scale = float(
                    payload.get("leg_scale", foot_ik_cfg.leg_target_scale)
                )
            if "debug_every" in payload:
                foot_ik_cfg.debug_every_n_frames = max(0, int(payload.get("debug_every", 0)))
            if "ik_reg_weight" in payload:
                foot_ik_cfg.ik_reg_weight = float(payload.get("ik_reg_weight", foot_ik_cfg.ik_reg_weight))
            if "ik_reg_hip_yaw" in payload:
                foot_ik_cfg.ik_reg_hip_yaw = float(payload.get("ik_reg_hip_yaw", foot_ik_cfg.ik_reg_hip_yaw))
            if "ik_reg_ankle_roll" in payload:
                foot_ik_cfg.ik_reg_ankle_roll = float(
                    payload.get("ik_reg_ankle_roll", foot_ik_cfg.ik_reg_ankle_roll)
                )
            if "ankle_offset" in payload:
                ao = tuple(payload.get("ankle_offset", foot_ik_cfg.ankle_target_offset_local))
                if len(ao) == 3:
                    foot_ik_cfg.ankle_target_offset_local = (
                        float(ao[0]),
                        float(ao[1]),
                        float(ao[2]),
                    )
        except Exception:
            return
        print(
            "[INFO] Leg IK UI: enable=%s reach=%.3f leg_scale=%.3f reg=%.3f offset=(%.4f,%.4f,%.4f)"
            % (
                "on" if foot_ik_cfg.enable else "off",
                float(foot_ik_cfg.max_reach_ratio),
                float(foot_ik_cfg.leg_target_scale),
                float(foot_ik_cfg.ik_reg_weight),
                float(foot_ik_cfg.ankle_target_offset_local[0]),
                float(foot_ik_cfg.ankle_target_offset_local[1]),
                float(foot_ik_cfg.ankle_target_offset_local[2]),
            )
        )

    set_joint_value_provider(lambda: joint_pos_deg_cache)
    set_playback_status_provider(_playback_status_for_ui)
    set_playback_transport_callbacks(_ui_toggle_pause, _ui_seek_frame, _ui_stop_playback)
    set_dance_play_callbacks(_dance_entries_for_ui, _on_dance_request_from_ui)
    set_dance_z_edit_callbacks(
        _dance_has_z_editted,
        _on_z_edit_request_from_ui,
        _z_edit_busy_for_ui,
        _dance_z_edit_ui_status,
        _on_delete_z_edit_from_ui,
    )
    set_dance_record_h5_callbacks(
        _on_record_h5_request_from_ui,
        _h5_record_busy_for_ui,
        _dance_has_h5,
        _on_delete_h5_from_ui,
        _dance_h5_deletable,
    )
    set_pd_drive_callbacks(_get_pd_drive_for_ui, _set_pd_drive_from_ui)
    set_z_offset_enable_callbacks(_get_z_offset_enable_for_ui, _set_z_offset_enable_from_ui)
    set_root_z_compress_callbacks(_get_root_z_compress_for_ui, _set_root_z_compress_from_ui)
    set_foot_ground_comp_callbacks(_get_foot_ground_comp_for_ui, _set_foot_ground_comp_from_ui)
    set_audio_volume_callbacks(_get_audio_volume_for_ui, _set_audio_volume_from_ui)
    set_root_quat_rpy_callbacks(_get_root_quat_rpy_for_ui, _set_root_quat_rpy_from_ui)
    set_foot_ik_callbacks(_get_foot_ik_for_ui, _set_foot_ik_from_ui)
    set_root_rot_bone_name_provider(lambda: str(ui_debug.root_rot_bone_name or ""))
    set_mapping_changed_callback(_on_mapping_ui_changed)
    create_mmd_config_ui()
    create_joint_rpy_mapping_ui()
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
        _cache_xyz("__root_target", foot_ik_state.last_target_root_world)
        _cache_xyz("__root_sim", foot_ik_state.last_dbg_sim_root_world)
        _cache_xyz("__ik_pred_l", foot_ik_state.last_left_ik_pred_world)
        _cache_xyz("__ik_pred_r", foot_ik_state.last_right_ik_pred_world)
        _cache_xyz("__ik_target_l", foot_ik_state.last_left_ik_target_world)
        _cache_xyz("__ik_target_r", foot_ik_state.last_right_ik_target_world)

        def _cache_scalar(key: str, val: float | None) -> None:
            if val is None:
                return
            joint_pos_deg_cache[key] = float(val)

        _cache_scalar("__dbg_root_to_red_l_m", foot_ik_state.last_dbg_root_target_to_red_l_m)
        _cache_scalar("__dbg_root_to_red_r_m", foot_ik_state.last_dbg_root_target_to_red_r_m)
        _cache_scalar("__dbg_red_to_ankle_l_m", foot_ik_state.last_dbg_red_to_ankle_l_m)
        _cache_scalar("__dbg_red_to_ankle_r_m", foot_ik_state.last_dbg_red_to_ankle_r_m)
        _cache_scalar("__dbg_red_to_pred_l_m", foot_ik_state.last_dbg_red_to_pred_l_m)
        _cache_scalar("__dbg_red_to_pred_r_m", foot_ik_state.last_dbg_red_to_pred_r_m)
        _cache_scalar("__dbg_pred_to_ankle_l_m", foot_ik_state.last_dbg_pred_to_ankle_l_m)
        _cache_scalar("__dbg_pred_to_ankle_r_m", foot_ik_state.last_dbg_pred_to_ankle_r_m)
        _cache_scalar("__dbg_ik_residual_l_m", foot_ik_state.last_left_ik_residual_m)
        _cache_scalar("__dbg_ik_residual_r_m", foot_ik_state.last_right_ik_residual_m)
        _cache_scalar("__dbg_sim_root_to_target_m", foot_ik_state.last_dbg_sim_root_to_target_m)
        _cache_scalar("__dbg_root_orient_err_deg", foot_ik_state.last_dbg_root_orient_err_deg)
        rpy_t = foot_ik_state.last_dbg_root_rpy_target_deg
        rpy_s = foot_ik_state.last_dbg_root_rpy_sim_deg
        rpy_d = foot_ik_state.last_dbg_root_rpy_delta_deg
        if rpy_t is not None:
            joint_pos_deg_cache["__dbg_root_rpy_target_r"] = float(rpy_t[0])
            joint_pos_deg_cache["__dbg_root_rpy_target_p"] = float(rpy_t[1])
            joint_pos_deg_cache["__dbg_root_rpy_target_y"] = float(rpy_t[2])
        if rpy_s is not None:
            joint_pos_deg_cache["__dbg_root_rpy_sim_r"] = float(rpy_s[0])
            joint_pos_deg_cache["__dbg_root_rpy_sim_p"] = float(rpy_s[1])
            joint_pos_deg_cache["__dbg_root_rpy_sim_y"] = float(rpy_s[2])
        if rpy_d is not None:
            joint_pos_deg_cache["__dbg_root_rpy_delta_r"] = float(rpy_d[0])
            joint_pos_deg_cache["__dbg_root_rpy_delta_p"] = float(rpy_d[1])
            joint_pos_deg_cache["__dbg_root_rpy_delta_y"] = float(rpy_d[2])

    def _read_sim_root_pose() -> tuple[tuple[float, float, float] | None, list[float] | None]:
        try:
            root_state = getattr(env.unwrapped.scene["robot"].data, "root_state_w", None)
            if torch.is_tensor(root_state) and root_state.shape[1] >= 7:
                # One GPU->CPU copy for pos+quat (avoids 7 scalar .item() syncs).
                vals = root_state[0, :7].detach().cpu().tolist()
                pos = (float(vals[0]), float(vals[1]), float(vals[2]))
                quat = [float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6])]
                return pos, quat
        except Exception:
            return None, None
        return None, None

    def _read_sim_root_world() -> tuple[float, float, float] | None:
        pos, _ = _read_sim_root_pose()
        return pos

    def _update_foot_root_debug_metrics(
        *,
        target_root_pos: tuple[float, float, float] | None,
        target_root_quat_wxyz: list[float] | None,
        left_ankle_world: tuple[float, float, float] | None,
        right_ankle_world: tuple[float, float, float] | None,
    ) -> None:
        sim_root, sim_quat = _read_sim_root_pose()
        foot_ik_state.last_dbg_sim_root_world = sim_root
        foot_ik_state.last_dbg_sim_root_quat_wxyz = sim_quat
        foot_ik_state.last_dbg_root_target_to_red_l_m = dist3(
            target_root_pos, foot_ik_state.last_left_foot_mmd_viz_world
        )
        foot_ik_state.last_dbg_root_target_to_red_r_m = dist3(
            target_root_pos, foot_ik_state.last_right_foot_mmd_viz_world
        )
        foot_ik_state.last_dbg_red_to_ankle_l_m = dist3(
            foot_ik_state.last_left_foot_mmd_viz_world, left_ankle_world
        )
        foot_ik_state.last_dbg_red_to_ankle_r_m = dist3(
            foot_ik_state.last_right_foot_mmd_viz_world, right_ankle_world
        )
        foot_ik_state.last_dbg_red_to_pred_l_m = dist3(
            foot_ik_state.last_left_foot_mmd_viz_world, foot_ik_state.last_left_ik_pred_world
        )
        foot_ik_state.last_dbg_red_to_pred_r_m = dist3(
            foot_ik_state.last_right_foot_mmd_viz_world, foot_ik_state.last_right_ik_pred_world
        )
        foot_ik_state.last_dbg_pred_to_ankle_l_m = dist3(
            foot_ik_state.last_left_ik_pred_world, left_ankle_world
        )
        foot_ik_state.last_dbg_pred_to_ankle_r_m = dist3(
            foot_ik_state.last_right_ik_pred_world, right_ankle_world
        )
        foot_ik_state.last_dbg_sim_root_to_target_m = dist3(sim_root, target_root_pos)
        foot_ik_state.last_dbg_root_orient_err_deg = quat_angular_error_deg(
            target_root_quat_wxyz, sim_quat
        )
        rpy_t = quat_wxyz_to_euler_xyz_deg(target_root_quat_wxyz)
        rpy_s = quat_wxyz_to_euler_xyz_deg(sim_quat)
        foot_ik_state.last_dbg_root_rpy_target_deg = rpy_t
        foot_ik_state.last_dbg_root_rpy_sim_deg = rpy_s
        if rpy_t is not None and rpy_s is not None:
            foot_ik_state.last_dbg_root_rpy_delta_deg = (
                float(rpy_s[0]) - float(rpy_t[0]),
                float(rpy_s[1]) - float(rpy_t[1]),
                float(rpy_s[2]) - float(rpy_t[2]),
            )
        else:
            foot_ik_state.last_dbg_root_rpy_delta_deg = None
        foot_ik_viz.update_root_debug(
            target_root_world=target_root_pos,
            sim_root_world=sim_root,
        )

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

    def _stop_playback_to_initial_pose() -> None:
        nonlocal current_motion, current_motion_label, is_playing, last_printed_frame
        nonlocal motion_track, playback_default_joint_pos
        nonlocal last_csv_motion_frame, mapping_reapply_requested
        nonlocal playback_paused, pause_hold_frame, pending_playback_toggle, pending_seek_frame
        nonlocal pending_playback_stop, motion_has_wav, pd_hold_joint_pos_cmd
        nonlocal last_frame_joint_pos_cmd, last_playback_target_root, last_playback_target_root_quat
        nonlocal h5_record_busy, h5_record_active, h5_recorder, h5_record_frame_cursor
        if h5_record_active or h5_recorder is not None:
            _cancel_h5_recording(reason="stop")
        pending_playback_stop = False
        audio_util.stop_wav()
        motion_has_wav = False
        is_playing = False
        current_motion = None
        current_motion_label = ""
        last_printed_frame = -1
        motion_track = None
        playback_default_joint_pos = None
        last_csv_motion_frame = None
        mapping_reapply_requested = False
        playback_paused = False
        pause_hold_frame = 0
        pending_playback_toggle = False
        pending_seek_frame = None
        pd_hold_joint_pos_cmd = None
        last_frame_joint_pos_cmd = None
        last_playback_target_root = None
        last_playback_target_root_quat = None
        foot_ik_state.reset()
        foot_ik_viz.clear()
        if initial_root_snapshot_row is not None:
            row = initial_root_snapshot_row
            apply_root_pos_instant(
                env,
                (float(row[0]), float(row[1]), float(row[2])),
                [float(row[3]), float(row[4]), float(row[5]), float(row[6])],
            )
        _reset_to_initial_pose(sync_ui_cache=True)
        print("[INFO] Playback stopped; reset to initial pose")

    def _cancel_h5_recording(*, reason: str) -> None:
        nonlocal h5_record_busy, h5_record_active, h5_recorder, h5_record_frame_cursor
        if not (h5_record_busy or h5_record_active or h5_recorder is not None):
            return
        h5_record_busy = False
        h5_record_active = False
        h5_recorder = None
        h5_record_frame_cursor = 0
        print(f"[INFO] H5 recording cancelled ({reason})")

    def _finalize_h5_recording() -> None:
        nonlocal h5_record_busy, h5_record_active, h5_recorder, h5_record_frame_cursor
        nonlocal h5_record_dance_key
        nonlocal is_playing, playback_paused, motion_has_wav, current_motion_label
        nonlocal dance_hdf5_motion_by_key
        recorder = h5_recorder
        label = str(current_motion_label or "")
        dkey = h5_record_dance_key
        h5_recorder = None
        h5_record_active = False
        h5_record_frame_cursor = 0
        h5_record_dance_key = None
        is_playing = False
        playback_paused = False
        motion_has_wav = False
        audio_util.stop_wav()
        foot_ik_viz.clear()
        try:
            if recorder is None:
                raise RuntimeError("recorder is missing")
            out_path = recorder.write()
            print(f"[INFO] H5 recording complete: {out_path} ({label})")
            if dkey:
                h5_bundle = load_motion(out_path)
                if h5_bundle is not None and str(h5_bundle.get("kind", "")) == "hdf5":
                    dance_hdf5_motion_by_key[str(dkey)] = (os.path.basename(out_path), h5_bundle)
                    print(f"[INFO] Updated in-session H5 cache for dance [{dkey}]")
        except Exception as exc:
            print(f"[ERROR] H5 recording failed ({label}): {exc}")
        finally:
            h5_record_busy = False

    def _begin_h5_recording_for_motion(csv_path: str, motion_bundle: MotionBundle) -> bool:
        nonlocal h5_recorder, h5_record_active, h5_record_frame_cursor, h5_record_busy
        _ensure_joint_info()
        if playback_default_joint_pos is None or joint_names is None:
            print("[ERROR] H5 recording failed: joint baseline unavailable")
            h5_record_busy = False
            return False
        if initial_root_snapshot_row is None:
            print("[ERROR] H5 recording failed: root anchor unavailable")
            h5_record_busy = False
            return False
        frame_list = motion_bundle.get("frame_list") or []
        if not frame_list:
            print("[ERROR] H5 recording failed: motion has no frames")
            h5_record_busy = False
            return False
        row = initial_root_snapshot_row
        try:
            h5_recorder = PlaybackH5Recorder.begin(
                source_csv=csv_path,
                runtime_joint_names=joint_names,
                baseline_joint_pos=playback_default_joint_pos,
                root_anchor_pos=(float(row[0]), float(row[1]), float(row[2])),
                root_anchor_quat_wxyz=[
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                ],
                max_frame=int(frame_list[-1]),
                has_hand_data=bool(motion_bundle.get("has_hand_data", False)),
                fps=float(VMD_FPS),
                knee_hinge_projection=bool(args_cli.mmd_knee_hinge_projection),
                root_quat_rpy_scale=tuple(root_quat_rpy_scale),
                root_quat_rpy_axis_idx=tuple(root_quat_rpy_axis_idx),
                mmd_center_to_root_offset_local_xyz=tuple(args_cli.mmd_center_to_root_offset_local_xyz),
                groove_pos_to_world=float(args_cli.groove_pos_to_world),
            )
        except Exception as exc:
            print(f"[ERROR] H5 recording failed to start: {exc}")
            h5_recorder = None
            h5_record_busy = False
            return False
        h5_record_active = True
        h5_record_frame_cursor = 0
        print(
            f"[INFO] H5 recording started: {os.path.basename(csv_path)} "
            f"({int(frame_list[-1]) + 1} frames, IK/ankle comp from playback pipeline)"
        )
        return True

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
        root_z_cfg = RootZCompressConfig(
            baseline_offset_m=float(root_z_baseline_offset_m_ui),
            outlier_scale=float(root_z_outlier_scale_ui),
        )
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
                root_z_compress_cfg=root_z_cfg,
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
            root_z_compress_cfg=root_z_cfg,
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
                pending_playback_stop = False
                pending_seek_frame = None
                pending_record_h5_key = None
                if h5_record_busy or h5_record_active:
                    _cancel_h5_recording(reason="env reset")
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

            if pending_playback_stop:
                _stop_playback_to_initial_pose()

            if pending_cycle_play:
                pending_cycle_play = False
                if not pose_motions:
                    print(f"[WARN] pose 目录无可播放 motion 文件: {POSE_DIR}")
                else:
                    _prepare_motion_switch()
                    current_pose_idx = (current_pose_idx + 1) % len(pose_motions)
                    name, _, data = pose_motions[current_pose_idx]
                    _switch_to_motion(data, f"pose[{current_pose_idx + 1}/{len(pose_motions)}] {name}")

            if pending_record_h5_key is not None and not h5_record_busy:
                dkey = pending_record_h5_key
                pending_record_h5_key = None
                entry = dance_motion_by_key.get(dkey)
                if entry is None:
                    print(f"[WARN] dance 键 [{dkey}] 未绑定文件")
                else:
                    entry, _used_z_editted = resolve_playback_motion_entry(
                        entry,
                        prefer_hdf5=False,
                        z_offset_enabled=z_offset_enabled_ui,
                    )
                    name, data = entry
                    if str(data.get("kind", "")) != "csv":
                        print(
                            f"[WARN] H5 recording requires CSV source for [{dkey}] "
                            f"(got {str(data.get('kind', 'unknown')).upper()})"
                        )
                    else:
                        csv_path = str(data.get("path", ""))
                        if not csv_path or not os.path.isfile(csv_path):
                            print(f"[WARN] CSV not found for dance [{dkey}]: {csv_path}")
                        else:
                            h5_record_busy = True
                            h5_record_dance_key = str(dkey)
                            _prepare_motion_switch()
                            _switch_to_motion(data, f"dance[{dkey}] {name}")
                            if not _begin_h5_recording_for_motion(csv_path, data):
                                is_playing = False
                                current_motion = None
                                current_motion_label = ""

            if pending_dance_key is not None:
                if h5_record_busy:
                    print("[WARN] Ignored dance play request during H5 recording")
                    pending_dance_key = None
                else:
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

                if h5_record_active:
                    frame = max(0, min(int(h5_record_frame_cursor), max_frame))
                    playback_paused = False
                else:
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
                    left_reach_clamped=bool(foot_ik_state.last_left_reach_clamped),
                    right_reach_clamped=bool(foot_ik_state.last_right_reach_clamped),
                )
                _apply_ankle_ground_comp_to_cmd(
                    joint_pos_cmd,
                    target_root_pos,
                    target_root_quat_wxyz,
                    joint_default=default_joint_pos,
                )

                if h5_record_active and h5_recorder is not None and joint_pos_cmd is not None:
                    rr, rp, ry = ui_debug.root_rpy_euler_scaled_deg
                    root_rpy_deg = None
                    if rr is not None and rp is not None and ry is not None:
                        root_rpy_deg = (float(rr), float(rp), float(ry))
                    h5_recorder.record_frame(
                        frame,
                        joint_pos_cmd,
                        root_pos=target_root_pos,
                        root_quat_wxyz=target_root_quat_wxyz,
                        root_rot_bone=ui_debug.root_rot_bone_name,
                        root_rpy_deg=root_rpy_deg,
                    )
                    h5_record_frame_cursor = frame + 1
                    if h5_record_frame_cursor > max_frame:
                        _finalize_h5_recording()
                        foot_ik_viz.clear()
                        current_motion = None
                        current_motion_label = ""
                        actions = zero_action
                        env.step(actions)
                        continue

                if use_root_teleport and target_root_pos is not None and target_root_quat_wxyz is not None:
                    if csv_root_rotation_lookup is False and not csv_root_track_warned:
                        print("[WARN] 当前 motion 未找到可用根旋转轨迹，root 朝向将保持动作起始值")
                        csv_root_track_warned = True

                if frame // PLAYBACK_LOG_FRAME_STRIDE != last_printed_frame:
                    last_printed_frame = frame // PLAYBACK_LOG_FRAME_STRIDE
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

                    if use_root_teleport and target_root_pos is not None and target_root_quat_wxyz is not None:
                        last_playback_target_root_quat = list(target_root_quat_wxyz)
                        applied_root = apply_root_pos_instant(
                            env, target_root_pos, target_root_quat_wxyz
                        )
                        last_playback_target_root = target_root_pos
                        if not applied_root and not root_track_warned:
                            print(
                                "[WARN] 当前环境不支持直接写 root 位姿，已跳过根位姿同步"
                                f"（平移骨: {mmd_root_trans_bone}）"
                            )
                            root_track_warned = True
                    else:
                        last_playback_target_root = target_root_pos
                        last_playback_target_root_quat = (
                            list(target_root_quat_wxyz) if target_root_quat_wxyz is not None else None
                        )

                    if pd_drive_enabled_ui and joint_pos_cmd is not None:
                        _apply_joint_pd_target(joint_pos_cmd)
                        actions = zero_action
                    else:
                        actions = torch.tensor(
                            result, dtype=torch.float32, device=env.unwrapped.device
                        ).unsqueeze(0)
                else:
                    actions = zero_action
                if frame >= max_frame and not h5_record_active:
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
                _apply_ankle_ground_comp_to_cmd(jp_cmd, tr_pos, tr_quat, joint_default=base_default)
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
            # Physics (contacts + joint PD) can rotate the floating pelvis after the
            # pre-step teleport; re-lock root/joints so sim matches the playback command.
            if use_root_teleport and last_playback_target_root is not None:
                apply_root_pos_instant(
                    env,
                    last_playback_target_root,
                    last_playback_target_root_quat,
                )
            if (not pd_drive_enabled_ui) and last_frame_joint_pos_cmd is not None and joint_ids is not None:
                apply_joint_state_instant(env, last_frame_joint_pos_cmd, joint_ids)
            if is_playing:
                _update_ankle_link_debug_viz(
                    last_playback_target_root,
                    last_playback_target_root_quat,
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
