"""
CSV 动作加载与 G1 关节重定向核心模块。

功能概览：
1) 读取 MMD 骨骼 CSV（欧拉角或四元数）；
2) 对缺失帧做插值并按骨骼查询；
3) 基于可编辑映射将骨骼旋转转换为 G1 目标关节角；
4) 为映射 UI 提供默认映射与运行时覆盖能力。

支持两类输入格式：
1) 欧拉角 CSV: frame,bone,pos_x,pos_y,pos_z,roll,pitch,yaw(度)
2) 四元数 CSV: frame,bone,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w

内部统一保存为弧度欧拉角 + 归一化四元数（bone dict 仅保留 `quat_wxyz`）。
- 肩/腿/腰链走 ``retarget_unitreeG1`` 专用反解；腰可选在合成前对上半身/上半身2各自取四元数共轭（见 ``get_waist_upper_pair_quat_conjugate``）。
- 其它单骨骼关节走 Swing-Twist 单轴提取。

注意：CSV 列 `quat_x/quat_y/quat_z/quat_w` 仍表示 x,y,z,w；仅在读入内存后转换为
`quat_wxyz`（w,x,y,z）以对齐 Isaac root_state API。
"""
import bisect
import csv
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    AxisMapRawEntry,
    G1_JOINT_AXIS_MAP_RAW,
    MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE,
)
from robot_mmd.train_workflow.utils.g1_foot_ik_geometry import (
    FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M,
    G1_FOOT_IK_HIP_OFFSET_Y_M,
    G1_FOOT_IK_HIP_OFFSET_Z_M,
    G1_FOOT_IK_SHIN_LENGTH_M,
    G1_FOOT_IK_THIGH_LENGTH_M,
)
from robot_mmd.train_workflow.utils.g1_leg_kinematics import (
    LegIkResult,
    g1_leg_clamp_target_to_reach,
    g1_leg_fk_pos,
    g1_leg_reach_clamped,
    g1_leg_remap_foot_ik_target,
    solve_g1_leg_ik_dls,
)
from robot_mmd.train_workflow.utils.mmd_fk import (
    FootIkVizConfig,
    compute_mmd_foot_ik_viz_bundle,
    foot_ik_panel_to_isaac_world,
    resolve_mmd_root_translation_pos,
)
from robot_mmd.train_workflow.utils.trans_util import (
    isaac_world_to_root_local,
    root_local_to_isaac_world,
)
from robot_mmd.train_workflow.retarget_unitreeG1 import (
    ANKLE_JOINT_TO_AXIS_INDEX,
    ANKLE_JOINT_TO_SIDE_BONE,
    HIP_JOINT_TO_AXIS_INDEX,
    HIP_JOINT_TO_SIDE_BONE,
    ELBOW_JOINT_TO_SIDE_BONE,
    SHOULDER_JOINT_TO_AXIS_INDEX,
    SHOULDER_JOINT_TO_SIDE_BONES,
    WAIST_JOINT_TO_AXIS_INDEX,
    WRIST_JOINT_TO_AXIS_INDEX,
    WRIST_JOINT_TO_SIDE_BONE,
    compute_ankle_angles,
    compute_elbow_angle,
    compute_hip_angles,
    compute_shoulder_angles,
    compute_waist_angles,
    compute_wrist_angles,
    leg_debug_info as _leg_debug_info,
    shoulder_debug_info as _shoulder_debug_info,
)

Axis3 = tuple[float, float, float]
AxisMapEntry = tuple[str | list[str], Axis3, float]


@dataclass
class FootIkConfig:
    """Config for VMD foot-target driven leg IK override."""

    enable: bool = False
    groove_pos_to_world: float = 0.1
    max_reach_ratio: float = 1.0
    leg_target_scale: float = 0.75
    leg_z_ground_clearance_m: float = 0.012
    leg_z_compress_power: float = 2.0
    leg_floor_z: float = 0.0
    is_static_pose: bool = False
    hip_offset_y: float = G1_FOOT_IK_HIP_OFFSET_Y_M
    hip_offset_z: float = G1_FOOT_IK_HIP_OFFSET_Z_M
    thigh_length: float = G1_FOOT_IK_THIGH_LENGTH_M
    shin_length: float = G1_FOOT_IK_SHIN_LENGTH_M
    hip_pitch_offset: float = 0.0
    hip_roll_gain: float = 0.85
    ankle_pitch_stabilize_gain: float = 0.75
    debug_every_n_frames: int = 0
    ik_max_iters: int = 20
    ik_warm_start: bool = True
    ik_warm_reset_target_delta_m: float = 0.04
    ik_max_apply_residual_m: float = 0.012
    ik_max_foot_z_local_m: float = 0.05
    ik_pos_tol_m: float = 1e-3
    ik_dls_lambda: float = 0.05
    ik_step_scale: float = 0.85
    ik_reg_weight: float = 0.15
    ik_reg_hip_yaw: float = 0.8
    ik_reg_ankle_roll: float = 0.8
    ik_min_knee_rad: float = 0.12
    # Keep ankle_pitch/roll from 足首 FK; IK only moves hip + knee to chase foot target.
    ik_pass_through_ankle: bool = True
    ankle_target_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.02)


@dataclass
class FootIkState:
    """Runtime state for foot IK reference anchoring."""

    last_left_target_local: tuple[float, float, float] | None = None
    last_right_target_local: tuple[float, float, float] | None = None
    last_left_foot_mmd_viz_world: tuple[float, float, float] | None = None
    last_right_foot_mmd_viz_world: tuple[float, float, float] | None = None
    last_left_toe_mmd_viz_world: tuple[float, float, float] | None = None
    last_right_toe_mmd_viz_world: tuple[float, float, float] | None = None
    last_left_foot_mmd_local_m: tuple[float, float, float] | None = None
    last_right_foot_mmd_local_m: tuple[float, float, float] | None = None
    last_left_foot_mmd_fk_world_m: tuple[float, float, float] | None = None
    last_right_foot_mmd_fk_world_m: tuple[float, float, float] | None = None
    last_left_toe_mmd_local_m: tuple[float, float, float] | None = None
    last_right_toe_mmd_local_m: tuple[float, float, float] | None = None
    last_left_toe_mmd_fk_world_m: tuple[float, float, float] | None = None
    last_right_toe_mmd_fk_world_m: tuple[float, float, float] | None = None
    last_target_root_world: tuple[float, float, float] | None = None
    last_dbg_sim_root_world: tuple[float, float, float] | None = None
    last_dbg_root_target_to_red_l_m: float | None = None
    last_dbg_root_target_to_red_r_m: float | None = None
    last_dbg_red_to_ankle_l_m: float | None = None
    last_dbg_red_to_ankle_r_m: float | None = None
    last_dbg_red_to_pred_l_m: float | None = None
    last_dbg_red_to_pred_r_m: float | None = None
    last_dbg_pred_to_ankle_l_m: float | None = None
    last_dbg_pred_to_ankle_r_m: float | None = None
    last_dbg_sim_root_to_target_m: float | None = None
    last_target_root_quat_wxyz: list[float] | None = None
    last_dbg_sim_root_quat_wxyz: list[float] | None = None
    last_dbg_root_orient_err_deg: float | None = None
    last_dbg_root_rpy_target_deg: tuple[float, float, float] | None = None
    last_dbg_root_rpy_sim_deg: tuple[float, float, float] | None = None
    last_dbg_root_rpy_delta_deg: tuple[float, float, float] | None = None
    last_left_reach_clamped: bool = False
    last_right_reach_clamped: bool = False
    last_left_ik_pred_world: tuple[float, float, float] | None = None
    last_right_ik_pred_world: tuple[float, float, float] | None = None
    last_left_ik_target_world: tuple[float, float, float] | None = None
    last_right_ik_target_world: tuple[float, float, float] | None = None
    last_left_ik_residual_m: float | None = None
    last_right_ik_residual_m: float | None = None
    last_left_ik_iters: int | None = None
    last_left_q_ik: tuple[float, float, float, float, float, float] | None = None
    last_right_q_ik: tuple[float, float, float, float, float, float] | None = None

    def reset(self) -> None:
        self.last_left_target_local = None
        self.last_right_target_local = None
        self.last_left_foot_mmd_viz_world = None
        self.last_right_foot_mmd_viz_world = None
        self.last_left_toe_mmd_viz_world = None
        self.last_right_toe_mmd_viz_world = None
        self.last_left_foot_mmd_local_m = None
        self.last_right_foot_mmd_local_m = None
        self.last_left_foot_mmd_fk_world_m = None
        self.last_right_foot_mmd_fk_world_m = None
        self.last_left_toe_mmd_local_m = None
        self.last_right_toe_mmd_local_m = None
        self.last_left_toe_mmd_fk_world_m = None
        self.last_right_toe_mmd_fk_world_m = None
        self.last_target_root_world = None
        self.last_dbg_sim_root_world = None
        self.last_dbg_root_target_to_red_l_m = None
        self.last_dbg_root_target_to_red_r_m = None
        self.last_dbg_red_to_ankle_l_m = None
        self.last_dbg_red_to_ankle_r_m = None
        self.last_dbg_red_to_pred_l_m = None
        self.last_dbg_red_to_pred_r_m = None
        self.last_dbg_pred_to_ankle_l_m = None
        self.last_dbg_pred_to_ankle_r_m = None
        self.last_dbg_sim_root_to_target_m = None
        self.last_target_root_quat_wxyz = None
        self.last_dbg_sim_root_quat_wxyz = None
        self.last_dbg_root_orient_err_deg = None
        self.last_dbg_root_rpy_target_deg = None
        self.last_dbg_root_rpy_sim_deg = None
        self.last_dbg_root_rpy_delta_deg = None
        self.last_left_reach_clamped = False
        self.last_right_reach_clamped = False
        self.last_left_ik_pred_world = None
        self.last_right_ik_pred_world = None
        self.last_left_ik_target_world = None
        self.last_right_ik_target_world = None
        self.last_left_ik_residual_m = None
        self.last_right_ik_residual_m = None
        self.last_left_ik_iters = None
        self.last_right_ik_iters = None
        self.last_left_q_ik = None
        self.last_right_q_ik = None


MMD_FINGER_PREFIXES: tuple[str, ...] = (
    "右親指",
    "左親指",
    "右人指",
    "左人指",
    "右人差指",
    "左人差指",
    "右中指",
    "左中指",
    "右薬指",
    "左薬指",
    "右小指",
    "左小指",
)

_HAND_JOINT_NAME_PARTS: tuple[str, ...] = (
    "_thumb_",
    "_index_",
    "_middle_",
    "_ring_",
    "_pinky_",
    "_little_",
    "_finger_",
)


def is_mmd_finger_bone(bone_name: str) -> bool:
    """Return True when bone name belongs to hand/finger chains."""
    name = str(bone_name or "")
    return any(name.startswith(p) for p in MMD_FINGER_PREFIXES)


def frame_has_hand_data(frame_data: dict[str, dict] | None) -> bool:
    """Return True when one frame contains any MMD finger bone."""
    if not frame_data:
        return False
    return any(is_mmd_finger_bone(name) for name in frame_data.keys())


def frames_have_hand_data(frames: dict[int, dict[str, dict]] | None) -> bool:
    """Return True when motion frames include at least one finger bone."""
    if not frames:
        return False
    for frame_data in frames.values():
        if frame_has_hand_data(frame_data):
            return True
    return False


def is_hand_joint_name(joint_name: str) -> bool:
    """Return True for runtime robot hand/finger joint names."""
    name = str(joint_name or "")
    if name.startswith("lh_") or name.startswith("rh_"):
        return True
    return any(part in name for part in _HAND_JOINT_NAME_PARTS)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """欧拉角 (XYZ 顺序) 转四元数 (x, y, z, w)"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # q = qz * qy * qx (XYZ intrinsic)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


def _quat_multiply(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """四元数乘法 q1 * q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """四元数转欧拉角 (XYZ)，返回 (roll, pitch, yaw)"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-10:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = max(-1, min(1, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """单位四元数共轭（逆）。"""
    x, y, z, w = _quat_normalize(q)
    return (-x, -y, -z, w)


def _quat_pow_xyzw(q_xyzw: tuple[float, float, float, float], exponent: float) -> tuple[float, float, float, float]:
    """单位四元数按指数缩放旋转角（exponent=0 为恒等，1 为自身；负指数反向）。"""
    x, y, z, w = _quat_normalize(q_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vn = math.sqrt(x * x + y * y + z * z)
    if vn < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    half = math.atan2(vn, w)
    if abs(half) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = x / vn, y / vn, z / vn
    nh = half * float(exponent)
    sn = math.sin(nh)
    return _quat_normalize((ax * sn, ay * sn, az * sn, math.cos(nh)))


def _bone_quat_from_xyzw(q_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """xyzw -> wxyz（bone dict 约定）。"""
    x, y, z, w = _quat_normalize(q_xyzw)
    return (w, x, y, z)


def _bone_quat_to_xyzw(q_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """wxyz -> xyzw（供内部数学计算）。"""
    w, x, y, z = q_wxyz
    return _quat_normalize((x, y, z, w))


def _finalize_bone_dict(bone_data: dict) -> dict:
    """标准化 bone dict：仅输出 pos/euler/quat_wxyz。"""
    quat_wxyz = bone_data.get("quat_wxyz")
    if quat_wxyz is None:
        quat_xyzw = bone_data.get("quat")
        if quat_xyzw is None:
            euler = bone_data.get("euler")
            if euler is None:
                quat_xyzw = (0.0, 0.0, 0.0, 1.0)
            else:
                quat_xyzw = _euler_to_quat(*euler)
        quat_wxyz = _bone_quat_from_xyzw(tuple(float(v) for v in quat_xyzw))
    else:
        quat_wxyz = tuple(float(v) for v in quat_wxyz)
        quat_wxyz = _bone_quat_from_xyzw(_bone_quat_to_xyzw(quat_wxyz))
    bone_data["quat_wxyz"] = quat_wxyz
    bone_data.pop("quat", None)
    return bone_data


def _quat_nlerp(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """单位四元数线性插值（带短弧修正）+ 归一化。"""
    x0, y0, z0, w0 = _quat_normalize(q0)
    x1, y1, z1, w1 = _quat_normalize(q1)
    dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
    if dot < 0.0:
        x1, y1, z1, w1 = -x1, -y1, -z1, -w1
    q = (
        (1.0 - t) * x0 + t * x1,
        (1.0 - t) * y0 + t * y1,
        (1.0 - t) * z0 + t * z1,
        (1.0 - t) * w0 + t * w1,
    )
    return _quat_normalize(q)


def swing_twist_angle(qx: float, qy: float, qz: float, qw: float, axis_xyz: Axis3) -> float:
    """提取四元数绕指定轴 axis_xyz 的扭转角（弧度）。"""
    ax, ay, az = axis_xyz
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-10:
        return 0.0
    ax, ay, az = ax / n, ay / n, az / n

    qx, qy, qz, qw = _quat_normalize((qx, qy, qz, qw))
    dot = ax * qx + ay * qy + az * qz
    tx, ty, tz, tw = (ax * dot, ay * dot, az * dot, qw)
    tx, ty, tz, tw = _quat_normalize((tx, ty, tz, tw))

    sin_half = math.sqrt(tx * tx + ty * ty + tz * tz)
    angle = 2.0 * math.atan2(sin_half, tw)
    if dot < 0.0:
        angle = -angle
    if angle > math.pi:
        angle -= 2.0 * math.pi
    elif angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _swing_twist_decompose_xyzw(
    q_xyzw: tuple[float, float, float, float],
    axis_xyz: Axis3,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """将旋转分解为 q = swing * twist（均为 xyzw）。"""
    ax, ay, az = axis_xyz
    n = math.sqrt(ax * ax + ay * ay + az * az)
    qx, qy, qz, qw = _quat_normalize(q_xyzw)
    if n < 1e-10:
        return (qx, qy, qz, qw), (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = ax / n, ay / n, az / n

    dot = ax * qx + ay * qy + az * qz
    q_twist = _quat_normalize((ax * dot, ay * dot, az * dot, qw))
    q_swing = _quat_normalize(_quat_multiply((qx, qy, qz, qw), _quat_conjugate(q_twist)))
    return q_swing, q_twist


def _quat_rotation_magnitude_deg_xyzw(q_xyzw: tuple[float, float, float, float]) -> float:
    """单位四元数对应的旋转角幅值（度），取最短弧（w 取绝对值）。"""
    _x, _y, _z, w = _quat_normalize(q_xyzw)
    w = max(-1.0, min(1.0, abs(w)))
    return math.degrees(2.0 * math.acos(w))


def knee_hinge_mapping_ui_extra(
    frame_data_raw: dict[str, dict] | None,
    *,
    projection_enabled: bool,
) -> dict[str, str]:
    """
    供映射 UI 显示：MMD 膝局部旋转在铰链投影下的分解与髋补偿（度）。
    返回键为 ``{left|right}_knee_joint__knee_mmd``，值为单行说明字符串。
    """
    out: dict[str, str] = {}
    if not frame_data_raw:
        return out

    side_cfg = {
        "left": (
            "left_knee_joint",
            "左足",
            "左ひざ",
            ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"),
        ),
        "right": (
            "right_knee_joint",
            "右足",
            "右ひざ",
            ("right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"),
        ),
    }

    mapping = get_mapping()
    for side, (knee_joint, hip_bone, knee_bone, hip_js) in side_cfg.items():
        if knee_joint not in mapping:
            continue
        if knee_bone not in frame_data_raw or hip_bone not in frame_data_raw:
            continue
        q_knee = _read_bone_quat_xyzw(frame_data_raw, knee_bone)
        if q_knee is None:
            continue
        _, axis, _scale = mapping[knee_joint]
        q_swing, _q_twist = _swing_twist_decompose_xyzw(q_knee, axis)
        sw_deg = _quat_rotation_magnitude_deg_xyzw(q_swing)
        hinge_deg = math.degrees(swing_twist_angle(*q_knee, axis))

        if not projection_enabled:
            out[f"{knee_joint}__knee_mmd"] = (
                f"hinge {hinge_deg:.1f} deg | swing {sw_deg:.1f} deg (proj off)"
            )
            continue

        fd = dict(frame_data_raw)
        for b in (hip_bone, knee_bone):
            if b in fd:
                fd[b] = dict(fd[b])
        _apply_knee_hinge_projection(fd, side, mapping)
        d_list: list[float] = []
        for hj in hip_js:
            a = get_g1_angle_from_frame(hj, frame_data_raw)
            b = get_g1_angle_from_frame(hj, fd)
            if a is not None and b is not None:
                d_list.append(math.degrees(b - a))
            else:
                d_list.append(0.0)
        dp, dr, dy = d_list[0], d_list[1], d_list[2]
        out[f"{knee_joint}__knee_mmd"] = (
            f"hinge {hinge_deg:.1f}deg swing {sw_deg:.1f} deg | "
            f"hip {dp:+.1f}/{dr:+.1f}/{dy:+.1f} deg"
        )
    return out


def elbow_hinge_mapping_ui_extra(
    frame_data_raw: dict[str, dict] | None,
    *,
    projection_enabled: bool,
) -> dict[str, str]:
    """
    Mapping UI: MMD elbow local rotation split (hinge/swing, deg).
    Keys: ``{left|right}_elbow_joint__elbow_mmd``.
    """
    out: dict[str, str] = {}
    if not frame_data_raw:
        return out

    side_cfg = {
        "left": ("left_elbow_joint", "左腕", "左ひじ"),
        "right": ("right_elbow_joint", "右腕", "右ひじ"),
    }

    mapping = get_mapping()
    for _side, (elbow_joint, arm_bone, elbow_bone) in side_cfg.items():
        if elbow_joint not in mapping:
            continue
        if elbow_bone not in frame_data_raw or arm_bone not in frame_data_raw:
            continue
        q_elbow = _read_bone_quat_xyzw(frame_data_raw, elbow_bone)
        if q_elbow is None:
            continue
        _, axis, _scale = mapping[elbow_joint]
        q_swing, _q_twist = _swing_twist_decompose_xyzw(q_elbow, axis)
        sw_deg = _quat_rotation_magnitude_deg_xyzw(q_swing)
        hinge_deg = math.degrees(swing_twist_angle(*q_elbow, axis))

        if not projection_enabled:
            out[f"{elbow_joint}__elbow_mmd"] = (
                f"MMD hinge {hinge_deg:.1f}deg swing {sw_deg:.1f}deg (proj off)"
            )
            continue

        out[f"{elbow_joint}__elbow_mmd"] = f"MMD hinge {hinge_deg:.1f}deg swing {sw_deg:.1f}deg"
    return out


# 轴索引定义：0=x, 1=y, 2=z
_AXIS_INDEX_TO_VEC: dict[int, Axis3] = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
}


def _build_axis_map(raw_map: dict[str, AxisMapRawEntry]) -> dict[str, AxisMapEntry]:
    out: dict[str, AxisMapEntry] = {}
    for joint_name, (bones, axis_idx, scale) in raw_map.items():
        axis = _AXIS_INDEX_TO_VEC.get(axis_idx)
        if axis is None:
            raise ValueError(f"invalid axis index {axis_idx} for joint {joint_name}")
        out[joint_name] = (bones, axis, scale)
    return out


# G1 关节 -> (MMD 骨骼或 [肩, 腕] 列表, Twist 轴(骨骼本地系), 缩放系数)
G1_JOINT_AXIS_MAP: dict[str, AxisMapEntry] = _build_axis_map(G1_JOINT_AXIS_MAP_RAW)


def _axis_to_index(axis: Axis3) -> int:
    ax, ay, az = axis
    vals = [abs(ax), abs(ay), abs(az)]
    return int(vals.index(max(vals)))


def _as_legacy_mapping(mapping: dict[str, AxisMapEntry]) -> dict[str, tuple[str | list[str], int, float]]:
    out: dict[str, tuple[str | list[str], int, float]] = {}
    for j, (bones, axis, scale) in mapping.items():
        out[j] = (bones, _axis_to_index(axis), scale)
    return out


# 兼容 UI：仍保留旧名称（骨骼, 欧拉索引, 缩放）
G1_JOINT_TO_MMD: dict[str, tuple[str | list[str], int, float]] = _as_legacy_mapping(G1_JOINT_AXIS_MAP)

# 运行时可编辑的映射（UI 修改后生效），None 时使用 G1_JOINT_AXIS_MAP
_editable_mapping: dict[str, AxisMapEntry] | None = None


def get_mapping() -> dict[str, AxisMapEntry]:
    """获取当前生效的映射（优先使用 UI 编辑后的）"""
    if _editable_mapping is not None:
        return _editable_mapping
    return G1_JOINT_AXIS_MAP


def set_editable_mapping(mapping: dict[str, AxisMapEntry] | None) -> None:
    """设置可编辑映射，None 时恢复默认"""
    global _editable_mapping
    _editable_mapping = mapping


def update_mapping_entry(joint_name: str, euler_idx: int, scale: float) -> None:
    """兼容旧 UI：通过 0/1/2 选择主轴，并更新缩放系数。"""
    global _editable_mapping
    base = G1_JOINT_AXIS_MAP.get(joint_name)
    if base is None:
        return
    bones = base[0]
    axis = _AXIS_INDEX_TO_VEC.get(int(euler_idx), (0.0, 0.0, 1.0))
    if _editable_mapping is None:
        _editable_mapping = dict(G1_JOINT_AXIS_MAP)
    _editable_mapping[joint_name] = (bones, axis, scale)


HINGE_SWING_ABSORB_JOINTS: frozenset[str] = frozenset(
    {
        "left_knee_joint",
        "right_knee_joint",
    }
)
# Knee hinge projection can inject high-frequency hip compensation on noisy frames.
# Keep projection enabled by default, but make it less aggressive.
_DEFAULT_HINGE_SWING_ABSORB: dict[str, float] = {k: 0.6 for k in HINGE_SWING_ABSORB_JOINTS}
_hinge_swing_absorb: dict[str, float] = dict(_DEFAULT_HINGE_SWING_ABSORB)

# Per-frame cap for swing component merged from knee to hip.
# Prevents sudden large compensation spikes that appear as lower-body jitter.
_KNEE_PROJECTION_MAX_SWING_DEG: float = 20.0

_waist_upper_pair_use_conj: list[bool] = [
    bool(MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE[0]),
    bool(MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE[1]),
]


def get_waist_upper_pair_quat_conjugate() -> tuple[bool, bool]:
    """腰 [上半身, 上半身2]：合成前是否对该骨四元数取共轭（逆旋转）。"""
    return bool(_waist_upper_pair_use_conj[0]), bool(_waist_upper_pair_use_conj[1])


def toggle_waist_upper_pair_quat_conjugate(which: int) -> tuple[bool, bool]:
    """切换第 ``which`` 骨（0=上半身, 1=上半身2）的共轭开关；返回当前 (c0, c1)。"""
    i = 0 if int(which) == 0 else 1
    _waist_upper_pair_use_conj[i] = not bool(_waist_upper_pair_use_conj[i])
    return get_waist_upper_pair_quat_conjugate()


def reset_waist_upper_pair_quat_conjugate() -> None:
    _waist_upper_pair_use_conj[0] = bool(MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE[0])
    _waist_upper_pair_use_conj[1] = bool(MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE[1])


def get_hinge_swing_absorb(joint_name: str) -> float:
    """膝/肘：非铰链 swing 并入父骨的强度（0=不并入，1=全量，与当前默认一致）。"""
    return float(_hinge_swing_absorb.get(joint_name, 1.0))


def set_hinge_swing_absorb(joint_name: str, value: float) -> None:
    if joint_name in HINGE_SWING_ABSORB_JOINTS:
        _hinge_swing_absorb[joint_name] = float(value)


def reset_hinge_swing_absorb() -> None:
    global _hinge_swing_absorb
    _hinge_swing_absorb = dict(_DEFAULT_HINGE_SWING_ABSORB)


def reset_mapping_to_default() -> None:
    """重置为默认映射"""
    global _editable_mapping
    _editable_mapping = None
    reset_hinge_swing_absorb()
    reset_waist_upper_pair_quat_conjugate()


def load_csv_motion(csv_path: str) -> dict[int, dict[str, dict]]:
    """
    加载 CSV 骨骼数据（自动识别四元数/欧拉角列）。
    返回: {frame_idx: {bone_name: {"pos": tuple, "euler": tuple(rad),
                                   "quat_wxyz": tuple(w,x,y,z)}}}
    """
    rows = None
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            with open(csv_path, encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise UnicodeDecodeError("", b"", 0, 0, "无法用 utf-8/cp932/shift_jis 解码 CSV")
    if not rows:
        return {}

    keys = set(rows[0].keys())
    has_euler = {"roll", "pitch", "yaw"}.issubset(keys)
    has_quat = {"quat_x", "quat_y", "quat_z", "quat_w"}.issubset(keys)
    if not has_euler and not has_quat:
        raise ValueError("CSV 必须包含 roll/pitch/yaw 或 quat_x/quat_y/quat_z/quat_w 列")

    frames: dict[int, dict[str, dict]] = {}
    for row in rows:
        frame = int(row["frame"])
        bone = row["bone"]
        pos = (
            float(row["pos_x"]),
            float(row["pos_y"]),
            float(row["pos_z"]),
        )
        if has_quat:
            quat = _quat_normalize(
                (
                    float(row["quat_x"]),
                    float(row["quat_y"]),
                    float(row["quat_z"]),
                    float(row["quat_w"]),
                )
            )
            euler = _quat_to_euler(*quat)
        else:
            # CSV 中欧拉角为角度，转为弧度供后续计算
            euler = (
                math.radians(float(row["roll"])),
                math.radians(float(row["pitch"])),
                math.radians(float(row["yaw"])),
            )
            quat = _euler_to_quat(*euler)
            quat = _quat_normalize(quat)
        if frame not in frames:
            frames[frame] = {}
        frames[frame][bone] = _finalize_bone_dict(
            {"pos": pos, "euler": euler, "quat_wxyz": _bone_quat_from_xyzw(quat)}
        )
    return frames


def get_frame_indices(frames: dict) -> list[int]:
    """获取所有帧号并排序"""
    return sorted(frames.keys())


def get_bone_frame_lists(
    frames: dict[int, dict[str, dict]],
    frame_list: list[int],
    all_bones: set[str],
) -> dict[str, list[int]]:
    """预计算每个骨骼的帧列表，用于 bisect 加速插值。"""
    return {
        bone: [f for f in frame_list if bone in frames.get(f, {})]
        for bone in all_bones
    }


def interpolate_bone(
    frame: int,
    bone: str,
    frames: dict[int, dict[str, dict]],
    bone_frame_list: list[int] | None = None,
) -> dict | None:
    """
    对指定帧、指定骨骼进行线性插值。
    若该帧有数据则直接返回；否则在前后关键帧之间插值。
    bone_frame_list: 该骨骼存在的帧号列表（已排序），传入则用 bisect 二分查找，否则内部计算。
    """
    if bone_frame_list is None:
        frame_list = get_frame_indices(frames)
        bone_frame_list = [f for f in frame_list if bone in frames.get(f, {})]
    if not bone_frame_list:
        return None

    if frame in frames and bone in frames[frame]:
        return _finalize_bone_dict(dict(frames[frame][bone]))

    # bisect 二分查找前后关键帧
    idx = bisect.bisect_right(bone_frame_list, frame)
    prev_f = bone_frame_list[idx - 1] if idx > 0 else None
    next_f = bone_frame_list[idx] if idx < len(bone_frame_list) else None

    if prev_f is None and next_f is None:
        return None
    if prev_f is None:
        if bone not in frames.get(next_f, {}):
            return None
        return _finalize_bone_dict(dict(frames[next_f][bone]))
    if next_f is None:
        if bone not in frames.get(prev_f, {}):
            return None
        return _finalize_bone_dict(dict(frames[prev_f][bone]))
    if prev_f == next_f:
        if bone not in frames.get(prev_f, {}):
            return None
        return _finalize_bone_dict(dict(frames[prev_f][bone]))

    # 欧拉角 / 四元数插值
    if bone not in frames.get(prev_f, {}) or bone not in frames.get(next_f, {}):
        return None
    d0, d1 = frames[prev_f][bone], frames[next_f][bone]
    t = (frame - prev_f) / (next_f - prev_f)
    pos = tuple((1 - t) * a + t * b for a, b in zip(d0["pos"], d1["pos"]))
    e0, e1 = d0["euler"], d1["euler"]
    euler = tuple((1 - t) * a + t * b for a, b in zip(e0, e1))
    q0_wxyz = d0.get("quat_wxyz")
    q1_wxyz = d1.get("quat_wxyz")
    q0 = _bone_quat_to_xyzw(tuple(float(v) for v in q0_wxyz)) if q0_wxyz is not None else _euler_to_quat(*e0)
    q1 = _bone_quat_to_xyzw(tuple(float(v) for v in q1_wxyz)) if q1_wxyz is not None else _euler_to_quat(*e1)
    quat_xyzw = _quat_nlerp(q0, q1, t)
    return _finalize_bone_dict({"pos": pos, "euler": euler, "quat_wxyz": _bone_quat_from_xyzw(quat_xyzw)})


def _read_bone_quat_xyzw(frame_data: dict[str, dict], bone_name: str) -> tuple[float, float, float, float] | None:
    bone_data = frame_data.get(bone_name)
    if bone_data is None:
        return None
    q_wxyz = bone_data.get("quat_wxyz")
    if q_wxyz is None:
        return _quat_normalize(_euler_to_quat(*bone_data["euler"]))
    return _bone_quat_to_xyzw(tuple(float(v) for v in q_wxyz))


def _read_first_available_bone_quat_xyzw(
    frame_data: dict[str, dict],
    bone_names: tuple[str, ...],
) -> tuple[float, float, float, float] | None:
    for bone_name in bone_names:
        q = _read_bone_quat_xyzw(frame_data, bone_name)
        if q is not None:
            return q
    return None


def _write_bone_quat_xyzw(frame_data: dict[str, dict], bone_name: str, q_xyzw: tuple[float, float, float, float]) -> None:
    bone_data = frame_data.get(bone_name)
    if bone_data is None:
        return
    q_norm = _quat_normalize(q_xyzw)
    bone_data["quat_wxyz"] = _bone_quat_from_xyzw(q_norm)
    bone_data["euler"] = _quat_to_euler(*q_norm)


_FOOT_IK_BONE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "left": ("左足ＩＫ", "左足IK"),
    "right": ("右足ＩＫ", "右足IK"),
}

_FOOT_IK_OVERRIDE_JOINTS: dict[str, tuple[str, str, str, str, str, str]] = {
    "left": (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    "right": (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
}

_FOOT_IK_LIMITS: dict[str, tuple[float, float]] = {
    "hip_pitch": (-2.3, 1.3),
    "hip_roll": (-0.9, 0.9),
    "knee": (-0.1, 2.9),
    "ankle_pitch": (-1.2, 1.0),
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _read_first_available_bone_pos(
    frame_data: dict[str, dict],
    candidates: tuple[str, ...],
) -> tuple[float, float, float] | None:
    for bone_name in candidates:
        d = frame_data.get(bone_name)
        if d is None:
            continue
        p = d.get("pos")
        if p is None or len(p) != 3:
            continue
        try:
            return float(p[0]), float(p[1]), float(p[2])
        except Exception:
            continue
    return None


def _leg_ik_merge_fk_ankle(
    q_ik: tuple[float, float, float, float, float, float],
    q_fk: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Replace ankle DOFs with FK retarget from 足首."""
    return (
        float(q_ik[0]),
        float(q_ik[1]),
        float(q_ik[2]),
        float(q_ik[3]),
        float(q_fk[4]),
        float(q_fk[5]),
    )


def _foot_ik_lock_ankle(cfg: FootIkConfig) -> bool:
    return bool(getattr(cfg, "ik_pass_through_ankle", True))


def _pick_leg_ik_seed(
    q_fk: tuple[float, float, float, float, float, float],
    warm_q: tuple[float, float, float, float, float, float] | None,
    target_ik: tuple[float, float, float],
    *,
    side: str,
    cfg: FootIkConfig,
    prev_target: tuple[float, float, float] | None,
) -> tuple[float, float, float, float, float, float]:
    """Choose IK iterate seed; fall back to FK when warm state is stale or unsafe."""
    if warm_q is None or not bool(getattr(cfg, "ik_warm_start", True)):
        return q_fk
    if prev_target is not None:
        dx = float(target_ik[0]) - float(prev_target[0])
        dy = float(target_ik[1]) - float(prev_target[1])
        dz = float(target_ik[2]) - float(prev_target[2])
        jump = math.sqrt(dx * dx + dy * dy + dz * dz)
        if jump > float(getattr(cfg, "ik_warm_reset_target_delta_m", 0.04)):
            return q_fk
    p_fk = g1_leg_fk_pos(q_fk, side=side)
    p_warm = g1_leg_fk_pos(warm_q, side=side)
    tx, ty, tz = target_ik
    r_fk_sq = (
        (tx - p_fk[0]) ** 2 + (ty - p_fk[1]) ** 2 + (tz - p_fk[2]) ** 2
    )
    r_warm_sq = (
        (tx - p_warm[0]) ** 2 + (ty - p_warm[1]) ** 2 + (tz - p_warm[2]) ** 2
    )
    max_foot_z = float(getattr(cfg, "ik_max_foot_z_local_m", 0.05))
    if float(p_warm[2]) > max_foot_z:
        return q_fk
    if r_warm_sq > r_fk_sq + 0.0004:
        return q_fk
    return warm_q


def _leg_ik_max_apply_residual_m(cfg: FootIkConfig) -> float:
    base = max(1e-6, float(getattr(cfg, "ik_max_apply_residual_m", 0.012)))
    if _foot_ik_lock_ankle(cfg):
        return base * 2.5
    return base


def _leg_ik_solution_acceptable(
    result: LegIkResult,
    *,
    cfg: FootIkConfig,
) -> bool:
    max_res = _leg_ik_max_apply_residual_m(cfg)
    if float(result.residual_m) > max_res:
        return False
    if float(result.foot_z_local) > float(getattr(cfg, "ik_max_foot_z_local_m", 0.05)):
        return False
    return True


def _leg_ik_result_foot_z_ok(result: LegIkResult, *, cfg: FootIkConfig) -> bool:
    return float(result.foot_z_local) <= float(getattr(cfg, "ik_max_foot_z_local_m", 0.05))


def _update_leg_ik_best(
    result: LegIkResult,
    *,
    cfg: FootIkConfig,
    best_any: LegIkResult | None,
    best_valid: LegIkResult | None,
) -> tuple[LegIkResult | None, LegIkResult | None]:
    if not _leg_ik_result_foot_z_ok(result, cfg=cfg):
        return best_any, best_valid
    if best_any is None or float(result.residual_m) < float(best_any.residual_m):
        best_any = result
    if _leg_ik_solution_acceptable(result, cfg=cfg) and (
        best_valid is None or float(result.residual_m) < float(best_valid.residual_m)
    ):
        best_valid = result
    return best_any, best_valid


def _solve_full_leg_ik_robust(
    target_ik: tuple[float, float, float],
    q_fk: tuple[float, float, float, float, float, float],
    *,
    warm_q: tuple[float, float, float, float, float, float] | None,
    prev_target: tuple[float, float, float] | None,
    side: str,
    cfg: FootIkConfig,
) -> tuple[tuple[float, float, float, float, float, float], float, int, bool]:
    """Multi-seed DLS IK; fast path uses one solve when warm start converges."""
    lock_ankle = _foot_ik_lock_ankle(cfg)
    if lock_ankle and warm_q is not None:
        warm_q = _leg_ik_merge_fk_ankle(warm_q, q_fk)

    def _finalize_q(
        q: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        return _leg_ik_merge_fk_ankle(q, q_fk) if lock_ankle else q

    target_use, _ = g1_leg_clamp_target_to_reach(
        target_ik,
        side=side,
        max_reach_ratio=float(cfg.max_reach_ratio),
    )
    q_seed = _pick_leg_ik_seed(
        q_fk,
        warm_q,
        target_use,
        side=side,
        cfg=cfg,
        prev_target=prev_target,
    )
    if lock_ankle:
        q_seed = _leg_ik_merge_fk_ankle(q_seed, q_fk)
    max_apply = _leg_ik_max_apply_residual_m(cfg)

    primary = solve_g1_leg_ik_dls(
        target_use,
        q_seed,
        q_reg=q_fk,
        side=side,
        cfg=cfg,
        lock_ankle_from_fk=lock_ankle,
    )
    if _leg_ik_solution_acceptable(primary, cfg=cfg):
        q_out = _finalize_q(primary.q)
        res = float(
            np.linalg.norm(
                np.asarray(target_use, dtype=np.float64)
                - np.asarray(g1_leg_fk_pos(q_out, side=side), dtype=np.float64)
            )
        )
        return q_out, res, int(primary.iterations), True

    best_any, best_valid = _update_leg_ik_best(
        primary, cfg=cfg, best_any=None, best_valid=None
    )

    extra_seeds: list[tuple[float, float, float, float, float, float]] = [q_fk]
    if float(q_fk[3]) < float(getattr(cfg, "ik_min_knee_rad", 0.12)) + 0.08:
        extra_seeds.append(
            (
                float(q_fk[0]),
                float(q_fk[1]),
                float(q_fk[2]),
                max(float(q_fk[3]), float(getattr(cfg, "ik_min_knee_rad", 0.12)) + 0.2),
                float(q_fk[4]),
                float(q_fk[5]),
            )
        )

    for seed in extra_seeds:
        if lock_ankle:
            seed = _leg_ik_merge_fk_ankle(seed, q_fk)
        if all(abs(float(seed[i]) - float(q_seed[i])) < 1e-9 for i in range(6)):
            continue
        result = solve_g1_leg_ik_dls(
            target_use,
            seed,
            q_reg=q_fk,
            side=side,
            cfg=cfg,
            lock_ankle_from_fk=lock_ankle,
        )
        best_any, best_valid = _update_leg_ik_best(
            result, cfg=cfg, best_any=best_any, best_valid=best_valid
        )
        if best_valid is not None and float(best_valid.residual_m) <= max_apply * 0.5:
            break

    if best_valid is None and best_any is not None and float(best_any.residual_m) > max_apply:
        retry_iters = min(48, max(int(cfg.ik_max_iters) * 2, int(cfg.ik_max_iters) + 12))

        class _RetryCfg:
            pass

        retry_cfg = _RetryCfg()
        for name in (
            "ik_max_iters",
            "ik_pos_tol_m",
            "ik_dls_lambda",
            "ik_step_scale",
            "ik_reg_weight",
            "ik_reg_hip_yaw",
            "ik_reg_ankle_roll",
            "ik_min_knee_rad",
        ):
            setattr(retry_cfg, name, getattr(cfg, name))
        retry_cfg.ik_max_iters = retry_iters
        retry_cfg.ik_reg_weight = float(cfg.ik_reg_weight) * 0.35

        retry = solve_g1_leg_ik_dls(
            target_use,
            _finalize_q(best_any.q),
            q_reg=q_fk,
            side=side,
            cfg=retry_cfg,
            lock_ankle_from_fk=lock_ankle,
        )
        best_any, best_valid = _update_leg_ik_best(
            retry, cfg=cfg, best_any=best_any, best_valid=best_valid
        )

    pick = best_valid if best_valid is not None else best_any
    if pick is None:
        q_fk_out = _finalize_q(q_fk)
        fk_res = float(
            np.linalg.norm(
                np.asarray(target_use, dtype=np.float64)
                - np.asarray(g1_leg_fk_pos(q_fk_out, side=side), dtype=np.float64)
            )
        )
        return q_fk_out, fk_res, 0, False

    q_out = _finalize_q(pick.q)
    res = float(
        np.linalg.norm(
            np.asarray(target_use, dtype=np.float64)
            - np.asarray(g1_leg_fk_pos(q_out, side=side), dtype=np.float64)
        )
    )
    accepted = best_valid is not None
    return q_out, res, int(pick.iterations), accepted


def _clamp_foot_ik_world_floor(
    foot_world: tuple[float, float, float],
    *,
    floor_z: float = 0.0,
    clearance_m: float = 0.005,
) -> tuple[float, float, float]:
    """Keep IK foot target from penetrating Isaac world floor (Z-up)."""
    min_wz = float(floor_z) + max(0.0, float(clearance_m))
    wz = float(foot_world[2])
    if wz >= min_wz:
        return foot_world
    return (float(foot_world[0]), float(foot_world[1]), min_wz)


def _remap_foot_ik_target_local(
    target_local: tuple[float, float, float],
    *,
    side: str,
    cfg: FootIkConfig,
) -> tuple[float, float, float]:
    return g1_leg_remap_foot_ik_target(
        target_local,
        side=side,
        xy_scale=float(cfg.leg_target_scale),
        max_reach_ratio=float(cfg.max_reach_ratio),
        ground_clearance_m=float(getattr(cfg, "leg_z_ground_clearance_m", 0.012)),
        z_compress_power=float(getattr(cfg, "leg_z_compress_power", 2.0)),
    )


def _scale_foot_ik_world_pos(
    foot_world: tuple[float, float, float] | None,
    *,
    side: str,
    cfg: FootIkConfig,
    root_pos_world: tuple[float, float, float] | None,
    root_quat_wxyz: list[float] | None,
) -> tuple[float, float, float] | None:
    if foot_world is None:
        return None
    if root_pos_world is None or root_quat_wxyz is None:
        return foot_world
    local = isaac_world_to_root_local(foot_world, root_pos_world, root_quat_wxyz)
    scaled_local = _remap_foot_ik_target_local(local, side=side, cfg=cfg)
    world = root_local_to_isaac_world(scaled_local, root_pos_world, root_quat_wxyz)
    world = _clamp_foot_ik_world_floor(
        world,
        floor_z=float(getattr(cfg, "leg_floor_z", 0.0)),
        clearance_m=float(getattr(cfg, "leg_z_ground_clearance_m", 0.012)),
    )
    return world


def _foot_ik_target_root_local(
    *,
    foot_world: tuple[float, float, float] | None,
    mmd_pos_raw: tuple[float, float, float] | None,
    frame_data: dict[str, dict] | None,
    cfg: FootIkConfig,
    side: str,
    root_pos_world: tuple[float, float, float] | None,
    root_quat_wxyz: list[float] | None,
    foot_ik_viz_cfg: FootIkVizConfig | None = None,
    center_mmd_pos: tuple[float, float, float] | None = None,
    center_bone_name: str | None = None,
) -> tuple[float, float, float] | None:
    """Root-local IK target from red-sphere Isaac world position (preferred)."""
    del center_bone_name
    need_scale = False
    if foot_world is None:
        if mmd_pos_raw is None or frame_data is None:
            return None
        foot_world = foot_ik_panel_to_isaac_world(
            mmd_pos_raw,
            frame_data,
            pos_scale=float(cfg.groove_pos_to_world),
            is_pose=bool(cfg.is_static_pose),
            side=side,
            viz_cfg=foot_ik_viz_cfg,
            target_root_pos=root_pos_world,
            target_root_quat_wxyz=root_quat_wxyz,
            center_mmd_pos=center_mmd_pos,
        )
        need_scale = True
    root_pos = root_pos_world if root_pos_world is not None else (0.0, 0.0, 0.0)
    root_quat = root_quat_wxyz if root_quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    local = isaac_world_to_root_local(foot_world, root_pos, root_quat)
    if need_scale:
        local = _remap_foot_ik_target_local(local, side=side, cfg=cfg)
        world = root_local_to_isaac_world(local, root_pos, root_quat)
        world = _clamp_foot_ik_world_floor(
            world,
            floor_z=float(getattr(cfg, "leg_floor_z", 0.0)),
            clearance_m=float(getattr(cfg, "leg_z_ground_clearance_m", 0.012)),
        )
        local = isaac_world_to_root_local(world, root_pos, root_quat)
    return local


def _apply_foot_ik_override_to_result(
    result: np.ndarray,
    frame_data: dict[str, dict],
    joint_names: list[str],
    default_joint_pos: np.ndarray,
    foot_ik_cfg: FootIkConfig | None,
    foot_ik_state: FootIkState | None,
    frame_idx: int | None = None,
    foot_ik_root_pos_world: tuple[float, float, float] | None = None,
    foot_ik_root_quat_wxyz: list[float] | None = None,
    foot_ik_viz_cfg: FootIkVizConfig | None = None,
    center_mmd_pos: tuple[float, float, float] | None = None,
) -> None:
    cfg = foot_ik_cfg
    state = foot_ik_state
    if cfg is None or state is None or (not bool(cfg.enable)):
        return
    state.last_left_ik_pred_world = None
    state.last_right_ik_pred_world = None
    state.last_left_ik_target_world = None
    state.last_right_ik_target_world = None
    state.last_left_ik_residual_m = None
    state.last_right_ik_residual_m = None
    state.last_left_ik_iters = None
    state.last_right_ik_iters = None

    center_pos = center_mmd_pos
    center_bone: str | None = None
    if center_pos is None and foot_ik_root_pos_world is not None:
        center_pos, center_bone = resolve_mmd_root_translation_pos(frame_data)

    jidx = {str(n): i for i, n in enumerate(joint_names)}
    for side in ("left", "right"):
        candidates = _FOOT_IK_BONE_CANDIDATES[side]
        mmd_pos = _read_first_available_bone_pos(frame_data, candidates)
        if mmd_pos is None:
            continue
        foot_world = (
            state.last_left_foot_mmd_viz_world
            if side == "left"
            else state.last_right_foot_mmd_viz_world
        )
        target = _foot_ik_target_root_local(
            foot_world=foot_world,
            mmd_pos_raw=mmd_pos,
            frame_data=frame_data,
            cfg=cfg,
            side=side,
            root_pos_world=foot_ik_root_pos_world,
            root_quat_wxyz=foot_ik_root_quat_wxyz,
            foot_ik_viz_cfg=foot_ik_viz_cfg,
            center_mmd_pos=center_pos,
            center_bone_name=center_bone,
        )
        if target is None:
            continue
        prev_target = (
            state.last_left_target_local if side == "left" else state.last_right_target_local
        )
        if side == "left":
            state.last_left_target_local = target
        else:
            state.last_right_target_local = target

        hj_p, hj_r, hj_y, kj, aj_p, aj_r = _FOOT_IK_OVERRIDE_JOINTS[side]
        req = (hj_p, hj_r, hj_y, kj, aj_p, aj_r)
        if any(jn not in jidx for jn in req):
            continue

        offset = cfg.ankle_target_offset_local
        target_ik = (
            float(target[0]) + float(offset[0]),
            float(target[1]) + float(offset[1]),
            float(target[2]) + float(offset[2]),
        )
        target_ik_world = None
        if foot_ik_root_pos_world is not None and foot_ik_root_quat_wxyz is not None:
            target_ik_world = root_local_to_isaac_world(
                target_ik,
                foot_ik_root_pos_world,
                foot_ik_root_quat_wxyz,
            )
        if side == "left":
            state.last_left_ik_target_world = target_ik_world
        else:
            state.last_right_ik_target_world = target_ik_world
        q_fk = tuple(
            float(result[jidx[jn]] - default_joint_pos[jidx[jn]]) for jn in req
        )
        warm_q = state.last_left_q_ik if side == "left" else state.last_right_q_ik
        ik_vals, ik_residual, ik_iters, ik_accepted = _solve_full_leg_ik_robust(
            target_ik,
            q_fk,
            warm_q=warm_q,
            prev_target=prev_target,
            side=side,
            cfg=cfg,
        )
        warm_keep = max(
            1e-6,
            _leg_ik_max_apply_residual_m(cfg) * 2.5,
        )
        if ik_accepted or float(ik_residual) <= warm_keep:
            if side == "left":
                state.last_left_q_ik = ik_vals
            else:
                state.last_right_q_ik = ik_vals
        pred_local = g1_leg_fk_pos(ik_vals, side=side)
        pred_world = None
        if foot_ik_root_pos_world is not None and foot_ik_root_quat_wxyz is not None:
            pred_world = root_local_to_isaac_world(
                pred_local,
                foot_ik_root_pos_world,
                foot_ik_root_quat_wxyz,
            )
        if side == "left":
            state.last_left_ik_pred_world = pred_world
            state.last_left_ik_residual_m = ik_residual
            state.last_left_ik_iters = ik_iters
        else:
            state.last_right_ik_pred_world = pred_world
            state.last_right_ik_residual_m = ik_residual
            state.last_right_ik_iters = ik_iters
        for jn, ik_angle in zip(req, ik_vals):
            ji = jidx[jn]
            result[ji] = float(default_joint_pos[ji] + float(ik_angle))
        debug_stride = int(max(0, int(cfg.debug_every_n_frames)))
        if (
            debug_stride > 0
            and frame_idx is not None
            and int(frame_idx) >= 0
            and int(frame_idx) % debug_stride == 0
        ):
            red_z = foot_world[2] if foot_world is not None else float("nan")
            print(
                "[IKDBG] f=%d side=%s red_z=%.3f target_root_local=(%.3f,%.3f,%.3f) "
                "residual=%.4fm iters=%s conv=%s"
                % (
                    int(frame_idx),
                    side,
                    float(red_z),
                    float(target_ik[0]),
                    float(target_ik[1]),
                    float(target_ik[2]),
                    float(ik_residual if ik_residual is not None else -1.0),
                    str(ik_iters),
                    "yes" if (ik_residual is not None and ik_residual <= float(cfg.ik_pos_tol_m)) else "no",
                )
            )


def _apply_knee_hinge_projection(
    frame_data: dict[str, dict],
    side: str,
    mapping: dict[str, AxisMapEntry],
) -> None:
    side_cfg = {
        "left": ("left_knee_joint", "左足", "左ひざ"),
        "right": ("right_knee_joint", "右足", "右ひざ"),
    }
    cfg = side_cfg.get(side)
    if cfg is None:
        return
    knee_joint, hip_bone, knee_bone = cfg
    knee_map = mapping.get(knee_joint)
    if knee_map is None:
        return
    _, axis, _scale = knee_map
    q_hip = _read_bone_quat_xyzw(frame_data, hip_bone)
    q_knee = _read_bone_quat_xyzw(frame_data, knee_bone)
    if q_hip is None or q_knee is None:
        return

    q_swing, _q_twist = _swing_twist_decompose_xyzw(q_knee, axis)
    absorb = get_hinge_swing_absorb(knee_joint)
    swing_deg = _quat_rotation_magnitude_deg_xyzw(q_swing)
    if swing_deg > 1e-6 and _KNEE_PROJECTION_MAX_SWING_DEG > 0.0:
        # Scale absorb down when swing is too large, keeping projection continuous
        # while avoiding impulsive hip correction.
        absorb *= min(1.0, float(_KNEE_PROJECTION_MAX_SWING_DEG / swing_deg))
    q_s_applied = _quat_pow_xyzw(q_swing, absorb)
    q_knee_new = _quat_normalize(_quat_multiply(_quat_conjugate(q_s_applied), q_knee))
    _write_bone_quat_xyzw(frame_data, knee_bone, q_knee_new)
    _write_bone_quat_xyzw(frame_data, hip_bone, _quat_multiply(q_hip, q_s_applied))


def _apply_elbow_hinge_projection(
    frame_data: dict[str, dict],
    side: str,
    mapping: dict[str, AxisMapEntry],
) -> None:
    """肘仅保留铰链 twist（pitch），移除 roll/yaw 的肩部补偿链路。"""
    side_cfg = {
        "left": ("left_elbow_joint", "左腕", "左ひじ"),
        "right": ("right_elbow_joint", "右腕", "右ひじ"),
    }
    cfg = side_cfg.get(side)
    if cfg is None:
        return
    elbow_joint, arm_bone, elbow_bone = cfg
    elbow_map = mapping.get(elbow_joint)
    if elbow_map is None:
        return
    _, axis, _scale = elbow_map
    q_elbow = _read_bone_quat_xyzw(frame_data, elbow_bone)
    if q_elbow is None:
        return

    _q_swing, q_twist = _swing_twist_decompose_xyzw(q_elbow, axis)
    # elbow 仅保留其铰链 twist（pitch），避免把 roll/yaw 继续留在 elbow。
    q_elbow_new = _quat_normalize(q_twist)
    _write_bone_quat_xyzw(frame_data, elbow_bone, q_elbow_new)


def get_g1_angle_from_frame(joint_name: str, frame_data: dict[str, dict]) -> float | None:
    """
    从帧数据中获取指定 G1 关节的目标角度偏移（弧度）。
    - 单骨骼（1 DOF：肘/腕/腿等）：对骨骼四元数做 Swing-Twist 单轴提取
    - 多骨骼组合（肩 = [肩, 腕]，腰 = [上半身, 上半身2]）：
        肩：专用链式反解；腰：``retarget_unitreeG1.compute_waist_angles`` 得 (pitch, roll, yaw)，
        再按映射里 axis_idx 的 0/1/2 取分量；scale 仅作符号或小幅增益。
    - 使用 get_mapping()，支持 UI 编辑后的映射

    肩链与 G1 URDF 一致：pitch(Y) → roll(X) → yaw(Z)，对应 R≈Rz·Rx·Ry，用 ``syxz``。
    腰链：yaw(Z) → roll(X) → pitch(Y)，对应 R≈Ry·Rx·Rz，用 ``szxy``（语义重排见 ``retarget_unitreeG1``）。
    """
    mapping = get_mapping()
    if joint_name not in mapping:
        return None
    bones, axis, scale = mapping[joint_name]

    # 髋关节 3DOF：专用链式反解
    if joint_name in HIP_JOINT_TO_AXIS_INDEX:
        side, leg_bone = HIP_JOINT_TO_SIDE_BONE[joint_name]
        q_leg = _read_bone_quat_xyzw(frame_data, leg_bone)
        if q_leg is None:
            return None
        pitch, roll, yaw = compute_hip_angles(side, q_leg)
        triple = (pitch, roll, yaw)
        return float(triple[HIP_JOINT_TO_AXIS_INDEX[joint_name]] * scale)

    # 踝关节 2DOF：专用链式反解
    if joint_name in ANKLE_JOINT_TO_AXIS_INDEX:
        side, ank_bone = ANKLE_JOINT_TO_SIDE_BONE[joint_name]
        q_ank = _read_bone_quat_xyzw(frame_data, ank_bone)
        if q_ank is None:
            return None
        pitch, roll = compute_ankle_angles(side, q_ank)
        pair = (pitch, roll)
        return float(pair[ANKLE_JOINT_TO_AXIS_INDEX[joint_name]] * scale)

    # 肩部 3DOF 走专用重定向（YXZ intrinsic + MMD->G1 基变换）
    if joint_name in SHOULDER_JOINT_TO_AXIS_INDEX:
        side, sho_bone, arm_bone = SHOULDER_JOINT_TO_SIDE_BONES[joint_name]
        q_sho = _read_bone_quat_xyzw(frame_data, sho_bone)
        q_arm = _read_bone_quat_xyzw(frame_data, arm_bone)
        q_elbow = _read_bone_quat_xyzw(frame_data, "左ひじ" if side == "left" else "右ひじ")
        if q_sho is None and q_arm is None:
            return None
        pitch, roll, yaw = compute_shoulder_angles(side, q_sho, q_arm, q_elbow)
        triple = (pitch, roll, yaw)
        base_val = triple[SHOULDER_JOINT_TO_AXIS_INDEX[joint_name]]
        return float(base_val * scale)

    # 肘部 1DOF：前臂长轴夹角弯曲（剔除烘焙进肘骨的前臂自转）。
    if joint_name in ELBOW_JOINT_TO_SIDE_BONE:
        side, elbow_bone = ELBOW_JOINT_TO_SIDE_BONE[joint_name]
        q_elbow = _read_bone_quat_xyzw(frame_data, elbow_bone)
        if q_elbow is None:
            return None
        bend = compute_elbow_angle(side, q_elbow)
        return float(bend * scale)

    # 腕部 3DOF 走专用重定向（YXZ intrinsic + MMD->G1 基变换），
    # 取代旧的三轴独立 swing-twist 提取。并把肘骨的前臂自转(pronation)转交给腕，
    # 恢复手心朝向。
    if joint_name in WRIST_JOINT_TO_AXIS_INDEX:
        side, wrist_bone = WRIST_JOINT_TO_SIDE_BONE[joint_name]
        elbow_bone = "左ひじ" if side == "left" else "右ひじ"
        q_wrist = _read_bone_quat_xyzw(frame_data, wrist_bone)
        q_elbow = _read_bone_quat_xyzw(frame_data, elbow_bone)
        if q_wrist is None and q_elbow is None:
            return None
        pitch, roll, yaw = compute_wrist_angles(side, q_wrist, q_elbow)
        triple = (pitch, roll, yaw)
        base_val = triple[WRIST_JOINT_TO_AXIS_INDEX[joint_name]]
        return float(base_val * scale)

    # 腰部 3DOF：专用链式反解（物理轴→语义 pitch/roll/yaw）
    if joint_name in WAIST_JOINT_TO_AXIS_INDEX:
        q_upper = _read_bone_quat_xyzw(frame_data, "上半身")
        q_upper2 = _read_bone_quat_xyzw(frame_data, "上半身2")
        q_lower = _read_first_available_bone_quat_xyzw(frame_data, ("下半身",))
        c0, c1 = get_waist_upper_pair_quat_conjugate()
        pitch, roll, yaw = compute_waist_angles(q_upper, q_upper2, q_lower, (c0, c1))
        triple = (pitch, roll, yaw)
        return float(triple[WAIST_JOINT_TO_AXIS_INDEX[joint_name]] * scale)

    if isinstance(bones, list):
        if len(bones) != 2:
            return None

        q_first = _read_bone_quat_xyzw(frame_data, bones[0])
        q_second = _read_bone_quat_xyzw(frame_data, bones[1])
        if q_first is None and q_second is None:
            return None
        if q_first is None:
            q = _quat_normalize(q_second)
        elif q_second is None:
            q = _quat_normalize(q_first)
        else:
            q = _quat_multiply(q_first, q_second)
        qx, qy, qz, qw = _quat_normalize(q)
        if qw < 0.0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        euler_xyz = _quat_to_euler(qx, qy, qz, qw)
        idx = _axis_to_index(axis)
        base_val = euler_xyz[idx]
    else:
        if bones not in frame_data:
            return None
        q_single = _read_bone_quat_xyzw(frame_data, bones)
        if q_single is None:
            return None
        q = _quat_normalize(q_single)
        base_val = swing_twist_angle(*q, axis)

    return float(base_val * scale)


def build_joint_positions_from_frame(
    frame_data: dict[str, dict],
    joint_names: list[str],
    default_joint_pos: np.ndarray,
    knee_hinge_projection: bool = True,
    enable_hand: bool = True,
    foot_ik_cfg: FootIkConfig | None = None,
    foot_ik_state: FootIkState | None = None,
    foot_ik_frame_idx: int | None = None,
    foot_ik_root_pos_world: tuple[float, float, float] | None = None,
    foot_ik_root_quat_wxyz: list[float] | None = None,
    foot_ik_viz_cfg: FootIkVizConfig | None = None,
    center_mmd_pos: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    从一帧的骨骼数据构建 G1 关节位置数组。
    - joint_names: 按 action 顺序的关节名列表
    - default_joint_pos: 默认关节位置
    - 返回: 目标关节位置，与 joint_names 同序
    """
    source_frame_data = frame_data
    if knee_hinge_projection:
        source_frame_data = dict(frame_data)
        for bone_name in ("左足", "左ひざ", "右足", "右ひざ", "左腕", "左ひじ", "右腕", "右ひじ"):
            bone_data = source_frame_data.get(bone_name)
            if bone_data is not None:
                source_frame_data[bone_name] = dict(bone_data)
        mapping = get_mapping()
        _apply_knee_hinge_projection(source_frame_data, "left", mapping)
        _apply_knee_hinge_projection(source_frame_data, "right", mapping)

    result = default_joint_pos.copy()
    for i, jname in enumerate(joint_names):
        if (not enable_hand) and is_hand_joint_name(jname):
            continue
        angle = get_g1_angle_from_frame(jname, source_frame_data)
        if angle is not None:
            result[i] = default_joint_pos[i] + angle
    _apply_foot_ik_override_to_result(
        result,
        source_frame_data,
        joint_names,
        default_joint_pos,
        foot_ik_cfg,
        foot_ik_state,
        frame_idx=foot_ik_frame_idx,
        foot_ik_root_pos_world=foot_ik_root_pos_world,
        foot_ik_root_quat_wxyz=foot_ik_root_quat_wxyz,
        foot_ik_viz_cfg=foot_ik_viz_cfg,
        center_mmd_pos=center_mmd_pos,
    )
    return result


def shoulder_retarget_debug_ui_extra(
    frame_data_raw: dict[str, dict] | None,
) -> dict[str, str]:
    """
    返回肩部 retarget 原始角度（未施加 scale）供 UI 显示。
    键: '__sho_left_raw' / '__sho_right_raw'
    值: 'P:±XX.X° R:±XX.X° Y:±XX.X°'
    """
    if not frame_data_raw:
        return {}
    return _shoulder_debug_info(frame_data_raw, _read_bone_quat_xyzw)


def retarget_leg_debug_ui_extra(
    frame_data_raw: dict[str, dict] | None,
) -> dict[str, str]:
    """
    返回腿部 retarget 原始角度（未施加 scale）供 UI 显示。
    键:
      '__leg_left_hip_raw' / '__leg_right_hip_raw' / '__leg_left_ank_raw' / '__leg_right_ank_raw'
    """
    if not frame_data_raw:
        return {}
    return _leg_debug_info(frame_data_raw, _read_bone_quat_xyzw)


def iter_motion_frames(
    csv_path: str,
    fps: float = 30.0,
) -> Iterator[tuple[int, dict[str, dict]]]:
    """
    按帧号顺序迭代动作数据。
    对于稀疏关键帧，会为每个整数帧插值生成数据。
    fps: VMD 标准帧率，用于确定帧范围
    """
    frames = load_csv_motion(csv_path)
    frame_list = get_frame_indices(frames)
    if not frame_list:
        return

    max_frame = frame_list[-1]
    all_bones = set()
    for f in frames.values():
        all_bones.update(f.keys())
    bone_frame_lists = get_bone_frame_lists(frames, frame_list, all_bones)

    for frame in range(max_frame + 1):
        frame_data: dict[str, dict] = {}
        for bone in all_bones:
            d = interpolate_bone(frame, bone, frames, bone_frame_lists.get(bone))
            if d is not None:
                frame_data[bone] = d
        if frame_data:
            yield frame, frame_data


def update_foot_ik_mmd_viz_world(
    foot_ik_state: FootIkState | None,
    frame_data: dict[str, dict],
    groove_pos_to_world: float,
    *,
    is_pose: bool = False,
    foot_ik_viz_cfg: FootIkVizConfig | None = None,
    target_root_pos: tuple[float, float, float] | None = None,
    target_root_quat_wxyz: list[float] | None = None,
    center_mmd_pos: tuple[float, float, float] | None = None,
    root_trans_bone: str | None = None,
    foot_ik_cfg: FootIkConfig | None = None,
) -> None:
    """Fill ``FootIkState`` red-sphere / IK target world positions for one frame."""
    del center_mmd_pos, root_trans_bone
    cfg = foot_ik_cfg if foot_ik_cfg is not None else FootIkConfig()
    if foot_ik_state is None:
        return
    foot_ik_state.last_left_foot_mmd_viz_world = None
    foot_ik_state.last_right_foot_mmd_viz_world = None
    foot_ik_state.last_left_toe_mmd_viz_world = None
    foot_ik_state.last_right_toe_mmd_viz_world = None
    foot_ik_state.last_left_foot_mmd_local_m = None
    foot_ik_state.last_right_foot_mmd_local_m = None
    foot_ik_state.last_left_foot_mmd_fk_world_m = None
    foot_ik_state.last_right_foot_mmd_fk_world_m = None
    foot_ik_state.last_left_toe_mmd_local_m = None
    foot_ik_state.last_right_toe_mmd_local_m = None
    foot_ik_state.last_left_toe_mmd_fk_world_m = None
    foot_ik_state.last_right_toe_mmd_fk_world_m = None
    foot_ik_state.last_target_root_world = target_root_pos
    if target_root_quat_wxyz is not None:
        foot_ik_state.last_target_root_quat_wxyz = [
            float(v) for v in target_root_quat_wxyz
        ]
    if not frame_data:
        return
    try:
        bundle = compute_mmd_foot_ik_viz_bundle(
            frame_data,
            pos_scale=float(groove_pos_to_world),
            is_pose=bool(is_pose),
            viz_cfg=foot_ik_viz_cfg,
        )
    except Exception:
        return

    def _apply(prefix: str, block: dict[str, tuple[float, float, float] | None], side: str) -> None:
        setattr(foot_ik_state, f"last_{prefix}_mmd_local_m", block.get("local_m"))
        setattr(foot_ik_state, f"last_{prefix}_mmd_fk_world_m", block.get("fk_world_m"))
        raw_world = block.get("isaac_world_m")
        setattr(
            foot_ik_state,
            f"last_{prefix}_mmd_viz_world",
            _scale_foot_ik_world_pos(
                raw_world,
                side=side,
                cfg=cfg,
                root_pos_world=target_root_pos,
                root_quat_wxyz=target_root_quat_wxyz,
            ),
        )

    _apply("left_foot", bundle["left"], "left")
    _apply("right_foot", bundle["right"], "right")
    _apply("left_toe", bundle["left_toe"], "left")
    _apply("right_toe", bundle["right_toe"], "right")


def update_foot_ik_reach_clamp_flags(
    foot_ik_state: FootIkState | None,
    foot_ik_cfg: FootIkConfig | None,
    *,
    root_pos_world: tuple[float, float, float] | None,
    root_quat_wxyz: list[float] | None,
    margin_m: float = FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M,
) -> None:
    """Set per-leg reach-clamp flags from red-sphere world targets (viz, not IK enable)."""
    if foot_ik_state is None or foot_ik_cfg is None:
        return
    foot_ik_state.last_left_reach_clamped = False
    foot_ik_state.last_right_reach_clamped = False
    root_pos = root_pos_world if root_pos_world is not None else (0.0, 0.0, 0.0)
    root_quat = root_quat_wxyz if root_quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    for side, attr in (
        ("left", "last_left_foot_mmd_viz_world"),
        ("right", "last_right_foot_mmd_viz_world"),
    ):
        foot_world = getattr(foot_ik_state, attr, None)
        if foot_world is None:
            continue
        target = isaac_world_to_root_local(foot_world, root_pos, root_quat)
        offset = getattr(foot_ik_cfg, "ankle_target_offset_local", (0.0, 0.0, 0.02))
        target_ik = (
            float(target[0]) + float(offset[0]),
            float(target[1]) + float(offset[1]),
            float(target[2]) + float(offset[2]),
        )
        clamped = g1_leg_reach_clamped(
            target_ik,
            side=side,
            max_reach_ratio=float(foot_ik_cfg.max_reach_ratio),
            margin_m=float(margin_m),
        )
        if side == "left":
            foot_ik_state.last_left_reach_clamped = clamped
        else:
            foot_ik_state.last_right_reach_clamped = clamped
