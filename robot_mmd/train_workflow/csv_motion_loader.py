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

内部统一保存为弧度欧拉角 + 归一化四元数（bone dict 仅保留 `quat_wxyz`），
并使用 Swing-Twist 提取指定轴角度，用于 MMD -> G1 的 1DOF 关节重定向。

注意：CSV 列 `quat_x/quat_y/quat_z/quat_w` 仍表示 x,y,z,w；仅在读入内存后转换为
`quat_wxyz`（w,x,y,z）以对齐 Isaac root_state API。
"""
import bisect
import csv
import math
from typing import Iterator

import numpy as np

from robot_mmd.train_workflow.g1_joint_axis_map_raw import AxisMapRawEntry, G1_JOINT_AXIS_MAP_RAW

Axis3 = tuple[float, float, float]
AxisMapEntry = tuple[str | list[str], Axis3, float]


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
                f"MMD hinge {hinge_deg:.1f}deg swing {sw_deg:.1f}deg (proj off)"
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
            f"MMD hinge {hinge_deg:.1f}deg swing {sw_deg:.1f}deg | "
            f"hip d(p/r/y) {dp:+.1f}/{dr:+.1f}/{dy:+.1f}deg"
        )
    return out


def elbow_hinge_mapping_ui_extra(
    frame_data_raw: dict[str, dict] | None,
    *,
    projection_enabled: bool,
) -> dict[str, str]:
    """
    Mapping UI: MMD elbow local rotation split + shoulder map delta (deg).
    Keys: ``{left|right}_elbow_joint__elbow_mmd``.
    """
    out: dict[str, str] = {}
    if not frame_data_raw:
        return out

    side_cfg = {
        "left": (
            "left_elbow_joint",
            "左腕",
            "左ひじ",
            (
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
            ),
        ),
        "right": (
            "right_elbow_joint",
            "右腕",
            "右ひじ",
            (
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
            ),
        ),
    }

    mapping = get_mapping()
    for side, (elbow_joint, arm_bone, elbow_bone, shoulder_js) in side_cfg.items():
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

        fd = dict(frame_data_raw)
        for b in (arm_bone, elbow_bone):
            if b in fd:
                fd[b] = dict(fd[b])
        _apply_elbow_hinge_projection(fd, side, mapping)
        d_list: list[float] = []
        for sj in shoulder_js:
            a = get_g1_angle_from_frame(sj, frame_data_raw)
            b = get_g1_angle_from_frame(sj, fd)
            if a is not None and b is not None:
                d_list.append(math.degrees(b - a))
            else:
                d_list.append(0.0)
        dp, dr, dy = d_list[0], d_list[1], d_list[2]
        out[f"{elbow_joint}__elbow_mmd"] = (
            f"MMD hinge {hinge_deg:.1f}deg swing {sw_deg:.1f}deg | "
            f"sho d(p/r/y) {dp:+.1f}/{dr:+.1f}/{dy:+.1f}deg"
        )
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
        "left_elbow_joint",
        "right_elbow_joint",
    }
)
_DEFAULT_HINGE_SWING_ABSORB: dict[str, float] = {k: 1.0 for k in HINGE_SWING_ABSORB_JOINTS}
_hinge_swing_absorb: dict[str, float] = dict(_DEFAULT_HINGE_SWING_ABSORB)


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


def _write_bone_quat_xyzw(frame_data: dict[str, dict], bone_name: str, q_xyzw: tuple[float, float, float, float]) -> None:
    bone_data = frame_data.get(bone_name)
    if bone_data is None:
        return
    q_norm = _quat_normalize(q_xyzw)
    bone_data["quat_wxyz"] = _bone_quat_from_xyzw(q_norm)
    bone_data["euler"] = _quat_to_euler(*q_norm)


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
    q_s_applied = _quat_pow_xyzw(q_swing, absorb)
    q_knee_new = _quat_normalize(_quat_multiply(_quat_conjugate(q_s_applied), q_knee))
    _write_bone_quat_xyzw(frame_data, knee_bone, q_knee_new)
    _write_bone_quat_xyzw(frame_data, hip_bone, _quat_multiply(q_hip, q_s_applied))


def _apply_elbow_hinge_projection(
    frame_data: dict[str, dict],
    side: str,
    mapping: dict[str, AxisMapEntry],
) -> None:
    """将ひじ的非铰链 swing 并回腕骨，让 shoulder 三轴（肩+腕组合）吸收。"""
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
    q_arm = _read_bone_quat_xyzw(frame_data, arm_bone)
    q_elbow = _read_bone_quat_xyzw(frame_data, elbow_bone)
    if q_arm is None or q_elbow is None:
        return

    q_swing, _q_twist = _swing_twist_decompose_xyzw(q_elbow, axis)
    absorb = get_hinge_swing_absorb(elbow_joint)
    q_s_applied = _quat_pow_xyzw(q_swing, absorb)
    q_elbow_new = _quat_normalize(_quat_multiply(_quat_conjugate(q_s_applied), q_elbow))
    _write_bone_quat_xyzw(frame_data, elbow_bone, q_elbow_new)
    _write_bone_quat_xyzw(frame_data, arm_bone, _quat_multiply(q_arm, q_s_applied))


def get_g1_angle_from_frame(joint_name: str, frame_data: dict[str, dict]) -> float | None:
    """
    从帧数据中获取指定 G1 关节的目标角度偏移（弧度）。
    - 单骨骼：对骨骼四元数做 Swing-Twist 提取
    - 肩部 [肩, 腕]：先组合四元数再做 Swing-Twist
    - 使用 get_mapping()，支持 UI 编辑后的映射
    """
    mapping = get_mapping()
    if joint_name not in mapping:
        return None
    bones, axis, scale = mapping[joint_name]
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
            q = _quat_normalize(_quat_multiply(q_first, q_second))
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
        _apply_elbow_hinge_projection(source_frame_data, "left", mapping)
        _apply_elbow_hinge_projection(source_frame_data, "right", mapping)

    result = default_joint_pos.copy()
    for i, jname in enumerate(joint_names):
        angle = get_g1_angle_from_frame(jname, source_frame_data)
        if angle is not None:
            result[i] = default_joint_pos[i] + angle
    return result


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
