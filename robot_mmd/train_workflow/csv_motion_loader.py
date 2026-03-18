"""
CSV 骨骼动作加载器 - 从欧拉角格式 CSV 加载并按帧提供数据
支持帧间插值，以及 MMD 骨骼到 G1 关节的映射
"""
import bisect
import csv
import math
from typing import Iterator

import numpy as np


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


def _combine_shoulder_euler(euler_肩: tuple[float, float, float], euler_腕: tuple[float, float, float]) -> tuple[float, float, float]:
    """将 肩+腕 的欧拉角组合为肩部真实 RPY。MMD 层级：肩 -> 腕，组合旋转 R_肩 * R_腕"""
    q_肩 = _euler_to_quat(*euler_肩)
    q_腕 = _euler_to_quat(*euler_腕)
    q_combined = _quat_multiply(q_肩, q_腕)
    return _quat_to_euler(*q_combined)


# G1 关节 -> (MMD 骨骼或 [肩, 腕] 列表, 欧拉分量索引, 缩放系数)
# 肩部需组合 肩+腕 得到真实 RPY（肩带动腕，层级：肩->腕）
# 欧拉分量: roll=0, pitch=1, yaw=2
G1_JOINT_TO_MMD: dict[str, tuple[str | list[str], int, float]] = {
    # 腿部
    "right_knee_joint": ("右ひざ", 1, 1.0),
    "left_knee_joint": ("左ひざ", 1, 1.0),
    "right_hip_pitch_joint": ("右足", 1, 0.8),
    "left_hip_pitch_joint": ("左足", 1, 0.8),
    "right_hip_roll_joint": ("下半身", 0, 0.3),
    "left_hip_roll_joint": ("下半身", 0, 0.3),
    "right_hip_yaw_joint": ("右足", 2, 0.5),
    "left_hip_yaw_joint": ("左足", 2, 0.5),
    "right_ankle_pitch_joint": ("右足首", 1, 0.6),
    "left_ankle_pitch_joint": ("左足首", 1, 0.6),
    "right_ankle_roll_joint": ("右足首", 0, 0.5),
    "left_ankle_roll_joint": ("左足首", 0, 0.5),
    # 手臂 - 肩部由 肩+腕 组合得到真实 RPY
    # 102 X 012 021 120 210 201
    "right_shoulder_pitch_joint": (["右肩", "右腕"], 0, -1.0),
    "right_shoulder_roll_joint": (["右肩", "右腕"], 1, 1.0),
    "right_shoulder_yaw_joint": (["右肩", "右腕"], 2, 1.0),
    "left_shoulder_pitch_joint": (["左肩", "左腕"], 0, -1.0),
    "left_shoulder_roll_joint": (["左肩", "左腕"], 1, -1.0),
    "left_shoulder_yaw_joint": (["左肩", "左腕"], 2, -1.0),
    "right_elbow_joint": ("右ひじ", 1, 1.0),
    "left_elbow_joint": ("左ひじ", 1, -1.0),
    "right_wrist_pitch_joint": ("右手首", 0, -1.0),
    "right_wrist_roll_joint": ("右手首", 1, -1.0),
    "right_wrist_yaw_joint": ("右手首", 2, -1.0),
    "left_wrist_pitch_joint": ("左手首", 0, 1.0),
    "left_wrist_roll_joint": ("左手首", 1, 1.0),
    "left_wrist_yaw_joint": ("左手首", 2, 1.0),
    # 躯干
    "waist_pitch_joint": ("上半身", 1, 1.0),
    "waist_roll_joint": ("上半身", 0, 1.0),
    "waist_yaw_joint": ("首", 2, 1.0),
}

# 运行时可编辑的映射（UI 修改后生效），None 时使用 G1_JOINT_TO_MMD
_editable_mapping: dict[str, tuple[str | list[str], int, float]] | None = None


def get_mapping() -> dict[str, tuple[str | list[str], int, float]]:
    """获取当前生效的映射（优先使用 UI 编辑后的）"""
    if _editable_mapping is not None:
        return _editable_mapping
    return G1_JOINT_TO_MMD


def set_editable_mapping(mapping: dict[str, tuple[str | list[str], int, float]] | None) -> None:
    """设置可编辑映射，None 时恢复默认"""
    global _editable_mapping
    _editable_mapping = mapping


def update_mapping_entry(joint_name: str, euler_idx: int, scale: float) -> None:
    """更新单个关节的映射（仅改 euler_idx 和 scale，骨骼名不变）"""
    global _editable_mapping
    base = G1_JOINT_TO_MMD.get(joint_name)
    if base is None:
        return
    bones = base[0]
    if _editable_mapping is None:
        _editable_mapping = dict(G1_JOINT_TO_MMD)
    _editable_mapping[joint_name] = (bones, euler_idx, scale)


def reset_mapping_to_default() -> None:
    """重置为默认映射"""
    global _editable_mapping
    _editable_mapping = None


def load_csv_motion(csv_path: str) -> dict[int, dict[str, dict]]:
    """
    加载欧拉角格式 CSV 骨骼数据。
    格式: frame, bone, pos_x, pos_y, pos_z, roll, pitch, yaw
    返回 {frame_idx: {bone_name: {pos, euler}}}
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
    if not rows or "roll" not in rows[0] or "pitch" not in rows[0] or "yaw" not in rows[0]:
        raise ValueError("CSV 必须包含 roll, pitch, yaw 列（欧拉角格式）")

    frames: dict[int, dict[str, dict]] = {}
    for row in rows:
        frame = int(row["frame"])
        bone = row["bone"]
        pos = (
            float(row["pos_x"]),
            float(row["pos_y"]),
            float(row["pos_z"]),
        )
        euler = (
            float(row["roll"]),
            float(row["pitch"]),
            float(row["yaw"]),
        )
        if frame not in frames:
            frames[frame] = {}
        frames[frame][bone] = {"pos": pos, "euler": euler}
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
        return dict(frames[frame][bone])

    # bisect 二分查找前后关键帧
    idx = bisect.bisect_right(bone_frame_list, frame)
    prev_f = bone_frame_list[idx - 1] if idx > 0 else None
    next_f = bone_frame_list[idx] if idx < len(bone_frame_list) else None

    if prev_f is None and next_f is None:
        return None
    if prev_f is None:
        if bone not in frames.get(next_f, {}):
            return None
        return dict(frames[next_f][bone])
    if next_f is None:
        if bone not in frames.get(prev_f, {}):
            return None
        return dict(frames[prev_f][bone])
    if prev_f == next_f:
        if bone not in frames.get(prev_f, {}):
            return None
        return dict(frames[prev_f][bone])

    # 欧拉角线性插值
    if bone not in frames.get(prev_f, {}) or bone not in frames.get(next_f, {}):
        return None
    d0, d1 = frames[prev_f][bone], frames[next_f][bone]
    t = (frame - prev_f) / (next_f - prev_f)
    pos = tuple((1 - t) * a + t * b for a, b in zip(d0["pos"], d1["pos"]))
    e0, e1 = d0["euler"], d1["euler"]
    euler = tuple((1 - t) * a + t * b for a, b in zip(e0, e1))
    return {"pos": pos, "euler": euler}


def get_g1_angle_from_frame(joint_name: str, frame_data: dict[str, dict]) -> float | None:
    """
    从帧数据中获取指定 G1 关节的目标角度偏移（弧度）。
    - 单骨骼：直接取对应欧拉分量
    - 肩部 [肩, 腕]：组合两骨骼旋转后取 RPY 分量
    - 使用 get_mapping()，支持 UI 编辑后的映射
    """
    mapping = get_mapping()
    if joint_name not in mapping:
        return None
    bones, euler_idx, scale = mapping[joint_name]
    if isinstance(bones, list):
        # 肩部：组合 肩+腕 得到真实 RPY
        if len(bones) != 2 or bones[0] not in frame_data or bones[1] not in frame_data:
            return None
        euler_肩 = frame_data[bones[0]]["euler"]
        euler_腕 = frame_data[bones[1]]["euler"]
        roll, pitch, yaw = _combine_shoulder_euler(euler_肩, euler_腕)
    else:
        if bones not in frame_data:
            return None
        roll, pitch, yaw = frame_data[bones]["euler"]
    euler = (roll, pitch, yaw)
    return euler[euler_idx] * scale


def build_joint_positions_from_frame(
    frame_data: dict[str, dict],
    joint_names: list[str],
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    """
    从一帧的骨骼数据构建 G1 关节位置数组。
    - joint_names: 按 action 顺序的关节名列表
    - default_joint_pos: 默认关节位置
    - 返回: 目标关节位置，与 joint_names 同序
    """
    result = default_joint_pos.copy()
    for i, jname in enumerate(joint_names):
        angle = get_g1_angle_from_frame(jname, frame_data)
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
