"""
CSV 骨骼动作加载器 - 从 galaxias_bones.csv 加载并按帧提供数据
支持帧间插值，以及 MMD 骨骼到 G1 关节的映射
"""
import bisect
import csv
import math
from pathlib import Path
from typing import Iterator

import numpy as np


def _quat_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """四元数转欧拉角 (XYZ 顺序)，返回 (roll, pitch, yaw) 弧度"""
    # 归一化
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-10:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    # XYZ 欧拉角
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


# MMD 骨骼 -> G1 关节映射：(G1 关节名, 从四元数提取的欧拉分量, 缩放系数)
# 欧拉分量: 'roll'=0, 'pitch'=1, 'yaw'=2
# MMD 坐标系与 G1 可能不同，缩放系数用于粗略对齐
MMD_TO_G1_MAPPING: dict[str, tuple[str, int, float]] = {
    # 腿部
    "右ひざ": ("right_knee_joint", 1, 1.0),  # 右膝 pitch
    "左ひざ": ("left_knee_joint", 1, 1.0),
    "右足": ("right_hip_pitch_joint", 1, 0.8),
    "左足": ("left_hip_pitch_joint", 1, 0.8),
    "右足首": ("right_ankle_pitch_joint", 1, 0.6),
    "左足首": ("left_ankle_pitch_joint", 1, 0.6),
    # 手臂
    "右ひじ": ("right_elbow_joint", 1, 1.0),
    "左ひじ": ("left_elbow_joint", 1, 1.0),
    "右肩": ("right_shoulder_pitch_joint", 2, 0.6),
    "左肩": ("left_shoulder_pitch_joint", 2, -0.6),
    "右腕": ("right_shoulder_roll_joint", 2, 0.5),
    "左腕": ("left_shoulder_roll_joint", 2, -0.5),
    "右手首": ("right_wrist_pitch_joint", 2, 0.5),
    "左手首": ("left_wrist_pitch_joint", 2, -0.5),
    # 躯干
    "上半身": ("waist_pitch_joint", 0, 0.4),
    "下半身": ("right_hip_roll_joint", 0, 0.3),  # 下半身影响髋部
    "首": ("waist_yaw_joint", 2, 0.2),
}


def load_csv_motion(csv_path: str) -> dict[int, dict[str, dict]]:
    """
    加载 CSV 骨骼数据，返回 {frame_idx: {bone_name: {pos, quat}}}
    """
    frames: dict[int, dict[str, dict]] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            bone = row["bone"]
            pos = (
                float(row["pos_x"]),
                float(row["pos_y"]),
                float(row["pos_z"]),
            )
            quat = (
                float(row["quat_x"]),
                float(row["quat_y"]),
                float(row["quat_z"]),
                float(row["quat_w"]),
            )
            if frame not in frames:
                frames[frame] = {}
            frames[frame][bone] = {"pos": pos, "quat": quat}
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
        d = frames[frame][bone]
        return {"pos": d["pos"], "quat": d["quat"]}

    # bisect 二分查找前后关键帧
    idx = bisect.bisect_right(bone_frame_list, frame)
    prev_f = bone_frame_list[idx - 1] if idx > 0 else None
    next_f = bone_frame_list[idx] if idx < len(bone_frame_list) else None

    if prev_f is None and next_f is None:
        return None
    if prev_f is None:
        return frames[next_f][bone]
    if next_f is None:
        return frames[prev_f][bone]
    if prev_f == next_f:
        return frames[prev_f][bone]

    # 线性插值
    d0, d1 = frames[prev_f][bone], frames[next_f][bone]
    t = (frame - prev_f) / (next_f - prev_f)
    pos = tuple((1 - t) * a + t * b for a, b in zip(d0["pos"], d1["pos"]))
    # 四元数球面插值 (slerp) 简化版：线性插值后归一化
    q0, q1 = np.array(d0["quat"]), np.array(d1["quat"])
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - t) * q0 + t * q1
    q = q / (np.linalg.norm(q) + 1e-8)
    quat = tuple(float(x) for x in q)
    return {"pos": pos, "quat": quat}


def mmd_bone_to_g1_angle(bone: str, quat: tuple[float, float, float, float]) -> float | None:
    """
    将 MMD 骨骼四元数转换为 G1 关节角度（弧度）。
    若骨骼不在映射中则返回 None。
    """
    if bone not in MMD_TO_G1_MAPPING:
        return None
    joint_name, euler_idx, scale = MMD_TO_G1_MAPPING[bone]
    roll, pitch, yaw = _quat_to_euler(quat[0], quat[1], quat[2], quat[3])
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
        for mmd_bone, (g1_joint, _euler_idx, _scale) in MMD_TO_G1_MAPPING.items():
            if g1_joint != jname:
                continue
            if mmd_bone not in frame_data:
                break
            quat = frame_data[mmd_bone]["quat"]
            angle = mmd_bone_to_g1_angle(mmd_bone, quat)
            if angle is not None:
                result[i] = default_joint_pos[i] + angle
            break
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
