#!/usr/bin/env python3
"""
生成 MMD->G1 关节映射测试用 CSV 文件。

仅上半身骨骼，每 100 帧动一个关节（从 0° 到 90°）：
  0-99:   右ひじ (right_elbow)
  100-199: 左ひじ (left_elbow)
  200-299: 右肩 (right_shoulder_pitch)
  300-399: 左肩 (left_shoulder_pitch)
  400-499: 右腕 (right_shoulder_roll)
  500-599: 左腕 (left_shoulder_roll)
  600-699: 右手首 (right_wrist_pitch)
  700-799: 左手首 (left_wrist_pitch)
  800-899: 上半身 (waist_pitch)
  900-999: 首 (waist_yaw)

播放时观察 G1 机器人哪个关节在动，即可验证 MMD_TO_G1_MAPPING 是否正确。
"""
import csv
import math
import os

# 10 个上半身 MMD 骨骼，顺序对应 0-99, 100-199, ... 帧块
# (bone_name, euler_axis)  euler: 0=roll(X), 1=pitch(Y), 2=yaw(Z)
TEST_BONES = [
    ("右ひじ", 1),   # pitch -> 绕 Y
    ("左ひじ", 1),
    ("右肩", 2),     # yaw -> 绕 Z
    ("左肩", 2),
    ("右腕", 2),
    ("左腕", 2),
    ("右手首", 2),
    ("左手首", 2),
    ("上半身", 0),   # roll -> 绕 X
    ("首", 2),
]


def quat_from_angle_axis(angle_rad: float, axis: int) -> tuple[float, float, float, float]:
    """根据绕轴旋转角度生成四元数 (x, y, z, w)。axis: 0=X, 1=Y, 2=Z"""
    half = angle_rad / 2
    s, c = math.sin(half), math.cos(half)
    if axis == 0:
        return (s, 0.0, 0.0, c)
    if axis == 1:
        return (0.0, s, 0.0, c)
    return (0.0, 0.0, s, c)


def generate_csv(output_path: str, num_frames: int = 1000, frames_per_joint: int = 100):
    """生成测试 CSV。"""
    rows = []
    for frame in range(num_frames):
        block_idx = frame // frames_per_joint
        frame_in_block = frame % frames_per_joint
        t = frame_in_block / max(1, frames_per_joint - 1)  # 0 到 1
        angle_rad = t * (math.pi / 2)  # 0 到 90 度

        for bone_idx, (bone_name, euler_axis) in enumerate(TEST_BONES):
            if bone_idx == block_idx:
                qx, qy, qz, qw = quat_from_angle_axis(angle_rad, euler_axis)
            else:
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # identity

            rows.append({
                "frame": frame,
                "bone": bone_name,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "pos_z": 0.0,
                "quat_x": round(qx, 6),
                "quat_y": round(qy, 6),
                "quat_z": round(qz, 6),
                "quat_w": round(qw, 6),
            })

    fieldnames = ["frame", "bone", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成: {output_path}")
    print(f"  帧数: {num_frames}, 每块 {frames_per_joint} 帧, 共 {len(rows)} 行")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(script_dir, "..", "media")
    os.makedirs(media_dir, exist_ok=True)
    output_path = os.path.join(media_dir, "joint_mapping_test.csv")
    generate_csv(output_path)
