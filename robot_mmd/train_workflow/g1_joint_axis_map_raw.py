"""
G1 关节 -> MMD 骨骼的紧凑轴映射（轴索引 0/1/2 对应 x/y/z，见 csv_motion_loader._AXIS_INDEX_TO_VEC）。

每项为 (MMD 骨骼名或 [肩, 腕] 列表, 轴索引, 缩放系数)。
"""

from __future__ import annotations

AxisMapRawEntry = tuple[str | list[str], int, float]

# G1 关节（紧凑写法） -> (MMD 骨骼或 [肩, 腕] 列表, 轴索引(0/1/2), 缩放系数)
G1_JOINT_AXIS_MAP_RAW: dict[str, AxisMapRawEntry] = {
    # 手臂（肩部：先组合 肩+腕 四元数）
    "left_shoulder_pitch_joint": (["左肩", "左腕"], 1, -1.0),
    "left_shoulder_roll_joint": (["左肩", "左腕"], 2, 1.0),
    "left_shoulder_yaw_joint": (["左肩", "左腕"], 0, 1.0),
    "left_elbow_joint": ("左ひじ", 1, -1.0),
    "left_wrist_pitch_joint": ("左手首", 0, 1.0),
    "left_wrist_roll_joint": ("左手首", 1, 1.0),
    "left_wrist_yaw_joint": ("左手首", 2, 1.0),
    "right_shoulder_pitch_joint": (["右肩", "右腕"], 1, 1.0),
    "right_shoulder_roll_joint": (["右肩", "右腕"], 2, 1.0),
    "right_shoulder_yaw_joint": (["右肩", "右腕"], 0, -1.0),
    "right_elbow_joint": ("右ひじ", 1, 1.0),
    "right_wrist_pitch_joint": ("右手首", 0, 1.0),
    "right_wrist_roll_joint": ("右手首", 1, 1.0),
    "right_wrist_yaw_joint": ("右手首", 2, 1.0),
    # 腿部
    "left_hip_pitch_joint": ("左足", 0, -1.0),
    "left_hip_roll_joint": ("左足", 2, -1.0),
    "left_hip_yaw_joint": ("左足", 1, -1.0),
    "left_knee_joint": ("左ひざ", 0, -1.0),
    "left_ankle_pitch_joint": ("左足首", 1, 1.0),
    "left_ankle_roll_joint": ("左足首", 0, 1.0),

    "right_hip_pitch_joint": ("右足", 0, -1.0),
    "right_hip_roll_joint": ("右足", 2, 1.0),
    "right_hip_yaw_joint": ("右足", 1, 1.0),
    "right_knee_joint": ("右ひざ", 0, -1.0),
    "right_ankle_pitch_joint": ("右足首", 1, 1.0),
    "right_ankle_roll_joint": ("右足首", 0, 1.0),
    
    # 躯干
    "waist_pitch_joint": (["上半身", "上半身2"], 1, 1.0),
    "waist_roll_joint": (["上半身", "上半身2"], 2, 1.0),
    "waist_yaw_joint": (["上半身", "上半身2"], 0, 1.0),
}
