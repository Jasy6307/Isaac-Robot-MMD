"""
G1 关节 -> MMD 骨骼的紧凑轴映射（轴索引 0/1/2 对应 x/y/z，见 csv_motion_loader._AXIS_INDEX_TO_VEC）。

每项为 (MMD 骨骼名或 [肩, 腕] 列表, 轴索引, 缩放系数)。

约定：
- 肩部 (左/右 shoulder_*) 走 ``retarget_unitreeG1.compute_shoulder_angles``：
  对组合 q_肩*q_腕 做 MMD->G1 基变换，再按 G1 链 (pitch_Y, roll_X, yaw_Z) intrinsic YXZ 反解。
  此处 axis_idx 选择三元组的位序: **0=pitch, 1=roll, 2=yaw**；scale 仅作 ±1 sign。
- 腿部：
  - hip (左/右 hip_*) 走 ``retarget_unitreeG1.compute_hip_angles``，axis_idx: **0=pitch, 1=roll, 2=yaw**；
  - ankle (左/右 ankle_*) 走 ``retarget_unitreeG1.compute_ankle_angles``，axis_idx: **0=pitch, 1=roll**。
- 腰部 (waist_*) 走 ``retarget_unitreeG1.compute_waist_angles``（外旋 szxy + 物理轴→语义重排）：
  axis_idx: **0=pitch, 1=roll, 2=yaw**（与肩/髋一致）；scale 仅作 ±1 微调。
  两骨骼在合成 **q_上半身·q_上半身2** 之前，可按 ``MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE`` 分别对该骨四元数取**共轭（逆旋转）**，便于与 UI 双按钮联动试方向。
- 肘 (左/右 elbow) 走 ``retarget_unitreeG1.compute_elbow_angle``：取前臂长轴在肘骨旋转
  前后的夹角作为铰链弯曲角（自动剔除被烘焙进肘骨的前臂自转 pronation）。
  axis_idx 在此分支不再使用；scale 仅作 ±1 sign（两肘默认 -1，均朝负向弯）。
- 腕 (左/右 wrist) 走 ``retarget_unitreeG1.compute_wrist_angles``（与肩同架构：
  MMD->G1 基变换 + YXZ 反解）；axis_idx: 0=pitch, 1=roll, 2=yaw；scale 仅作 ±1 sign。
- 其余单骨骼 (膝等) 仍按 Swing-Twist 单轴提取。
"""

from __future__ import annotations

AxisMapRawEntry = tuple[str | list[str], int, float]

# CSV 根四元数 → world 后 ``quat_to_euler_xyz`` 得 (roll,pitch,yaw)，按下行选分量×scale 再拼回四元数。
# 三行 root R/P/Y：axis_idx 选源分量 0/1/2，scale 为符号/增益。
MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT: tuple[int, int, int] = (0, 1, 2)
MMD_ROOT_QUAT_RPY_SCALE_DEFAULT: tuple[float, float, float] = (1.0, 1.0, -1.0)

# 腰链 [上半身, 上半身2]：在 q_first * q_second 之前是否对该骨骼局部四元数取共轭（逆旋转）。
# True = 该骨先取逆再参与相乘；映射 UI 中「Waist」标题下两枚按钮可运行时覆盖，Reset 时回到此处。
MMD_WAIST_UPPER_PAIR_QUAT_CONJUGATE: tuple[bool, bool] = (False, False)

# G1 关节（紧凑写法） -> (MMD 骨骼或 [肩, 腕] 列表, 轴索引(0/1/2), 缩放系数)
G1_JOINT_AXIS_MAP_RAW: dict[str, AxisMapRawEntry] = {
    # 手臂：肩+腕 组合 -> retarget_unitreeG1.compute_shoulder_angles -> (pitch, roll, yaw)
    # axis_idx: 0=pitch, 1=roll, 2=yaw；scale 只承担符号
    "left_shoulder_pitch_joint": (["左肩", "左腕"], 0, 1.0),
    "left_shoulder_roll_joint": (["左肩", "左腕"], 1, 1.0),
    "left_shoulder_yaw_joint": (["左肩", "左腕"], 2, 1.0),
    "left_elbow_joint": ("左ひじ", 1, -1.0),
    "left_wrist_pitch_joint": ("左手首", 0, -1.0),
    "left_wrist_roll_joint": ("左手首", 1, -1.0),
    "left_wrist_yaw_joint": ("左手首", 2, 1.0),
    "right_shoulder_pitch_joint": (["右肩", "右腕"], 0, 1.0),
    "right_shoulder_roll_joint": (["右肩", "右腕"], 1, 1.0),
    "right_shoulder_yaw_joint": (["右肩", "右腕"], 2, 1.0),
    "right_elbow_joint": ("右ひじ", 1, -1.0),
    "right_wrist_pitch_joint": ("右手首", 0, -1.0),
    "right_wrist_roll_joint": ("右手首", 1, -1.0),
    "right_wrist_yaw_joint": ("右手首", 2, 1.0),
    # 腿部：hip -> (pitch, roll, yaw), ankle -> (pitch, roll)
    "left_hip_pitch_joint": ("左足", 0, 1.0),
    "left_hip_roll_joint": ("左足", 1, 1.0),
    "left_hip_yaw_joint": ("左足", 2, 1.0),
    "left_knee_joint": ("左ひざ", 0, -1.0),
    "left_ankle_pitch_joint": ("左足首", 0, 1.0),
    "left_ankle_roll_joint": ("左足首", 1, 1.0),

    "right_hip_pitch_joint": ("右足", 0, 1.0),
    "right_hip_roll_joint": ("右足", 1, 1.0),
    "right_hip_yaw_joint": ("右足", 2, 1.0),
    "right_knee_joint": ("右ひざ", 0, -1.0),
    "right_ankle_pitch_joint": ("右足首", 0, 1.0),
    "right_ankle_roll_joint": ("右足首", 1, 1.0),

    # 躯干：上半身+上半身2 -> retarget_unitreeG1.compute_waist_angles -> (pitch, roll, yaw)
    "waist_pitch_joint": (["上半身", "上半身2"], 0, 1.0),
    "waist_roll_joint": (["上半身", "上半身2"], 1, 1.0),
    "waist_yaw_joint": (["上半身", "上半身2"], 2, 1.0),
    # O6 Hand runtime joint names (lh_/rh_) — single-axis swing-twist; scale sign tuned for MMD curl.
    "lh_thumb_cmc_yaw": ("左親指０", 2, -1.0),
    "lh_thumb_cmc_pitch": ("左親指１", 2, -1.0),
    "lh_thumb_ip": ("左親指２", 2, -1.0),
    "lh_index_mcp_pitch": ("左人指１", 2, -1.0),
    "lh_index_dip": ("左人指２", 2, -1.0),
    "lh_middle_mcp_pitch": ("左中指１", 2, -1.0),
    "lh_middle_dip": ("左中指２", 2, -1.0),
    "lh_ring_mcp_pitch": ("左薬指１", 2, -1.0),
    "lh_ring_dip": ("左薬指２", 2, -1.0),
    "lh_pinky_mcp_pitch": ("左小指１", 2, -1.0),
    "lh_pinky_dip": ("左小指２", 2, -1.0),
    "rh_thumb_cmc_yaw": ("右親指０", 2, 1.0),
    "rh_thumb_cmc_pitch": ("右親指１", 2, 1.0),
    "rh_thumb_ip": ("右親指２", 2, 1.0),
    "rh_index_mcp_pitch": ("右人指１", 2, 1.0),
    "rh_index_dip": ("右人指２", 2, 1.0),
    "rh_middle_mcp_pitch": ("右中指１", 2, 1.0),
    "rh_middle_dip": ("右中指２", 2, 1.0),
    "rh_ring_mcp_pitch": ("右薬指１", 2, 1.0),
    "rh_ring_dip": ("右薬指２", 2, 1.0),
    "rh_pinky_mcp_pitch": ("右小指１", 2, 1.0),
    "rh_pinky_dip": ("右小指２", 2, 1.0),
}
