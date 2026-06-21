# train_workflow/utils

MMD → G1 流水线的**共享库**（非入口脚本）。被 `g1_vmd_0_replay.py`、`g1_vmd_1_train.py`、`g1_vmd_2_eval.py`、UI 与 `scripts/` 引用。

G1 **机器人资产与执行器 PD** 配置在 `my_task/robots/`（`g1_29dof_o6_cfg.py`、`actuator_pd.py`），不在本目录。

---

## 目录结构

```
utils/
├── README.md
├── retarget/        # MMD → G1 重定向
│   ├── unitree_g1.py
│   └── joint_axis_map.py
├── motion/          # 动作资产 I/O、训练侧解析
│   ├── loader.py
│   ├── resolve.py
│   ├── sync.py
│   └── window.py
├── format/          # CSV / HDF5 序列化
│   ├── csv_loader.py
│   └── hdf5.py
├── ik/              # 足端 / 腿几何
│   ├── mmd_fk.py
│   ├── geometry.py
│   ├── leg_kinematics.py
│   └── ankle_ground.py
├── playback/        # replay 运行时
│   ├── targets.py
│   ├── sim_robot.py
│   ├── recorder.py
│   ├── root_z.py
│   └── cli.py
├── math/
│   └── trans_util.py
└── media/
    └── audio_util.py
```

**import 示例：**

```python
from robot_mmd.train_workflow.utils.retarget.joint_axis_map import G1_JOINT_AXIS_MAP_RAW
from robot_mmd.train_workflow.utils.format.csv_loader import FootIkConfig
from robot_mmd.train_workflow.utils.motion.resolve import resolve_dance_h5_by_name
from robot_mmd.my_task.robots.actuator_pd import apply_robot_pd_profile
```

---

## 模块一览

| 路径 | 基本功能 | 主要调用方 |
|------|----------|------------|
| `retarget/unitree_g1.py` | 肩/腿/腰/腕 MMD→G1 四元数反解与 tune | csv_loader、UI |
| `retarget/joint_axis_map.py` | G1↔MMD 关节映射表 | csv_loader、UI |
| `format/csv_loader.py` | MMD CSV → G1 关节 + Foot IK | replay、H5 编译、UI |
| `format/hdf5.py` | H5 读写；CSV 预编译 | replay、train buffer、sync |
| `motion/loader.py` | CSV/H5 → `MotionBundle`；yaml 登记 | replay |
| `motion/resolve.py` | 舞蹈名 → H5、训练 log 目录 | train、eval |
| `motion/sync.py` | VMD 扫描 → CSV/H5 | replay 启动 |
| `motion/window.py` | 窗口帧数 ↔ 秒 | train、eval |
| `playback/targets.py` | 单帧关节 + 根位姿目标 | replay |
| `playback/root_z.py` | 根 Z 补偿 | replay UI、脚本 |
| `playback/cli.py` | replay CLI | replay |
| `playback/sim_robot.py` | Isaac 关节/根写入 | replay |
| `playback/recorder.py` | Record H5 | replay UI |
| `ik/*` | 足端 FK/IK 几何 | csv_loader、playback |
| `math/trans_util.py` | 四元数（wxyz） | 全局 |
| `media/audio_util.py` | WAV 伴音 | replay、UI |

---

## 依赖关系（简图）

```
入口 g1_vmd_* / scripts / UI
        │
   motion/*
        │
 format/csv_loader  ◄── retarget/*
        │
 format/hdf5 ── playback/*
        │
   ik/* + math/trans_util + media/audio_util

PD / USD 机器人配置 ──► my_task/robots/
```

---

## 相关目录

| 路径 | 说明 |
|------|------|
| `../g1_vmd_0_replay.py` | MMD 回放主入口 |
| `../g1_vmd_1_train.py` / `../g1_vmd_2_eval.py` | 训练 / 评估 |
| `../scripts/` | 离线 CLI |
| `../../my_task/robots/` | G1 USD spawn、执行器 PD |
| `../../my_task/motion_reference/` | RL H5 参考缓冲 |
