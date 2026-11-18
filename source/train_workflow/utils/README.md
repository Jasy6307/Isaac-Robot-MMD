# train_workflow/utils

MMD → G1 流水线的**共享库**（非入口脚本）。被 `g1_vmd_0_replay.py`、`g1_vmd_1_train.py`、`g1_vmd_2_eval.py`、UI 与 `scripts/` 引用。

G1 **机器人资产与执行器 PD** 配置在 `my_task/robots/`（`g1_29dof_o6_cfg.py`、`actuator_pd.py`），不在本目录。

源鞋底高度与骨盆修复、固定根避碰平滑已整理为 [grounding/](grounding/README.md)，入口为 `scripts/g1_ground_motion.py`。它显式运行并另存输出；原有转换入口默认不变。

保存遥测的离线分析由 `jump_reports.py`、`jump_comparison.py`、`turn_analysis.py` 实现，公用读取及接触窗口统计位于 `jump_analysis.py`，分类入口为 `scripts/g1_diagnostics.py`。三种历史腿部优化在 `motion/legacy_leg_full.py`、`legacy_leg_local.py`，共用数据读写与指标；优化器和拒绝导出条件分别保留，不能作为新版 grounding 的替代验收。

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
from source.train_workflow.utils.retarget.joint_axis_map import G1_JOINT_AXIS_MAP_RAW
from source.train_workflow.utils.format.csv_loader import FootIkConfig
from source.train_workflow.utils.motion.resolve import resolve_dance_h5_by_name
from source.my_task.robots.actuator_pd import apply_robot_pd_profile
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
| `playback/collision_h5_job.py` | 独立进程修复、进度与取消 | replay UI |
| `motion/self_collision.py` | USD 几何包络、FK、碰撞投影 | 修复脚本 |
| `motion/collision_reference.py` | 保真碰撞修复 + 自动局部脚步尖峰修复 | 无碰撞 H5 导出 |
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

### 在 replay 中录制防穿模参考 H5

运行 `python source/train_workflow/g1_vmd_0_replay.py`，在 **G1 MMD config** 选择 CSV 舞蹈，停止播放后点击 **Record H5 without collision**。
也可以仅加载任意符合本项目 MMD CSV 格式的文件：

```powershell
python source\train_workflow\g1_vmd_0_replay.py `
  --motion_csv "media\dance\ji_le_jing_tu\ji_le_jing_tu.csv" `
  --num_envs 1 --sim_fps 60
```

录制复用当前 replay 的映射、IK、脚底补偿、Root 设置，并保存实际关节基准。若开启 `Z_offset_enable` 且存在对应 CSV，则录制当前选用的 `*_z_editted.csv`。
随后独立 CPU 子进程按以下步骤修复；窗口显示阶段和进度，**Cancel** 可取消，退出 replay 会停止自己启动的修复进程：

1. 从当前 G1 USD 读取关节与几何包络，投影自碰撞，优先保留原动作及脚端轨迹。
2. 对剩余穿模局部加力投影。按“绝对异常幅度 + 相对原动作放大”检测修复引入的脚步尖峰，不按歌曲名或固定帧号处理。
3. 仅在检测窗口内生成稳定腿部轨迹，以平滑权重接回原修复轨迹，再进行局部碰撞投影；不对全段脚步应用统一速度上限。
4. 对完整轨迹每帧和相邻帧中点进行检测，要求穿透不超过 1 mm，且不再有检测到的新增脚步尖峰。通过后才发布 H5。

输出为 CSV 同目录的 `*_without_collision.h5` 和 `*.collision.json`；重复录制自动添加 `_001` 等后缀。原始 CSV、普通 H5 和旧修正版不会被覆盖。
成功后用 **Play repaired H5** 回放最新通过检测的文件；普通 **Play H5** 仍使用原先普通 H5。
录制快照、状态和详细日志保存在 `output/collision-h5-jobs/<任务ID>/`。验证失败时保留报告及 `*.candidate.npz` 诊断数据，不发布未通过的 H5。

核心入口是 `motion.collision_reference.repair_reference_h5(source, output, robot_usd, progress=None)`；
已录制的原始 H5 也可由 `scripts/record_h5_without_collision.py --input ... --output ...` 离线修复。
自动 CSV 录制与自碰撞修正使用 `--motion_csv ... --auto_record_collision_corrected_h5 --exit_after_collision_record`。旧参数 `--auto_record_h5_without_collision` 保留为兼容别名，执行相同流程。

一键 VMD → H5 → 带音乐的 H5 回放 MP4（在项目根目录、Isaac Python 环境运行）：

```powershell
python source/train_workflow/g1_vmd_pipeline.py `
  --vmd "media/dance/IRIS_OUT/IRIS_OUT.vmd" `
  --audio "media/dance/IRIS_OUT/IRIS_OUT.wav" `
  --collision_correction
```

省略 `--collision_correction` 输出普通 H5。省略 `--audio` 自动寻找 VMD 同名音频。
默认修正 H5 为 `IRIS_OUT_collision_corrected.h5`，普通 H5 为 `IRIS_OUT_rebuilt.h5`，不加时间戳；重复生成在末尾加 `_v2`、`_v3`，整组产物使用同一版本。修正前的 CSV/H5 使用 `_raw` 标识。
CSV/原始 H5/修正 H5 位于 `media/dance`，AVI、带音乐 MP4 和 `.pipeline.json` 清单位于 `media/recordings`。
可用 `--name` 指定最终文件名前缀、`--output_dir` 和 `--video_dir` 指定目录；已有文件不会覆盖。
音乐从第 0 秒对齐；音频较短时补静音，较长时截到视频结束。需要 FFmpeg（PATH 或 `imageio-ffmpeg`，也可 `--ffmpeg` 指定）。
此入口会依次打开并自动关闭 H5 录制、H5 回放窗口；不需要手动勾选录像。
回放逐帧录制，校验视频帧数和 FPS 与最终 H5 一致；碰撞修正失败、录像失败或窗口提前关闭均中止后续步骤。
这是参考动作回放视频，不运行训练策略。
每份生成的 H5 还保存同目录 `.motion.json`，记录原始 VMD、原始舞蹈名及音乐路径。
训练验收 MP4、策略验证及检查点对比录像会自动读取该来源记录；旧产物兼容从 `media/recordings/*.pipeline.json` 按完整 H5 路径匹配配乐。`_v2`、`_v3` 或自定义名字不会影响音乐对应关系。
验收渲染场景也会保存解析到的音频路径；显式 `--audio` 优先。音乐预览按起始秒数和播放速率同步。

检测基于当前 G1 的选定非相邻身体包络（含张开手掌），不是精确三角网格或 PhysX 的全面碰撞认证，也不处理地面/背景碰撞。
极端输入可能无法自动找到可行轨迹，此时会明确失败；不会因为换了按钮就保证任意不兼容格式或任意姿态都可修复。

| 路径 | 说明 |
|------|------|
| `../g1_vmd_0_replay.py` | MMD 回放主入口 |
| `../g1_vmd_1_train.py` / `../g1_vmd_2_eval.py` | 训练 / 评估 |
| `../scripts/` | 离线 CLI |
| `../../my_task/robots/` | G1 USD spawn、执行器 PD |
| `../../my_task/motion_reference/` | RL H5 参考缓冲 |
