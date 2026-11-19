# G1 命令行工具

本目录仅放可执行工具。G1 测试已迁入 [../tests/](../tests/README.md)，日常通过根目录 `run_tests.py` 运行。

| 工作 | 推荐入口 |
| --- | --- |
| 已保存遥测的跳跃/转向离线诊断 | `g1_diagnostics.py jump decompose/phases/reference-change/compare/summary`、`turn analyze/compare` |
| 修复脚底 IK、固定根避碰平滑、硬验收 | `g1_ground_motion.py prepare/solve/finalize-ik/smooth/audit` |
| 从 VMD 生成并验收 G1 训练参考 H5 | `sim_main.py --mode g1-prepare-motion` |
| 旧 Cartesian 避碰/平滑复现 | `g1_ground_motion.py legacy-refine --method fidelity/smooth` |
| 可见网格 FCL 复查 | `audit_full_meshes.py --pair-scope hands/all` |
| 认可动作回归 | `regress_ground_motion.py` |
| 原三步工作流和 VMD 流水线 | 上级 `g1_vmd_0_replay.py`、`g1_vmd_1_train.py`、`g1_vmd_2_eval.py`、`g1_vmd_pipeline.py` |

保留的旧命令兼容入口：

- `audit_hand_meshes.py` → 同一网格检查实现，默认 `hands`，保持旧手部碰撞对和 head 包含检查范围。VMD 流水线现已显式调用 `audit_full_meshes.py --pair-scope hands`。
- `refine_collision_fidelity.py`、`smooth_collision_segment.py` → `utils/grounding/legacy_refinement.py` 的两个数值参数模式，保持原参数、文件输出、续算与窗口选项。实际优化算法不再复制两份。

旧优化模式保留历史的仅脚底下界约束，不能冒称已包含新版源支撑/腾空和左右脚高度上下界验收。新修复 IK 工作使用 `g1_ground_motion.py smooth`，范围见 [正式链路说明](../utils/grounding/README.md)。

第三轮将 `smooth_h5_full_legs.py`、`smooth_h5_leg_window.py`、`repair_h5_leg_spikes.py` 收敛为 `utils/motion/legacy_leg_full.py`、`legacy_leg_local.py`，旧入口均保留；也可用 `g1_ground_motion.py legacy-legs full/window/spikes`。三种优化策略、预算、采样位置及硬验收分别保留，共享 H5 读取、脚底指标和保护其他通道的导出。它们继续属于历史方法，不代表新版固定根 grounding 验收。

七个旧诊断入口保留参数及可导入函数，业务实现收敛为 `utils/jump_reports.py`、`jump_comparison.py`、`turn_analysis.py`。原 `analyze_jump_diagnostics.py` 和 `analyze_auto_jump.py` 已是薄入口，不重复计数。

`decompose_jump_tracking.py`、`mark_jump_phases.py`、`analyze_turn.py` 默认将新报告另存 `output/g1_diagnostics/<任务>/<时间戳>/`，可用 `--output-dir` 指定目录。已有 JSON/图片拒绝覆盖；显式指定原遥测目录仍可生成尚不存在的报告。其他显式输出路径也拒绝覆盖已有报告。跳跃分解/阶段工具保持 stdout JSON，输出目录提示写入 stderr。转向比较可用 `--control-diagnostics`、`--treatment-diagnostics` 指向单独保存的分析目录。

全段旧平滑拒绝隐式覆盖已有 solver state/report；显式 `--resume_npz` 仍支持原续算/`--export_state` 语义。成品 H5、媒体和历史输出不做搬移。第三轮整理前源码与 SHA-256 清单见 [压缩快照](../../archive/script_round3_20261117/README.md)。

## 生成 G1 训练参考 H5

在项目的 `isaac61` PowerShell 中运行。

```powershell
python sim_main.py --mode g1-prepare-motion --vmd "media\dance\IRIS_OUT\IRIS_OUT.vmd" --no-video
```

默认从第 0 帧处理前 90 秒，源动作更短时用实际长度。`--seconds` 可指定时长。输出默认另存到 `output/dance_auto` 下的带时间戳目录，或通过 `--output-dir` 指定新目录。

验收后的训练输入是 `deliverables/arm_stabilized.h5`。把 `result.json` 中的 `training_h5` 路径传给 `python sim_main.py --mode g1-train` 的 `--motion_h5`。同时指定 `--dance IRIS_OUT` 时，舞蹈名用于归档训练日志，参考轨迹取显式指定的 H5。

默认还会生成三列复核录像。需要录像时去掉 `--no-video`，并用 `--accept-eula` 确认 Isaac Sim 许可，或沿用已配置的 `OMNI_KIT_ACCEPT_EULA=YES`。`--audio` 可指定音乐，默认查找同目录同名音频，无匹配时录制静音视频。

- 从原始 VMD 烘焙爱音动作。源足 IK/FK 由足 IK 关键帧数量选择，IK 模式尊重源开关。`--leg-mode` 可显式选择。G1 使用当前 O6 USD、同侧腿蒙皮鞋底、骨盆坐标变换和贴地 IK。确定支撑时的鞋底范围为 −0.3 至 3 mm，真正腾空按源动作复核。
- 左列从当前贴地 IK 计算避碰基线，中列在同一基线上做源手腕和掌心引导的双臂平滑，右列用爱音 PMX 原动作及 PhysX 次级物理。爱音从中性姿态预热再进入首帧，正式缓存恢复原 VMD 时间码。录制沿用共享的角色外观、地面和相机。
- `--retries` 默认 3。碰撞复查发现漏检配对时加可见网格凸包约束，难解区间用固定序列的姿态偏置重算。双臂阶段会根据原动作的速度与高频信号识别异常，在局部改用渐入渐出的高斯初值，按抽搐评分与碰撞距离筛选，再用原约束重解。重试与筛选均由代码完成。
- 验收检查关节角度、速度、mimic、根通道、接触/腾空、双臂超阈值帧、逐帧和四分之一帧处的 0.5 mm 所选包络间距、FCL 可见网格及训练 H5 读取格式。该 G1 资产已知的机械 CAD 接触会记录。检查范围是离散运动学采样。
- `result.json` 分别记录 `training_h5_ready` 和 `video_state`。动作验收通过后先发布 H5，再录制视频。录像失败以非零状态退出并报告原因，已发布的 H5 仍可传给训练入口。动作验收失败时，`reports/` 保留各次检查、选用初值和违规片段，`logs/` 保存进程日志。
