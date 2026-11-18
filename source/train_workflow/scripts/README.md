# G1 命令行工具

本目录仅放可执行工具。G1 测试已迁入 [../tests/](../tests/README.md)，日常通过根目录 `run_tests.py` 运行。

| 工作 | 推荐入口 |
| --- | --- |
| 已保存遥测的跳跃/转向离线诊断 | `g1_diagnostics.py jump decompose/phases/reference-change/compare/summary`、`turn analyze/compare` |
| 修复脚底 IK、固定根避碰平滑、硬验收 | `g1_ground_motion.py prepare/solve/finalize-ik/smooth/audit` |
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
