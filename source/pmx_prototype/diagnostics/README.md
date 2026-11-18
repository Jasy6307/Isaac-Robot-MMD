# 诊断与验收工具

这里保存离线分析、物理检查、策略评估和角色验收实现。直接运行本目录脚本，或通过 `python source/pmx_prototype/pmx.py diagnostics <工具名> [参数]` 调用。此前根目录旧转发文件已移除；本轮兼容入口仍保留在本目录。

| 类别 | 工具 |
| --- | --- |
| 发裙与动作保真 | `analyze_mutsumi_cloth_cache.py`、`analyze_mutsumi_skirt_jitter.py`、`audit_mutsumi_reference_motion.py` |
| 步态与参考动作 | `audit_anon_gait.py`、`audit_gait_controller.py`、`audit_running_arms.py`、`measure_contiguous_variation.py`、`measure_gait_slip.py`、`find_mocap_cycles.py` |
| 物理与实时链路 | `check_physx_quality.py`、`check_realtime.py`、`check_live_music.py` |
| 完整角色与模仿环境 | `validate_physical_character.py`、`validate_full_motion.py`、`validate_imitation_heading.py`、`validate_imitation_idle.py` |
| 诊断视频合成 | `compose_gait_diagnostic.py` |

不同工具仍需原有资产、依赖和运行环境：`check_live_music.py` 会访问音频设备；角色验收会启动 Isaac Sim。这些工具不能全部作为普通 Python 单元测试运行。
项目根目录与同级公共模块通过 `source_layout.py` 定位；已有训练快照与发布链路仍由 `source_bundle.py` 收集本地依赖。

## Locomotion 诊断

睦的 USD 缓存检查现由 `character_cache.py cloth/temporal/reference` 统一管理（可通过 `pmx.py diagnostics character_cache` 调用）。`analyze_mutsumi_cloth_cache.py`、`analyze_mutsumi_skirt_jitter.py`、`audit_mutsumi_reference_motion.py` 保留原参数和完成标记；共享骨架查询，分别保留代理穿插、时间抖动和参考动作差异的指标，不混为一个评分。这三个模式仍按睦的骨架和材质部件适配，不能直接宣称支持其他角色。输出 JSON 拒绝覆盖，`--dry-run` 只显示参数，不打开 USD。原 `--frames 7011`、cloth `--stride 30` 默认值和 30Hz 时间指标保持。

14 个历史脚本归入 `analyses/` 下 7 个分类模块，原文件名保留为兼容入口。
`locomotion_tools.py` 统一参数与发现，`workspace.py` 统一输入证据、结果写入和串行推理任务。
`analyses/catalog.json` 保存原文件名、分类、默认历史数据目录与是否需要推理。

| 分类 | 子命令 | 原脚本 |
|---|---|---|
| sampling | audit / verify | audit_low_speed_sampling / verify_low_speed_fix |
| response | windows / replay | audit_response_windows / replay_acceptance_v2 |
| reference | kinematics / transition | audit_reference_kinematics / audit_transition_chain |
| planner | compare / split | compare_command_plans / compare_split_command_plans |
| tracking | margin / reverse | diagnose_tracking_margin / repeat_reverse_turn |
| acceptance | limits / smoke | audit_scratch_acceptance_limits / smoke_acceptance_v2 |
| reporting | rewards / summary | audit_low_speed_rewards / summarize_low_speed_diagnosis |

```powershell
python source/pmx_prototype/pmx.py diagnostics locomotion --help
python source/pmx_prototype/pmx.py diagnostics locomotion response windows --dry-run
```

每个子命令支持 `--input-dir`、`--output-dir` 和 `--dry-run`。
不指定输出目录时创建 `output/locomotion_diagnostics/<原脚本名>/<时间戳>/`，拒绝覆盖已有结果；历史报告与录像不变。旧脚本仍读取原默认输入，但新结果也写入此独立目录，这是本轮有意调整的输出行为。
`--report-folder` 和 `--run` 兼容输入目录；limits/replay 支持旧的 `--output <新报告文件>`。
`sampling audit --configs <json>` 可指定名称到配置文件路径的映射。

planner、tracking 和 acceptance smoke 会启动 Isaac 推理，其余分类只分析已有数据；预览不导入训练环境，也不启动进程。正式推理继续遵守项目 Isaac 启动要求。本轮回归没有启动训练或推理。

默认设置仍保留历史实验对照：例如 reference transition 使用 retention_strength 的 baseline/original，验收 limits 额外对照此前固定基线；这些跨实验来源没有被改成新的默认基线。若要复现同一报告，需要保留清单指定的数据、模型与引用动作。自定义输入只替换该任务的主数据目录，不承诺打包所有跨实验依赖。
smoke 自定义输入需含 `models.json` 和 `smoke_config.json`；默认运行保留历史模型来源。

`probe_low_speed_tracking.py` 是运行中 balanced 实验的后续依赖，暂不合并；正式 evaluator 和其他诊断入口也不属于这 14 个历史脚本。

CPU 回归检查快照完整性、控制流程与验收逻辑 AST、旧导入身份、配方隔离、采样边界、响应计算及输入/输出隔离。依赖真实历史数据的额外检查报告另存 `output/script_round2_20261117/`。
