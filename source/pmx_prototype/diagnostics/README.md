# 诊断与验收工具

这里保存离线分析、物理检查、策略评估和角色验收实现。直接运行本目录脚本，或通过 `python source/pmx_prototype/pmx.py diagnostics <工具名> [参数]` 调用。根目录旧转发文件已移除，工具参数与退出码保留。

| 类别 | 工具 |
| --- | --- |
| 发裙与动作保真 | `analyze_mutsumi_cloth_cache.py`、`analyze_mutsumi_skirt_jitter.py`、`audit_mutsumi_reference_motion.py` |
| 步态与参考动作 | `audit_anon_gait.py`、`audit_gait_controller.py`、`audit_running_arms.py`、`measure_contiguous_variation.py`、`measure_gait_slip.py`、`find_mocap_cycles.py` |
| 物理与实时链路 | `check_physx_quality.py`、`check_realtime.py`、`check_live_music.py` |
| 完整角色与模仿环境 | `validate_physical_character.py`、`validate_full_motion.py`、`validate_imitation_heading.py`、`validate_imitation_idle.py` |
| 诊断视频合成 | `compose_gait_diagnostic.py` |

不同工具仍需原有资产、依赖和运行环境：`check_live_music.py` 会访问音频设备；角色验收会启动 Isaac Sim。目录迁移不改变这些要求，也不意味着全部工具可以作为普通 Python 单元测试运行。

项目根目录与同级公共模块通过 `source_layout.py` 定位。修改工具时维护这里的实现，不在根目录另建同名入口。训练快照与发布包由 `source_bundle.py` 补齐本地导入及兼容入口对应的实现文件。
