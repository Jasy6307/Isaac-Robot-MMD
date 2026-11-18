# Locomotion 实验控制

本目录统一登记历史实验、共享 JSON/子进程记账，并存放尚有复现价值的工作流。
`catalog.json` 是入口清单，`workflows/` 是实现；训练环境、采样器与验收器继续留在原正式模块。

- 9 个非运行中的控制脚本保留原 `training/*.py` 文件名，改为兼容入口。
- `continue_sampling_pivot` 与 `run_transition_retention` 共用同一工作流，后者使用独立模块实例配置，避免污染前者。
- 其余 7 个控制脚本迁入 `workflows/`，保留各自训练预算、恢复身份检查、停止条件及验收逻辑。
- `jobs.py` 合并 JSON 读写、串行任务启动与状态记录；特殊的暂停、限时、OOM 回退与并行度比较仍保留在对应工作流。
- 原模块导入继续返回同一实现对象，兼容恢复脚本对 `run_split_scratch` 全局配置的修改；启动器仍以原文件名建立子进程，兼容限时控制器的进程身份检查。

正在运行的 `training/run_balanced_scratch.py` 和它随后调用的
`diagnostics/probe_low_speed_tracking.py` 原样保留。本轮不调整训练核心、配置、验收器或该实验的依赖；清单将其标记为 `active-preserved`，统一启动器拒绝重复启动。

```powershell
python source/pmx_prototype/pmx.py experiments list
python source/pmx_prototype/pmx.py experiments run run_transition_retention --dry-run
```

去掉 `--dry-run` 会真实执行历史控制器，可能继续训练、暂停历史进程或写入历史实验目录；必须先核对清单、预算与运行状态。这里保留历史设置，不自动重命名产物或绕过已有重复运行检查。

整理前 23 个控制/诊断脚本的逐字节快照及 SHA-256 清单位于
[`archive/experiments/locomotion_20261116_17`](../archive/experiments/locomotion_20261116_17/README.md)。快照用于审阅和核对历史算法，运行复现应走本目录登记的入口，不直接运行改变了目录深度的快照。

低速、响应、转向等诊断见 [`diagnostics/README.md`](../diagnostics/README.md)。
