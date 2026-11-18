# 第二轮整理前快照

`catalog.json` 登记 9 个实验控制脚本与 14 个诊断脚本整理前的原路径、快照路径、SHA-256 和行数。
这些文件逐字节保留，用于核对历史算法、参数、验收门槛和复现来源；不是日常工具入口。
请勿直接执行这些快照：其相对目录深度已改变，原始输出路径也可能已有成品。

运行入口见 [`experiments/README.md`](../../../experiments/README.md) 和
[`diagnostics/README.md`](../../../diagnostics/README.md)，旧 `training/`、`diagnostics/` 文件名仍兼容。
本轮未复制或修改在运行的 balanced 控制器、训练核心及专用 tracking probe。
