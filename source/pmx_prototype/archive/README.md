# 历史实验与配置归档

睦回放的当前标准单独维护在上层 `PLAYBACK.md`。后续回放标准化已将本轮性能/启动临时脚本和旧试验数据归入 `output/archive/mutsumi_playback_history_*/`；清单保留原路径、归档路径和逐文件SHA-256，当前验收样片仍在 `output/mutsumi_startup_v4/`。历史记录集中于 `MUTSUMI_PLAYBACK_HISTORY.md`。这是针对本轮实验的归档，不涉及训练日志、策略、官方参考片及其备份；下文“不搬移output”的说明对应更早一轮目录分类。

本目录用于复现历史实验，不改变当前预览、录像或训练的默认选择。

- `experiments/`：100STYLE v4 与 continuous-wrists 的固定发布脚本、旧参考库组装和原点校正脚本。通过分类路径或统一入口 `pmx.py archive <工具名>` 运行；根目录同名转发文件已移除。
- `configs/`：21 份历史 v5/v6 训练覆盖与已被替代的物理参数快照。`catalog.json` 记录旧路径、归档路径及 SHA-256。

配置使用文件已迁入上层 `configs/training/` 与 `configs/physics/`，根目录不再保留 JSON 配置副本。归档版本仍作为冻结参数保留；`catalog.json` 的 `source_path` 指向分类后的使用文件，`legacy_path` 是历史名称，`archive_path` 与 SHA-256 继续指向冻结快照。测试验证使用文件与归档及哈希一致，发布流程一并复制所请求配置及其归档和索引。

物理版本、默认状态和模型限制见 `configs/physics/secondary_profiles_catalog.json`（相对 PMX 根目录）。索引的 `path` / `snapshot` 已更新到分类路径，`archive_path` 保留历史快照位置。当前默认与认可的参数内容不变。

以下历史对照开关仍保留在运行模块内，不拆成第二套物理实现：

| 选项 | 保留原因 |
| --- | --- |
| `--legacy-secondary-physics` | 历史对照；批录也使用它避免再次覆盖已合并的默认参数 |
| `--quality legacy` | 第一版 PhysX 质量的历史比较 |
| `--legacy-compute` | 优化前后计算、姿态与性能对比 |

这些分支仍有调用和验收价值。删除它们会改变旧命令或批录行为，本轮只登记归档背景，不删除运行代码。既有 `output/`、`logs/` 和录像不做搬移或重新打包；新快照使用补齐依赖后的复制流程。
