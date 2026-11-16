# 配置分类

所有路径相对于 `source/pmx_prototype/`。根目录只保留布局索引 `module_locations.json`；运行参数文件全部放在本目录。

| 目录 | 用途 |
| --- | --- |
| `training/amp/` | AMP 训练与评估覆盖 |
| `training/continuous/` | 连续指令、相位和边界评估覆盖 |
| `training/v5/`、`training/v6/` | 历史版本训练/评估覆盖 |
| `training/clearance/` | 空间间隙学习覆盖 |
| `physics/secondary_default.json` | 当前共享物理默认；内容与搬移前一致 |
| `physics/secondary_profiles_catalog.json` | 物理版本、状态和路径索引 |
| `physics/snapshots/` | 已认可或曾作为默认的版本快照 |
| `physics/presets/` | 历史 Anon 发裙参数预设 |
| `physics/models/` | 模型专属参数 |

例如：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --config-override source/pmx_prototype/configs/training/amp/anon_amp_forward.json --help
python source/pmx_prototype/physx_dance.py --secondary-profile source/pmx_prototype/configs/physics/secondary_default.json --help
```

代码通过 `resolve_source()` 定位内置配置。`config_path()` 兼容旧项目根目录配置路径：文件存在时优先使用原文件，只有已登记且不存在的旧项目路径才重定向，不替换自定义配置或任意同名文件。预览、录像、训练及 worker 的配置选项接入相同规则；无参数启动仍加载同一默认内容。

历史冻结副本继续放在 `archive/configs/`，由 `archive/catalog.json` 记录其来源和哈希。训练日志、策略发布包及 `output/` 中既有 JSON 不移动、不改写。
