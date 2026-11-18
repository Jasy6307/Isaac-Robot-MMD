# tools

发布、数据和外观辅助工具。

角色预览统一实现于 `dance_preview.py`，`preview_dance.py` 和 `preview_mutsumi.py` 为兼容入口。公共函数不再由通用入口反向依赖角色专用脚本；原参数、舞蹈别名、自定义 `--config`、输出目录默认值及 dry-run JSON 保持。睦专用入口默认写 `output/mutsumi_playback/`，通用入口写 `output/dance_playback/`。共享外观、地面、角色资产、物理配置和 worker 行为不在本轮调整范围内；`source_bundle.py` 可继续静态收集预览实现与配置。

文件原名到分类路径的映射见上层 `module_locations.json`；公共导入和资源定位由上层 `source_layout.py` 管理。可运行工具通过 `python source/pmx_prototype/pmx.py list` 查看。
