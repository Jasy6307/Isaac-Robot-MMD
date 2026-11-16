# 角色舞蹈回放标准

统一入口：`tools/preview_dance.py`（旧 `preview_mutsumi.py` 保持兼容）；共享设置：`configs/playback/mutsumi_deepbluetown.json`；舞蹈资产表：`configs/playback/mutsumi_dances.json`。默认仍播放 Deep Blue Town，`--dance` 可选择现有6支已验收动作的USD，不重新转换动作。各舞蹈的参数与资产解析均已检查；完整实时性能和预热效果验收仍以 Deep Blue Town 为基准。

## 运行

2K（默认，保持4:5竖版）：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --resolution 2k
```

1K：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --resolution 1k
```

选择舞蹈（以下均为2K，换成 `--resolution 1k` 即为1K）：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance deepbluetown --resolution 2k
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance girls --resolution 2k
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance ji_le_jing_tu --resolution 2k
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance IRIS_OUT --resolution 2k
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance ni_zui_zhong_yao --resolution 2k
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --dance hal_ia_yeah_oh_ahhh_oh --resolution 2k
```

`irisout`、`jilejingtu` 也可分别作为 `IRIS_OUT`、`ji_le_jing_tu` 的别名；名称不区分大小写。`--list-dances` 列出舞蹈；已退役的 Galaxias 不提供。

无实时物理的性能对照：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model mutsumi --resolution 2k --no-secondary-physics --render-fps 60
```

默认输出按运行时间及进程号分目录存放于 `output/mutsumi_playback/mutsumi_<舞蹈>/<分辨率>/`，避免同时启动或重复运行覆盖日志。`--output` 可指定输出位置；`--dry-run` 只检查资产并打印有效参数，不启动 Isaac。

爱音 Girls：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model anon --dance girls --resolution 2k
```

实验性精简裙边碰撞代理（仅爱音，默认仍为完整V1）：

```powershell
python source\pmx_prototype\tools\preview_dance.py --model anon --dance girls --resolution 1k --physics-detail reduced
```

该选项每隔一列保留裙边纵向胶囊及其关节，减少手、腿和头发的裙边碰撞覆盖；保留120Hz、身体动作、裙子和头发节点与V1参数。

实验性60Hz插值显示可在该命令后加 `--render-fps 60 --interpolate-60`：物理仍以120Hz运行，后台每秒输出30个物理姿态，前台在相邻姿态间插值，增加约一个姿态间隔的延迟；实际显示帧率和物理推进速度应分别查看，不能以显示帧率判定舞蹈与音乐实时同步。`--physics-detail sparse` 每隔三列保留裙边代理，可能增加穿模，属未定型的视觉/性能实验。音乐始终原速原音高。
它是视觉/性能试验，可能更易穿模，不会修改默认模板。2026-11-14 Girls 前30秒1K测试：完整V1 27.7 FPS、精简版29.4 FPS，仍有长帧；同动作前30秒视频与结果位于 `output/anon_girls_physics_detail_compare/`。

实时播放的 **Dance Physics Monitor** 会停靠在右下侧的 Property 面板组。`PHYSICS / ACTUAL` 按物理子步计数与墙钟时间显示实际 Hz 和目标 Hz；`DISPLAY / PRESENTED` 是画面更新率。插值模式还单列已解算姿态率，因此会明确区分 120 Hz 物理子步、30 Hz 解算姿态和 60 FPS 显示。面板同时显示实时速度、单帧工作耗时和碰撞状态。

爱音目前在此入口仅登记 Girls，其他组合会明确报错；可使用 `--model anon --list-dances` 查询。爱音使用原生模型数据构建物理，不加载睦的 rig。 音乐按原速和原音高播放，爱音预览容许估计音画偏差达0.25秒再重同步；物理持续慢于实时速度时仍会发生跳转，实际听觉效果和偏差需自行确认。 爱音的 sparse + 60Hz 插值预览默认给物理进程8个任务线程、前台4个任务线程；其他爱音预览物理进程默认16个，睦仍为4个。不改变120Hz或约束设置；实际效果受并发负载影响。

## 固定标准

| 项目 | 当前标准 |
| --- | --- |
| 物理 | 共享默认 V1（原 V10） / `hair_contact_drape_v1` / cloth / 120 Hz |
| 并行 | 独立CPU物理进程，每显示帧4个子步，最多预取一帧 |
| 显示 | 默认目标30 FPS；1K为1080×1350，2K为1440×1800 |
| 外观 | 共享批准的mmd-toon微调版；原版地面、材质及反射 |
| 启动 | 并行初始化、按USD内容校验的相机边界缓存、加载期间隐藏角色 |
| 首姿态 | 固定源动作第0帧预热2秒/240子步，之后显示角色并启动动作与音乐 |
| 动作及几何 | 既有身体动作和缩短10%的裙子，均不重建或修改 |

预热是启动设置，不写入V10参数。`--startup-settle-seconds 0` 可恢复无预热初始状态；正数设置预热秒数。重置和循环也重新预热，可能有短暂等待。无物理模式不预热。物理默认模板不再匹配预设声明时，入口会报错，需先审核新版本，避免未验收参数被默默替换。

这套预览不使用舞蹈物理缓存。后台不可见进程使用最小图形上下文，显示端仍使用完整批准外观。原低层入口 `physx_dance.py` 保留用于录制、诊断和兼容旧命令；录像、外部碰撞场景及接触验收仍使用内联物理路径，不走此标准预览的独立进程模式。

## 验收和已知边界

当前保留样片及记录：

- `output/mutsumi_startup_v4/deepbluetown_prewarm_before_after_2k.mp4`：左无预热、右两秒预热，每侧原生2K；前10秒动作，额外1.5秒首帧定格。
- `output/mutsumi_startup_v4/initial_pose_detail_before_after.mp4`：四秒首帧局部对比。
- `output/mutsumi_startup_v4/verification.json`、`final/startup_report.json`：参数哈希、动作/相机校验及启动阶段数据。

本机启动测得串行对照约48.4秒，并行首次约24.5秒，最终缓存命中约20秒；首次加载、代码变更导致Numba或渲染缓存失效、并发负载较高时可能明显更久。初始裙子对腿部胶囊代理的穿入明显减少，仍有最大约2.63mm残余；这不是网格对网格零穿模保证。已有完整舞蹈测试平均约30 FPS，但并发训练时仍有渲染长帧，不能据此承诺所有负载下严格稳定30 FPS。

旧实验数据按归档清单保留，不作为当前运行入口。默认参数、官方G1参考片及历史视频备份不因测试清理而改变。

## 运行资产与清理边界

以下是有效资产，不是临时性能测试：

- `media/mutsmi/`、`media/dance/deepbluetown/`：PMX、贴图、动作、音源和当前参考片。
- `output/mutsumi_validation_120hz_bright/deepbluetown/character.usdc`、`body.usdc`、`rig.json`：已验收的动画和物理rig。
- `output/mutsumi_pmx/character.usdc`：上述USD引用的基础模型。
- `output/mutsumi_validation_references/`、`output/mutsumi_validation_v10_1080/`：G1参考登记、缓存及旧版备份。
- `output/pmx_camera_cache/`：可校验、自动失效的当前相机缓存，保留可加快启动。

这些USD具有外部引用，搬迁不能只复制主文件。其他训练的 `logs/`、配置、策略、数据和模型不在本次清理范围。历史实验说明见 `archive/MUTSUMI_PLAYBACK_HISTORY.md`。
