# PMX 角色骨骼原型

已验收的睦舞蹈回放请先看 [PLAYBACK.md](PLAYBACK.md)：统一1K/2K入口及当前默认设置。

## 代码导航

第三轮预览合并见 [tools 导航](tools/README.md)：`preview_dance.py`、`preview_mutsumi.py` 两个入口共用一个实现。睦的缓存、裙摆和参考动作检查统一使用 `diagnostics/character_cache.py` 的 `cloth`、`temporal`、`reference` 子命令。源码快照见 [第三轮压缩归档](../archive/script_round3_20261117/README.md)。

实验控制见 [experiments/README.md](experiments/README.md)：9 个历史控制入口迁入工作流，继续采样与 transition 共用实现；正在运行的 balanced 实验原样保留。低速/转向等 14 个历史诊断归入 [7 类诊断模块](diagnostics/README.md)。统一发现入口为 `pmx.py experiments list` 与 `pmx.py diagnostics locomotion --help`；旧文件名继续兼容，新诊断结果另存。整理前源码及校验清单见 [历史快照](archive/experiments/locomotion_20261116_17/README.md)。

G1 / Anon 同屏比较统一由 `play_g1_anon_comparison.py` 调用 `runtime/comparison_player.py`，支持原两列、三列、四列及附加 `--smooth-h5` 的五列。`--start-seconds/--end-seconds` 选择同步片段，录像音轨按源帧时间截取；`--dry-run` 只检查路径并输出布局，不启动 Kit。

Girls 整段三列及片段五列统一走 `sim_main.py --mode compare-g1-pmx`，分别传入 `--preset girls-full` 和 `--preset girls-segment`。片段预设默认 16–21 秒，原列标签、外观和音画时间轴沿用共享播放器。

本目录同时保留当前运行链路与历史实验。先按下表定位入口；历史章节中的能力描述只对应当时阶段，当前外观和物理默认以本文开头的说明为准。

| 功能 | 入口与公共实现 |
| --- | --- |
| PMX 转换、VMD 烘焙、资产查看 | `convert_pmx.py`、`load_vmd.py`、`view_character.py` |
| 舞蹈实时预览 | `tools/preview_dance.py`（角色舞蹈标准入口），`physx_dance.py`（原生 PhysX）、`realtime_dance.py` / `realtime_xpbd.py`（XPBD） |
| 录像与缓存重录 | `record_physx_video.py`、`record_cached_secondary.py`、`record_mutsumi_references.py` |
| 统一外观 | `appearance_defaults.py` → `character_appearance.py`；`character_semantics.py`、`bangs_shadow.py`、`cel_studio.mdl` |
| 头发、裙摆物理 | `secondary_profiles.py`、`configs/physics/secondary_default.json` → `physx_secondary.py`；XPBD 实现在 `secondary_motion.py` |
| 策略训练与播放 | `train_anon_lab.py`（Isaac Lab Direct）、`train_anon_locomotion.py`（可选后端）、`play_anon_policy.py`、`inspect_anon_policy.py`；详见 `ANON_LOCOMOTION.md` |
| 单人／多人物行走验证 | 统一入口 `inspect_anon_policy.py` 默认单人；加 `--characters 4` 控制 1 人、其余 3 人随机行走，可切换控制对象。Lab Direct＋CPU，人物互碰默认关闭，保留独立发裙物理；实时预览默认 60 FPS 姿态插值，策略 30Hz、物理 120Hz；多人内部实现在 `runtime/anon_crowd_viewer.py`；详见 `ANON_LOCOMOTION.md` |
| 发裙物理档位 | 统一行走验证默认 `light`（light1），单人／多人均生效；可显式选择 `full`、`light2` 或更低预算的 `light3`。多人加 `--mixed-secondary` 时受控者默认 light、背景 light3，随控制对象自动交接；加 `--controlled-secondary-mode light2` 可调整前台。保留 120Hz、节点与碰撞覆盖。四人混合模式本机一分钟实测前台 light／light2 分别为 Solved 29.22／29.88Hz，均约 60 FPS 插值显示；对照与限制见 `ANON_LOCOMOTION.md` |
| 物理角色构建 | `physical_character.py`、`build_anon_v3.py`（25 DOF）、`build_anon_v4.py`（31 DOF）；详见 `PHYSICAL_CHARACTER.md` |
| 动作参考与处理 | `*_reference.py`、`build_*motion*.py`、`retarget_100style.py` 及平衡、平滑、审计脚本 |
| 验证 | `test_*.py` 为单元测试；`check_*.py`、`validate_*.py` 多数需要 Isaac Sim、资产或音频设备 |

`anon_*.json` 是训练/评估参数覆盖；`secondary_*.json` 是物理预设，版本与状态见 `configs/physics/secondary_profiles_catalog.json`。文件名中的旧版本号不代表可以删除：构建器仍由环境按配置选择，部分配置和脚本仍被发布打包与训练快照使用。

整理约定：先保持现有文件路径、命令行参数、导入名称和启动顺序。Isaac Sim 入口中的 `SimulationApp` 初始化与后续 `pxr` / `omni` 导入有先后要求，不能简单按普通 Python 导入排序。输出、缓存和历史录像继续写入项目既有目录。

### 分类结构与统一入口（2026-11-14）

根目录从 108 个 Python 文件减为 17 个，只保留常用转换、预览、录像、训练入口和布局/打包支持。入口命令沿用原路径；不常用脚本不再保留根目录转发文件。

| 位置 | 内容 |
| --- | --- |
| `runtime/` | 外观、发裙物理、角色构建基础、音频等共享实现 |
| `training/` | 训练环境、策略通信与 UI、AMP、模型版本与迁移 |
| `motion/` | 动作参考、重定向、参考库构建、平衡与平滑处理 |
| `diagnostics/` | 检查、评估、审计及诊断录像 |
| `tools/` | 发布打包、数据下载、外观对比和外部渲染工具 |
| `tests/` | 单元与兼容性测试 |
| `configs/` | 按用途分类的训练覆盖与物理配置，见 [配置说明](configs/README.md) |
| `archive/` | 历史实验与冻结配置 |

统一列出和运行工具（仍在项目根目录执行）：

```powershell
python source/pmx_prototype/pmx.py list
python source/pmx_prototype/pmx.py diagnostics measure_gait_slip --help
python source/pmx_prototype/pmx.py tools package_anon_policy --help
```

分类前缀可以省略，也可直接运行分类目录里的脚本。旧名称到新位置的完整映射见 `module_locations.json`；上文历史命令已改到对应分类路径。外部代码建议通过 `source.pmx_prototype.<原模块名>` 导入；独立脚本如使用原来的裸模块导入，先 `import source_layout` 初始化分类搜索路径。

共享源文件定位统一走 `resolve_source()`；源码快照由 `source_bundle.py` 解析本地依赖并保留分类目录，不会在复制时启动模拟器。v3/v4 继续复用同一个资产构建器，保留版本差异。JSON 参数已按用途迁入 `configs/`；MDL 资源保留原路径。配置参数内容不变，旧项目配置路径可由入口参数兼容解析。

统一测试入口：`python run_tests.py --suite cpu --area pmx`；需要本机角色、动作或策略文件的测试用 `--suite assets --area pmx`。依赖与报告说明见 [测试指南](../../TESTING.md)。旧的 `python -m unittest discover -s source/pmx_prototype/tests` 仍可用，但会尝试执行整个目录，不按依赖分层筛选。

## 默认角色外观：mmd-toon（cel-studio，2026-11-25）

共享 `mmd-toon` 使用已确认的 cel-studio 外观、分材质阴影配色和衣料明暗分阶。预览、策略展示与新录像沿用原参数名。地面保留场景原版材质、显示与反射。

| 设置 | 默认值 |
| --- | --- |
| look | `mmd-toon`（cel-studio，默认） |
| 环境光倍率 | `0.75`，Dome 强度 `412.5` |
| 角色材质亮度倍率 | `0.88` 对应批准的分材质亮度，脸部 `0.537`、头发 `0.511`、上衣 `0.537`、裙子 `0.529` |
| 刘海遮挡阴影 | 亮部的 `0.68`，边缘平滑 |
| 脸部塑形 | 开启；暗部 `0.77`，暖色 `(1, 0.925, 0.89)` |
| 阴影配色 | 按材质语义和贴图底色给头发、衣料和皮肤分配暗面色 |
| 衣料明暗 | 衣料最深暗面的线性亮度降低 `7%`，中间暗面降低 `3.5%` |
| 发丝描边 | 保留主发束细描边，剔除微小独立发丝的轮廓壳 |
| 地面 | 原版；材质、反射参数不改，随环境照明自然变化 |

参数统一在 `runtime/appearance_defaults.py`，由 `runtime/character_appearance.py`、`cel_studio.mdl` 和 `runtime/bangs_shadow.py` 实现。当前验收样片是 `media/recordings/DeepBlueTown_fabric_shadows_20261125/` 中爱音和睦前 30 秒三列对比的中列，左列是仅启用分材质阴影配色的版本，右列为 `lit-anime` 提亮版。

### 可用模式与入口

当前仅保留三个外观选项，预览与录像统一使用 `appearance_defaults.LOOK_CHOICES`：

| look | 用途 |
| --- | --- |
| `original` | 最初普通三维材质与照明，原版地面 |
| `soft-pbr` | 提亮低反射 PBR 与脸部柔光，原版地面 |
| `mmd-toon` | 默认 cel-studio 赛璐璐，完整参数见上表，原版地面 |

`realtime_dance.py`、`record_physx_video.py`、`physx_dance.py` 支持这三个选项。策略预览/录像 `play_anon_policy.py`、轨迹录像 `record_anon_locomotion.py` 和资产查看 `view_character.py` 默认接入共享 `DefaultCharacterLook`。其他角色批录通过原生 PhysX 入口继承默认；不会自动覆盖已生成的视频。

五个中间实验模式及专用材质代码已经移除。已有八格对比视频和原片仅作为历史产物保留，不再代表当前可选项。比较脚本 `record_look_comparison.py` 现在只录制和比较三个保留模式，写入独立的 `media/recordings/IRIS_OUT_look_comparison_current/`，不会覆盖历史对比目录。

独立录像 `record_physx_video.py` 在编码器结束、视频与报告写入后释放自身物理资源，再使用 `SimulationApp.close(skip_cleanup=True)` 快速结束批处理，避开 Windows 上可能卡住的完整 profiler/框架关闭阶段。正常退出码为 0；录制或资源释放失败返回非零。此退出方式仅用于该独立入口，不修改预览和训练的生命周期，也不提前截断编码或报告写入。

### 使用

```powershell
python source\pmx_prototype\realtime_dance.py --usd "output\iris-out-audit\IRIS_OUT_pmx.usdc" --music "media\dance\IRIS_OUT\IRIS_OUT.wav" --render-fps 60
```

省略 look 即使用精调赛璐璐；也可显式写 `--look mmd-toon`。XPBD 预览和独立录像仍支持亮度参数覆盖、`--face-shaping` 与 `--no-face-shaping`。显式选择普通原版或提亮 PBR 时，未指定的附加参数使用其基础值。

### 新角色适配与限制

共享组件支持不同角色根路径（默认 `/World/Character`，轨迹录像为 `/World/Anon/Visual`），不修改源 USD、贴图或物理参数。脸部材质识别 `face`、`顔`、`脸`；刘海识别 `前发`、`前髪`、`bang`、`fringe`。缺失时输出明确提示，保留赛璐璐与描边，不能认为投影已启用；新角色命名不同需适配这些语义。

cel-studio 保留清楚的主明暗分区，加入中间色、柔和脸部边界、哑光衣料和发束高光色带。角色部位按材质语义匹配；未识别的普通表面使用通用赛璐璐配置。刘海投影仍由实际变形几何计算。以下历史章节保留原实验背景，涉及当前外观默认时以本节为准。

睦的分片 USD 通过 `character_semantics.py` 在渲染层适配：按头部蒙皮权重划分脸部材质，按前发骨骼识别刘海，并向可见脸部网格写入遮挡权重；保留眼睛原材质。发丝描边沿用原贴图透明度。适配不修改源模型网格、动作或物理参数。

PhysX `cloth` 模式可通过物理 profile 的 `appearance.hair_skirt_contacts=true` 启用头发节点与裙子节点、纵向裙边胶囊的接触。当前默认模板已启用该开关；初始实验版保存在 `output/mutsumi_iris_hair_cloth_v2/secondary_profile.json`。裙边使用独立碰撞组，继续排除裙边自身及裙节点的重复接触；新增报告 `hair_cloth_contact_steps` 和 `hair_cloth_contact_peak_newtons` 测量头发与裙边的接触，未涵盖所有头发与裙节点接触。仍为骨骼代理碰撞，不能保证可见三角网格完全不穿模。

局部裙腰贴合可选 `appearance.skirt_waist_support`：`node_radii_by_row_m`、`contact_offsets_by_row_m` 按 `qun_<row>_<column>` 的行设置代理半径、接触偏移（米），`shape_gain_by_row` 设置该行恢复力倍率；超出数组的行沿用原参数。它不修改网格、腰线、骨骼静态位置或裙长约束。Girls 前 40 秒的 V4 实验位于 `output/mutsumi_girls_waist_v4/`，沿用 V3 布料，仅局部渐变贴合；用户确认后已合入默认模板。

模型专属侧发可选 `appearance.local_hair_support`：`bone_prefixes` 显式匹配发链，`pin_roots` 固定匹配链第 0 行的位置，`shape_gain_by_row` 与 `bend_multiplier` 调整回形/抗弯倍率。未匹配的发链保持原参数，不更改网格或碰撞开关。爱音 `rclh_ / lclh_ / rch_ / lch_` 的 Girls 前 40 秒试验在 `output/anon_girls_sidehair_v5/`，用户确认后已合入默认模板。

《极乐净土》当前参考片按原竖版比例录制为 `1080×1350 / 30 FPS`，使用 `record_cached_secondary.py --width 1080 --height 1350` 原生渲染。G1 合成脚本可用 `--output-height 1350` 保留该参考片的原生像素；右侧已有 G1 验收录像按比例缩放，时钟、评分及录制内容不变。

`record_cached_secondary.py` 可用既有完整 UsdSkel 物理缓存重录外观，要求原物理报告为完整的 120 Hz 版本。录制时不启动物理，直接读取缓存每帧姿态，检查缓存哈希、抽样姿态、原地面属性与相机构图，并输出 `render_report.json`。2026-11-13 首次仅将《极乐净土》验收参考切换为 `mmd-toon`，保留既有 120 Hz 缓存及 10% 缩短裙长；其他舞蹈仍保持原视频。


当前实时头发／裙子物理默认模板为 `configs/physics/secondary_default.json`（V1，原 hair_contact_drape_v10），**120 Hz**、30 FPS 时每帧 **4** 子步、`cloth` 质量。适用于原生物理舞蹈播放、录制、睦批录和爱音在线播放。模型网格尺寸自动解析；静态裙长、动作、骨骼编号不放入公共模板。历史索引见 `configs/physics/secondary_profiles_catalog.json`，旧版参数文件已清理；历史实验文字仅供参考。

将 PMX 的网格、UV、基础贴图、骨骼及蒙皮权重直接转换成 USD UsdSkel，
在 Isaac Sim 6.1 中循环展示左臂、左肘、左髋和左膝的独立旋转。
该目录独立于现有 G1/VMD 训练流程，不需要 Blender 或额外安装插件。

在项目根目录 PowerShell 中运行：

```powershell
cd I:\robot_isaac
.\run_isaac61.bat source\pmx_prototype\convert_pmx.py "media\千早爱音_by_小麦弥望\Anon Chihaya.pmx"
.\run_isaac61.bat source\pmx_prototype\view_character.py
```

第二条命令持续播放八秒循环，关闭窗口即可退出。已转换过可直接运行第二条。
角色按网格包围盒缩放至 1.6 米；这不是对角色官方身高的认定。
转换时可通过 `--height 1.6` 调整。

生成文件位于 `output/pmx_prototype/`：

- `character.usdc`：含骨骼、蒙皮、241 帧动画的角色资产。
- `textures/`：复制后的贴图；原始素材不变。
- `conversion_report.json`：模型统计、日文骨骼与 USD joint 路径映射、CPU 蒙皮检查结果。
- `scene.usda`：查看器生成的带相机及灯光的场景。
- `rest.png` / `arm.png` / `elbow.png` / `hip.png` / `knee.png`：使用下述截图命令生成。

有界渲染验证：

```powershell
.\run_isaac61.bat source\pmx_prototype\view_character.py --headless --steps 30 --captures
```

控制角度与旋转轴在 `convert_pmx.py` 的 `controls` 中，修改后重新转换。
动画依次为左臂 0–2 秒、左肘 2–4 秒、左髋 4–6 秒、左膝 6–8 秒，
每段均从静止姿态转动再复位。使用标准日文 MMD 骨骼名；其他模型需检查名称。
此模型的腿部权重绑定在 D 骨骼上，原型会递归计算追加旋转，驱动对应 D 骨骼。

这是骨骼动画原型，尚未创建 PhysX articulation、碰撞或重力控制。
未实现 IK、表情 morph、追加位移、头发/裙摆物理、MMD toon/sphere/描边材质。
追加旋转使用共同静止坐标轴求值，不等同于完整 MMD 求解器；SDEF/QDEF 暂按线性蒙皮近似。
该素材实际只包含 BDEF1/2/4。基础贴图与 MMD 原渲染效果可能不同。

素材署名：**Bushiroad Games/BungleScrungle/海新_/小麦弥望**。
素材使用条款仍以原文件夹中的说明为准；生成资产仅供当前本地原型使用。

## 在爱音身上播放《极乐净土》VMD

先完成上述 PMX 转换，然后运行：

```powershell
cd I:\robot_isaac
.\run_isaac61.bat source\pmx_prototype\load_vmd.py
.\run_isaac61.bat source\pmx_prototype\view_character.py --usd output\pmx_prototype\ji_le_jing_tu.usdc --output output\pmx_prototype\dance
```

已烘焙过时只需运行第二条。窗口按实际时间循环播放整段约 218 秒的舞蹈，
关闭窗口退出。可加 `--start-frame 300` 从第 10 秒开始。

`load_vmd.py` 默认读取项目中的 `media/dance/ji_le_jing_tu/ji_le_jing_tu.vmd`，
按骨骼名匹配爱音 PMX，支持全角/半角名称归一化、VMD 贝塞尔位置插值和四元数球面插值。
采用 PMX 本身的骨长和膝盖限位求解两节腿部 IK，再解单节脚尖 IK，
随后计算追加旋转（包括绑定腿部网格的 D 骨骼）。不经过 G1 的动作映射。
VMD 内的 IK 开关会按关键帧生效。

完整动作含 6,543 帧，匹配 77 条轨道，最终烘焙 85 根骨骼。
`ji_le_jing_tu.usdc` 引用原来的 `character.usdc`，二者及贴图需要保留在本机。
`ji_le_jing_tu.json` 记录匹配、未匹配轨道、IK 误差和抽样蒙皮包围盒。
源 VMD 标注的模型名为“しまかぜ”，骨长差异可能使个别足部目标超出爱音腿部可达范围。

本版本播放身体、手指、视线与根位移；暂未加载表情 morph、伴音或头发/裙摆物理。
裙摆不会自动避开腿和手臂，部分姿态会穿插；不等同于 MMD 完整渲染与物理解算。
已有动作借物信息见项目主 README（《极乐净土》动作 by Yurie）。

跨全曲的渲染检查：

```powershell
.\run_isaac61.bat source\pmx_prototype\view_character.py --usd output\pmx_prototype\ji_le_jing_tu.usdc --output output\pmx_prototype\dance --headless --steps 90 --captures --capture-frames 0 300 600 1500 3000 4500 6000
```

## 头发与裙摆的物理烘焙

```powershell
cd I:\robot_isaac
.\run_isaac61.bat source\pmx_prototype\runtime\secondary_motion.py
.\run_isaac61.bat source\pmx_prototype\view_character.py --usd output\pmx_prototype\ji_le_jing_tu_physics.usdc --output output\pmx_prototype\physics
```

已经烘焙后只需第二条，播放完整舞蹈与物理结果。骨骼舞蹈发生变化时重新运行第一条。
该工具读取 PMX 中 484 个刚体、879 个约束的拓扑，提取裙摆、长发、短发与饰品骨骼，
构建 496 个粒子节点（494 个动态骨骼节点和两个外部锚点）、2,253 条拉伸／剪切／弯曲约束。
裙腰和头皮首排节点跟随身体，其他节点受重力、惯性、阻尼和形状回复作用。
25 个 PMX 身体胶囊／球形代理随已有舞蹈移动，碰撞投影将布料与头发节点推到代理外侧；
底边还会与地面接触。默认每动画帧四个子步（120 Hz），每个子步 12 次约束迭代。

这是**离线 XPBD 粒子模拟后烘焙成 UsdSkel 动画**，不是把原始 MMD Bullet 刚体逐个导入 PhysX。
原模型中的零厚度片、质量、角度弹簧等参数没有直接照搬；使用了适合粒子布料的约束参数。
Isaac Sim 播放的是已计算的物理轨迹，不会在回放时对新加入的外部物体实时产生碰撞响应。
查看器会从只读缓存中抽样当前姿态，再写入场景的骨骼默认值，避免每次渲染都处理整段大缓存。
导出的 `scene.usda` 仍引用完整动画，不保存查看器的临时姿态覆盖。
现有 G1 训练与验证流程不受影响。

结果资产：`output/pmx_prototype/ji_le_jing_tu_physics.usdc`，它引用原来的舞蹈和角色文件。
同名 JSON 记录求解规模、主身体位移检查和蒙皮顶点的穿入指标。
穿入指标使用同一组身体碰撞代理比较模拟前后，仅统计受动态骨骼影响的顶点，
不是三角网格间的精确穿模检测。饰品的既有嵌入、布料顶点与骨骼点的偏移、
代理与真实身体轮廓的差异，仍可能带来残留穿模；尚无三角形级自碰撞。

当前全曲的 11 个抽样姿态中，上述代理穿入深度总和下降约 75.9%；
第 1,500 帧（50 秒）最大代理穿入深度从 8.42 cm 降至 3.56 cm。
这不代表所有姿态都改善了同样比例，也不代表精确网格穿模消除了 75.9%。
全曲骨骼求值、主身体位移一致性和七个时间点的 Isaac Sim 渲染已经验证。

有界试算可使用 `--frames 1600 --output output/pmx_prototype/physics_trial.usdc`。
可通过 `--substeps` 和 `--iterations` 调整模拟精度，默认值适用于当前模型。
工具使用 Isaac Sim 自带的 NumPy、SciPy、Numba；没有修改已安装的训练依赖。
为兼容捆绑版本，独立烘焙进程会在导入 Numba 时临时禁用不兼容的可选 coverage 钩子。

## 实时 XPBD 舞蹈

```powershell
cd I:\robot_isaac
.\run_isaac61.bat source\pmx_prototype\realtime_dance.py
```

直接使用 `ji_le_jing_tu.usdc` 的身体动画，每帧推进头发／裙摆粒子，并将当前结果驱动到蒙皮。
不读取 `ji_le_jing_tu_physics.usdc`；无需先运行离线物理烘焙。
身体动画仍是预先从 VMD 转换好的骨骼动作，头发／裙摆则在播放过程中实时求解。
输入物理缓存会被拒绝，避免误将缓存回放当成实时模拟。

默认以 60 FPS 渲染，舞蹈保持正常速度。物理仍按 30 Hz 批次更新，每批四个子步、
每子步 12 次约束迭代，固定时间步仍为 1/120 秒。
新增的渲染帧使用相邻物理姿态的位移线性插值与四元数最短路径 SLERP，
不会额外推进物理或重复上一帧。只保留相邻两帧实时状态，最多提前计算一个 1/30 秒批次。
整数 VMD 帧对应的物理结果与原版保持一致。
窗口中 `Live XPBD` 面板显示帧率和 XPBD 毫秒耗时；`Gravity` 会立即影响后续模拟，
`Reset dance and particles` 重置舞蹈和粒子状态。全曲结束后重置状态并循环，
不会把结尾的速度直接带到开头。性能不足时降低播放速度，保持固定时间步，不跳过物理步骤。

可用 `--render-fps 30` 恢复原来显示频率，`--render-fps 60` 明确指定 60 FPS。
当前支持 30 的整数倍。`--steps` 统计渲染帧数，所以 60 FPS 的整曲为 13,086 帧。

### 伴奏开关

Live 面板中的 **Play music (accompaniment)** 默认勾选，使用本地
`media/dance/ji_le_jing_tu/ji_le_jing_tu.wav`，初始音量为 20%。取消勾选立即停止，重新勾选后从当前舞蹈时间恢复。
重置和舞蹈循环会同步重启音乐，关闭窗口会停止播放。
播放器以解码后的 PCM 音频切片定位，避免 SDL_mixer 对 WAV 音乐跳转支持不一致的问题。
若估算音频时钟与舞蹈时间相差超过 150 ms，会重新定位；实际扬声器延迟取决于设备缓冲。

可用 `--no-music` 以不勾选状态启动，或 `--music "完整 WAV 路径"` 更换 16-bit PCM WAV。
加速 `--benchmark` 模式会自动静音并禁用开关，避免加速舞蹈与正常速度音乐错位。
音频设备不可用时，面板会显示原因并禁用开关，舞蹈和物理解算仍可运行。
音频不会改变渲染频率或 120 Hz 物理时间步。

```powershell
.\run_isaac61.bat source\pmx_prototype\diagnostics\check_live_music.py
```

此检查以零音量调用实际音频设备，验证开启、关闭、12.5 秒定位和舞蹈重置。
实时报告的 `music` 字段记录播放状态、启停事件及估算同步漂移。

60 FPS 有界可见窗口验证已通过：32 秒渲染 1,920 帧，平均实际频率 60.0 FPS，
物理步长保持 1/120 秒；数据位于 `output/pmx_prototype/realtime_60_validation/`。

原 30 FPS 模式的完整可见窗口性能测试（不限制播放速度，物理步长保持不变）：

```powershell
.\run_isaac61.bat source\pmx_prototype\realtime_dance.py --render-fps 30 --benchmark --steps 6543 --output output\pmx_prototype\realtime_benchmark --capture-frames 300 1500 3000 4500 6000
```

`realtime_report.json` 记录含骨骼更新、求解、USD 上传和 `app.update()` 渲染的帧耗时分布，
单独记录 XPBD 耗时，并用与离线版相同的蒙皮顶点／身体代理指标比较穿入深度。
`capacity_fps` 排除启动、诊断探针和截图暂停；`observed_wall_fps` 包含这些诊断暂停。
`paced_fps` 是帧工作和限速等待的平均频率。`frame_timings.csv` 在有界测试结束或关闭窗口时保存。
实时窗口会每 300 个渲染帧更新一次报告，`--benchmark` 可测不加渲染限速时的处理能力。
报告分别记录 `target_render_fps`、`physics_batch_hz`、`solver_hz` 和实际物理批次数。
XPBD 耗时按真正执行的物理批次统计，不把未求解的插值帧计为一次求解。
启动时的 Numba 编译和着色器预热不计入帧率。

运行状态与数值一致性检查：

```powershell
.\run_isaac61.bat source\pmx_prototype\diagnostics\check_realtime.py
```

检查实际模型的逐帧／批量求解一致性、速度状态延续、重置可复现，以及运行中改变重力后的响应。
还会验证 127 次半帧渲染只执行 64 个物理批次，整数帧物理结果一致，新增画面有真实插值变化。
实时求解仍使用自定义 XPBD 与身体胶囊／球形代理，不是 PhysX 场景求解器。
其他场景物体尚不会自动参与碰撞，仍无三角形级自碰撞，少数姿态会保留局部穿模。

本机完整可见窗口测试（1000×1000 渲染、6,543 帧、4 子步／12 迭代）结果：

| 指标 | 实测 |
| --- | --- |
| 帧工作耗时，均值 / P95 / P99 | 9.20 / 10.88 / 13.19 ms |
| XPBD 求解耗时，均值 | 2.38 ms |
| 不限速处理能力，排除诊断与截图 | 108.65 帧/秒 |
| 包含诊断及截图暂停的整体速度 | 106.94 帧/秒 |
| 超过 33.33 ms 的帧比例 | 0.18% |
| 11 个抽样姿态的代理穿入深度总和降幅 | 75.6% |

上述高帧率测试采用原来的 30 FPS 渲染配置加速遍历全曲；当前正常观看默认为 60 FPS 渲染、120 Hz 求解。
统计排除了应用启动、Numba 编译和着色器预热。穿入降幅仍是身体代理指标，
不是“实际穿模消除了 75.6%”。完整数据和逐帧 CSV 位于 `output/pmx_prototype/realtime_benchmark/`。

## 原生 PhysX 头发／裙摆与场景碰撞

```powershell
python source\pmx_prototype\physx_dance.py
```

独立的原生 PhysX 原型，保留原来的 XPBD 入口。固定使用 1/120 秒的 PhysX 步，
正常观看按墙钟累计应执行的物理步，目标渲染为 60 FPS，身体继续由 VMD 驱动。
渲染稍慢时会在下一次渲染前补做物理步，不放大时间步、不跳过碰撞步骤；
单次最多补做 8 步，以便窗口操作后逐步追上时间。头发与裙摆使用 465 个动态球形节点、
31 个运动学锚点和 2,209 条原生距离约束（含 40 条发梢限制）；积分、约束和碰撞由 `omni.physx` CPU 后端处理。
共享的 Python/Numba 代码仅用于读取身体动画、构建骨骼拓扑和重建蒙皮姿态，
不会调用 XPBD 的粒子积分／约束求解函数。

默认 hair 配置采用 PhysX PGS 求解器、40 次位置迭代、8 次速度迭代，Fabric 同步物理状态。
节点启用原生 CCD、接触修正速度限制和阻尼，弯曲使用软距离弹簧。
PMX 自带的身体碰撞组屏蔽会转换为原生 FilteredPairs，避免短发／饰品与其嵌入部位产生冲突。
默认仅非相邻头发节点启用自碰撞，裙摆内部与头发／裙摆之间仍屏蔽自碰撞；
同一场景中的其他 PhysX Collider 可正常与动态节点碰撞。
少量形状回复力通过 PhysX 施力接口施加，不在 Python 中进行位置投影或碰撞修正。

默认放置两个普通静态碰撞球：蓝色接触裙摆、橙色接触头发。
`Scene contact peak` 显示 PhysX tensor 接口对这两个测试物体返回的接触力，
不是根据可见重叠估算的值。音乐复选框默认开启，重置和循环时同步重启。
`Reset dance and PhysX bodies` 会重新设置节点及身体碰撞代理的位置与速度。

加载自己的 USD 场景，或隐藏内置测试球：

```powershell
python source\pmx_prototype\physx_dance.py --scene "场景文件完整路径.usd" --no-obstacles
```

环境物体需要具有 `UsdPhysics.CollisionAPI`，运动物体还需相应的 RigidBody 配置，
并与角色处于同一个 Physics Scene。单纯可见的网格不会自动产生物理碰撞。
PhysX 的 broadphase/contact solver 会处理这些场景 Collider，不用把外部物体转换进自定义 XPBD 列表。
接触力面板只追踪内置测试球，不统计所有新导入物体的接触力。

这版仍是**刚体节点代理驱动蒙皮**，不是三角形级布料或头发求解：
局部网格仍可能穿插，小物体可能从节点间隙穿过，褶皱可能比 XPBD 版更明显。
身体是运动学动画，不会因为头发或裙摆受力而改变舞步或维持平衡。
输出 `physx_report.json` 记录真实 PhysX 接触力、步数和性能；`scene.usda` 可用于查看场景结构，
实时驱动身体和蒙皮仍需要运行该 Python 入口，不能把导出的场景当作独立完成的舞蹈回放。

本机完整可见窗口测试已通过：13,086 个渲染帧、218.1 秒模拟时间，
1000×1000 渲染下不限速平均约 56 FPS，平均帧工作耗时 17.76 ms，
每次渲染对应的 PhysX 工作平均 7.52 ms（测试固定每渲染帧两步）。
原生接触报告在 9,825 个物理步中记录了与内置场景碰撞球的接触。
此版本比 XPBD 更重，当前不保证锁定 60 FPS；正常观看仍按独立墙钟推进物理与伴奏。
测试报告和全曲关键姿态截图位于 `output/pmx_prototype/physx_full/`。
正常时间调度加音乐的初次可见窗口运行约 47–50 FPS，实际物理频率接近 120 Hz；
该结果与不限速固定两步／渲染帧的 56 FPS 测试口径不同，不应混作稳锁 60 FPS。

### PhysX 发束与裙摆保形优化

第一轮保形配置为 `--quality smooth`，仍由原生 PhysX 积分、求解距离约束及处理场景碰撞。
改进内容包括：

- 发束按根部到发梢逐渐降低保形强度；裙摆使用独立的纵向与横向参数。
- 885 个三点弯曲组对比当前曲率与身体坐标系下的原始曲率，通过 PhysX 施力接口施加回复力和阻尼。
  这些是附加的物理力，不是在 Python 中重新投影节点位置，也不是改回 XPBD。
- 头发与裙摆采用各自的低摩擦物理材质，减少碰撞时粘住、反复折叠的情况。
- 蒙皮骨骼方向改用邻近节点的中心方向估计，减少单条短边抖动造成的碎折。
  骨骼支点保留 PhysX 返回的位置，没有为了视觉平滑而移动碰撞节点。

旧版仍可用于同舞姿比较：

```powershell
python source\pmx_prototype\physx_dance.py --quality legacy
python source\pmx_prototype\physx_dance.py --quality smooth
```

相同前 50.02 秒动作、相同场景碰撞球的对照中，发束归一化曲率偏差 RMS 从 1.472 降到 1.004
（下降约 31.8%），裙摆从 1.295 降到 1.127（约 13.0%）。
该指标表示偏离原始弯曲形状的程度，不是“画质提高 32%”或“穿模减少 32%”。
保形更强也可能减少自由飘动幅度；这轮 smooth 配置仍有部分发梢折角、网格穿插和自碰撞缺失的限制。
对照数据在 `output/pmx_prototype/physx_quality_comparison.json`，
同舞姿截图分别在 `physx_quality_legacy/` 与 `physx_quality_smooth/`。

数值检查：

```powershell
python source\pmx_prototype\diagnostics\check_physx_quality.py
```

检查回复力方向、合力平衡、平移不变性、阻尼耗散，以及蒙皮方向平滑后原生节点位置不变。
新增的 CPU 计算内核在开始计时、播放音乐前预热。
优化配置另已通过完整 218.1 秒舞蹈／13,086 个渲染帧的可见窗口检查，
未检测到非有限值或位置发散，保留原生场景接触；不限速固定步数测试平均约 53.5 FPS。
完整报告和后半段截图位于 `output/pmx_prototype/physx_quality_full/`。

### 发梢防折与发束分离

上一轮头发配置为 `--quality hair`，旧 smooth 配置仍可用于对照。

- 非相邻头发球节点启用原生 PhysX 自碰撞；原始姿态距离小于 29 mm 的近邻及约束连接的节点仍屏蔽碰撞，避免相邻层互相挤压。
- 发束横向与斜向连接改成较软的原生距离弹簧，纵向保留长度约束，减少碰撞时整片头发被横向拉成一团。
- 加强纵向发梢曲率回复力，降低横向曲率耦合；40 条原生最小距离约束限制末端两段过度反折。
  名义折角上限为 55 度，若原始造型本身更弯则保留其角度加 15 度；有限迭代及约束误差会影响实际折角。
- 头发节点半径由根部 10 mm 渐变到末端 8 mm，降低摩擦，并提高求解迭代次数；本轮优先保证形态与碰撞质量。

这仍是球节点碰撞代理，不是发丝、线段或三角网格级自碰撞，节点间隙和相邻发片仍可能穿插。
报告新增末端折角 P95、超过 75 度的末端比例，以及原始间距大于 50 mm、当前距离小于 12 mm 的头发节点对数量。
这些指标每 30 个物理步抽样，报告取时间平均，用于同动作对照，不等同于网格穿模或视觉质量评分。

同样前 50 秒、200 个抽样时刻的 smooth → hair 对照：

| 头发节点指标 | smooth | hair |
| --- | ---: | ---: |
| 末端折角 P95 的时间平均 | 65.61° | 19.90° |
| 末端折角超过 75° 的比例，时间平均 | 4.06% | 0% |
| 非局部节点距离小于 12 mm 的对数，时间平均 | 0.305 | 0.060 |
| 归一化曲率偏差 RMS | 0.938 | 0.542 |

数据在 `output/pmx_prototype/physx_hair_comparison.json`。这些结果同时包含自碰撞、
回复力、连接强度、半径和迭代数变化的影响，不将全部改善归因于某一个参数。
更强的纵向保形会减少部分自由弯曲；当前限制仍包括相邻发片与节点间隙的网格穿插。

hair 配置已完成整首 218.1 秒、26,172 个原生物理步的无窗口渲染验证，
未检测到非有限值或位置发散；9,751 个物理步记录到与内置场景物体的接触。
全曲末端超过 75 度的折角抽样比例约 0.0115%，并非完全没有反折。
报告和同舞姿截图位于 `output/pmx_prototype/physx_hair_full/`。


### 裙摆自碰撞实验（尚未通过效果验收）

默认继续使用上一轮验证过的 `--quality hair`。新的 `--quality cloth` 是实验配置：
裙摆球节点半径 24 mm，静／动摩擦 0.08/0.045；非相邻裙摆节点启用原生 PhysX 自碰撞，
原始距离小于 75 mm 的邻近布片仍屏蔽碰撞；头发与裙摆之间仍屏蔽交叉自碰撞。
裙摆纵／横向曲率回复刚度为 8/6，形状回复加速度系数 45，PGS 迭代数为 96/16，
裙摆节点最大接触修正速度 3 m/s。固定步频仍为 120 Hz，所有场景碰撞保留。
裙摆蒙皮朝向补充环向信息，支点位置保持 PhysX 原始输出。

本轮测试了额外防折距离约束、纵向跨段限制、展开回复力及 240 Hz 步进；
这些试验未得到整体稳定的效果提升，最终没有保留额外裙摆防折约束或展开力，
也没有将实验配置设为默认。较少的非局部节点挤近不代表大折角和网格穿插一并解决。

同样前 50 秒，最终保留的实验配置（`physx_cloth_relaxed/`）与 hair 对照：
非局部裙摆节点距离小于 30 mm 的对数均值从 5.44 降至 1.13；
超过 90 度的三点组比例从 4.59% 升至 10.26%；
蓝色场景球最大代理穿入深度的时间平均从 16.58 mm 升至 18.45 mm。
该结果属于混合得失，不能作为裙摆整体效果改善的证据。

指标每 0.25 秒抽样；穿入深度针对球形节点代理，不代表可见三角网格的穿模率。
报告保存于 `output/pmx_prototype/physx_cloth_baseline/`、`physx_cloth_relaxed/`，
其他试验目录记录各自参数试验时的结果，不应当作当前默认配置。
数值检查验证纵向不变、布片绕纵向旋转 90 度时能够恢复蒙皮扭转，并保持原生节点位置。

## 睦：原生 PhysX 头发与裙摆

模型作者：很ying的哈呵。输入为用户确认的 `output/mutsumi_pmx/IRIS_OUT_clearance.usdc`。
先运行一次 `python source\pmx_prototype\prepare_mutsumi_physx.py`，在独立的
`output/mutsumi_physx/` 中建立 21 列、9 行的裙摆骨骼与双线性蒙皮；原始 PMX 不修改。
头发沿用 PMX 发束链，裙腰固定到髋部。身体沿用输入动作。

```powershell
python source\pmx_prototype\physx_dance.py --usd output\mutsumi_physx\character.usdc --pmx "media\mutsmi\mutsmi v1.pmx" --rig-data output\mutsumi_physx\rig.json --music media\dance\IRIS_OUT\IRIS_OUT.wav --output output\mutsumi_physx\live --quality cloth --no-obstacles
```

PhysX CPU 以 120 Hz 推进；头发和裙摆分别启用非相邻节点自碰撞，头发与裙摆之间的交叉碰撞关闭。
这是刚体节点代理驱动蒙皮，不是三角网格布料，仍可能有局部穿模。
本模型的 `cloth` 配置已通过完整 15.5 秒测试，不改变爱音入口的默认 `hair` 配置。
`--record-video PATH --ffmpeg PATH` 录制当前解算，`--bake-usd PATH` 保存 PhysX 生成的动画缓存。
录制使用 `--benchmark --steps 930` 时输出 15.5 秒、60 FPS 视频；不代表实时运行能锁定 60 FPS。
`IRIS_OUT_physx.usdc` 是已解算缓存，可由 `view_character.py --usd ...` 回放；实时交互使用上面的入口。

### 睦的前发锁定与平滑裙摆（v2）

`prepare_mutsumi_physx.py --profile smooth --output output/mutsumi_physx_v2` 生成第二版。
`legacy` 保留第一版绑定；`smooth` 锁定 front hair 链，后发仍由 PhysX 求解。
裙摆使用 16 权重的三次 B 样条位移蒙皮，节点旋转不额外叠加到裙摆网格。
物理网格保留原节点位置，允许有限剪切与环向伸展，并加入 147 条原生距离防折约束。
这套设置通过 rig.json 启用，爱音和旧 rig.json 的行为不变。

```powershell
python source\pmx_prototype\physx_dance.py --usd output\mutsumi_physx_v2\character.usdc --pmx "media\mutsmi\mutsmi v1.pmx" --rig-data output\mutsumi_physx_v2\rig.json --music media\dance\IRIS_OUT\IRIS_OUT.wav --output output\mutsumi_physx_v2\live --quality cloth --no-obstacles
```

第二版完成 930 帧/1860 步测试，前发局部姿态和身体原动画一致。
八个抽样姿态的裙摆相对原形折角变化 P95 的平均值，从第一版 17.38 度降至 10.06 度。
这是局部网格变形指标，不是整体画质或无穿模保证。

### 睦的短裙版本

`prepare_mutsumi_physx.py --profile smooth --skirt-length-scale 0.9 --output output/mutsumi_physx_short`
保持腰线与水平轮廓，将裙长从约 42.04 cm 缩至 37.83 cm，下摆抬高约 4.20 cm。
顶点、法线、裙摆骨架和约束的原始长度同步重建；UV 与原 PMX 保持不变。
省略 `--skirt-length-scale` 默认为 1.0。短裙的实时入口沿用 `physx_dance.py`，
将 usd 与 rig-data 分别改为 `output/mutsumi_physx_short/character.usdc` 和同目录 `rig.json`。
完成 930 帧、1860 个物理步测试；前发锁定和身体舞蹈一致性验证通过。

### 裙长保持修正：防止解算过程中被拉长

旧版本只验证静态几何缩短，未验证运行后的长度：30% 短裙链条平均初始约 30.70 cm，
在原距离关节下结尾约 50.76 cm。该结果是物理约束伸长，不能当作原始裙长的可靠对照。
新生成 rig.json 默认带 `skirt_max_stretch=1.02`，由原生 D6 径向上限约束限制纵向链接与腰部到各行的跨度。
使用该字段时采用 240 Hz、PGS、255 次位置迭代，成本高于旧版本；已有旧 rig.json 保持原行为。

```powershell
python source\pmx_prototype\prepare_mutsumi_physx.py --profile smooth --skirt-length-scale 0.7 --output output\mutsumi_physx_short30_fixed
python source\pmx_prototype\physx_dance.py --usd output\mutsumi_physx_short30_fixed\character.usdc --pmx "media\mutsmi\mutsmi v1.pmx" --rig-data output\mutsumi_physx_short30_fixed\rig.json --music media\dance\IRIS_OUT\IRIS_OUT.wav --output output\mutsumi_physx_short30_fixed\live --quality cloth --no-obstacles
```

完整 465 个动作帧的检查中，腰部到下摆节点的最大直线跨度为初始链条弧长的 1.0122 倍，
结尾链条平均长度为 29.52 cm；身体动作及前发锁定验证通过。
该检查衡量节点跨度与链长，不等同于三角网格的精确不可伸长性或无穿模证明。
数据见 `output/mutsumi_physx_short30_fixed/full_length_verification.json`。
旧版对照资产与录制已保留；新视频为 `media/recordings/IRIS_OUT_mutsumi_physx_short30_fixed.mp4`。

### 10% 短裙的接触减震版本

`output/mutsumi_physx_short10_contact/` 保持 10% 缩短及 D6 长度上限。
rig.json 的 `skirt_contact_smoothing` 仅作用于动态裙摆节点：contact_offset_m=0.006，
max_pushout_m_s=0.3，linear_damping=10；不改变腰线、网格尺寸与前发锁定。
实时运行时将 physx_dance.py 的 --usd/--rig-data 指向该目录的 character.usdc/rig.json。
全曲 60 Hz 节点轨迹在髋部坐标系中，高频残差 RMS（9 帧、三阶 Savitzky-Golay 去趋势）
从 1.4805 mm 降至 1.2016 mm；加速度 P95 从 41.10 降至 32.72 m/s²。
这些是整曲代理节点指标，不是严格的接触事件分段分析或完全无抖动保证。
数据位于 contact_verification.json，视频为 media/recordings/IRIS_OUT_mutsumi_physx_short10_contact.mp4。

### 碰腿释放阶段减震

`output/mutsumi_physx_short10_release/` 与上一版 contact 的 character.usdc 二进制一致。
新增 skirt_contact_release 只对接触过身体的裙摆节点施加相对腿部运动的速度阻尼，
接触消失后以 0.10 秒时间常数平滑衰减。物理积分和约束仍为原生 PhysX；无画面后处理滤波。
保留原复位力与长度上限。接触推出限速 0.15 m/s，线性阻尼 14；不采用试验性的复位力法向裁剪。
全曲对照：节点加速度最大值 793.47 → 463.10 m/s²，P99 74.77 → 71.46 m/s²；
高频残差 RMS 1.2016 → 1.1053 mm。该指标基于髋部坐标系中的整曲节点轨迹，
不等同于逐次接触事件的冲击力或完全无抖动保证。前发和身体逐帧一致性验证通过。
实时运行仍用 physx_dance.py，usd 和 rig-data 改为本目录的 character.usdc 与 rig.json。
视频：media/recordings/IRIS_OUT_mutsumi_physx_short10_release.mp4。

## 睦的舞蹈参考视频与 G1 验收对照

批录入口：`python source\pmx_prototype\record_mutsumi_references.py`。
默认覆盖现有七支完整舞蹈及 hal 短版；支持 `--dances NAME ...` 和 `--force`。
它使用现有 VMD、睦的已修正模型、10% 短裙、前发锁定、长度上限与接触释放减震配置。
动作按睦骨骼单位换算，并沿用已确认的腿距与代理避碰修正；不更改训练 H5 或验收指标。
模型作者：很ying的哈呵。

输出位于每个舞蹈的 `media/dance/<dance>/<stem>_mutsumi_reference.mp4`；
30 FPS、720×900、固定正面全身取景，视频第 0 帧对应 VMD 第 0 帧。
完整物理动画缓存、录制日志及可恢复批次清单位于 `output/mutsumi_validation_references/`。
每段电影通过完整解码计帧后才标记 verified；全部请求完成后原子更新
`source/train_workflow/validation_videos.json` 的默认参考片选择。

新启动的 G1 自动训练沿用已有验收录制流程，生成“睦参考在左、G1 多种子验收在右”的对照片。
比较按验收的 source_start_s、playback_rate 和帧步长对齐；参考结束后保持末帧。
验收音轨作为唯一合成音轨；参考视频仅用于观看对比，不参与验收评分。
已经运行的训练保存了启动时的参考选择，配置更新不会热替换该选择；
可使用 `source/train_workflow/scripts/compare_validation_video.py` 为已有验收 MP4 重新合成。


### 参考片脚步保真修正（2026-11-12）

`record_mutsumi_references.py` 默认关闭额外腿部避碰；仅用 `--leg-clearance-dances 舞蹈名 ...` 显式启用。IRIS OUT 上调出的避碰参数不可泛化到全部编舞：旧版极乐净土每只脚额外横移最高 25 cm，改变了脚步。新策略保留单位换算、Anon 初始腿距和基础 IK，不追加逐帧外移。缓存签名包含策略版本及避碰开关。旧库 manifest 的动作保真检查与视频解码/时钟检查分开标注；未重新检查的旧片不能仅凭 verified 状态判定动作保真。

批录默认通过 `--preserve-existing-dances IRIS_OUT` 保留已验收的 IRIS OUT 视频（包括 `--force` 时），复用前检查 sidecar SHA256；完整重新渲染需要显式用空的 `--preserve-existing-dances` 覆盖此保护。其余舞蹈的修正记录见 `output/mutsumi_other_dances_fixed/clearance_audit.json`，逐帧骨骼对照使用 `character_cache.py reference`。

最终逐帧核查：HAL 完整版、Girls、Galaxias 的实际脚踝曾被额外外移，已取消额外避碰并完整重录；你最重要与 Deep Blue Town 的 VMD 全程关闭腿部 IK，控制目标外移未作用于腿部，保留原视频。HAL 短版额外修正全程为零，保留。IRIS OUT 按用户要求保留原视频并通过 SHA256 核验。`load_vmd.py` 的 `active_mean_m`/`active_max_m` 只统计启用 IK 的帧，禁用时为 null；旧 `mean_m`/`max_m` 为兼容保留，包含禁用帧，不能独立作为解算失败判据。核查汇总：`output/mutsumi_other_dances_fixed/review_summary.json`。

### Girls 新物理对照试验

`output/mutsumi_girls_physics_update/rig.json` 在现有 Girls 动作与缩短 10% 的裙子上合并 `configs/physics/presets/anon_secondary_hand_contacts.json`，保留前发锁定、防拉长约束和接触释放设置，并将旧 `skirt_contact_smoothing.linear_damping` 改为新配置的 2，避免覆盖新阻尼。实际场景检查通过：57 个头发动态节点阻尼 2.2，168 个裙子动态节点阻尼 2，半径均 3 mm，168 个裙边胶囊及两个手掌代理。身体动作与镜头保持一致。此为独立试验片，完整对比为 `girls_physics_before_after.mp4`，左旧右新。新版裙摆卷折明显增加，未替换原验收参考片；所有原舞蹈视频均通过 SHA256 保留检查。

批录器支持显式 `--secondary-profile source/pmx_prototype/configs/physics/presets/anon_secondary_hand_contacts.json`，默认不迁移新参数；缓存签名包含配置内容。启用新配置时同步处理裙子阻尼覆盖。

### 睦 Girls 防卷折校准

`configs/physics/models/mutsumi_secondary_girls.json` 为睦专用试验配置，仅验证 Girls，不改变 USD 静态裙长、蒙皮、身体动作或相机。保留新版后发物理和前发锁定，分别控制裙子/头发的自由垂落，恢复裙子支撑与接触厚度；将手部代理拟合范围从整只张开手（约 16 cm）改为掌部（约 7.5 cm）；裙子下段边胶囊允许与大腿/膝腿代理接触，上段排除新增腿部接触；168 条纵向边胶囊之间增加 147 个 PhysX SphericalJoint 弯曲限位，使用 Z 轴 30° 双锥参数。该参数为关节设置，不等于保证所有渲染顶点折角小于 30°。所有新增行为均须由配置显式启用，默认不改变爱音。

验收文件在 `output/mutsumi_girls_physics_tuned/`：`girls_before_after_tuned.mp4` 左侧是上次直接迁移爱音配置的卷裙版本，右侧为睦校准版；`results.json` 提供整首同时间点的卷折及共用腿部碰撞代理指标，`body_motion_audit.json` 提供身体矩阵逐帧检查。`character_cache.py cloth` 评估真实蒙皮裙子顶点对共用 PMX 腿部胶囊的穿入，不能作为三角网格精确无穿模的证明。完整旧参考库保留不变。

爱音先前的行走使用 quality=hair / 120 Hz，Girls 睦使用 quality=cloth / 240 Hz，并非同条件对照。爱音主裙网格垂直长约 65.7 cm，当前睦约 37.8 cm；控制链、蒙皮、碰撞体不同。两者裙子相邻纵向节点间距都约 4.6–4.8 cm，不能简单将问题归因于节点密度。

Girls 减抖版本（仅前 30 秒）：`output/mutsumi_girls_physics_smooth/girls_jitter_before_after_30s.mp4` 左侧上一版，右侧减抖版。专用配置启用 `skirt_reduced_constraints`，移除重复两跳/纵向 DistanceJoint 与旧弦长防折约束，保留防拉长 D6 和原生弯曲角限位；杆件质量从 0.5 g 调整为 6 g，统一节点/杆件碰撞推出上限为 0.08 m/s，增加杆件线性和角阻尼，降低剪切与弯曲弹簧刚度。`results.json` 记录骨盆局部坐标下的时序指标：高频残差 RMS 降低约 55%，6–14 Hz 功率降低约 87%；该滤波仅用于诊断指标，没有对动作、物理缓存或视频做平滑后处理。仅验证了前 30 秒；其他舞蹈与原参考视频未改。

Girls 接触突跳版本：`output/mutsumi_girls_contact_cushion/girls_contact_before_after_30s.mp4` 左侧上一轮减抖版，右侧本轮最终 D 版。仅 900 帧/30 秒。原生球关节的 cone 通过 `PhysxLimitAPI` 改为隐式弹簧阻尼软限位：stiffness=0.002 torque/degree，damping=0.0001 torque*s/degree（SDK schema 使用角度单位）；长度 D6 硬上限保持。接触释放阻尼使用 25 ms 建立时间及 20 ms 法线平滑，切向比例 0.3。最坏节点加速度尖峰、最大高频残差均下降约 53%，同一腿部代理穿入指标不变；纵向 >90° 卷折采样从约 0.16% 到 0.54%，为柔性限位的弯曲容许量代价，仍可能有局部抖动/穿模。数据见 results.json。预接触缓冲与 480 Hz 的 A/B 实验未被选用，最终仍是 240 Hz；新增缓冲辅助保持可选且默认关闭。`girls_contact_detail_4x_slow.mp4` 是前 30 秒内 17.8–20.2 秒的局部放大、四倍慢放，仅用于观察。身体矩阵逐帧一致，原参考库视频未替换。

### 爱音复用睦接触参数的 Girls 试验

`output/anon_girls_mutsumi_physics/anon_girls_30s.mp4`：爱音原生 PMX/UsdSkel 上应用当前睦的柔性限位与渐进接触阻尼配置，仅 Girls 前 30 秒（900 帧/30 fps）。新入口 `physx_dance.py --secondary-profile` 在 BodyMotion 拓扑与蒙皮初始化之前载入配置；不调用睦专属的 rig-data 骨骼重映射。爱音保留原生 16×21 裙子网格和原裙长，派生 294 个柔性防折关节、609 个防拉长 D6 上限、315 个裙边胶囊，使用同样的 240 Hz、1.02 伸长上限。`secondary_profile.json` 保存实际参数快照，`model_audit.json` 验证身体动作、静态网格和 translations-only 模式，音视频完整解码及现有参考片 SHA256 检查通过。

### 当前默认与历史物理版本

历史记录（旧参数文件现已清理）：2026-11-14 用户确认 V10 头发优化成为默认。冻结快照 `configs/physics/snapshots/secondary_hair_contact_drape_v10_120hz.json`；升级前 V7 完整保存在 `configs/physics/snapshots/secondary_hairlength_v7_120hz.json`，另有本次字节备份 `output/secondary_default_v10_promotion/previous_default_v7.json`。V10 仅新增 `hair_constraint_solver`，原裙腰、裙子支撑、侧发、防拉长和 **120 Hz / cloth** 参数逐项保留。视觉依据为爱音 Girls 前 40 秒 V9/V10 对比；仍有瞬时残余穿入，其他角色未作完整舞蹈视觉复验。现有视频和缓存不覆盖，后续默认启动使用新参数。

V8 头发接触实验（未应用默认）：`hair_constraint_solver` 在原生 PhysX 子步后交替处理原始曲率、发段长度、身体代理、裙节点/边及非相邻发束接触，最后保留长度上限，回写刚体位置/速度。`iterations=24`、`bend_relaxation=.12` 的爱音 Girls 前 40 秒对比在 `output/anon_hair_contact_v8/`。身体穿入为骨骼代理指标，手掌盒使用原始胶囊近似，不能等同于可见三角网格穿透深度；仍有瞬时残余穿入。该历史实验阶段默认仍为 V7。

V9 自然垂落实验（未应用默认）：`hair_constraint_solver.velocity_mode=native_tangent` 保留原生速度，仅在发段接近长度上限时消除沿发链分离的相对速度，不再把位置排斥校正直接转换成向外速度。弯曲松弛使用 `.08`、迭代 48 次，启用 `contact_velocity=true`，对接触后的法向速度单独修正，保留切向滑动并限制法向反弹，使用 `.25` 的校正速度混合以平衡接触稳定性；长度上限保留。Girls 前 40 秒 V8/V9 对比位于 `output/anon_hair_drape_v9/`；裙子参数、动作、裙长、120 Hz 和外观保持不变。

V10（已批准为默认）：`contact_refine_iterations=24` 在主约束求解后追加不施加曲率回形的接触/长度收敛，主求解仍为 48 次，`bend_relaxation=.06`、`projection_velocity_blend=.15`。爱音 Girls 前 40 秒 V9/V10 对比保存于 `output/anon_hair_drape_v10/`，V10 参数已合入默认配置；此次升级不重录视频。

2026-11-14：V7 修复头发在交互后持续拉长。`hair_max_stretch=1.015` 按模型静态发骨长度创建原生限位，并在每个 120 Hz 子步后按根到梢顺序校正超长发段，回写 PhysX 位置与线速度；渲染和下一子步均使用校正状态。不是修改原网格长度或单独缩短显示骨骼。快照 `configs/physics/snapshots/secondary_hairlength_v7_120hz.json`；V6 裙腰、侧发与发裙碰撞参数保留。爱音 Girls 前 40 秒验证：后发整束最大长度比例约1.015，所有发段最大比例低于1.016，身体动作逐帧一致。仍可能瞬时压缩或发生代理网格穿模，不保证所有碰撞问题一并解决。对比在 `output/anon_hair_length_v7/`，原正式参考视频不覆盖。

历史：2026-11-14：用户暂定将爱音 V6 裙腰至裙身过渡支撑记录为默认，冻结快照为 `configs/physics/snapshots/secondary_fabric_transition_sidehair_v6_120hz.json`。`skirt_waist_support.shape_gain_by_grid_rows` 在原生网格解析后选用对应支撑曲线：当前 16 行爱音使用 `[1,12,12,11,8,5.5,3.5,2,1.25,1]`；9 行睦及未匹配网格继续用已确认的 V5 基础曲线。其余参数逐项保留，包括侧发、发裙碰撞、裙长约束和 **120 Hz**。视觉依据为 `output/anon_girls_waist_transition_v6/` 中 Girls 前 40 秒第二轮实验；平均外鼓改善温和、仍有残余外撑。此次仅记录并更新默认，不重录或覆盖既有视频。V5 及更早快照保留。

历史：2026-11-14：用户确认将爱音 V5 侧发优化加入默认模板，继续 **120 Hz / cloth**。当前冻结参数为 `configs/physics/snapshots/secondary_fabric_waist_sidehair_v5_120hz.json`，V4 快照继续保留。仅新增 `local_hair_support`，V4 裙子与发裙碰撞参数逐项保持不变；侧发只匹配 `rclh_ / lclh_ / rch_ / lch_`，不匹配这些名称的角色保持原有头发设置。依据为爱音 Girls 前 40 秒 V4/V5 对比；此次升级不重录或覆盖任何现有动作视频、缓存或验收参考记录。

历史：2026-11-13：用户确认将 V4 裙腰贴合、V3 裙料支撑/回落和 V2 发裙碰撞一起应用到默认模板，继续 **120 Hz / cloth**。冻结参数见 `configs/physics/snapshots/secondary_fabric_waist_v4_120hz.json`，上一版默认完整保存在 `configs/physics/snapshots/secondary_soft_contact_v1_120hz.json`。新默认不包含模型骨骼编号或网格/裙长变更；上部 `qun_` 行使用渐变半径、接触偏移与恢复力。视觉依据为睦 Girls 前 40 秒 V3/V4 对比，其他角色未作完整舞蹈视觉复验。此次只更新默认配置，不重录、替换或重新标记现有参考视频/缓存。

默认模板由 `secondary_profiles.py` 统一定位。`physx_dance.py`、`record_physx_video.py`、`record_mutsumi_references.py`、`play_anon_policy.py` 默认载入同一份参数；显式 secondary-profile 可选择旧配置，legacy-secondary-physics 可跳过公共模板。睦批录先合并参数到模型 rig，再禁用工作进程的二次默认覆盖，确保自定义或历史配置不被覆盖。爱音在线播放在拓扑/蒙皮初始化前加载参数，子步数量按实际 Hz/30 计算；默认依然 120 Hz、每帧 4 步。

历史主要版本依次包括：内置 legacy；Anon natural（5 mm，中等回形）；Anon drape（3 mm，自由垂落）；Anon hand_contacts（手掌与裙边交互，旧行走默认）；睦 short10_release（前发锁定、防拉长和接触释放）；Girls 直接移植版（卷裙，已淘汰）；Girls hard_bend（防卷折但抖动）；Girls reduced_jitter（减少重复约束、平衡杆件）；accepted soft_contact（柔性角限位、接触阻尼渐进启用）；旧 soft_contact_v1 默认（120 Hz）；旧 fabric_waist_v4 默认（发裙碰撞、增强布料支撑、渐变裙腰贴合）；旧 fabric_waist_sidehair_v5 默认（侧发优化）；旧 fabric_transition_sidehair_v6 默认（按网格适配长裙过渡）；旧 V7 增加发链防拉长；当前 V10 默认改进接触与自然垂落，120 Hz。具体路径和状态见 JSON 索引。旧 JSON 只是参数预设，历史质量档位、频率与加载阶段可能不同，历史效果以对应缓存和视频为准。缩裙 10/20/30% 是几何变体，不是独立的通用物理模板。

历史 soft_contact_v1 验证：4 项模板/网格测试；爱音与睦在当时默认 120 Hz 各运行 60 帧，物理步数均为 236（第 0 帧无推进）；爱音在线入口运行 6 帧，预热 120 步后每帧 4 步，共 144 步；原生录制入口两帧共 4 步。没有重新录制完整舞蹈，原有参考视频与 IRIS OUT 保留。此前 Girls 30 秒视觉验收使用 240 Hz，120 Hz 的本次检查仅为短测，不等同于完整舞蹈质量复验。

爱音 Girls 120/240 Hz 同外观对照：`output/anon_girls_120_240_bright/anon_girls_240_vs_120_bright_30s.mp4` 左侧按旧 240 Hz 物理参数重录，右侧为当前默认 120 Hz；两侧均为 soft-pbr 提亮外观、原版反射地面、同一镜头和前 30 秒动作资产，物理参数唯一差异为频率。两侧各 900 帧/30 fps，物理步数分别 7192/3596，第 0 帧不推进；完整解码、音频与相机一致性通过。`motion_geometry_audit.json` 验证剪辑层未修改原骨架动画及裙子静态网格。旧 240 Hz 原片和全局 120 Hz 默认均保持原样，验证信息见 verification.json。

### 睦完整 G1 验收参考库：120 Hz + 提亮外观

`output/mutsumi_validation_120hz_bright/` 保存本次七个非 IRIS OUT 动作的完整重录：你最重要、HAL 完整版、极乐净土、Girls、Galaxias、Deep Blue Town、HAL 短版。统一使用本次冻结的默认物理参数（120 Hz）、30 FPS / 720×900、soft-pbr 角色外观和原版地面；从 VMD 第 0 帧开始，身体动作与已确认版本逐帧核验一致，静态裙长仍为原版缩短 10%。IRIS OUT 的原 MP4、原配置及参考记录保持不变。逐首完整解码及音频检查后才替换 `media/dance/.../*_mutsumi_reference.mp4`，旧 MP4 与 sidecar 备份在本次目录 `previous/`。G1 `validation_videos.json` 及主参考库 manifest 已同步，8 个参考视频、24 个 H5 变体解析检查通过。Galaxias 沿用已有短音源，尾部静音补齐。

`recording_index.json` 为文件清单，`final_verification.json` 为总验证结果；各动作 `motion_appearance_audit.json` 包含动作、裙子几何、原地面及提亮灯光核验。`ji_le_jing_tu_G1_iter10000_comparison.mp4` 使用既有 iter10000 验收录像合成新版参考对照，原策略、原视频中的判定结果及评分没有改变。已生成的历史合成视频保持原内容，后续合成读取更新后的默认参考文件。批录缓存签名现包含外观、分辨率、帧率和参数内容，防止旧外观误跳过；IRIS OUT 保护优先于强制重录。

HAL 短版最终补充取景修正：原辅助骨骼包围范围令相机高度约 1.4 cm，低姿态时拍到地面侧面黑带。此片改为 `fit-camera-mesh`，按蒙皮顶点范围取景并将相机高度下限设为 0.55 m，保留原动作、原裙长、原地面及物理参数。1240 帧逐帧黑带扫描通过；其他六片取景和 IRIS OUT 均未改。批录器同时检查渲染成功标志，避免 Kit 脚本异常但退出码为零时误用旧报告。

### 当前睦参考库与标准回放

已验收的实时回放标准见 [PLAYBACK.md](PLAYBACK.md)，统一入口为 `tools/preview_dance.py --model mutsumi|anon --dance <舞蹈名> --resolution 1k|2k`。默认 V1（原 V10）、120 Hz、mmd-toon、原地面、并行物理、相机缓存、加载隐藏及两秒第0帧预热；预设位于 `configs/playback/mutsumi_deepbluetown.json`。支持现有6支舞蹈及各自音源，资产表位于 `configs/playback/mutsumi_dances.json`；完整实时性能验证基准仍为睦 Deep Blue Town。

现有 G1 参考库保留六支动作的原标准路径：`source/train_workflow/validation_videos.json` 与 `output/mutsumi_validation_references/manifest.json`。Galaxias 已退役，Deep Blue Town 音源已恢复音量；旧参考片备份保留于 `output/mutsumi_validation_v10_1080/previous/`。本次启动优化不重录或替换这些参考片。

当前通过的预热对比样片、验收数据保留在 `output/mutsumi_startup_v4/`。阶段性的性能实验和临时脚本进入 `output/archive/mutsumi_playback_history_*/`，清单包含原路径、新路径和 SHA-256。历史说明集中在 [archive/MUTSUMI_PLAYBACK_HISTORY.md](archive/MUTSUMI_PLAYBACK_HISTORY.md)，不再把各实验版混列为当前默认。
