# 配置分类

所有路径相对于 `source/pmx_prototype/`。根目录只保留布局索引 `module_locations.json`；运行参数文件全部放在本目录。

| 目录 | 用途 |
| --- | --- |
| `training/amp/` | AMP 训练与评估覆盖 |
| `training/continuous/` | 连续指令、相位和边界评估覆盖 |
| `training/v5/`、`training/v6/` | 历史版本训练/评估覆盖 |
| `training/clearance/` | 空间间隙学习覆盖 |
| `training/unified/` | 统一走跑的当前平衡低速配方及续训变体 |
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

## 统一走跑：低速覆盖与第二版验收

`training/unified/low_speed_balanced_v2.json` 是待训练的新配方；续训使用
`training/unified/low_speed_balanced_v2_resume.json`，保留 checkpoint 中已学习的探索标准差。
两者不会自动启动训练，也不改写已保存的策略。

环境池按编号互斥分配：后退20%、左右侧移25%、原地转向8%、原地连续换向7%、高速转向切换5%、
持续低速15%、低速行走/停步交替10%、通用命令10%。持续低速覆盖0.15–0.40 m/s，
包含离散速度锚点和连续采样，至少60%的方向选择明确为前进；80%的低速命令不带转向。
行走保持12–16秒，停步保持6–8秒。历史 `continuous_low_speed_fraction` 不作用于统一分支；
新配方使用显式的 `unified_low_speed_env_fraction` 和 `unified_low_stop_env_fraction`。

统一入口的新运行默认使用 `response_steady_v3`。复现历史结果时可显式使用
`response_steady_v2` 或 `legacy_v1`。以下响应和稳态规则自第二版开始沿用。

- 固定命令：先估算规划器达到目标附近的时间；再允许一个参考步态周期完成支撑交换及0.5秒确认，
  此余量至少1.5秒。总响应预算最少2秒、最多5秒，按参考而非被测策略表现预先确定。
  线速度使用居中1秒均值，角速度使用居中2秒均值，进入原跟踪容差并保持0.5秒才算响应完成。
  未改变命令的保持段不新增响应检查。
- 稳态窗口在预先确定的响应预算之后开始，至少观察2秒。稳态线速度、角速度及停步速度阈值不变。
  完整序列中的停步从5秒延长到7秒，总时长355秒；短筛查缺少稳态观察时标记为观察不足，必须进入完整验收。
- 连续变化命令同时报告原始命令误差与相对规划命令的跟踪误差；圆周方向变化另限制等效方向延迟不超过1秒。
  单调速度斜坡不能独立辨识延迟与速度偏差，不伪造其延迟判定。
- 零跌倒、姿态、空间避让、腾空及既有滑步/速度保持要求继续保留。观察不足不会直接判合格；
  筛查只能触发完整测量，不能替代完整验收。不同协议的训练验收记录不能混写。

第三版把头部门槛集中在起始站立、前走停步、左转停、右转停。其余停步继续计算头部朝向数据，
正式计分仍使用身体指标。完整验收在同一仿真场景中检查 0.15 和 0.40 m/s 两端，每档使用
四组初始步态相位和朝向并测试走停，合计八个条件，要求零跌倒。中间速度由原有 0.30 m/s
前走和走停段覆盖。低速结果写在完整报告的 `low_speed` 字段，轻量筛查沿用原场景。

统一走跑验收默认在数值结果写入后回放 `rollout.npz`，生成单策略 1080p、30 FPS 片段。
录像使用共享第 8 版外观和头发、裙摆物理。完整验收选取四个正式头部场景，轻量筛查使用已有片段。
单独启动训练时，从最终 checkpoint 执行一次现有轻量筛查。从零训练及续跑按原节奏验收。
完整验收出数值后暂停训练录制，失败的录像在训练进程退出后重试。轻量筛查只保存数值和轨迹。
录像中保留通过和失败的实测动作。结果位于验收目录的 `video_preview/`，重复验收使用独立录像目录。
`acceptance_timing.json` 分别记录数值验收与录像耗时，从零训练的每个训练目录另存 `acceptance_videos.json`。
只测性能或已自带验收编排的训练进程可使用 `--no-acceptance-video`。可通过
`diagnostics/record_gait_diagnostic.py --acceptance-report` 重录已有轨迹。

判定实现见 `training/locomotion_acceptance.py`，完整验收见 `diagnostics/evaluate_unified_walk_run.py`，
训练采样见 `training/low_speed_commands.py`。

## 左右侧跑镜像增强（尚未训练验收）

`training/unified/lateral_mirror_v1.json` 基于上述从零配方；
`lateral_mirror_v1_resume.json` 用于后续续训，不覆盖继承的探索标准差。
另备 `lateral_mirror_augmentation_only.json` 和 `lateral_mirror_loss_only.json`
用于分别检验数据增强和软约束。它们均不自动启动训练。

- `train_lateral_mirror` 接入 RSL-RL 原生 Symmetry；镜像只在原始输入上进行，之后使用模型原有归一化。
  当前动作、参考、未来参考、角速度、四元数、手腕以及接触标签均按解剖映射变换。
- PPO 增强及软损失仅覆盖请求、当前规划、0.1秒未来规划均为纯侧跑且已进入完整跑步参考的样本。
  `mirror_eligible` 是保存在 rollout 中的辅助标签，不进入 actor/critic；策略仍为248输入、31输出。
  其他运动保持原样，避免把非镜像的长动捕片段强制成同一个动作。
- 时钟使用运行时实际 `unified_contact_offsets` 的左右差；当前银行为-0.5625周期。
  完整连续动作、步间变化和未来参考均保留，不平均为短循环，不假定半周期。
- `train_mirror_augmentation` 将有效样本的镜像加入 PPO 小批次及训练归一化统计。
  `train_mirror_loss_coeff: 0.05` 是待短训对照的初值，尚不代表已验证的最优强度。
  约束作用于策略输出均值；推理时没有镜像动作替换、输出平均或驱动补偿。
- `amp_mirror_augmentation` 对专家和策略回放的原始四帧特征同时增强，保留时间顺序；
  上身相位奖励尺度也从原始及镜像参考共同统计，消除对应关节尺度偏差。
  AMP 奖励取原始/镜像判别分数的对称均值，保证复用镜像优势时不会携带原判别器的方向偏置；
  这增加一次判别器前向计算，实际训练速度需后续实测。该均值只用于奖励，不修改策略动作。
  加载旧 checkpoint 时仍完整继承其判别器及归一化状态，不进行输入重置。
- 原采样池、物理参数、避让、手腕和摆臂设置及验收门槛保持不变。

实现：`training/lateral_symmetry.py`。配对诊断：`diagnostics/probe_lateral_symmetry.py`，
`--verify-hooks` 在实际环境中检验增强、反向梯度及加载一致性，执行零次优化器更新。
诊断中的短时镜像配对动作仅用于排查物理/奖励差异，不是发布或验收策略。

### 仅镜像增强的从零训练准备

`training/unified/lateral_mirror_augmentation_scratch.json` 是从零训练配方：
启用 PPO/AMP 镜像增强和奖励对称化，`train_mirror_loss_coeff: 0.0`，
RSL-RL 的 `use_mirror_loss` 明确关闭；actor/critic/AMP 判别器随机初始化，优化器和策略观察统计从零开始，
AMP 特征统计从镜像动捕专家重新计算，不继承旧 checkpoint。
保留从零初始化的腿部探索标准差0.08、完整长动捕、低速池、31自由度手腕/摆臂和自主避让设置。

准备入口为 `training/prepare_mirror_scratch.py`。默认及 `--prepare` 都只写准备计划和配置，
不导入 Isaac/Kit、不构造模型、不启动训练进程；用户明确说开始后才能使用 `--start`。
准备目录为 `output/anon_locomotion/scratch_mirror_augmentation_v1`，
实际训练日志预留到 `logs/rsl_rl/anon_locomotion/scratch_mirror_augmentation_v1`。
启动前核对源码、配置、动捕/物理输入哈希，拒绝覆盖旧训练；不恢复已停止的短训对照。

计划使用8192并行、seed42、最多6000轮；每500轮轻量筛查、每1000轮完整验收。
`response_steady_v3` 将八组低速条件纳入完整验收，继续采用原身体、滑步和速度保持门槛。
记录首次通过轮次和训练用时，验收与初始化分别计时。准备阶段只生成计划和配置。
