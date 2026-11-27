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

第三版统一走跑训练默认用 `unified_backward_diagonal_speed_cap: 0.859` 限制后退和斜向目标的平移合成速度，
前进和纯侧向继续用 `max_speed`（当前配方为 1.20 m/s）。抽到超限目标时沿原方向缩放；此设置作用于训练采样。

统一入口的新运行默认使用 `response_steady_v3`，也支持 `response_steady_v2` 和 `legacy_v1`。
复现历史计分应使用报告对应的评分代码版本。以下规则适用于当前响应和稳态计分。

- 固定命令：先估算规划器达到目标附近的时间；再允许一个参考步态周期完成支撑交换及0.5秒确认，
  此余量至少1.5秒。规划器及线速度响应预算最少2秒、最多5秒，由参考动作预先确定。
  `response_steady_v3` 的停步段（含右转后的 `stand_end`）根部偏航确认预算为3.00秒，其余适用段落为2.10秒。线速度使用居中1秒均值，偏航角速度使用居中2秒均值，进入跟踪容差并保持0.5秒才算确认。
  `response_steady_v3` 的低速倒走线速度确认额外容许四个控制帧，30 Hz 下为约 0.133 秒。稳态计分从原线速度预算后开始。
  未改变命令的保持段不新增响应检查。
- 稳态窗口在线速度响应预算之后开始，至少观察2秒，使用原稳态线速度、角速度及停步速度阈值。
  稳态数据不足的段落标记为观察不足。
- 连续变化命令同时报告原始命令误差与相对规划命令的跟踪误差；圆周方向变化另限制等效方向延迟不超过1秒。
  单调速度斜坡不能独立辨识延迟与速度偏差，不伪造其延迟判定。
- 完整验收要求全程零跌倒，计分段的姿态、空间避让、腾空及滑步和速度保持等指标达到门槛，稳态数据充足。
  不同协议的训练验收记录应分别保存。

第三版的站立和停步仍采集头部偏航、俯仰及其稳态统计，正式计分使用身体指标。
常规验收中的低速动作由 0.30 m/s 前走及走停、0.40 m/s 倒走覆盖。
复现早期第三版的额外八组低速检查时可指定 `unified_eval_low_speed: true`。

第三版常规完整验收的起始站立与右转停稳各为 12 秒。达到常规完整门槛的候选追加右转后
40 秒长站复查。复现采用旧时长的报告时可指定 `unified_eval_stand_hold_seconds: 40`。

当前完整验收在停步处将 40 段分成四组，各角色独立推进并采集足底载荷。开场慢走和渐减速后的停步分别作为渐加速、下一次前跑的准备动作，保留遥测和录像。正式验收计分 38 段，准备段的跌倒计入全程。
八段跑步后的停步为 6.5 秒。慢走停步总长 5.5 秒，线速度响应预算 3 秒，偏航确认预算 3 秒，稳态计分 2.5 秒。其余停步沿用原时长。
后三组各预热 4 秒，并行推进 91.5 秒仿真，录制轨迹共 321.5 秒。报告保留分组边界和实际用时。
复现 323 秒或 327 秒的历史序列时，`unified_eval_stop_timing` 分别设为 `compact` 或 `original`。历史 40 段计分使用 `unified_eval_merge_preparation: false`。续训默认沿用验收计划记录的停步时长。
训练的 `acceptance_plan.json` 记录 `unified_eval_parallel_envs`，历史配置可沿用一路验收。
右转后长站复查沿用一条连续序列。录像按实测轨迹分组切镜。
四组完整验收默认先按原顺序计算头发和裙摆，再用 `recording/anon_acceptance_video.json` 中的四个录像进程并行渲染。
成片按原章节顺序拼接并解码核对帧数。分组录制或拼接失败时自动改用串行录像。

统一走跑验收默认在数值结果写入后回放 `rollout.npz`，生成单策略 1080p、30 FPS 视频。
录像使用共享第 8 版外观和头发、裙摆物理。完整验收按顺序录制全部准备和计分段，保留每段原时长和独立角色的轨迹边界。
从零训练、续训和单独训练默认每完成 1000 个迭代执行一次完整验收。
训练自然结束的 checkpoint 若未在间隔边界验收，结束时补一次完整验收。可通过 `--acceptance-interval` 调整训练间隔。

### 爱音抗扰续训

`training/unified/push_recovery_pilot.json` 用于从统一走跑 checkpoint 开新 run。用
`--checkpoint`、`--config-override` 和新的 `--run-dir` 启动 `train_anon_locomotion.py`。
约一半训练回合启用扰动。角色进入稳定指令后，腰部在随机方向受力一个 PhysX 子步，
两次受力间隔至少 4 秒。受力后的 1.2 秒内逐步恢复对动作风格与参考接触的奖励权重。
训练记录在 `status.json` 的 `push_recovery` 字段。后续从该 run 的 checkpoint
继续训练时自动读入同目录的课程配置，环境复位后重新采样受力时机。
所有普通验收和预览默认无扰动，可另用固定幅度、方向和步态相位的施力矩阵做独立验收。
第二轮配置 `training/unified/push_recovery_round2.json` 将挑战档提高到 0.125–0.175 m/s，并在后退和前走左转时提高该档的采样概率。
大推力补步配置 `training/unified/push_recovery_corrective_steps.json` 将主挑战档设为 0.22–0.40 m/s，并混入少量 0.40–0.55 m/s 样本。
较强受推后短暂放宽参考动作终止条件，奖励向补偿方向落脚，并在最初的失衡响应之后降低躯干倾斜与滚转俯仰速度。该配置沿用原跌倒线和无扰动走跑验收。
数值验收和录像期间暂停训练，失败的录像在训练进程退出后重试。
录像中保留通过和失败的实测动作。完整录像位于验收目录的 `video_full/`，诊断片段位于 `video_preview/`。
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
默认零速站姿采用已标定的掌缘和指尖间隙参考。当前动作库位于
`output/anon_locomotion/release_v7_unified_walk_run_idle_clearance/full_motion.npz`，同目录的 `reference.json` 记录来源和校验哈希。

准备入口为 `training/prepare_mirror_scratch.py`。默认及 `--prepare` 都只写准备计划和配置，
不导入 Isaac/Kit、不构造模型、不启动训练进程；用户明确说开始后才能使用 `--start`。
准备目录为 `output/anon_locomotion/scratch_mirror_augmentation_v1`，
实际训练日志预留到 `logs/rsl_rl/anon_locomotion/scratch_mirror_augmentation_v1`。
启动前核对源码、配置、动捕/物理输入哈希，拒绝覆盖旧训练；不恢复已停止的短训对照。

计划使用8192并行、seed42、最多6000轮，每1000轮完整验收。
`response_steady_v3` 继续采用原身体、滑步和速度保持门槛。
记录首次通过轮次和训练用时，验收与初始化分别计时。准备阶段只生成计划和配置。
