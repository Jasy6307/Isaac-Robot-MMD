# 爱音速度指令行走

独立于 G1 舞蹈训练。物理身体来自 `anon_physical.usda`，训练不加载蒙皮和贴图，
以原生 PhysX GPU articulation 批量仿真，通过 RSL-RL 5.4.1 的 PPO 学习。

## 代码与历史配置位置

训练、预览和策略检查仍使用原入口。历史 v5/v6 JSON 归档于 `archive/configs/`，使用文件按版本归入 `configs/training/v5/` 和 `configs/training/v6/`，旧项目路径参数由入口兼容解析，本文既有命令继续可用；索引见 `archive/catalog.json`。离线诊断实现位于 `diagnostics/`，通过分类路径或统一入口 `pmx.py` 调用，根目录旧转发入口已移除。

训练代码快照与相关发布入口通过 `source_bundle.copy_sources()` 补齐本地 Python 依赖，包括 v3/v4 的公共构建器和迁移脚本的实现。快照只读取源码，不导入或启动模拟器；旧 run 与已发布策略不改写。身体版本仍由 `full_body_reference` / `articulated_wrists` 选择，未改变观测、动作、奖励或 checkpoint 格式。

## 完整抗扰训练和验收

```powershell
python -u source\pmx_prototype\train_anon_push_full.py `
  --course mixed4 `
  --baseline "logs\rsl_rl\anon_locomotion\lab_direct_backward_start_continue_02409_to_10000_8192_20261124\scratch_010000.pt" `
  --num-envs 8192 `
  --output "output\anon_push_full\round15_mixed4"
```

四段课程共 5,102 次 PPO 迭代。每段启动时只读取一次配置，整段使用同一套采样范围和比例。阶段间从当次仿真采集并拟合走动和站立补步示范。底座需保留相邻的两份配置和其中引用的动作、AMP 和身体资产。

| 阶段 | PPO 迭代 | 内容 |
| --- | ---: | --- |
| 走跑抗扰 | 731 | 从无抗扰权重续训，混合轻推、挑战推和少量强推，增加倒走和左转样本。中间存档和段末权重用于生成移动补步示范。 |
| 腰腿纠偏 | 1,426 | 从拟合过的补步纠偏开始，冻结底座，固定混合轻推与强推。 |
| 站立承重 | 595 | 冻结走跑纠偏，学习站立前后及两侧受推后的补步和落脚承重。段末生成站立侧向补步示范。 |
| 站立侧推 | 2,350 | 从拟合过的站立侧向纠偏开始，混合左右方向、强度和四种推力波形，训练落脚承重及第二步制动。 |

四段覆盖参数见 `configs/training/unified/push_recovery_mixed4.json`。默认使用 8192 并行。加 `--check` 可检查输入、各段迭代和并行配置。加 `--no-video` 可省略结尾的配对录像。中断后用相同参数和输出目录加 `--resume` 继续，已完成的阶段由 `progress.json` 记录。

历史 26 段配方仍由 `--course replay` 启动，省略 `--course` 时沿用这个模式。配方见 `configs/training/unified/push_recovery_full.json`。可在该模式加 `--prefix-num-envs 2048` 复现早期六个打底阶段的并行数。

最终报告 `result.json` 写明权重、原走跑的 38 项验收、本次 0.20/0.525 m/s 恢复结果、强推方向网格、四种波形和并排视频路径。保底门槛包括走跑 38/38 且零跌倒、0.20 m/s 至少 833/896、0.525 m/s 至少 269/896，以及 0.70 m/s 前后推的保持。另以第十五轮实测的 0.20、0.525、0.70–0.80 m/s 和高冲量波形结果逐项比较。退出码 0 表示达到第十五轮基准，3 表示通过保底门槛但部分指标低于基准，2 表示保底门槛未通过。

旧 26 段和新四段共用 `training/push_pipeline/` 中的训练、验收和录像工具。完整重训的用时和成绩以各次运行的报告为准。

## 控制约定

指令为 `(forward, left, yaw)`：前方为角色局部 -Y，左方为局部 +X，正转向绕 Z 逆时针。
平移速度按圆盘采样，初期范围 0.05～0.12 m/s，700 次迭代逐步扩大至 0.4 m/s；
转向上限逐步扩大到 0.6 rad/s。约 20% 指令为停步、10% 为原地转向，其余包含任意方向平移。
每四秒换一次指令，训练回合最长 12 秒。

这是 **步态参考 + PPO 残差策略**。解析足部落点与 1.25 Hz 步态相位提供参考，
策略输出左右腿共 12 轴的关节位置残差；腰与手臂用固定姿态 PD 保持。
观测为机身速度、角速度、重力方向、指令、腿部关节位置/速度、上一动作和相位，共 50 维。
奖励包含速度/转向跟踪、直立与高度、足部接触节奏、抬脚高度、抑制滑步和动作变化。

仿真 120 Hz，控制 30 Hz。正常控制不改写根姿态、不施加辅助支撑力；只有回合重置写入根状态。
关节 PD、惯量、碰撞过滤沿用前置物理身体。当前只覆盖平地，不代表具备抗扰或复杂地形能力。

## 训练、续训与验收

```powershell
python source/pmx_prototype/train_anon_locomotion.py --num-envs 2048 --iterations 1200 --headless
```

指定 `--run-dir` 可以固定运行目录。日志位于 `logs/rsl_rl/anon_locomotion/`：
`env_config.json`、`agent_config.json`、代码快照、TensorBoard、每 100 次迭代 checkpoint、
`status.json`、最终 `policy_final.pt` 和 `policy_jit.pt`。

续训使用 `--checkpoint "实际 checkpoint 路径"`，同时给新 `--run-dir`。
checkpoint 所在目录必须保留 `env_config.json`；续训的 `--iterations` 表示增加的学习次数。

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode eval --num-envs 1 --checkpoint "实际 checkpoint 路径" --output output/anon_locomotion/evaluation --headless
```

验收连续切换站立、前进、后退、左右侧移、斜走、左右转向、原地转向和停步，
保存实际物理连杆姿态、关节、指令、测得速度、重置标记及分段误差。
验收回合不会因训练中的 12 秒超时而重置。`--zero-policy` 用于测量未学习残差的步态参考基线。

初版数值门槛：连续演示无跌倒，各移动段平均平移速度误差均 <0.10 m/s，
各段平均转向误差均 <0.15 rad/s，停步段平均速度 <0.08 m/s。
这些只是初步门槛，仍需结合转向误差和录像检查步态与滑步，不能只看奖励。

## 视频

```powershell
python source/pmx_prototype/record_anon_locomotion.py --rollout output/anon_locomotion/evaluation/rollout.npz --output output/anon_locomotion/anon_walking.mp4 --headless
```

视频用原始爱音蒙皮回放 **实际策略物理轨迹**，显示目标与实测速度。渲染不再积分物理，
不平滑或修正根运动，相机只跟随位置；触发安全重置会显式标注。
这是身体行走策略的展示，头发和裙摆仍随骨骼，未加入独立次级物理。
`--max-frames`、`--start-frame` 支持短预览；默认完整输出 30 FPS MP4。

## 已交付的 v1

验证后的范围先定为 **平移 0～0.25 m/s、转向 ±0.35 rad/s**。
0.4 m/s、±0.6 rad/s 是训练课程的扩展目标，尚不作为本版验收承诺。

- 策略：`output/anon_locomotion/release_v1/policy.pt`。
- 残差网络 JIT：`output/anon_locomotion/release_v1/residual_actor_jit.pt`。
- 完整演示：`output/anon_locomotion/anon_walking_v1.mp4`。
- 77 秒、16 个方向、左右行走转向/原地转向、多次停步的独立验收：`output/anon_locomotion/compass_200/report.json`。
- 结果：0 次跌倒，平均移动速度误差 0.0207 m/s，最差方向分段误差 0.0357 m/s，
  最大分段转向误差 0.0716 rad/s，停步段最高平均速度 0.0112 m/s。
- 选用 `model_200.pt`：201 次 PPO 更新，2048 环境，19,759,104 条转移。
  首版达到验收要求后停止训练；暂定的 1200 次上限没有耗尽，选用经过完整验收的策略而非最后一次更新。

完整复验命令：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode eval --num-envs 1 --checkpoint output/anon_locomotion/release_v1/policy.pt --suite compass --linear-speed 0.25 --yaw-speed 0.35 --output output/anon_locomotion/recheck_v1 --headless
```

JIT 输入为本环境定义的 50 维观测，输出为 12 维残差动作；仍需按环境约定裁剪到 [-2,2]，
乘以 0.18 rad，再加解析步态参考并交给关节 PD。它不是直接输出力矩或身体根速度的网络。
release 中的 Python 文件是原始训练代码快照；日常运行使用项目 `source/pmx_prototype/` 下的当前入口。
训练目录放置 `stop.request` 文件，可在下一次更新结束时保存最终 checkpoint 并退出。

## v2：速度扩展与协调摆臂

v2 的目标范围为任意方向平移 0～0.4 m/s、转向 ±0.6 rad/s。从 v1 权重继续 PPO 训练，
保留 50 维观测和 12 维腿部残差输出。`--v2` 启用以下控制/奖励设置，并写入 checkpoint 旁的环境配置：

- 肩部按步态相位和前后速度交替摆动，肘部轻微弯曲；摆动指令通过物理关节 PD 执行。
  腿部策略在这些真实的手臂动力学影响下续训。手臂轨迹属于相位协调参考，并非新增的神经网络动作维度。
- 步态和摆臂使用平滑后的速度指令，30 Hz 下每步插值 0.15，减少起停和变向时的参考突变。
- 随侧向速度适度扩大落脚间距，避免大步侧走时双脚相互挤压。
- 增加支撑脚倾斜、滑动与动作变化的代价；用实际骨盆高度约束姿态，避免 PhysX 自动选择的根连杆造成高度含义歧义。

从 v1 启动新训练的命令：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --checkpoint output/anon_locomotion/release_v1/policy.pt --v2 --curriculum-iterations 400 --iterations 700 --num-envs 2048 --run-dir logs/rsl_rl/anon_locomotion/new_v2 --headless
```

验收上限时使用 `--suite compass --linear-speed 0.4 --yaw-speed 0.6`。
`audit_anon_gait.py` 可从实际物理轨迹计算支撑脚滑动/倾斜、足部离地高度、腿部加速度、骨盆起伏和双臂摆幅/相关性。
对照自然度时，必须使用相同速度和指令序列；高速结果不直接与 v1 的低速结果比较。

### 已交付的 v2

- 模型与配置：`output/anon_locomotion/release_v2/`，选用全局编号 500 的 checkpoint。
- 演示：`output/anon_locomotion/anon_walking_v2.mp4`，77 秒、1280×720、30 FPS。
- 上限验收：16 个方向均以 0.4 m/s 发指令，左右行走转向与原地转向均测试 ±0.6 rad/s，
  连续运行 77 秒，0 次跌倒。平均平移误差 0.0220 m/s，最差分段 0.0508 m/s，
  最大分段转向误差 0.0785 rad/s，停步段最高平均速度 0.0280 m/s。
- 同速对照：在 v1 的 0.25 m/s、±0.35 rad/s 序列下，骨盆上下起伏减少约 17.5%，
  腿部加速度 RMS 减少约 4.5%；肩部前后摆幅约 18°，左右臂相关性 -0.967，呈交替摆动。
- 几何接触点估算的滑动指标未改善：双脚均值约从 0.0444 增至 0.0624 m/s。
  因此本版仍有轻微滑步，不能将自然度改进理解为所有步态指标都优于 v1。
- 手臂摆动在训练和验收中真实参与身体动力学，录像回放没有额外添加或修正摆臂。

复验 v2 上限：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode eval --num-envs 1 --checkpoint output/anon_locomotion/release_v2/policy.pt --suite compass --linear-speed 0.4 --yaw-speed 0.6 --output output/anon_locomotion/recheck_v2 --headless
```

继续训练 v2：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --checkpoint output/anon_locomotion/release_v2/policy.pt --num-envs 2048 --iterations 500 --run-dir logs/rsl_rl/anon_locomotion/next_v2 --headless
```

无需再传 `--v2`；checkpoint 旁的 `env_config.json` 已保存控制版本与全部相关设置。

## v3：程序全身步态参考与学习

参考来源为本项目编写的程序曲线与解剖骨长 IK，参数见 `natural_walk_reference.py`。
参考包含步频随速度变化、平滑抬脚、腰部小幅协调转动、屈肘摆臂，以及颈部对躯干运动的部分补偿。
它是人工设计的运动参考；当前采用 PPO 加上身姿态/速度参考奖励进行训练。

### 身体与策略接口

- 独立身体资产：`output/anon_physical_v3/anon_physical.usda`，29 个刚体、25 个驱动自由度。
- 新增颈部 yaw/pitch 两轴；腰、肩、肘采用较柔和的 PD。旧版资产保持独立。
- 策略输出：腿 12 + 腰 3 + 颈 2 + 双臂 8 = 25 维；上身在参考曲线附近学习有界残差。
- 观测 89 维：保留 v2 前 50 个通道，再加入上身关节位置、速度和上一动作。
- 扩维迁移保留 v2 腿部输出，新增输入权重和上身动作均值从零开始；扩维张量使用新的优化器状态。
  `migration.json` 记录腿部输出一致性检查。
- 速度/转向、站立和足部奖励继续保留，上身参考奖励不要求身体强行逐帧复制参考。

### 参考动画与物理策略

`--mode reference --v3` 导出参考关节姿态，通过 FK 生成预览。预览根运动按指令生成，
不积分身体动力学，不能把参考动画视为策略已能保持平衡的证明。
`--mode eval` 导出的则是实际闭环物理策略轨迹；视频和元数据分别标注这两种结果。

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode reference --v3 --num-envs 1 --output output/anon_locomotion/v3_procedural_reference --headless
python source/pmx_prototype/train_anon_locomotion.py --checkpoint output/anon_locomotion/release_v2/policy.pt --v3 --iterations 600 --num-envs 2048 --run-dir logs/rsl_rl/anon_locomotion/new_v3 --headless
```

参考动画输出为 `output/anon_locomotion/anon_v3_reference.mp4`。参考本身包含站立、前后左右行走、转向和停步，
共 29 秒；已检查 25 轴姿态全部有限且没有超出关节限位。

### 已交付的 v3

策略包位于 `output/anon_locomotion/release_v3/`，使用全局编号 800 的 checkpoint。
完整物理验收：0.4 m/s 的 16 个方向、±0.6 rad/s 的双向转向和多次停步，77 秒、0 次跌倒；
平均平移误差 0.0187 m/s，最大分段平移误差 0.0632 m/s，最大分段转向误差 0.0840 rad/s。

同指令与 v2 对照：骨盆起伏约减少 19%，腿部加速度 RMS 约减少 3%，几何接触滑动估计约减少 5%。
腰部 yaw 活动范围约 8°，颈部 yaw 约 7°，双臂前后摆幅约 36～40°。
头部倾斜的第 95 百分位约 8.2°，略高于固定颈部 v2 的 7.5°；上身活动更丰富不代表每项稳定性指标都更小。

- 程序参考动画：`output/anon_locomotion/anon_v3_reference.mp4`，29 秒，运动学预览。
- 训练后策略效果：`output/anon_locomotion/anon_walking_v3.mp4`，77 秒，真实物理闭环轨迹。
- 参考数据：`output/anon_locomotion/v3_procedural_reference/rollout.npz`。
- 策略验收数据：`output/anon_locomotion/v3_compass_800/rollout.npz`。

复验：

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode eval --num-envs 1 --checkpoint output/anon_locomotion/release_v3/policy.pt --suite compass --linear-speed 0.4 --yaw-speed 0.6 --output output/anon_locomotion/recheck_v3 --headless
```

复验/续训时直接加载 checkpoint 旁的配置即可，无需再次加 `--v3`。当前版本仍有轻微滑步，且未接入头发与裙摆的独立次级物理。

## v4：100STYLE Neutral 真人动捕参考

数据作者：Ian Mason、Sebastian Starke、Taku Komura。官方来源 https://www.ianxmason.com/100style/ ，
数据记录 https://doi.org/10.5281/zenodo.8127870 ，许可证 CC BY 4.0。
本项目修改包括坐标变换、骨架比例和关节限位适配、周期提取、低通处理、静态姿态校准与方向混合。
原始 BVH、裁剪表、下载地址及 SHA256 保存在 `media/mocap/100style/`。

`download_100style.py` 下载 FW/BW/SW/ID/TR1 五个 Neutral 片段；
`retarget_100style.py` 根据官方 Frame_Cuts 去除准备动作，匹配真人肢体朝向与爱音关节链，
保留原始根运动生成运动学预览。`mocap_walk_reference.py` 从 FW/BW/SW 提取完整周期，
排除周期内根 yaw 变化超过 0.6 rad 的片段，保留前四阶傅里叶分量以闭合循环；
用 ID 校准颈部静态偏置和躯干倾斜。TR1 目前仅作为原始重定向预览，未用于训练库。

训练继续使用 PPO：上身采用动捕目标的姿态和速度奖励，腿部新增低权重动捕姿态奖励；
腿部的基础足端 IK、速度指令和平衡/接触奖励继续参与控制。
这是动捕姿态参考训练，不是 AMP 判别器。原始动捕速度约 0.4～0.9 m/s，
训练通过周期重采样、速度相关幅度混合和物理策略适应 0～0.4 m/s；不直接复制根位移。
左右侧移通过关节镜像适配，前后/侧向按指令混合。训练命令通过 `--mocap-library` 指定参考库，
使用 `--run-dir` 时会冻结复制一份，后续修改素材不影响当前 run。

```powershell
python source/pmx_prototype/tools/download_100style.py
python source/pmx_prototype/motion/retarget_100style.py
python source/pmx_prototype/motion/mocap_walk_reference.py
python source/pmx_prototype/train_anon_locomotion.py --checkpoint output/anon_locomotion/release_v3/policy.pt --mocap-library output/anon_locomotion/100style_reference/library.npz --iterations 300 --num-envs 2048 --run-dir logs/rsl_rl/anon_locomotion/new_100style --headless
```

`anon_100style_reference.mp4` 是五段真人动捕重定向预览，视频明确标注非物理策略结果。

### 已完成的 100STYLE 试验版

- 策略包：`output/anon_locomotion/release_v4_100style/`，选择 checkpoint 1000。
- 从 v3 的 800 续训，最终选择增加 200 个更新的版本；本轮共运行至 1030，约 536 秒。
- 动捕预览：`output/anon_locomotion/anon_100style_reference.mp4`，40 秒；已按 ID 做中立姿态校准，与训练库一致。
- 物理策略录像：`output/anon_locomotion/anon_walking_v4_100style.mp4`，77 秒。
- 上限验收：16 个方向 0.4 m/s、转向 ±0.6 rad/s、多次停步，0 次跌倒。
  平均平移误差 0.0211 m/s，最差分段 0.0354 m/s，最大分段转向误差 0.0748 rad/s，
  停步最高分段平均速度 0.0136 m/s。
- 与相同指令的 v3 对比：双脚几何接触滑动估计平均约从 0.0554 降至 0.0483 m/s；
  头部角速度 RMS 约减少 9.3%。腿部加速度 RMS 约增加 3.6%，头部倾斜 P95 从 8.19° 增至 8.84°。
  仍有轻微滑步，后退/侧移时转头较明显，不能将单一指标改善等同于自然度已经全面解决。
- JIT 导出对 64 组 89 维输入与 checkpoint 的 25 维残差输出完全一致。

## v5：鞋底滑动与腿部连续性优化

基于 100STYLE v4 续训，配置入口为 `configs/training/v5/anon_v5_config.json`，通过 `--config-override` 加载。
所有新增行为由配置开关控制，旧策略配置继续使用原有路径。

- 采用实际鞋底八角点的相邻帧世界坐标位移，在连续近地时惩罚最低角点的水平滑动。
  这一测量与离线滑步审计一致。PhysX `get_link_velocities()` 返回质心速度，不能直接当作踝部原点速度。
- 足端摆动段采用五次多项式，起止位置、速度、加速度与支撑段一致；
  抬脚曲线使用 `64*s^3*(1-s)^3`，边界速度和加速度均为零。
  低速与停步时逐渐收小抬脚高度。
- 腿部残差动作做轻度一阶滤波（alpha 0.65），加入关节加速度惩罚。
  滤波参与训练和推理，视频回放不额外平滑动作。
- 验收继续使用 77 秒、0.4 m/s、±0.6 rad/s 的同一序列，并同时比较接触比例、鞋底滑动和腿部加速度。

```powershell
python source/pmx_prototype/train_anon_locomotion.py --checkpoint output/anon_locomotion/release_v4_100style/policy.pt --config-override source/pmx_prototype/configs/training/v5/anon_v5_config.json --iterations 400 --num-envs 2048 --run-dir logs/rsl_rl/anon_locomotion/new_v5 --headless
```

### 已完成的 v5

选择 `v5_flow_matched/model_1200.pt`，策略包 `output/anon_locomotion/release_v5/`。
先从 v4 的 1000 训练至 1100 学习防滑；再使用 `configs/training/v5/anon_v5_flow_config.json` 训练至 1200，
将抬脚高度降至 0.038 m、腿部滤波 alpha 调为 0.55、加速度惩罚权重调为 0.8，
并使离地高度奖励与平滑足端轨迹一致。

同一 77 秒验收，0 次跌倒；双脚持续近地滑动估计平均从 0.0483 降至 0.0125 m/s（约 -74%），
腿部加速度 RMS 从 18.19 降至 16.25 rad/s²（约 -11%），骨盆起伏从 0.0188 降至 0.0158 m（约 -16%）。
最大分段平移误差 0.0290 m/s，最大分段转向误差 0.0545 rad/s，停步最高分段平均速度 0.0152 m/s。
移动时近地接触比例仍约 77%，速度跟踪正常。
头部角速度 RMS 略增，本轮的改善重点为脚部和腿部；详细限制见策略包 README。
新录像为 `output/anon_locomotion/anon_walking_v5.mp4`，回放实际物理轨迹，不额外修正动作。

## 实时策略播放：默认开启头发与裙摆物理

新增入口 `play_anon_policy.py`，默认加载 v5，默认开启原生 PhysX 头发与裙摆次级物理。
窗口中的 Forward / Left / Yaw 滑块控制前后、侧向、转向，Stop 按钮停步；
平移向量长度限制为 0.4 m/s，转向限制为 ±0.6 rad/s。

```powershell
python source/pmx_prototype/play_anon_policy.py
```

策略每一步现场推理，身体在独立 GPU PhysX 场景中闭环运行；
实际身体刚体姿态传给显示进程，驱动发根、裙腰以及 PMX 身体碰撞代理。
头发和裙摆在 CPU PhysX 场景中以 1/120 s 固定步长模拟，使用已有 `hair` 质量配置，
包含 465 个动态节点、弹性/距离约束、身体/地面碰撞和头发自碰撞。
裙摆也参与动力学与身体碰撞；该配置不启用裙布自碰撞或发裙互碰。
蒙皮由在线物理结果更新，没有读取旧的行走或头发/裙摆缓存。

这里采用单向耦合：外观物理随身体运动，不将其接触反作用力反馈给 RL 身体。
因此本次没有重训策略，也没有改变 25 维动作接口；这也不是完整三角形布料或 Bullet PMX 求解器的逐项复刻。
物理步长与视频时间步固定，不代表墙钟播放一定达到 30 FPS，实际速度取决于计算和渲染开销。
完整基准会把各阶段耗时写入输出目录的 `report.json`。

自动展示前进、后退、侧移、转向和停步（37 秒模拟时间）：

```powershell
python source/pmx_prototype/play_anon_policy.py --demo
```

现场推理并录制：

```powershell
python source/pmx_prototype/play_anon_policy.py --headless --demo --record-video output/anon_locomotion/anon_walking_v5_secondary.mp4 --output output/anon_locomotion/live_secondary_demo
```

关闭外观物理做对照时加 `--no-secondary-physics`。关闭窗口后，入口会结束其策略子进程。
此功能由运行时创建，直接打开旧的 `anon_physical.usda` 不会自动启动该 Python 求解流程。

本机完成的在线验收：37 秒前后左右行走、转向和停步，1110 个策略/显示帧，0 次跌倒；
465 个动态节点，含预热共 4560 个次级物理子步，身体碰撞持续有效，完整视频通过解码检查。
推理与当前帧的次级物理/渲染重叠执行，但每个身体状态仍按顺序消费，不跳帧。
本次含录制的墙钟耗时约 58 秒（约 19.1 显示帧/秒），视频按 30 FPS 的模拟时间输出；
这不代表交互模式已保证墙钟 30 FPS。
结果见 `output/anon_locomotion/live_secondary_demo/report.json` 与 `video_verification.json`。

## 训练耗时汇报

### 头发与裙摆外观更新

最新默认策略为 `release_v6_continuous/policy.pt`，支持连续方向、连续速度（平移模长 ≤0.4 m/s）及转向 ±0.6 rad/s。
连续全身动捕参考按接触相位校准混合，命令参考平滑响应，自主避让保留，驱动补偿关闭。
140 秒主验收与另起 68 秒双上限边界复查均零跌倒并通过；79 秒完整物理在线演示零跌倒。
视频 `output/anon_locomotion/anon_continuous_1080p.mp4` 为原生 1920×1080、TAA、CRF 16、30 FPS。
本轮 8192 并行的四阶段训练合计 34 分 45 秒，V6 累计约 2 小时 26 分 12 秒，含试验，排除验收和录像。
完整方法、误差门槛和播放指令见最新发布目录 README.md。

上一版策略为 `release_v6_learned_clearance/policy.pt`，通过训练奖励学习近髋避让。
8192 并行测试通过，本轮（含短测试）9 分 52 秒，V6 累计约 1 小时 51 分 27 秒。
训练/验收无驱动补偿，63 秒控制验收、66 秒带头发/裙摆物理录制均零跌倒。
运行时肩部补偿代码已删除；learn_arm_clearance 仅控制训练中的空间避让奖励。
手部近髋平均外移约 4.4～4.6 cm，前后摆幅基本保留；完整指标及范围见最新发布目录 README.md。
对比视频 `output/anon_locomotion/anon_learned_clearance_comparison.mp4`。

历史摆臂避让曾采用推理阶段驱动补偿；该配置文件和开关现已删除，下面仅保留历史实验说明。策略权重/JIT 均未重训或修改，单独调用原策略/JIT 不包含此补偿。
根据手相对骨盆的位置，仅在近髋窗口弱化肩部 roll 的策略残差并混合完整动捕参考，
叠加最多约 6° 的平滑参考偏移，偏移变化率不超过 0.25 rad/s。
6° 仅指显式参考偏移，并非相对原策略电机目标的总变化上限；残差混合也会改变目标。
其余关节目标、前后摆臂参考和手裙碰撞保留，关节由真实 PhysX 动力学执行。
63 秒连续控制验收通过、零跌倒，66 秒含头发/裙摆物理录制零跌倒、全片解码通过。
前进 5～13 秒测量，近髋手部外移左右各约 2.4 cm；前后摆幅左 19.2→19.1 cm、右 15.7→17.3 cm。
对比视频：`output/anon_locomotion/anon_arm_clearance_comparison.mp4`，明确标注驱动补偿、未重训。
轨迹与验收记录：`output/anon_locomotion/arm_clearance_final/verification.json` 和
`output/anon_locomotion/arm_clearance_final_control/report.json`。

当前默认配置为 `configs/physics/presets/anon_secondary_hand_contacts.json`。在垂落版本上，仅恢复手部/前臂与裙摆碰撞并补齐覆盖：
原 PMX 的左右前臂、手首代理（索引 5、6、9、10）屏蔽了裙摆组，现对裙摆定向解除过滤。
左右手部使用原模型蒙皮顶点拟合的定向盒，随手首旋转，替换原来的粗胶囊。
纵向布条补入 315 根半径 3 mm 的原生 PhysX 动态胶囊，以球关节连接原节点；
只与手部/前臂开启身体碰撞，内部及其余身体代理过滤，避免重新扩大整条裙子的碰撞厚度。
每根胶囊质量 0.5 g，不改变 RL 身体、动作、摆臂轨迹或策略；属于单向外观物理。
66 秒在线测试零跌倒，7599 个子步记录到新增碰撞段与手部的实际接触（含预热），全片解码通过。
对比视频：`output/anon_locomotion/anon_hand_skirt_comparison.mp4`。
记录：`output/anon_locomotion/hand_contacts_final/report.json`、`verification.json`。
这些是骨骼节点与边段代理，片内尚非逐三角形连续布料碰撞，不保证所有局部穿插都消失。
上一版对照可通过 `--secondary-profile source/pmx_prototype/configs/physics/presets/anon_secondary_drape.json` 加载。

之前的垂落优化使用 `configs/physics/presets/anon_secondary_drape.json`，更早的 `configs/physics/presets/anon_secondary_natural.json` 也保留供对照。
移除同一网格隔两段节点的弦长约束，改用软弯曲力；横向/斜向约束允许收拢，保留长度上限。
后部长发与裙摆减少撑形及阻尼，前发单独保留造型支撑；节点半径 3 mm，身体碰撞代理半径为原值 90%。
66 秒在线验证完成（1980 帧、零跌倒），检查完整抽帧及匹配侧面画面，视频全片解码通过。
本轮对照为 `output/anon_locomotion/anon_secondary_drape_comparison.mp4`，左侧是上一版优化，右侧是本轮。
最终配置与物理记录见 `output/anon_locomotion/secondary_drape_final/report.json`。

上一轮的 `configs/physics/presets/anon_secondary_natural.json`：裙摆碰撞球半径从 18 mm 降到 5 mm，
头发从约 8～10 mm 降到 5 mm。裙摆横向和斜向允许有限剪切，纵向长度约束保留；
降低弯曲/撑形力与阻尼，裙摆蒙皮保留原骨骼朝向并应用物理位移，减少相邻布片旋开。
这属于骨骼节点代理的外观改进，未替换为三角形布料；原模型的分片、开口仍保留。

66 秒在线演示（1980 帧）零跌倒、465 个动态节点，视频解码及抽帧检查通过。
新视频为 `output/anon_locomotion/anon_secondary_natural.mp4`；
同策略同动作序列的前后并排对照为 `output/anon_locomotion/anon_secondary_before_after.mp4`。
记录中保存实际生效配置，见 `output/anon_locomotion/secondary_natural/report.json`。
对照旧配置时加 `--legacy-secondary-physics`，自定义配置用 `--secondary-profile 路径.json`。
本次没有重训策略。

V6 完整动捕版本发布于 `output/anon_locomotion/release_v6_full_motion/`。
采用全身关节、骨盆、躯干与接触时序参考，206 维输入、25 维动作。
选择 model_1900 并使用 1.5 秒参考过渡，63 秒连续验收零跌倒；
验证范围为前后 0.4 m/s、左右 0.35 m/s、转向 ±0.6 rad/s 及停步。
当前按参考库最近命令选择动作，尚未实现任意连续方向插值。
更高速度 0.6～0.79 m/s 尚未通过完整连续验收。
8 轮 V6 训练（含未选实验）累计约 1 小时 41 分 34 秒，不含准备、验收和录像。
完整指标、逐轮耗时、播放入口和限制见发布目录 README.md。

每次训练完成都应报告训练累计耗时，多次续训与本轮试验需要合并计入总数。
数据准备、环境启动、验收与录像时间单独说明，不混称为训练时间。
新 run 完成时会写入 `training_time.json`；旧 run 可用最后的 `status.json.elapsed_seconds` 估算，
并注明其不含最终导出、环境启动等开销。

## AMP 全身动作先验：自然前进实验版

新增 `output/anon_locomotion/release_v6_amp_forward/`，8192 并行，选择 model_2500。
使用 100STYLE 两个稳定前进周期及平均参考训练 4 帧全身对抗先验，放宽上身参考与刚度，
通过髋部附近手腕空间奖励学习避让，运行时手臂补偿关闭。仍采用动捕参考加残差 PD 控制。
本阶段范围为 0.4 m/s 前进与站立、起停，不替换此前任意方向连续策略。
67 秒验收零跌倒，前进平均 0.4072 m/s；腰胸反向配合与腰臂协调改善，上半身高频成分下降。
37 秒同命令、同提亮柔光、同头发裙摆物理的 1080p 对比视频：
`output/anon_locomotion/anon_amp_forward_comparison_1080p.mp4`。
本轮含短测与未选迭代共约 18 分 6 秒；V6 累计约 2 小时 44 分 18 秒，
不含准备、启动、验收与录像，早期四轮为日志估算。
完整方法、指标、限制与复现入口见该发布目录的 README.md。

## 自然度排查与参考/避让兼容实验

证据与两段侧面、背面三组对比见 `output/anon_locomotion/naturalness_diagnostic/README.md`。
原始动捕的主要手臂几何经重定向保留，但平均短周期减少步间变化；参考手腕在髋部附近
与空间避让目标存在约 18 mm 的冲突。对前进与站立参考做小幅服装感知手臂 IK 适配，
同步适配 AMP 专家，保留腿部、骨盆、躯干和接触时序，减少上身探索噪声并加入相位奖励。
修复过程中补上站立/停步偏航验收；未通过的中间模型保留拒绝记录。
最终 `v6_amp_garment_idle/policy_final.pt` 通过 67 秒控制验收、零跌倒，仍关闭运行时补偿。
这是整体观感变化有限的局部修复实验，不宣称明显真人自然度，不改全方向策略默认入口。
本轮四次训练共约 14 分 15 秒，V6 累计约 2 小时 58 分 33 秒，不含准备、验收与录像。
录像使用最新共享第 8 版外观和头发/裙摆物理，1920×1080、30 fps。

## 连续六周期动捕参考

最新前进实验见 `output/anon_locomotion/continuous_mocap/README.md`。
使用 100STYLE Neutral_FW 连续 8.20 秒、六个步态周期，适配到 0.4 m/s 后为 10.371 秒，
保留不同周期间的姿态与时序差异，不再跨周期平均。原路径转向归一化、足底接触和裙摆适配
写入参考，仅末端 0.25 秒做循环衔接；AMP 专家采样避开接缝。
新增参考步态相位编码，保持 206 维输入与 25 维动作，8192 并行、无驱动补偿。
选择 `v6_contiguous_long/policy_final.pt`，97 秒验收零跌倒，前进均速 0.4013 m/s；
真实关节的周期差异与参考相关性 0.921，近地足底滑移代理值约为上一版的一半。
24 秒侧背面三组 1080p 视频保存在该目录，使用第 8 版外观与头发/裙摆物理。
本轮三次训练约 15 分 32 秒；V6 累计约 3 小时 14 分 4 秒，不含准备、验收与录像。
仍是前进与起停实验，不改全方向策略默认入口；模型运行需要本轮的长参考和相位编码支持。


### 长参考脚高换算问题（2026-11-13）

跑步准备时发现 `build_continuous_walk.py` 将约 49.4 mm 骨盆/髋部校准量误加到踝部目标高度。源码已修正。以上连续六周期走路的控制指标属于历史实测，但参考动作质量验收撤回；其外观不能作为原始动捕质量的证据。当前跑步使用 `running_mocap/corrected/` 修正参考，详见该目录上一级 README。


## 100STYLE 前进慢跑实验

`output/anon_locomotion/running_mocap/README.md` 记录 Neutral_FR 连续4周期参考与修正后的踝高换算、离线支撑适配。8192并行，最终 `v6_run_stop/policy_final.pt` 通过67秒起停/持续跑验收、零跌倒，均速1.2138m/s，地面力腾空约4.30%。仅固定1.2m/s前进慢跑，未替换全方向走路入口。标准PD含参考腿部速度目标，浮动根无辅助力，运行时摆臂补偿关闭。腕部固定，手掌朝向仍有限制。1080p在线演示 `output/anon_locomotion/anon_running_1080p.mp4`，共享外观08及头发裙摆物理。本轮含失败尝试31分32秒，V6累计约3小时45分37秒，不含准备、验收、录像。


## 慢跑手臂与手腕重定向（31 DOF）

独立 `output/anon_physical_v4/` 为每侧增加前臂旋转和腕部两轴活动，保留旧v3模型；31动作、248维观测。最终参考对肘部/手腕做约2.5/4.5cm平滑外移，保留骨盆、腿部、躯干和时序。新关节采用捕获的前臂/手腕旋转和静止骨架轴向校准。8192并行，完整全身AMP、手掌朝向与手臂间距奖励，旧驱动避让补偿关闭；名义PD包含腿部、前臂和腕部参考速度，浮动根无辅助力。

`v6_run_wrists_refine/policy_final.pt` 通过67秒起停和持续慢跑验收、零跌倒，跑速1.2134m/s。实测肘部横向中位数左/右20.13/19.57cm，手腕13.34/12.47cm，手掌参考跟踪误差约2.90°。仍是低腾空1.2m/s前进慢跑，不宣称全方向跑步或自然手指抓握。详细方法、实测和录像见 `output/anon_locomotion/running_wrists/README.md`。本轮包括中间参考尝试14分31秒，V6累计约4小时00分08秒，不含准备、验收、录像。


## Isaac Lab Direct 训练与验证（2026-11-22）

训练、验收和策略预览统一使用 Isaac Lab Direct。通用入口是 `train_anon_locomotion.py`，`train_anon_lab.py` 继续转发相同参数。旧 checkpoint 仍读取同目录的环境配置。纯程序参考和全身动作参考均由 Lab Direct 步进，旧配置里的后端字段随加载归一为 `lab-direct`。

这条路径继承真正的 `DirectRLEnv`，由基类 `step()` 管理物理子步、终止和复位，并使用官方 `RslRlVecEnvWrapper`。原控制器拆分为准备动作、写入 PD 目标、子步采样、奖励计算、复位和观测阶段；Lab 不调用旧环境的完整 `step()`。为保持现有参考数据语义，控制与评分继续使用原生 PhysX Tensor 的 XYZW 顺序和关节名映射。保留 120 Hz 物理、30 Hz 控制、31 DOF/248 观测、参考速度前馈、AMP 和原奖励尺度；奖励仍只乘一次 dt。

目前支持 full-body imitation 配置（当前统一走跑使用 31 DOF）；不把旧的纯程序化 12 DOF 入口静默当作 Lab 任务。Lab 3 的 USD 与 PhysX 批量复制、CPU/GPU 环境碰撞隔离以及 120 Hz 足底接触报告均已接入。训练目录若已有配置或权重会拒绝覆盖。

从当前 10,000 模型开始 Lab 续训的示例，`--run-dir` 必须使用新的目录：

```powershell
python source\pmx_prototype\train_anon_lab.py --mode train --headless --checkpoint "logs\rsl_rl\anon_locomotion\resume_plagenhoef_085_acceptance_to_10000_8192_20261122\scratch_010000.pt" --num-envs 8192 --iterations 100 --run-dir "logs\rsl_rl\anon_locomotion\lab_direct_walk_new"
```

独立完整验收时使用新模型的实际路径：

```powershell
python source\pmx_prototype\train_anon_lab.py --mode eval --headless --checkpoint "logs\rsl_rl\anon_locomotion\lab_direct_walk_new\policy_final.pt" --config-override source\pmx_prototype\configs\training\unified\lab_direct_eval.json --output "output\anon_locomotion\lab_direct_walk_eval_new"
```

自动验收及录像流程继续使用原入口。验证未显式指定环境数时默认为 1；上述配置选择四人物完整套件。CPU 对照可通过独立 JSON 覆盖 `simulation_device`。键盘界面默认在 CPU 上做身体仿真和推理，需要 CUDA 时加 `--body-device cuda:0`。

本次迁移的原始测试、旧代码快照和报告保存在 `output/anon_lab_direct_migration_20261122/`。阶段拆分前后原生链路的 180 步观测、动作、奖励、姿态、关节和接触力逐项一致。已完成 64 环境两迭代与 8192 环境一迭代的真实 GPU PPO/AMP 测试、保存和 JIT 导出，以及新模型后端自动继承的重载测试。原 10,000 模型在 Lab CUDA 后端通过完整四人物 40 段验收，零跌倒；这是已有权重的迁移验证，不是短训模型已经全套合格。新保存模型通过独立的站立、行走、停止三段诊断，并接回原外观与发裙物理的 30 帧预览。

首次 CPU 完整 40 段对照中，旧 10,000 模型在原生下有 1 次跌倒、Lab 下有 2 次。后续已定位并修复 CPU 场景原点初始化：原生预热一步、Lab 预热两步，原代码从预热后的物理根坐标取得原点，第二步的约 0.74mm 平移通过世界坐标接触计算的浮点差异影响闭环轨迹。Lab CPU 现在按 PhysX 关节树顺序读取 USD 静止根骨骼世界坐标作为原点，不改变接触参数、模型或奖励。

迁移时，双环境 360 控制步（1440 物理子步，含多次复位）的观测、动作、奖励、姿态、关节和接触力与原生逐项一致；四人物完整 40 段的全部 20 个 rollout 数组也逐项一致。CPU 两条链路现在均为同一处右转跑跌倒一次，另有后退步行响应验收失败，尚不能称该旧模型在 CPU 完整验收合格。正式训练/完整验收仍优先使用已完整通过的 CUDA 路径。初次迁移记录保留在原目录；本次根因、原始对照及修复验收见 `output/anon_lab_cpu_diagnosis_20261122/result.json`。

## 键盘可视化验收界面

单人与多人验收默认使用 **DLSS Quality**（RTX Real-Time 2.0，AA=3、DLSS execMode=2），在视口初始化完成后应用，减少平移镜头时远处网格丢线、发白和拖影。可用 `--dlss-mode performance|balanced|quality|auto` 显式选择其他档位。运行报告同时记录请求的 DLSS 档位及实际生效的抗锯齿、DLSS 数值，便于核对 Kit 是否覆盖设置。

启动 `python source\pmx_prototype\inspect_anon_policy.py`，默认 1 人，使用 **Lab Direct＋CPU**，加载 2026-11-24 通过正式验收的新倒走起步采样策略 `WALK + RUN | Iteration 10,000`，权重为 `logs/rsl_rl/anon_locomotion/lab_direct_backward_start_continue_02409_to_10000_8192_20261124/scratch_010000.pt`。指定 `--checkpoint` 时只加载所选模型。加 `--characters 4` 在同一入口启用多人场景，详见下节；显式 `--characters 1` 与不带人数参数相同，保留原单人界面、相机、默认 60Hz 显示插值及发裙配置。一个网络连续控制走路和跑步；窗口内可直接调速，换 checkpoint 时重新启动窗口。

启动进度 JSON 仅用于诊断。Windows 临时占用文件、拒绝替换时，保留上次完整快照并在后续进度更新重试，终端显示 `ANON_STARTUP_WARNING`；此类写入失败不会中断策略或发裙初始化，也不会等待文件解锁而阻塞界面。

W/S 前后，A/D 左右，Q/E 转向，松键停步。+（=）/- 每次调速 0.05 m/s，滑块支持连续速度，斜向归一化到 0～1.2 m/s。空格停步，V 切换相机。键盘验收摔倒后会显示 `Fallen [R]`，也没有回合时限。按 R 或点击 Reset 才复位。默认使用 CPU 做身体仿真和推理，沿用外观 08、共享的头发裙摆 PhysX 和策略自主避让。

键位映射已移到视口左下角的半透明 HUD，显示 Q/W/E、A/S/D、速度/转向设置及 Space/R、-/+、V 快捷键。按键时高亮，自动验收/演示时随输入指令高亮；隐藏右侧 Policy Inspector 面板后 HUD 仍可见。HUD 不接收鼠标操作。它属于视口界面层，原有独立相机截图/录像不会自动包含这个 HUD。

手动推力在底部 **Policy Controls** 设置，范围为等效速度增量 0.05–1.25 m/s，默认 0.525 m/s。取消 **Random** 后可指定相对角色朝向的角度，0° 向前，+90° 向左，−90° 向右，±180° 向后。点击 **Push once [P]** 或按 **P** 施加一次，也可按住移动键在走跑中施加。力度按角色质量和物理步长换算，施加于腰部一个物理子步。手动推力会中断自动播放，摔倒后按 R 或 RESET 复位。多人模式作用于当前受控角色。每次实际推力记录在 `policy_worker.log` 的 `CHARACTER_PUSH` 行中。

**Auto Run Eval (Short)** 在原有 105 秒演示后增加最后一个阶段 `22/22 Push Character`，总长 110 秒。进入该阶段时施加一次 0.50–0.55 m/s 的随机水平推力，随后观察 5 秒，每约 0.5 秒自动切换 WASDQE 单键或组合键对应的策略指令（移动速度至多 0.5 m/s、转向速度至多 0.35 rad/s），HUD 同步显示阶段、计时及按键高亮。此阶段摔倒不会自动复位，结束后停止自动指令并恢复手动操作；Space/R 或切换人物可提前中断。长版验收不增加该阶段。

独立的 **Policy Controls** 面板默认停靠在底部，与 **Content** 同组选项卡，并默认激活。Run Acceptance、Auto Demo、Stop/Reset、速度/转向滑块，以及 Camera 的 Angle/Side/Back/Front 视角按钮均集中在此，V 键仍可切换视角。Keyboard (viewport HUD) 勾选框位于 Camera 栏下方，保持原有键盘启用/停用行为。播放状态、手动/自动输入同步和播放期间的控件锁定沿用原逻辑。**Policy Inspector 沿用右下角 Property 的停靠位**，与 Property 同组并默认选中，Stage 位于其上方；不占满右侧整栏。模型、状态读数和已确认的 9 行平铺关节内容保持不变；人物外观及发裙物理设置不变。

自动验收/演示的阶段提示（如 `05/21 Ramp down`）使用 25 号字，在底部控制面板内左对齐显示，保留阶段编号、动作名称和当前/总时间。

面板主体按 Checkpoint playback、Speed/Yaw、Camera 从左到右三栏并排；相机四个视角按钮在最右栏以两行两列显示。

Policy Inspector 默认启用只读 **Joint motion**，界面名称、提示和状态全部使用英文，沿用原有字体。**Motion status** 直接平铺在面板顶部，不使用下拉框；显示指令、实测速度、求解/渲染频率和跌倒/复位统计。移除模型标题行及点击视口获取焦点的提示，失去焦点时仍停止键盘运动输入。关节直接平铺在标题下方，不设关节/身体部位的子下拉框：Hip、Knee、Ankle、Shoulder、Elbow、Forearm、Wrist、Waist、Neck 共 9 行。左右同名部位配对，中间名称，左右各自的运动条和速度；同侧同部位的 Roll/Pitch/Yaw 等轴合并成一个关节，共 16 个关节覆盖 31 个驱动轴。悬停查看各轴速度、摆幅，点击进入对应侧的轴详情。不再显示细碎的多列表格，也不需要横向滚动。

每轴速度使用最近 0.3 秒实际角速度绝对值的采样平均，避免往返转动相互抵消；关节组显示其中最快轴的速度，保留原刻度与颜色，不将其解释为三维合成角速度。所有关节共用 180°/s 满条刻度，超过满条仍显示实际数值。灰色 Still (<3°/s)、蓝色 Gentle (3–30°/s)、绿色 Moving (30–90°/s)、金色 Fast (>=90°/s) 只表示运动快慢，不是故障或验收判定。悬停与详情中的 **swing** 是最近 1 秒对应轴实际角度的最大值减最小值。顶部显示活动关节数、驱动轴总数及最活跃轴。以上均不是力矩、功率或电机负载。

名称和速度保持 15–16 号字，减少重复标签、折叠层级和行间空白；列表只纵向滚动。已删除 Expand 按钮及独立总览窗口、共享数据和刷新/关闭代码，也移除列表顶部的 Group speed / Full bar 两行说明；运动程度的计算、条形刻度、悬停说明和点击详情保持原行为。

点击关节速度进入详情，默认选中当前最快轴，用 Roll / Pitch / Yaw 等轴按钮切换；单轴关节不显示额外轴选择器。查看该轴运动程度、最终 PD 目标角、实际角、跟踪误差、瞬时角速度及最近 5 秒曲线，附关节限位、最近边界余量与无量纲策略残差；Action 不是力矩或负载。点击标题栏 **Back** 返回平铺列表。每个 30Hz 控制步从身体 worker 读取一次，界面约 10Hz 更新，最多保留最近 5 秒仿真时间的全部关节历史。复位、切换模型或仿真时钟倒退时清空历史；数据异常时显示 unavailable，不展示旧读数。监测只读，不向驱动、策略或发裙物理写入设置。默认运行指令不变；诊断对照可临时加 `--no-joint-monitor`。

第一版验证在 `output/anon_joint_monitor_v1_20261122/` 和 `output/anon_joint_monitor_v1_english_20261122/`。默认 CPU/Lab 的监测开关对照共 240 步，身体姿态、速度、指令、终止和复位数组逐项相同；原生 CPU 60 步同指令对照也一致，CUDA/Lab 已验证 30 个有效采样。英文真实人物预览完成 180 步并显示完整 5 秒曲线，保持共享外观和 120Hz 发裙物理。这些是监测功能与行为隔离检查，不是对策略重新训练或重新宣称完整质量验收通过。

### 同一入口切换多人行走验证

```powershell
python source\pmx_prototype\inspect_anon_policy.py --characters 4
```

#### 可选轻量发裙物理

单人和多人都可追加 `--secondary-mode light`、`light2` 或更低预算的 `light3`，例如两人轻量2模式：

```powershell
python source\pmx_prototype\inspect_anon_policy.py --characters 2 --secondary-mode light2
```

更偏向性能的四人轻量3模式：

```powershell
python source\pmx_prototype\inspect_anon_policy.py --characters 4 --secondary-mode light3
```

也可让受控人物使用 `light`（light1），背景人物使用 `light3`：

```powershell
python source\pmx_prototype\inspect_anon_policy.py --characters 4 --mixed-secondary
```

前台改为 light2、背景仍为 light3：

```powershell
python source\pmx_prototype\inspect_anon_policy.py --characters 4 --mixed-secondary --controlled-secondary-mode light2
```

`--controlled-secondary-mode` 仅与 `--mixed-secondary` 一起使用，可指定前台档位；省略时仍是 `light`，保留既有混合命令的行为。切换控制对象时，所选前台档位自动交给新人物，其余人物回到 light3。该参数独立于 `--secondary-mode`，后者只在统一档位模式中决定所有角色的档位。

前台 light2／背景 light3 的四人同条件 1800 步／60 秒实测（CPU、1600×1000、seed 42、固定 `scratch_009500.pt`）：Solved 平均 **29.88Hz**、Render **59.60 FPS**，150 个更新间隔的滑动频率为 **28.68–30.62Hz**；解算帧间隔 P95 47.94ms、P99 63.91ms，仍有长帧，不能称为完全稳锁 30。相同输入的先前前台 light1／背景 light3 为 Solved 29.22Hz，前台 worker 平均工作耗时由 31.72 降为 26.58ms。两组身体输入相同，但在不同墙钟时段测量，收益取决于负载。

另通过四人 300 步角色切换／相机／暂停／复位回归，每步前台 light2、背景 light3 分配正确，换档瞬间位置和速度保持不变。初始前台在首次交接前与统一 light2 的骨骼和粒子状态逐项相同，未被提升的背景与统一 light3 逐项相同；各组均无跌倒。记录在 `output/anon_mixed_light2_20261124/result.json`，不覆盖原前台 light1 的数据。

`--mixed-secondary` 仅用于 2–8 人且启用发裙物理的手动控制场景，覆盖 `--secondary-mode` 的统一档位设置；与 `--random-only`、关闭发裙、inline 或 legacy 模式互斥。初始前台人物由 `--controlled-character` 决定；通过人物按钮／数字键切换时，原人物降为 light3，新人物采用所选前台档位。档位与对应的控制请求按相同序号派发，沿用最多一步预取，不额外排队。原生位置／速度迭代及头发投影／细化在原场景中更新，保留位置、速度及既有物理状态，避免重建场景和额外复位；120Hz、碰撞代理、角色互碰设置和共享外观继续沿用。面板显示 `Secondary light / BG light3` 或对应的所选前台档位，报告记录 `controlled_secondary_mode`、`initial_secondary_modes`、`final_secondary_modes`；混合模式的 `commands.jsonl` 逐步记录实际 `secondary_modes`。

2026-11-24 四人、CPU、1600×1000、同一 `scratch_009500.pt`、seed 42、1800 步／60 秒测试：全 light1 的先前同输入结果为 Solved 27.31Hz、Render 59.79 FPS；混合模式为 **Solved 29.22Hz、Render 59.71 FPS**，解算均值提高约 7%，但 150 个更新间隔的滑动频率仍为 26.28–30.41Hz，未稳锁 30。受控人物发裙 worker 平均 31.72ms，三个背景分别约 22.58／22.62／22.49ms，重档人物仍可能限制整体节拍。两组身体轨迹相同，均无跌倒；基准来自相同代码路径的不同墙钟时段，不代表所有负载下的固定收益。

四人 300 步切换／相机／背景暂停／复位回归中，每步均只有对应受控角色使用 light1；升降档瞬间，动态发裙节点与裙边代理的位置和速度逐项不变。未被提升的背景人物与统一 light3 的物理状态逐项相同；初始受控人物在首次交接前与统一 light1 逐项相同。两人操作回归也通过。原生单场景链条的独立进程对照进一步确认：预算变更确实影响后续求解，而非只更新 USD 标签；切换时未重建场景。权威结果为 `output/anon_mixed_secondary_20261124/result.json` 和 `live_budget_validation.json`，早先未采用的探针日志仅作排错记录。

统一验证入口现在默认使用 `--secondary-mode light`（light1），单人和多人均生效；仅加 `--characters 4` 时四人全部采用 light1。混合模式仍需显式添加 `--mixed-secondary`。需要完整档可指定 `--secondary-mode full`；其他舞蹈播放入口及共享 V1 配置不随本次 Inspector 默认值调整。四档都使用原生 PhysX、相同的 120 Hz 物理频率、动态节点和碰撞覆盖：

| 模式 | 原生位置／速度迭代 | 头发主投影／接触细化 |
| --- | --- | --- |
| full | 255／16 | 48／24 |
| light | 64／8 | 48／8 |
| light2 | 32／8 | 16／8 |
| light3 | 8／8 | 4／2 |

`light` 当前的 48 次头发主投影原样保留。`light2`、`light3` 逐档降低预算，保留防拉长、120Hz 和现有碰撞代理；light3 原生速度迭代仍为 8，只减少位置、头发投影及接触细化迭代。约束收敛和贴身接触的余量更小，急转时可能更松。动态覆盖由 `secondary_profiles.apply_secondary_mode` 执行，适用于单人、多人和单人 inline 物理。关闭发裙或 legacy 模式与所有轻量档互斥。运行报告记录 `secondary_mode`，多人状态栏显示当前模式。

当前继续减量后的 light3：原生位置 16→8、头发主投影 8→4、接触细化 4→2，仍沿用 light3 名称，速度迭代 8、120Hz、节点和碰撞代理不变。四人可见窗口 1600×1000、CPU、seed 17、每组 600 步／20 秒，当前 checkpoint `lab_direct_scratch_10000_8192_20261123/scratch_009500.pt`（SHA256 `aa126d94ca038c24bbba25501b96e9d82dd546928de9fc172e65c1f4903da0bf`）同条件前后对照，身体指令及根轨迹字节相同，均无跌倒。发裙 worker 平均工作耗时 24.52→23.00ms，降低约 6.2%；显示 59.35→59.34 FPS、策略解算 28.43→28.54Hz，整体帧率基本相同，不能把预算减半解释为帧率翻倍。四人 210 步人物切换、相机、单人／全员复位及关节监测检查通过，复位计数 `[1,1,2,1]`，无跌倒。独立记录在 `output/anon_light3_retune_20261123/performance_comparison.json`；计时阶段没有并行视频处理或数值审计。

质量对照使用原 10,000 次 checkpoint 的相同 600 步／20 秒身体输入；旧记录复用前核对了输入、几何、求解源码及非预算参数相同，不与上述当前 checkpoint 的性能计时混算。头发最大长度比仍约 1.015，裙摆局部纵向边长比 P99 为 1.0533→1.0826、局部最大值 1.3716→1.6916；这是局部节点边的指标，不代表整件裙子按此比例拉伸。头发／裙摆高频二阶差分 RMS 为 4.767→4.775mm／6.213→6.131mm。六个时间点及最大局部拉伸帧的画面未见明显持续发散，但裙摆更松、局部拉伸增大，不保证完整网格无穿模。详见同目录 `quality_comparison.json`、`contact_sheet.png` 和 `after_quality.mp4`。旧首版视频与检查记录保留，不能当作新 light3 的效果录像。

下列两段为**首版 light3**（原生位置／速度 16／8，头发主投影／接触细化 8／4）的历史对照；当前 light3 已继续减量为表中的 8／8、4／2，运行参数仍是 `--secondary-mode light3`。

light2／首版 light3 的同输入质量对照包含站立、走跑、转向、侧移、后退及停止，各 600 步／20 秒；两组身体输入逐项相同，均无跌倒。头发最大长度比均约 1.015，裙摆局部纵向边长比 P99 为 1.0348／1.0533，局部最大值 1.3511／1.3716；light3 更松，防拉长目标不等于所有局部边都严格满足目标。骨盆坐标系中头发高频二阶差分 RMS 为 4.785／4.767mm，裙摆为 6.279／6.213mm；本次未出现指标上升或抽帧可见的持续发散，不保证完整网格无穿模。节点、裙边代理及所有非预算参数保持一致。数值记录：`output/anon_light3_20261123/quality_comparison.json`；同目录 `light2_vs_light3.mp4` 左 light2、右首版 light3，以仿真时间播放，不能用于判断墙钟帧率。

首版 light3 性能以排除并行视频编码／数值审计的复测为准：四人、可见窗口 1600×1000、CPU、seed 17、相同 600 步持续活动，light2／首版 light3 的发裙 worker 平均工作耗时 27.94／25.41ms，降低约 9.1%；但策略解算 27.40／27.12Hz、显示 59.16／59.04 FPS，未测到稳定的整帧提升。首组 light2 同期进行了视频审阅处理，不能把其 23.72Hz 与 light3 的 28.26Hz 差距归为模式收益。完整流水线仍受显示和其他开销影响，减少求解迭代不保证整帧加速。两组身体指令与根轨迹字节相同，均无跌倒。两人首版 light3 另测 300 步／10 秒，显示 59.98 FPS、解算 30.00Hz。四人 210 步切换／相机／复位／关节监测检查通过，复位计数 `[1,1,2,1]`，无跌倒。记录在同目录 `benchmark_comparison.json`，主要对照为 `crowd4_light2_v2` 和 `crowd4_light3_v2`；light3 仅为可选档，不改变 full 默认值。

多人路径先串行完成首帧显示，让蒙皮／Fabric 显示缓存初始化，再采用有界的一步预取：身体 N+1 与发裙 N 并行，发裙 N+1 与显示 N 并行；每个人仍逐步执行全部 120Hz 子步，不跳过或合并物理状态。2026-11-24 起，下一步身体结果的接收及发裙任务派发由一个独立线程完成，不再等主线程的 Kit 刷新结束；输入采样、人物选择、相机及全部 UI 操作继续在主线程执行。结果严格按序消费，最多预取一步，关闭时先回收该线程再关闭工作进程。刘海阴影的 Numba 内核释放 Python GIL，以允许接收线程并行推进；阴影算法、采样与阈值不变。

按键／复位最多有一步已排队输入，角色监测仍读取当前显示帧对应的人物。等待回调不立即重复绘制刚显示过的旧姿态；实时运行采用固定节拍，避免逐帧休眠误差累积。`--serial-playback` 仅供多人串行诊断。运行报告除 Solved／Render 均值外，新增解算帧间隔 P95／P99 及 150 个更新间隔的滑动窗口频率范围，用于区分平均达到 30Hz 与帧间抖动。两人／四人的人物切换、单人复位、背景暂停及全员复位均有实测；旧验证在 `output/anon_duo_light2_20261123/controls_validation.json`，新验证在 `output/anon_crowd_30hz_20261124/validation.json`。早先四人首帧直接预取的卡住日志保留，当前仍保留首帧串行初始化。

2026-11-24 四人 light3、可见窗口 1600×1000、Lab Direct＋CPU、同一 `scratch_009500.pt` 的分段测量：身体工作进程平均计算约 12.96ms，原主线程取到结果平均等待约 21.18ms（其中约 15.65ms 与显示工作重叠，不能再与绘制耗时相加）；身体结果就绪到被读取平均约 8.16ms，发裙工作约 23.13ms。同步预取 600 步 Solved 28.51Hz，异步接收／派发后同输入 600 步 30.03Hz，身体轨迹字节相同。

最终版本以 seed 42 连续运行 1800 步／60 秒，排除前 30 步后平均 Solved **30.027Hz**、Render **59.676 FPS**，四人均无跌倒；150 个更新间隔的滑动频率范围 **29.525–30.600Hz**，解算帧间隔 P95 **50.77ms**、P99 **59.55ms**。这是持续吞吐达到约 30Hz，尚未做到每个更新间隔都严格为 33.3ms；仍有偶发长帧。尝试过 4ms 和一个显示周期的取数优先等待，两者均未改善抖动，后者还把显示降至约 50 FPS，已全部撤回，没有增加相应启动参数。最终记录为 `long_nogil_60s`，不能用未采用的 `long_grace_60s` 或 `priority_20s` 当作当前性能。

四人带角色切换、相机、背景暂停／恢复、单人及全员复位的 300 步对照中，每位角色的骨骼位置、旋转、发裙粒子坐标与同步预取路径逐项相同，物理参数、120Hz 子步及碰撞覆盖未改。阴影内核持锁／释放锁的四组数值对照结果逐项相同。两人操作回归通过，51 项相关单元测试通过。完整验证在上述新目录的 `validation.json`、`profile_summary.json` 与 `shadow_equivalence.json`；旧数据和失败试验保留。本段是当前调度实现的测量，下文保留此前不同实现与负载的历史计时。

两人、可见窗口、1600×1000、CPU、相同 300 步输入的阶段对照（排除前 30 步、关闭墙钟限速、不录像）：串行 light 17.32 FPS，预取 light 31.41 FPS，预取 light2 36.84 FPS；其 P95 帧耗时分别约 64.82／37.55／31.98ms。light 在串行和预取下的每帧骨骼位置、旋转及粒子坐标逐项相同，所有模式的身体输入与根轨迹相同。随后减少旧帧重复绘制并调整节拍，两人持续活动 600 步（20 秒，受控者走跑转向侧移后退，另一个随机行走）实测稳态 30.00 FPS，P95 33.71ms、P99 35.96ms，两人均无跌倒；节拍调整前后 600 步物理状态逐项相同。该数据描述当前测试负载，不代表所有动作和后台负载下严格锁定每帧 30 FPS。报告位于 `output/anon_duo_light2_20261123/benchmark_comparison.json` 和 `active_comparison.json`。

当前 light 与 light2 的相同 20 秒质量对照：`media/recordings/Anon_duo_light_vs_light2_20261123.mp4`（左 light、右 light2，仿真时间播放）。均保持 120Hz、461 个动态主节点和 sparse 档 105 个裙边杆；均无跌倒，头发最大长度比约 1.015。裙摆纵向边长比 P99 为 1.0256／1.0348；light2 略松，局部仍有瞬时拉伸。节点的高频变化指标未上升，抽帧未见明显新增大面积穿模；这些检查不保证完整网格零穿插。详情见同目录 `quality_comparison.json`。下文为早先版本的历史计时。

2026-11-23 四人带渲染、1600×1000、Lab Direct＋CPU、不录像的同条件 300 步测试使用当时的轻量档（头发主投影／接触细化 16／8）。排除前 30 步后，完整档为 10.52 FPS，旧轻量档为 15.30 FPS。单个人物发裙 worker 平均耗时从 60.66 ms 降到 31.62 ms。两组身体指令、根状态、速度、跌倒和复位逐项相同，均无跌倒。

质量对照采用相同 20 秒身体输入，包含站立、走路、1m/s 跑步、转向、侧移、后退和停止，每组 600 帧／30fps。对照视频：`media/recordings/Anon_secondary_full_vs_light_20261123.mp4`（左 full、右 light）；原两路视频和物理节点审计另存于 `output/anon_secondary_light_20261123/`。两组均零跌倒，头发最大长度比约 1.015；裙摆局部纵向长度比 P99 为 1.0224／1.0256，轻量略松。局部边仍存在瞬时拉伸，长度检查不代表完整网格无穿插；抽帧复核未见明显新增大面积穿模或持续发散，急转和极端姿态仍可能局部穿模。详细性能、长度与高频变化诊断见该目录 `result.json`。完整模式继续作为原效果基准。

#### 多人物控制与运行记录

四人模式使用与单人 Inspector 相同的 checkpoint，Isaac Lab Direct＋CPU 批量在线推理；手动控制 Anon 01，其余 3 人独立随机行走。右侧 Policy Inspector 的人物按钮或数字键 1–4 可切换控制对象（`--controlled-character N` 指定初始人物）。切换会清除旧按键、待处理复位和关节历史；原人物恢复随机指令，新人物等候手动输入。

统一入口的 `--characters` 默认 1，支持 1–8 人；2–8 人进入多人场景。`--policy`、`--checkpoint`、`--output`、`--body-device`、`--physics-threads`、`--physics-detail`、`--frontend-task-threads`、`--no-secondary-physics` 和 `--record-video` 按当前模式转交。单人发裙 worker 默认 16 线程，多人每位默认 4 线程，沿用合并前各自设置；多人支持 Lab Direct、独立发裙 worker 和 30／60 FPS 显示，不接受inline 或 legacy 发裙选项。物理频率仍为每人 120Hz。多人实现已收进 `runtime/anon_crowd_viewer.py`，由统一入口直接调用，旧的独立多人启动脚本已移除。

多人实时预览现在默认启用 **60 FPS 显示插值**，适用于 full、light、light2、light3；可用 `--render-fps 60` 明确指定，或加 `--render-fps 30` 恢复原显示频率。`--interpolate-60` 是 60 FPS 的别名。每个角色独立保存相邻 30Hz 解算姿态，位置线性插值、旋转使用最短路径 SLERP；复位只清空该人物的插值缓存，相机与柔光跟随显示姿态。插值约增加一帧解算姿态的显示延迟，不改策略输入、发裙求解参数或 120Hz 子步。启动时先建立初始蒙皮与脸部遮挡显示缓存，再发送发裙初始姿态，避免初始化显示与发裙工作重叠造成等待。

两人 light2、CPU、1600×1000 可见窗口、相同 600 步／20 秒持续活动实测：30 FPS 对照显示 30.01 FPS；新默认显示 **59.70 FPS**，策略 **30.03Hz**，P95 显示帧间隔 23.32ms，两人均无跌倒。共显示 1,193 帧，其中 1,182 帧处于真实姿态插值阶段；抽取的 60 个实际显示帧、120 个人物姿态均与相邻解算状态的插值逐项一致，且不同于原始帧。两组 600 步身体轨迹字节相同，每人的骨骼位置、旋转及物理粒子逐项相同。记录在 `output/anon_crowd_display60_20261123/display_validation.json`。此结果对应本次负载，不保证四人或 full 档锁定 60 FPS。面板 `Solved` 与 `Render` 分别显示解算／显示频率。

多人录像仍按 30 FPS 仿真时间输出；`--record-video`、`--no-real-time` 不限速诊断和关闭发裙诊断默认 30 FPS，显式 60 FPS 与录像／不限速互斥。`--steps` 始终表示 30Hz 控制步，不随显示帧率翻倍。

60 FPS 插值另通过两人／四人各 210 步的实际界面回归：人物切换、相机按钮、单人 R 复位、背景暂停／恢复、全员复位及 31 轴监测均正常，均无跌倒。两人的复位计数为 `[1,2]`，四人为 `[1,1,2,1]`。四人短测显示 57.70 FPS、策略解算 24.27Hz，仿真仍慢于实时；插值不等于提高物理解算能力。记录见同目录 `controls_validation.json`，短测包含复位和界面截图，不作为纯稳态性能基准。

受控人物沿用单人界面的 WASD、Q/E、速度/转向滑条、Space 停止、R 复位、Auto Demo、Run Acceptance 和关节监测。停止或复位仅作用于受控人物；背景人物继续执行指令。受控人物跌倒后等待 R，背景人物仍自动独立复位。验收/演示针对当前人物，随机多人预览本身不等同正式验收通过。原面板停靠位置保留。默认相机跟随受控人物，Camera 按钮或 V 切换角度；取消 Follow 可看全场总览，再操作相机按钮或 V 会恢复跟随。

Policy Inspector 顶部的 **Pause background** 只向背景人物发送零速度，手动人物仍可控制；**Reset all** 复位全员并重新开始随机指令时间轴。身体和发裙物理始终继续仿真。所有身体在同一个 Lab Direct PhysX 场景，共用地面；默认关闭跨人物身体碰撞，保留各自与地面的碰撞以及资产内部的碰撞过滤。`--inter-character-collisions` 可显式开启人物互碰。每人使用共享 `mmd-toon`（`cel-studio` 外观）和独立的 120Hz 原生发裙物理解算器，对应脸部柔光仅作用于该人物。

背景随机指令每 2–5 秒更新，前 1 秒站立，默认平面速度上限 0.55m/s、转向上限 0.4rad/s，含随机停止。`--seed`（默认 42）使采样可复现；`--characters`、`--max-speed`、`--max-yaw`、`--arena-radius` 可调整规模与背景指令范围，手动人物的速度范围沿用单人 Inspector。靠近默认 7m 软边界时仅通过普通 policy 指令让背景人物向场内行走。`--random-only` 可恢复全员随机模式；加上 `--inter-character-collisions` 对应原版录像和性能测试的互碰设置。

输出保存到独立 `output/anon_locomotion/crowd/<时间_PID>/`：`commands.jsonl` 为逐步指令、受控人物编号、根状态、跌倒和复位，`report.json` 记录共享场景、碰撞组、权重哈希与每人统计，各人物有独立 secondary worker 日志。`--steps N` 可有限运行；`--headless` 必须给有限步数，`--save-preview` 保存最后视口，`--no-secondary-physics` 仅用于明确的身体诊断。已有非空输出目录不会覆盖。

交互版实测位于 `output/anon_crowd_manual_20261123/`：真实界面回调驱动前后/侧移/转向、切换人物、单人复位、暂停/恢复背景与全员复位。完整发裙版本 `full_secondary_final` 共 210 步，四人均无跌倒，复位次数为 1/1/2/1；监测读取当前人物的 31 个关节轴。碰撞组仅允许本人物与公共地面，不允许其他人物。检查脚本与结果同目录保存；短测试不代表全方向长时间步态验收。

`--record-video <新MP4路径>` 可录制每个控制步的真实视口，以 30fps 仿真时间编码；`--video-width 1920 --video-height 1080` 设置原生 1080p。录像沿用视口的共享外观、地面和 120Hz 发裙物理，含人物编号、仿真时钟和跌倒次数，明确注明播放速度不代表实时性能。已有视频拒绝覆盖。录像期间不按墙钟限速，截图和编码额外开销计入该次 report 的 wall_seconds/control_fps，不能将录制速度作为非录像模式的实时帧率。

原策略没有人群避让输入。发裙是各自的单向身体驱动物理，尚无人物之间的发裙接触或发裙对 RL 身体的反作用。以下为改造前全员随机且身体互碰开启时的历史实测：四人 CPU＋发裙 240 步/8 秒仿真均无跌倒，独立指令与运动通过检查；启动约 24.53 秒，8 秒仿真约用 22.50 秒墙钟（约 10.7 控制帧/秒），不代表达到实时 30fps。首次开发截图/退出故障日志保留在 run01，最终可复核报告与预览为 `output/anon_crowd_validation_20261123/run02/`。

四人随机行走录像：`media/recordings/Anon_crowd_random_walk_20261123.mp4`，1920×1080、30fps、900 帧/30 秒，保留共享外观和四套 120Hz 发裙物理。编号和字幕为英文，明确按仿真时间播放。录制实际耗时约 109.85 秒（含截图/编码，不含约 24.51 秒启动），不能把该次约 8.19 控制步/秒当作非录像实时速度。01 号在约 29.8 秒发生一次跌倒并自动复位，原样保留；其余三人无跌倒，不能声称此 30 秒全程零跌倒。报告与完整解码/900 帧验收见 `output/anon_crowd_video_20261123/run01/report.json` 和上级 `video_validation.json`，另目视核对了开头、10 秒、20 秒和片尾画面。

性能测量条件补充：上述多人预览、分段计时及关闭发裙对照，与后台 8192 环境的 Lab Direct 后退起步 A/B 训练任务时间重叠；属于并发任务负载下的结果，不是机器空闲时的性能上限。当时仍有训练运行，未暂停或调整它。整卡采样约 49–56% 利用率、显存约 11GB，并非该训练进程独占统计。证据和测量条件已补入 `output/anon_crowd_bottleneck_20261123/background_training_snapshot.json` 及对应比较报告。

训练结束后于 2026-11-23 17:28 起重新实测，启动前未发现训练进程，GPU 约 3%、显存约 2.7GB。两组均保留四套 120Hz 发裙物理、Lab Direct＋CPU、1600×1000，不录像或截图，各运行 300 控制步并排除前 30 步。历史相同配置（headless、全员随机、人物互碰开启）由 10.63 提升至 11.51 个新解算控制帧/秒，约 +8.26%，指令、根轨迹、速度、跌倒和复位与旧次逐项相等。当前统一入口的可见四人交互窗口（一人手动等待输入、三人随机、人物互碰关闭）为 10.84 个新控制帧/秒，平均 92.21ms、P95 101.42ms；300 步均无跌倒。两种场景的 UI、相机和命令不同，不能将它们的差值归因于训练。历史同配置复测的最慢发裙 worker 仍约 59.28ms/控制帧，CPU 发裙解算仍是主要瓶颈。未修改运行代码、物理参数、外观或模型；小幅变化也包含正常调度波动。计时副本、进程快照、GPU 采样和对比报告见 `output/anon_crowd_idle_benchmark_20261123/comparison.json` 及同目录文件。

统一走跑的训练、自动验收、手动界面均按平滑后的平移指令模长限幅。0.55 m/s 及以下允许 ±0.60 rad/s，到 0.90 m/s 平滑收紧到 ±0.40 rad/s，更高速度沿用 ±0.40 rad/s。面板的 `Request` 显示按键请求，`Yaw Limit` 显示当前上限，`Applied` 显示策略收到的指令，`Actual` 显示仿真实测速度。

CPU/GPU同权重325秒完整验收全部通过、均零跌倒；72秒1080p在线演示也零跌倒。检查包含连续走跑调速、全方向、前后侧运动转向、直接起停、原地转向和两次40秒站立。当前统一版采用控制器角色航向统计速度，另报胸腰摆动误差；足底支撑力按120Hz子步采样。面板仍显示命令与实际状态。按用户要求未自动操作可见窗口；发布模型的后台预加载/通信已无界面验证，只有一个模型、同一进程/场景处理方向命令。

当前发布目录内 `comparison_1080p.mp4` 为18秒新旧并排对比，`demo_1080p.mp4` 为72秒完整在线演示，均为原生1080p/30FPS、共享外观08与当前头发裙摆物理。8192并行续训加入支撑点位移、摆腿相位清地和关节平顺性奖励，并在训练阶段复习旧策略转向/急停；完整动捕、手腕/摆臂和自主避让配置保持不变。默认CPU后退/左侧/右侧跑的几何滑移分别降低33.8%/39.7%/24.5%；CPU/GPU各方向均达到至少20%改善，且实测跑速保留至少95%。仍有残余拖脚，腿部加速度RMS未全面下降，具体数据见发布目录README。

本轮六次试训/续训共1小时28分36秒，累计11小时22分37秒，不含准备、场景启动、单独验收和录像，包含验证期间继续训练的未选迭代。逐轮计时见发布目录 `training_summary.json`。此前统一版 `release_v6_unified_walk_run/` 的本轮训练为3小时32分12秒、当时累计9小时54分，其原录像 `output/anon_locomotion/anon_unified_walk_run_1080p.mp4` 保留为对照。

## 旧版31自由度全方向连续策略

发布 `output/anon_locomotion/release_v6_continuous_wrists/`。0～0.4m/s任意平移、±0.6rad/s转向，保留前臂/手腕活动及摆臂外移；8192并行，运行时补偿关闭。修正不同片段的根旋转中心及对应速度项后，152秒主套件与68秒边界套件均通过、零跌倒。低速跟踪仍偏慢；不是全方向1.2m/s跑步。按用户要求已从验收窗口移除，旧模型文件保留，默认为从零训练版。本轮含所有尝试1小时05分35秒，V6累计约5小时05分43秒，计时口径见该目录README。1080p在线演示 output/anon_locomotion/anon_continuous_wrists_1080p.mp4。


## 运行时肩部补偿已移除

主动肩部偏移/残差混合执行分支、偏移状态、旧配置文件和播放器补偿开关已删除。训练的learn_arm_clearance空间奖励、参考适配和标准PD跟踪继续保留。历史null配置可加载并清理；非空旧补偿配置明确报错，不会被静默执行。当前播放轨迹使用policy_trace.npz（hands/commands/dt），不再输出伪补偿偏移。历史日志、报告、录像和冻结代码不重写，其旧命令中的补偿参数不适用于当前入口。


## 实时验收优化（历史配置实测）

默认步行验收入口现采用CPU身体仿真/策略推理、GPU渲染及小型策略线程池。保留1080p、批准外观和120Hz头发裙摆物理。该历史配置的带界面测试约26.1 FPS（静止键盘界面，600帧），零跌倒，尚未稳定30 FPS；用户确认可接受后暂停调优。CPU完整连续指令验收通过，物理/阴影一致性检查通过。详情见 `output/anon_locomotion/realtime_optimization/README.md`。启动仍为 `python source\pmx_prototype\inspect_anon_policy.py`，需要GPU身体后端时附加 `--body-device cuda:0`。


## 舞蹈计算加速同步至策略验收（2026-11-14）

默认验收与录像现启用舞蹈播放的编译头发速度约束及接触求解加速，启动静置和运行子步均接入。物理参数、外观和策略权重不变；无需新增启动参数，已有窗口重启后生效。诊断对照可加 `--reference-hair-compute`，恢复接入前的头发计算分支。

当前共享V1物理（原V10，120Hz、48+24次约束求解）、1080p TAA、CPU身体后端，同600帧/20秒动作指令串行对照，排除前30帧：7.93 → 21.50 FPS，约2.71倍；平均帧耗时126.11 → 46.52ms。该结果来自无界面实际渲染，未操作可见窗口，尚未达到30FPS，窗口由用户自行测试。历史26.1FPS使用更早物理配置，不作为此次同条件对照。

## 走路验收独立外观物理进程（2026-11-14）

本节保留当时完整 V1 档的历史实现和计时；当前统一 Inspector 的默认发裙档位已改为 light1，见前文。

`inspect_anon_policy.py` 默认在独立CPU进程执行完整头发和裙摆物理，策略身体状态逐帧按顺序输入，当前帧渲染时可并行计算下一帧；物理维持完整V1、120Hz及全部裙边代理。外观、动作策略及控制输入保持原样。用 `--inline-secondary` 可切回原有同进程物理以便比较；`play_anon_policy.py` 的默认模式不变，直接调用时可使用 `--secondary-worker`。

无窗口1080p对照（同240帧动作、去除前30帧，CPU身体后端），身体状态及骨骼平移/旋转逐元素一致；原版20.02 FPS、帧耗时P95 56.30 ms，独立进程8个任务线程26.06 FPS、P95 42.94 ms。独立进程16个任务线程的不同短测有明显约100 ms长帧，因此默认8个任务线程；物理线程仍为16。实际带窗口帧率依机器负载而变，此结果不保证实时30 FPS。测试记录位于 `output/anon_locomotion/secondary_process_compare/`。

键盘预览默认使用 `sparse` 裙边和 4 个前台任务线程。与 VMD 预览共用的 `--physics-detail` 可覆盖这一设置：`full` 保留全部裙边代理，`reduced` 每两列保留一列，`sparse` 每三列保留一列。三种设置都保留 120 Hz 头发和裙摆解算，减少裙边代理会改变接触覆盖和裙摆运动。性能对照另存于 `output/anon_locomotion/vmd_physics_port_20261118_224117_416/`。

键盘预览默认以 60 FPS 插值显示。身体策略仍按 30 Hz 更新，头发和裙摆以 120 Hz 解算；显示在相邻姿态间插值，跌倒和重置时直接对齐新姿态。插值增加约一帧解算姿态的显示延迟。面板 `Solved` 和 `Render` 分别统计最近一秒完成的解算姿态与渲染帧，每约 0.2 秒刷新。可显式加 `--render-fps 30`；内联物理、关闭次级物理或录像时默认以 30 FPS 显示。

同一72秒身体输入的2160帧、8640物理子步重放中，粒子位置、速度及显示骨骼逐元素完全相同，最大误差0；刘海阴影掩码一致。两次默认16线程在线基准的身体轨迹、显示骨骼、手部轨迹和指令也逐元素一致，均零摔倒。配置/外观/策略及底层物理源文件SHA256保持不变。原始报告、轨迹、截图与方法见 `output/anon_locomotion/dance_acceleration_sync/README.md`。

## 2026-11-14 从零训练实测

单次seed42、8192并行、31 DOF/248观测，全新actor/critic/归一化/优化器/AMP，未加载历史策略。400次、30分15秒首次通过原主/边界功能门槛；1000次、1小时16分06秒首次通过预定接近旧版的动作指标门槛。608次后在同一随机初始化链路追加接触奖励强化（0.8到4），未改验收阈值，所有训练时间均累计。CPU/GPU两套复核均通过、零跌倒；方向切换窗口几何滑移代理仍高于旧版，不宣称逐项完全等质。独立发布 `output/anon_locomotion/release_v6_scratch_31dof/`，旧默认模型保留，详情、审计与1080p同机位对比见该目录README.md。

窗口内切换复核补充：CPU后端持续40秒零指令时，从零版在第524、1049帧触发跌倒重置（首次约17.5秒）；旧版同测零跌倒。单模型与三模型预加载模式的1200帧身体轨迹逐元素一致，最大差0，说明此长时间站立弱项不由切换/归一化串用引入。该复核时从零版仍为默认，其后已由统一走跑版接替。该次界面改动未修改权重或训练行为。记录见 `output/anon_locomotion/inspector_switch_validation/idle_results.json` 与 `idle_equivalence.json`。
