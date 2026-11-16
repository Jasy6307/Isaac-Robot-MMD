# 爱音速度指令行走

独立于 G1 舞蹈训练。物理身体来自 `anon_physical.usda`，训练不加载蒙皮和贴图，
以原生 PhysX GPU articulation 批量仿真，通过 RSL-RL 5.4.1 的 PPO 学习。

## 代码与历史配置位置

训练、预览和策略检查仍使用原入口。历史 v5/v6 JSON 归档于 `archive/configs/`，使用文件按版本归入 `configs/training/v5/` 和 `configs/training/v6/`，旧项目路径参数由入口兼容解析，本文既有命令继续可用；索引见 `archive/catalog.json`。离线诊断实现位于 `diagnostics/`，通过分类路径或统一入口 `pmx.py` 调用，根目录旧转发入口已移除。

训练代码快照与相关发布入口通过 `source_bundle.copy_sources()` 补齐本地 Python 依赖，包括 v3/v4 的公共构建器和迁移脚本的实现。快照只读取源码，不导入或启动模拟器；旧 run 与已发布策略不改写。身体版本仍由 `full_body_reference` / `articulated_wrists` 选择，未改变观测、动作、奖励或 checkpoint 格式。

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


## 键盘可视化验收界面

启动 `python source\pmx_prototype\inspect_anon_policy.py`，默认仅加载 `WALK + RUN | Unified 31 DOF (default)`，权重位于 `output/anon_locomotion/release_v7_unified_walk_run_slip_refine/`。一个网络连续控制走路和跑步，走跑之间不需切换模型或重启窗口；已运行的窗口需重新启动才能加载本次更新。旧权重保留在历史目录，当前预加载列表只有统一模型。

W/S前后、A/D左右、Q/E转向，松键停步；+（=）/-每次调速0.05m/s，滑块支持连续速度。平移向量模长0～1.2m/s，转向±0.6rad/s，斜向归一化。空格停步，R重置，V切换相机。默认CPU身体仿真/推理，保留批准外观08、当前共享头发裙摆PhysX与学习的裙摆避让，运行时驱动补偿已移除。

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

`inspect_anon_policy.py` 默认在独立CPU进程执行完整头发和裙摆物理，策略身体状态逐帧按顺序输入，当前帧渲染时可并行计算下一帧；物理维持完整V1、120Hz及全部裙边代理。外观、动作策略及控制输入保持原样。用 `--inline-secondary` 可切回原有同进程物理以便比较；`play_anon_policy.py` 的默认模式不变，直接调用时可使用 `--secondary-worker`。

无窗口1080p对照（同240帧动作、去除前30帧，CPU身体后端），身体状态及骨骼平移/旋转逐元素一致；原版20.02 FPS、帧耗时P95 56.30 ms，独立进程8个任务线程26.06 FPS、P95 42.94 ms。独立进程16个任务线程的不同短测有明显约100 ms长帧，因此默认8个任务线程；物理线程仍为16。实际带窗口帧率依机器负载而变，此结果不保证实时30 FPS。测试记录位于 `output/anon_locomotion/secondary_process_compare/`。

同一72秒身体输入的2160帧、8640物理子步重放中，粒子位置、速度及显示骨骼逐元素完全相同，最大误差0；刘海阴影掩码一致。两次默认16线程在线基准的身体轨迹、显示骨骼、手部轨迹和指令也逐元素一致，均零摔倒。配置/外观/策略及底层物理源文件SHA256保持不变。原始报告、轨迹、截图与方法见 `output/anon_locomotion/dance_acceleration_sync/README.md`。

## 2026-11-14 从零训练实测

单次seed42、8192并行、31 DOF/248观测，全新actor/critic/归一化/优化器/AMP，未加载历史策略。400次、30分15秒首次通过原主/边界功能门槛；1000次、1小时16分06秒首次通过预定接近旧版的动作指标门槛。608次后在同一随机初始化链路追加接触奖励强化（0.8到4），未改验收阈值，所有训练时间均累计。CPU/GPU两套复核均通过、零跌倒；方向切换窗口几何滑移代理仍高于旧版，不宣称逐项完全等质。独立发布 `output/anon_locomotion/release_v6_scratch_31dof/`，旧默认模型保留，详情、审计与1080p同机位对比见该目录README.md。

窗口内切换复核补充：CPU后端持续40秒零指令时，从零版在第524、1049帧触发跌倒重置（首次约17.5秒）；旧版同测零跌倒。单模型与三模型预加载模式的1200帧身体轨迹逐元素一致，最大差0，说明此长时间站立弱项不由切换/归一化串用引入。该复核时从零版仍为默认，其后已由统一走跑版接替。该次界面改动未修改权重或训练行为。记录见 `output/anon_locomotion/inspector_switch_validation/idle_results.json` 与 `idle_equivalence.json`。
