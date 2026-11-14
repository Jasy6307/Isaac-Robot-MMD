# 爱音物理身体：行走 RL 的前置资产

此实现独立于 G1 舞蹈训练，不修改 PMX、原始 `character.usdc` 或既有播放入口。
它建立符合爱音骨长的简化刚体人形，并用实际 PhysX 连杆姿态驱动原有 628 根蒙皮骨骼。

25 DOF / 31 DOF 参考身体继续通过 `build_anon_v3.ensure_asset()` / `build_anon_v4.ensure_asset()` 生成，公共写出步骤位于 `reference_asset_builder.py`。原 `output/anon_physical_v3/`、`output/anon_physical_v4/` 路径和已有资产复用规则保持不变；v3 肘部增益与 v4 手腕/固定父关节映射仍分别保留。构建必须在 Kit 启动后调用。

## 身体与控制约定

- Z 向上，角色前方为 **-Y**，左侧为 +X。后续速度任务须显式适配该朝向。
- 27 个刚体，23 个驱动自由度：腰 3、每腿 6、每臂 4；头和手采用固定关节。
- 根为自由浮动骨盆；仅 `joints` 验证临时固定骨盆并抬高身体，避免地面阻碍逐轴测试。
- 总质量 50.52 kg，为仿真工程估计；质量分配、惯量、力矩和关节范围均非实测人体参数。
- 身体使用胶囊，骨盆与脚底使用盒体；衣服和头发不作为承重碰撞形状。
- 启用自碰撞，过滤关节图中四跳内的近邻连杆，避免串联虚拟关节附近互相挤压。
- manifest 中关节位置/速度使用弧度，刚度为 Nm/rad，阻尼为 Nm·s/rad。
  USD 的角度限制、目标和角速度用度；角驱动刚度/阻尼用每度量纲，导出时显式转换。
- `SkinBridge` 根据实际连杆世界姿态计算骨骼世界姿态，再还原为骨骼局部姿态。
  腿部 D 骨显式映射到相应腿段；其余辅助骨继承所属身体段。原示范动画不参与控制。

## 验证入口

在配置好的 Isaac Sim 6.1 Python 环境运行：

```powershell
python source/pmx_prototype/diagnostics/validate_physical_character.py --case stand --seconds 30 --headless --capture
python source/pmx_prototype/diagnostics/validate_physical_character.py --case joints --seconds 15 --headless --capture
python source/pmx_prototype/diagnostics/validate_physical_character.py --case drop --seconds 3 --headless --capture
```

也可使用项目的 `run_isaac61.bat` 调用相同脚本。去掉 `--headless` 可显示窗口，
加 `--interactive` 可在有界验证结束后继续观察。`joints` 至少需要 13.8 秒完成全部轴的检查。

`stand` 不固定根、不改写根位置、不施加外部支撑力，只保持零目标关节 PD。
`drop` 初始抬高 0.25 m 并关闭全部关节驱动刚度、阻尼，验证真实重力响应。
`joints` 依次向各轴发送平滑小幅目标，检查全部轴有实际响应和限位约束。
三种模式均检查有限状态、驱动单位、CPU 蒙皮求值及骨骼层级的世界姿态重建误差。
站立还检查两只脚持续分担重量，合力接近整个人形重力。

## 输出

- `output/anon_physical/anon_physical.usda`：默认根 `/Anon` 的可引用物理资产，不含地面、相机和固定支撑。
  它引用 `stand/scene.usda` 的 articulation 子树；分发时应保留整个输出目录和原角色 USD/贴图依赖。
- 每种模式的 `scene.usda`：带地面、灯光和相机的初始验证场景。
- `manifest.json`：连杆、关节、参数、骨骼映射及坐标约定。
- `report.json`：验收结果和量化指标，`policy: null` 表示尚无 RL 策略。
- `trace.npz`：240 Hz 的实际关节状态、目标、根/连杆姿态与双脚接触力。
- `physical.png`：请求 `--capture` 时的截图。

`scene.usda` 和独立资产含初始蒙皮姿态。普通 USD 查看器不会自动运行 Python 的 `SkinBridge`；
要让蒙皮实时跟随物理身体，应使用本验证入口，或在后续环境中调用桥接器。
头发和裙摆目前随身体骨骼移动，未接入次级物理。

## 当前边界

这是可受重力、接触和关节驱动控制的身体基础，以及平地静态站立基线。
静态站立通过不代表具备抗扰平衡、踏步或行走能力；没有训练任何行走 policy。
后续再添加 Isaac Lab 批量环境、重置、速度指令、观测/动作、奖励及终止条件，
并在动态接触中继续校准惯量、脚底和执行器参数。

## 本机验收结果

Isaac Sim 6.1 / PhysX CPU、240 Hz 下，三种模式全部通过：

- 30 秒自由站立：最大骨盆倾角 0.785°，无关节越界；双脚最后一秒平均竖直接触力约 250.36 N、245.24 N。
- 15 秒逐轴测试：23 个轴均在各自指令窗口内产生超过 0.04 rad 的实际响应；无关节越界。
- 3 秒无驱动下落：确认所有驱动刚度与阻尼为零，身体受重力下落并倒地；状态有限，最大限位超差约 0.00273 rad。
- 三种模式的 CPU 蒙皮和骨骼世界姿态重建均通过；独立 USD 的 27 个刚体、23 个转动关节、正质量/惯量和关系目标均已复核。

汇总见 `output/anon_physical/validation_summary.json`；每种模式保留对应报告、完整轨迹和截图。

## 策略播放中的外观物理

上述 v1 身体资产的验证仍只涵盖身体。现在可通过 `play_anon_policy.py` 加载 v5 策略，
默认启用头发和裙摆在线 PhysX 节点网络，由真实身体姿态驱动根部与身体碰撞代理。
这属于运行时的单向外观物理，不向身体 articulation 反馈接触力，未修改旧 USD 的身体质量和自由度。
启动方式及限制见 `ANON_LOCOMOTION.md` 的“实时策略播放”部分。
