# 修复 IK、固定根避碰与时间平滑

正式入口为 [g1_ground_motion.py](../../scripts/g1_ground_motion.py)。本包从已认可的 Girls 前 80 秒和后续五首前缀批处理中提取算法，不读取或执行 `output/` 中的 Python 脚本。数值腿部求解继续使用原有 `RobotLeg.solve_sequence`。

## 模块与处理阶段

| 模块 | 职责 |
| --- | --- |
| `motion.py` | H5 世界坐标/关节绝对值解码、四元数连续化、独占创建输出、保留未改动通道 |
| `sole.py` | 从显式指定的角色网格和蒙皮权重测量源鞋底高度；目前只注册 Anon 适配 |
| `ik.py` | 骨盆旋转往返恒等式、鞋底高度缩放、骨盆 Z 拟合、腿部求解、可选支撑根 Z 反馈 |
| `geometry.py` | G1 O6 包络扩展的显式历史版本：腕髋组合、手掌本体盒、真实腰部障碍、右小拇指/肩壳支撑平面 |
| `smoothing.py` | 固定根位姿的分窗 SQP、双向平滑、重叠接缝约束、强平方松弛恢复、可选局部偏置种子；独立硬验收 |
| `pipeline.py` | 各阶段编排、输入哈希绑定、源接触验收、候选与最终发布分离 |

源动作需先用 `load_vmd.py` 烘焙。必须按实际 VMD 选择源 FK/IK；prepare 要求源 USD 同名 JSON 的 `leg_mode` 与显式参数一致。该模式与 G1 数值腿部 IK 无关。

骨盆世界旋转使用 `R = A.T @ R_source_lower_body @ C.T`，并检查 `A @ R @ C @ R_source_lower_body.T ≈ I`。默认显示对齐为 `A=Rz(180°)`。鞋底目标为扣除 3mm 源高度容差、按腿长比缩放后的高度，保留真实腾空；不按水平移动速度压平跳跃。上肢通道与根 XY 保留，根 Z 与旋转重拟合，相邻四元数符号连续。

## 新建一次处理

以下示例中的输入文件需替换为实际路径；每次使用新的输出目录。prepare 当前消费从动作起点开始的前缀，`--frames` 省略则覆盖输入 H5 全长，不能超过源 USD 时间范围。

```powershell
python source/train_workflow/scripts/g1_ground_motion.py prepare --input "base.h5" --source "source.usdc" --robot "assets/g1_29dof_o6_hand_V3.usd" --output-dir "output/new_ground_run" --frames 2400 --source-leg-mode ik --character anon
python source/train_workflow/scripts/g1_ground_motion.py solve --run "output/new_ground_run" --side left
python source/train_workflow/scripts/g1_ground_motion.py solve --run "output/new_ground_run" --side right
python source/train_workflow/scripts/g1_ground_motion.py finalize-ik --run "output/new_ground_run"
```

产出 `source_sole.npz`、`targets.npz`、两侧解算结果和 `fixed_ik.h5`。输入 H5、源 USD、机器人 USD 的哈希写入 config，后续阶段发现输入变化即拒绝继续。骨盆拟合至少需要九帧。输入其他角色会明确拒绝，新增角色必须适配实际网格/蒙皮和鞋底选点，不能套用 Anon 选择规则。

只有源支撑/腾空、关节角度/速度/mimic 检查通过才写 fixed IK。此文件尚未做避碰。Deep Blue Town 快速换脚修订使用 `finalize-ik --root-feedback`：在保持关节、根 XY/旋转的情况下，以有界凸优化修正根 Z；该选项必须在后续固定根避碰之前执行。

## 固定根避碰与平滑

```powershell
python source/train_workflow/scripts/g1_ground_motion.py smooth --reference "output/new_ground_run/fixed_ik.h5" --seed "collision_seed.h5" --robot "assets/g1_29dof_o6_hand_V3.usd" --geometry batch --targets "output/new_ground_run/targets.npz" --output-dir "output/new_ground_smooth" --device cpu
```

seed 必须与修复 IK 的关节顺序、帧数、FPS、baseline、根锚点和根通道完全一致；可直接提供修复 IK 本身作为初始 seed，但复杂交叉轨迹可能需要多次局部恢复。`--device cuda:0` 可用于批量几何计算，不启动 Isaac Kit。

默认 60 帧窗口、30 帧步长、两次正反向 sweep，内部边界固定四帧；每一步复核全轨迹的限位、速度和加速度边界。`--start/--end` 以帧限定局部修复范围，重叠窗口负责接缝。可用 `--escape-center 帧号` 显式生成确定性的平滑偏置种子，随后仍需 SQP 和硬验收。强平方松弛权重默认 2e8，与非线性平方违约量一致；复现原始 Girls 线性松弛时指定 `--restoration-quadratic-weight 0`。

每次输出目录必须不存在。`candidate.h5` 始终标为 staging，并保存独立的 `validation.json`；失败保留诊断，返回非零，**不生成 final.h5**。只有重新读取编码后的 H5 并通过全部以下检查才生成 final：

- 修复 IK 的根位置、根四元数及锚点逐项不变；关节角度、速度和 mimic 正确。
- 全帧和 1/4、1/2、3/4 帧插值处所选包络至少 0.5mm 间距。
- 相对参考的左右脚底高度上下界全部通过，参考支撑无异常双脚离地，参考腾空保留。
- 提供 `--targets` 时，额外核验源动作支撑与腾空。建议始终提供，以区分源接触与机器人参考接触。
- 上肢 jerk RMS 相对修复 IK 参考下降，沿用认可成品的时间平滑发布条件。

final 仅表示上述运动学/包络验收通过。完整可见网格复查仍使用 [audit_full_meshes.py](../../scripts/audit_full_meshes.py) 在新的输出目录执行；机械 CAD 接触必须逐对记录，不能称为全网格零相交。此整理不自动启动训练、录像或动力学认证。

## 几何版本与只读回归

| `--geometry` | 范围 | 对应历史基准 |
| --- | --- | --- |
| `girls` | 原始 G1 所选包络 | Girls 前 80 秒 |
| `batch-hips` | 加入腕/前臂/手指到髋壳的组合，保留中性手盒 | IRIS OUT 15.5 秒 |
| `batch-hips-palm` | 上项加收紧的手掌本体盒，保留动态手指包络 | Deep Blue Town 前 90 秒 |
| `batch` | 上项加真实 pelvis/waist 障碍、已识别小拇指/肩壳凸包支撑平面 | 后续 HAL、极乐净土、你最重要前 90 秒；新运行推荐版本 |

实验期间辅助脚本被更新过，不能把最新几何下的指标冒称为所有历史成品的原验收指标。回归清单为 [accepted_runs.json](../../../../tests/fixtures/grounding/accepted_runs.json)，冻结每份认可 H5 的哈希、处理帧数、源模式和对应几何版本。Deep Blue Town 在最新 `batch` 的更大保守腰部盒下存在包络违约；这不等于实际网格相交。旧成品保留，新版本需另存并重新优化/复查，不能靠缩小验收范围发布新结果。

```powershell
python run_tests.py --suite assets --match "*accepted_grounding*"
python source/train_workflow/scripts/regress_ground_motion.py --output-dir "output/new_ground_regression" --rebuild-targets --solve-smoke-frames 32 --smooth-smoke-frames 60
```

回归重新测量六段完整源鞋底并构建目标，与保存的旧目标比较；使用冻结的全段腿部解算结果回放最终组装，比较 H5 关节数据和根位姿，再执行完整帧/四分之一帧硬验收。另运行 32 帧新数值 IK 和 60 帧双向平滑冒烟。**这不是六段完整数值 IK 与全部避碰优化的重新求解**，也不会重复完整 FCL 扫描。protected_before 与 summary 哈希报告核对原成品未被改写。

现有普通 replay/训练转换默认尚未切换到此链路；先通过显式入口产出和验收新 H5，再在原工作流中指定它。历史视频、H5、VMD、PMX、缓存和实验脚本全部保留。

## 旧入口兼容

原 `refine_collision_fidelity.py` 与 `smooth_collision_segment.py` 现在是薄入口，统一调用本包 `legacy_refinement.py`。也可通过 `g1_ground_motion.py legacy-refine --method fidelity/smooth` 使用。旧模式保留原来的端点冻结、时间权重、仅脚底下界约束、输出文件参数、`--resume-checkpoint` 与 `--window-limit`，不自动升级验收语义；seed 根通道与参考不一致会明确拒绝。

新版 repaired-sole 算法继续使用 `smooth` 子命令，历史模式与新版共用基础 SQP 求解器，但保留明确的约束差异。不要将旧模式输出改名当成修复 IK 新链路的结果。
