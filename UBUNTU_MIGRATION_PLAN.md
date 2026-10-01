# 保留 Isaac Sim 5.1 的 Ubuntu 迁移方案

评估日期：2026-09-30。目标：保留 Windows 10，保留 Isaac Sim 5.1.0 / Isaac Lab 2.3.0，恢复项目的回放、训练与评估。

本次仅进行了环境查询、代码阅读和转换脚本 `--help` 检查，没有安装软件、修改驱动或启动仿真。

## 结论

代码迁移到原生 Ubuntu 可行，主要工作是重新安装 Linux 环境、迁移本地数据以及整理启动脚本，预计比升级到 Isaac Lab 3.0 小得多。

现有 WSL 可以用于开发和离线动作处理，但不推荐作为完整 Isaac Sim 5.1 的运行方案。CUDA 可见不代表 Vulkan/RTX 可用。NVIDIA 支持人员在 2025-11 的答复中明确指出 Isaac Sim 的 WSL RTX 限制；目前本机常见 Vulkan ICD 目录也没有 NVIDIA ICD。这是平台限制，不能靠改项目的 Python 导入解决。[NVIDIA 支持答复](https://forums.developer.nvidia.com/t/wsl2-cannot-find-gpus-with-isaacsim/348783)

推荐路径：Windows 10 保持原样，在另一块 SSD 或独立分区安装原生 Ubuntu 24.04 LTS，通过双系统运行 Isaac；也可使用另一台有 RTX 显卡的原生 Ubuntu 机器。系统安装、分区和驱动安装属于后续单独执行的工作。

## 已核实的本机状态

| 项目 | 检查结果 |
| --- | --- |
| Windows | Windows 10，10.0.19045.6466 |
| GPU | RTX 5090 D v2，24455 MiB 显存 |
| Windows 驱动 | 616.64 |
| WSL | 2.7.14.0；WSLg 1.0.73.2 |
| 已安装发行版 | Ubuntu-24.04，WSL 2 |
| Ubuntu 系统 | Ubuntu 24.04.4 LTS；用户 jasy |
| WSL GPU | `/dev/dxg` 存在，`nvidia-smi` 能识别 GPU，驱动仍为 616.64 |
| WSL 环境 | 能找到 Python 和 Git；此次 shell 未找到 Conda、ffmpeg、vulkaninfo |
| Isaac 目录 | 用户家目录的常见 Isaac/Conda 路径未发现现成安装；不代表其他路径绝对不存在 |
| WSL 可见内存 | 约 30 GiB，swap 8 GiB |
| 数据访问 | 当前 Windows 仓库可从 `/mnt/i/robot_isaac` 读取 |
| 简单检查 | `vmd_2_csv.py --help` 在现有 WSL 中正常退出 |

WSL 显示的约 1 TB 根文件系统容量是虚拟磁盘容量；实际可增长空间由存放 VHDX 的 Windows 卷决定，不能据此认定物理磁盘有 1 TB 可用空间。

## 版本和平台边界

Isaac Sim 6.1 的 Windows 官方支持平台为 Windows 11，Windows 10 已不受支持；这是官方支持范围，不等于程序必然在每台 Windows 10 上无法启动。[6.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)

Isaac Sim 5.1 官方支持 Ubuntu 22.04 / 24.04，文档列出的 Linux 测试驱动是 580.65.06。该版本是验证参考，不能未经检查就认定适合 RTX 5090 D v2；应选择同时支持这张具体显卡和 Isaac Sim 5.1 的 Linux 驱动，并用最小场景验证。[5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

WSL 使用宿主机 Windows 驱动提供 CUDA，不能在 WSL 内安装普通 Linux NVIDIA 显示驱动来替换它。原生 Ubuntu 的驱动则独立于 Windows，能分别选择版本。[CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)

| 路径 | 用途与判断 |
| --- | --- |
| 现有 WSL Ubuntu | 可做编辑、Git、VMD/CSV 等离线处理；不作为完整 Isaac 仿真的交付目标 |
| WSL 内 Docker | 仍依赖相同 GPU 接口，不能据此解决 RTX 限制 |
| 原生 Ubuntu 双系统 | 推荐；可运行完整 GUI / 仿真 / 训练，并独立配置 Linux 驱动 |
| 另一台原生 Ubuntu RTX 机器 | 可行；Windows 保留为编辑和文件管理端，Linux 执行仿真 |

## 阶段 1：先验证原生 Ubuntu 和 GPU

1. 保留 Windows，优先在独立 SSD 上安装 Ubuntu 24.04 LTS；若使用现有磁盘分区，需要先确认空间、备份和启动方式。
2. 配置支持具体 GPU 的 NVIDIA 驱动。核对 Vulkan 是否枚举到 NVIDIA GPU、所需光追能力是否存在。仅 `nvidia-smi` 正常不足以验收。
3. 下载 Linux x86_64 的 Isaac Sim 5.1.0，运行 Compatibility Checker 和 Test Kit，再验证空场景及一个机器人场景。
4. 若这一阶段失败，先排查 Linux 驱动、Vulkan 和 Kit 日志，暂不改项目的强化学习逻辑。

验收：原生 Ubuntu 上 Isaac Sim 5.1 能正常创建 GPU 设备、显示场景并运行物理步骤。官方提供兼容性检查器和最小 Kit 测试。[5.1 安装与检查器](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)

原生 Ubuntu 改变了驱动栈，但尚未实测，不能保证自动解决当前 Windows 的崩溃。当前也未取得 5.1 启动日志，因此“驱动太新是唯一原因”仍未被验证。

## 阶段 2：建立同版本 Linux 环境

建议布局：

```text
~/projects/robot_isaac/          项目、assets、media、配置
~/isaac_workspace_5_1/
  IsaacSim/                     Linux 5.1.0 安装
  IsaacLab/                     固定 v2.3.0
```

- 使用 Linux 原生文件系统存放运行代码、仿真包和训练输出。
- 建立新的 Linux Conda 环境，Python 固定为 3.11；Ubuntu 系统 Python 不直接替代它。[Lab 2.3 Python 要求](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/binaries_installation.html)
- Isaac Lab 固定到 `v2.3.0` tag，而不是浮动的 main。
- 创建 Linux 的 `_isaac_sim` 和项目 `isaac_workspace` 符号链接，不能照搬 Windows 联接。
- 重装 Python 依赖，不复制 Windows Conda 环境、DLL 或 Isaac Sim Windows 安装目录。
- PyTorch 采用 Lab 2.3 x86_64 官方 CUDA 12.8 构建基线（2.7.0 / torchvision 0.22.0）；RSL-RL 按该 Lab tag 的依赖安装并记录实际版本，避免误装最新大版本。[Lab 2.3 依赖说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- 在新环境执行 `pip install -e .`，补充项目的 NumPy、h5py、PyYAML；音视频功能按需安装 pygame、OpenCV、ffmpeg / imageio-ffmpeg。
- 验证 GPU 张量运算，以及空 Isaac Lab 场景；保存依赖清单和版本信息。

初期建议为系统、仿真、缓存和训练结果预留 150–250 GB 空间；这是工程预算，不是官方最低要求，后续按日志增长调整。

## 阶段 3：搬运代码和本地数据

Git 源码以外还需单独复制：

- `media/dance/` 和 `media/pose/` 中的 VMD、VPD、CSV、H5、WAV。
- `source/train_workflow/dances_config.yaml`：它被 Git 忽略，不能只复制 example。
- `policy/` 与需要保留的 `logs/` checkpoint、`params/env.yaml`、`params/agent.yaml`。
- `assets/` 中 O6 USD / USDZ 及其依赖资源；必须验证 USD 引用和纹理路径。

保留现有 H5 schema、WXYZ 四元数和关节映射。Linux 迁移且 Lab 版本不变，不需要引入 Lab 3.0 的 XYZW / ProxyArray 改造。

检查路径大小写、盘符绝对路径和反斜杠；优先使用仓库相对路径。现有 `source/paths.py` 已按文件位置生成路径，大部分源码可直接保留。

在 Linux 上重建链接。复制过程中不要跟随原有 `isaac_workspace -> I:\isaac` 链接，把大型 Windows 安装也搬过去。

## 阶段 4：少量仓库适配

| 文件/模块 | 拟调整 |
| --- | --- |
| `setup_env.sh` | 移除无条件清空工作目录的行为；检查已有安装版本；正确处理路径、下载重试、依赖、LF 行尾和执行权限 |
| Linux 启动脚本 | 增加统一入口，激活 Linux 环境、校验 Isaac 路径，支持 replay/train/eval |
| `pyproject.toml` 或依赖文件 | 记录本项目依赖及可选音视频依赖，固定已验证的版本组合 |
| `pyrightconfig.example.json` / README | 增加 Linux 环境路径和运行方法 |
| `audio_util.py` | 已有 pygame 后端；Linux 没有 winsound 回退，需测试音频设备、暂停和跳转 |
| AVI / ffmpeg 模块 | 验证 Linux OpenCV 编码器是否可用，必要时明确选择可用编码器 |
| USD 和舞蹈配置 | 修复发现的 Windows 绝对路径及大小写问题 |

特别注意：现有 `setup_env.sh` 有 `rm -rf ${WORKSPACE}`，随后才下载仿真包。不能把它直接作为迁移执行入口，否则可能删除已有工作空间。先修正安装流程，再执行。

项目核心重定向、IK、奖励、观测、残差控制预计不需要因操作系统改变而重写；需要通过运行验证这一判断。

## 阶段 5：逐层验收

1. **离线数据**：用一个 VMD 实际转换 CSV；读取已有 H5，核对帧数、FPS、关节名称和根姿态。当前仅完成了 CLI 帮助检查。
2. **机器人资产**：加载 O6 模型，核对关节名、数量、限位、PD 参数、根链接和碰撞。
3. **单环境回放**：先关闭音视频，检查 T-pose、转向、足部 IK、根 Z 修正；录制临时 H5 并重新读取。
4. **最小训练**：4 个环境、1–5 次 PPO 迭代，确认观测/动作形状、奖励有限、reset/终止正常、checkpoint 可保存。
5. **策略评估**：先评估新 checkpoint；旧 checkpoint 需要核对网络结构、观测归一化、关节顺序后再使用。能加载不等于效果相同。
6. **扩容**：从 64 到 256，再按显存和速度逐步增加；2048 环境不是起步验收条件。
7. **交互功能**：最后验收 Mapping UI、快捷键、音乐同步、AVI 和音频合成。

训练时间步先保持原实现。现有配置中 `dt=1/60`、`decimation=1` 实际对应 60 Hz 控制，与 30 Hz 注释不一致；迁移时记录实际 `env.step_dt`，避免顺手改成 30 Hz 导致参考动作时序和策略行为一起改变。

建议按环境/启动、数据/资产、回放、训练/评估分批提交，便于发现问题和回退。Windows 源码与旧数据作为对照保留。

## WSL 的保留用途

现有 Ubuntu 无需为离线开发而重新安装。可先整理代码、运行转换脚本、检查数据，之后将同一套源码部署到原生 Ubuntu。

若在 WSL 内长期使用 Linux 工具，应把工作副本放在 Linux 文件系统中，再从 Windows 导入需要的数据；微软建议这样改善文件访问性能。[WSL 开发环境说明](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)

本方案不把 WSL 完整仿真列为交付目标。若用户坚持仅使用 WSL，则先以最小兼容性/RTX 实验作为可行性门槛；失败后停止安装整套训练环境，不通过 Docker 或 `--headless` 假定限制已解除。
