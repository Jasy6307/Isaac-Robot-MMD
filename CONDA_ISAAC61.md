# Isaac Sim 6.1 的独立 Conda 环境

## 路径

- 环境名称：`isaac61`。
- Python：`I:\isaac61\envs\isaac61\python.exe`，Python 3.12。
- Sim：安装在该环境中的官方 `isaacsim==6.1.0.0` pip 包。
- Lab：`I:\isaac61\IsaacLab-conda`，固定 `v3.0.0-EA` / `ae37b028ea415c91ea2bc32609efcd759ed2b974`。
- 运行缓存、临时文件、Kit 配置与日志：`I:\isaac61\conda-runtime`。
- pip / Conda 下载缓存：`I:\isaac61\cache`；大型 wheel：`I:\isaac61\downloads`。

现有 `I:\isaac61\IsaacSim`、`I:\isaac61\IsaacLab` 与 `run_isaac61.bat` 保留。两套环境使用不同的 Python 和 Lab 源码目录。项目代码、动作数据仍共享 `I:\robot_isaac`。

## 日常运行

在支持 Conda 的 PowerShell / Anaconda Prompt 中执行：

```powershell
conda activate isaac61
cd I:\robot_isaac
python source\train_workflow\g1_vmd_0_replay.py
```

该入口启动普通动作查看和重定向界面，不需要训练或 checkpoint。在 `G1 MMD config` 面板选择舞蹈并播放。

最小场景：

```powershell
python source\train_workflow\smoke_isaac61.py --viz kit --steps 300
```

`--steps 0` 保持窗口开启；`--viz none --steps 60` 进行无窗口检查。

退出环境：

```powershell
conda deactivate
```

Conda 激活时自动设置本环境的路径和 I 盘缓存变量；停用时恢复此前的值。无需再运行 `run_isaac61.bat`，该脚本始终启动保留的独立版。

### 环境路径隔离

`isaac61` 激活脚本明确设置 `PYTHONPATH`、`ISAAC_PATH`、`CARB_APP_PATH` 和 `EXP_PATH`，使 Python 与 Kit 插件使用当前 6.1 环境。`PYTHONPATH` 指向项目根目录。独立版 `run_isaac61.bat` 也明确设置自身的 Kit 路径。

修改激活配置后，已打开的终端需要重新激活环境：

```powershell
conda deactivate
conda activate isaac61
cd I:\robot_isaac
python source\train_workflow\g1_vmd_0_replay.py
```

Conda 回放 GUI 和独立版 CUDA 无窗口仿真均完成 60 步隔离验证，记录见 `output\isaac61-conda\replay-isolation-fixed.log` 与 `output\isaac61-conda\standalone-isolation-fixed.log`。

CSV → H5 的四元数列表转换与 IK 关键帧参数修复也已验证：真实 `IRIS_OUT.csv` 自动生成 H5 成功，回放 GUI 完成 60 次循环，记录见 `output\isaac61-conda\replay-pasted-error-fixed.log`。

## 编辑器

在 VS Code / Cursor 中用 `Python: Select Interpreter` 选择：

```text
I:\isaac61\envs\isaac61\python.exe
```

新建终端后检查 `python -c "import sys; print(sys.executable)"`，输出应为上面的路径。

项目的编辑器默认解释器及补全路径已配置为新环境。此前的编辑器设置备份位于 `output\isaac61-conda\editor-backup`。如果编辑器仍显示旧解释器，手动执行一次 `Python: Select Interpreter`。

## 验证记录

2026-10-01 在本机 Windows 11 / RTX 5090 D v2 / 驱动 617.14 上完成：

- Python 3.12.14、Sim 6.1.0.0、Lab v3.0.0-EA、PyTorch 2.11.0+cu128、RSL-RL 5.4.1。
- `pip check`：无缺失或冲突依赖；记录见 `output\isaac61-conda\pip-check.log`。
- CUDA 无窗口仿真 60 步通过，51 个关节，根姿态和关节状态均为有限值；非单位旋转的 WXYZ → XYZW 写入断言通过。
- Kit GUI 仿真 300 步通过，截图见 `output\isaac61-conda\smoke.png`。
- 原重定向回放入口启动、动作列表和 Mapping / Retarget UI 加载正常，完成 60 次循环后正常退出。
- 测试结果中的 `python_executable` 与 `isaacsim_path` 均指向新 Conda 环境，未调用保留的独立版。
- 新 Python 编译检查和 `git diff --check` 通过。

日志和报告：`output\isaac61-conda\smoke-headless.log`、`smoke-kit.log`、`replay.log`、`smoke_replay_headless.json`、`smoke_replay.json`。

完整依赖版本：`I:\isaac61\logs\conda-installed-packages.txt`；环境及激活变量导出：`I:\isaac61\logs\conda-environment.yml`。

## 依赖维护

pip 版 Sim 固定了部分依赖，Lab EA 的安装脚本则会升级其中一些包。本环境使用官方 Sim 元数据生成的 `conda_isaac61_constraints.txt` 重新解析了兼容依赖。保留 Sim 要求的 Newton 1.5.0、llvmlite 0.46.0，Numba 使用满足 Lab 最低要求的 0.64.0；Warp 为 1.16.0，USD provider 为 usd-exchange 2.3.0。

本项目使用 PhysX 后端和原有 PD 控制方式。当前已验证该路径，未验证 Newton 后端。不要在此环境直接重跑不带版本约束的 Lab 全量安装脚本；添加依赖时使用约束文件，例如：

```powershell
python -m pip install -c conda_isaac61_constraints.txt -e .[audio]
python -m pip check
```

三个大型官方扩展缓存 wheel 已校验 SHA256，完整文件保存在 `I:\isaac61\downloads`。下载和安装日志保存在 `I:\isaac61\logs\conda-*.log`。

Isaac Lab 3.0.0-EA 是抢先体验版；旧版 checkpoint 不自动迁移。本次未重新验证完整舞蹈播放/录制、视频导出或长时间训练。
