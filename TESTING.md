# 统一测试入口

在仓库根目录运行 `python run_tests.py`，默认选择 **cpu** 层。沿用现有 unittest 测试，不移动旧文件；每个文件使用当前 Python 的独立子进程运行，避免导入缓存、mock 和全局状态跨文件污染。CPU 层隐藏 CUDA 设备，不启动 Isaac Sim。

## 测试分层

| 层 | 内容 | 前提 |
| --- | --- | --- |
| `cpu`（默认） | 算法、数据读写、配置、工作流编排、IPC 与回归测试 | 项目 Python 依赖；部分测试使用 CPU PyTorch、RSL-RL、SciPy、Clarabel、Pillow 和 FFmpeg，无需启动 Kit |
| `usd` | 机器人几何、关节限制、上下肢 IK、碰撞修正 | CPU 依赖、可导入的 `pxr`、仓库内 G1 USD；不启动 Kit |
| `assets` | 当前角色预览预设、舞蹈清单和策略目录验证 | 本机 `media/`、`output/`、`policy/` 等实际资源；不启动 Kit |
| `isaac` | 现有 `smoke_isaac61.py`：单环境、60 步、无窗口仿真 | 已配置的 Isaac Sim 6.1 / Lab 3、兼容驱动及运行环境 |

分类以整个测试文件的最重依赖为准。`assets` 文件中的部分断言虽然是纯逻辑，整文件仍需本地资源。纯 CPU 表示不运行仿真/GPU，不表示仅依赖 Python 标准库。视频测试会调用 FFmpeg，IPC 测试会使用回环 socket。

建议使用项目已有 Python 环境。补齐离线测试辅助依赖可安装 `python -m pip install -e ".[test]"`；此 extra 不安装或升级 Isaac、PyTorch、RSL-RL 和 USD，请按 [环境说明](CONDA_ISAAC61.md) 配置这些版本。缺依赖、缺资源或断言失败均原样报告失败，不会静默跳过。

## 常用命令

```powershell
python run_tests.py
python run_tests.py --suite cpu --area pmx
python run_tests.py --suite cpu --match "*audio*"
python run_tests.py --suite usd
python run_tests.py --suite assets
python run_tests.py --suite isaac
python run_tests.py --suite cpu --suite usd
python run_tests.py --suite isaac --list
```

`--area` 支持 `g1`、`pmx`、`infra`；`--match` 匹配不带 `.py` 的文件名。`--list` 校验清单并展示选择，不导入测试、不启动仿真。Isaac 层必须显式选择，没有默认全层运行。

每个文件默认超时 900 秒，可用 `--timeout 120` 覆盖。测试逐文件串行运行；失败后继续收集其他文件结果。退出码 0 表示所选文件全部成功，1 表示至少一个失败/超时，2 表示参数或清单错误。unittest 自身的 skip 信息保留在日志中，文件成功不代表其中每个用例都执行。

报告写入 `output/tests/<运行编号>/summary.json`，记录 Python、命令、状态、耗时和日志路径；每个文件有独立 `.log`。Isaac 冒烟报告写入本次目录的 `isaac/`。其他原有测试可能在 `output/` 下创建临时目录。

## 增加或调整测试

1. G1 测试继续放在 `source/train_workflow/scripts/test_*.py`，PMX 放在 `source/pmx_prototype/tests/test_*.py`；入口自身测试放在 `tests/test_*.py`。
2. 在 [tests/suites.json](tests/suites.json) 登记 `path`、`suite`、`area`。清单检查会拒绝漏登记、重复登记和不存在的文件，避免新测试悄悄漏跑。
3. 每个测试文件按 unittest 标准组织。新增需要真实仿真的脚本时，显式登记到 `isaac` 并提供有限步数参数；不要自动收集所有 `validate_*` 或训练入口。
4. 修改依赖、增加本地资产访问时同步调整分层。旧的直接执行和 unittest discover 命令仍可用，但它们不经过此分层筛选。

推荐日常先运行 CPU 层；修改 USD 几何/IK 时加 USD 层；改角色资源配置时加 assets；改仿真适配、初始化或场景流程时再运行 Isaac 冒烟。60 步冒烟不代表 PPO 训练验收或画面质量验收。

## IPC 超时与执行环境

CPU 组中的 `test_anon_policy_transport.py` 会建立经过认证的本机 TCP 连接。执行沙箱可能限制回环通信；解释器路径相同，也不代表网络权限相同。

遇到 `IPC response timed out` 时，先在失败上下文运行有限时探针：

```powershell
python -c "import socket; socket.setdefaulttimeout(3); pair=socket.socketpair(); print('SOCKETPAIR_OK'); [s.close() for s in pair]"
```

如果探针在 `accept()` 超时，再在允许回环通信的终端运行同一探针和原测试。Codex 中需要通过执行工具的权限审核，在沙箱外运行；不要修改防火墙，也不要把延长测试超时或跳过测试当作修复。

2026-11-16 本机对照：`I:\isaac61\envs\isaac61\python.exe` 在沙箱内的探针超时，沙箱外立即成功；未经修改的 IPC 测试在沙箱外通过（0.11 秒）。这条记录仅解释此次故障，后续超时仍需以当次对照结果判断。完整 CPU 组需在允许回环通信的上下文执行。
