# G1 测试

G1 的单元、USD 和认可动作回归测试统一放在本目录。33 个原 `scripts/test_*.py` 已整体迁入，仓库根定位深度不变；测试中的工具引用继续指向 `../scripts/`。

推荐使用仓库根的分层入口：

```powershell
python run_tests.py --suite cpu --area g1
python run_tests.py --suite usd --area g1
python run_tests.py --suite assets --area g1
```

原直接执行命令的文件路径改为 `source/train_workflow/tests/test_*.py`。测试之间不保留 33 份旧路径转发文件，避免工具目录再次混入测试；通用 unittest discover 可改用 `-s source/train_workflow/tests`，但不会按依赖分层。

完整依赖、测试清单和报告说明见 [TESTING.md](../../../TESTING.md)。
