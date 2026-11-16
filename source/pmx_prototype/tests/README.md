# PMX 测试

统一分层入口与依赖说明见 [TESTING.md](../../../TESTING.md)。

```powershell
python run_tests.py --suite cpu --area pmx
python run_tests.py --suite assets --area pmx
```

CPU 层不启动 Isaac；本地角色/动作/策略文件相关测试归入 assets。分类清单位于 [tests/suites.json](../../../tests/suites.json)，新增测试必须登记。

原有 `python -m unittest discover -s source/pmx_prototype/tests` 继续可用，但会尝试运行整个目录，包括依赖本地资产的测试，不执行分层筛选。
