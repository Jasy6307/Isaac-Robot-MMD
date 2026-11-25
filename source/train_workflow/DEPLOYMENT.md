# 不依赖原始 H5 的部署包

训练仍使用 H5；已有模型不需要重新训练。部署包保存指定检查点的原始字节、
当前有效的 Isaac Lab 环境/策略配置，以及已经完成关节映射和控制频率重采样的参考张量。
上半身仍直接跟随轨迹，腿部仍使用参考角度加策略残差。

## 导出

使用项目原有的 Isaac Lab Python 环境。保留该模型验证时实际使用的 task、dance、
window_frames、PD 配置和其他控制参数，在原验证命令末尾添加：

```powershell
--export_deployment_bundle output/deploy/dance_bundle.pt
```

建议显式指定 `--checkpoint`，确定打包的模型。导出会创建环境、验证检查点可加载，
保存部署包后退出，不执行舞蹈播放，也不会重新训练。
省略 `--window_frames` 时，沿用原验证行为，使用所选 H5 的完整长度。
导出只代表指定配置下的推理包，不会自动推断原训练命令中的所有覆盖参数。

## 加载与播放

```powershell
python source/train_workflow/g1_vmd_2_eval.py `
  --deployment_bundle output/deploy/dance_bundle.pt `
  --num_envs 1 `
  --play
```

加载时不再查找、打开原始 H5，也不需要原检查点和训练日志目录。
轨迹从部署包恢复为共享缓存，观测、动作、奖励、重置和终止条件均使用该缓存。
Runner 所需的检查点会从包内释放到临时目录，在进程退出时清理。

包固定 task、策略、舞蹈、窗口、控制频率、关节顺序、默认关节角及控制配置。
部署包按内置配置加载，兼容性取决于当前代码和运行环境是否仍提供包内引用的类和函数。
不允许用 CLI 覆盖 dance/checkpoint/window_frames/physics_hz/decimation/
residual_alpha/use_reference_residual/pd_profile/self_collisions；UI 内也不切换策略或窗口。
更换这些内容应重新导出或加载另一份部署包。

这是一份用于当前项目及匹配 Isaac Lab/PyTorch 环境的 Python 部署包，并非 ONNX
或独立机器人控制程序。只加载可信的包。机器人 USD、Isaac 运行环境等资源仍需提供；
音频和场景资源没有嵌入，场景可以用 `--no_stage` 关闭。

## 检查

```powershell
python source/train_workflow/tests/test_deployment_bundle.py
```

CPU 回归覆盖：移除源 H5 和原检查点后的包恢复、关节/根节点/相位及下一帧参考的
逐项一致、环境共享缓存的无 H5 加载、不兼容配置和损坏轨迹拒绝。
真实模型还应在 Isaac 环境中导出一次，再将原 H5 临时移开，在新进程加载包完成播放。
