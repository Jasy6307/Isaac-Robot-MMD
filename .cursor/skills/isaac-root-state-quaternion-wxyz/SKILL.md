---
name: isaac-root-state-quaternion-wxyz
description: >-
  Documents Isaac Lab / Isaac Sim articulation root world pose quaternion layout
  in root_state_w as wxyz (not xyzw). Use when reading or writing robot
  root_state_w, write_root_state_to_sim, root quat columns 3–6, or when
  quaternion math in Python assumes xyzw (e.g. CSV exports, numpy, custom
  helpers). Prevents axis-swapped or somersault-like root rotation bugs.
---

# Isaac `root_state_w` 四元数顺序（wxyz）

## 核心规则

- `Articulation.data.root_state_w`（以及 `write_root_state_to_sim` 写入的 13 维状态）里，**姿态四元数占第 4–7 列（索引 3:7），顺序为 `w, x, y, z`**，即 **wxyz**。
- 许多脚本、CSV、以及手写四元数乘法默认使用 **xyzw**（`x, y, z, w`）。**混用会导致旋转完全错误**，常见表现是本应绕竖直轴的转向变成绕水平轴大翻转。

## 读写约定

读 Isaac → 参与 xyzw 运算：

- `w, x, y, z = state[..., 3], state[..., 4], state[..., 5], state[..., 6]`
- `xyzw = [x, y, z, w]`

写回 Isaac（从 xyzw）：

- `[w, x, y, z] = [qw, qx, qy, qz]`

## 实现提示

- 在 `_apply_root_pos_instant` 一类函数里：**先 wxyz→xyzw 再算 delta，写回前 xyzw→wxyz**。
- 日志若标注 `quat_xyzw`，应对从 `root_state_w` 读出的四元数先做转换再打印，避免误导调试。

## 与本仓库

- `robot_mmd/train_workflow/run_stand.py` 中已用 `_quat_wxyz_to_xyzw` / `_quat_xyzw_to_wxyz` 封装上述约定；新增 Isaac 根姿态逻辑时应复用同一约定。
