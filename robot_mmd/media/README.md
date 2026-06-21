# robot_mmd/media — 本地媒体目录

本目录**不会**随 Git 仓库上传。克隆项目后，请在此放置你自己的 MMD 动作与音频文件。

## 目录结构

```text
media/
├── README.md          ← 本说明（会进仓库）
├── dance/             ← 舞蹈：VMD / CSV / H5 / WAV
└── pose/              ← Pose 循环用 CSV（可选）
```

## dance/ 下应放什么

| 文件类型 | 说明 |
| -------- | ---- |
| `*.vmd` | MMD 骨骼动作源文件；启动 `run_g1_mmd_playback.py` 时可自动转为 CSV/H5 |
| `*.csv` | 骨骼四元数 CSV（由 `vmd_2_csv.py` 生成或自行准备） |
| `*.h5` / `*.hdf5` | 编译后的 G1 关节轨迹（由 `csv_2_hdf5.py` 生成，**RL 训练常用**） |
| `*.wav` | 可选伴音（Windows 回放；路径在 `dances_config.yaml` 中配置） |

路径均**相对于本目录** `robot_mmd/media/`。  
例如在配置里写 `dance/my_motion.h5`，对应文件为 `robot_mmd/media/dance/my_motion.h5`。

可选：同目录下可有 `<stem>_z_editted.csv` / `<stem>_z_editted.h5`（根 Z 修正版；Mapping UI 勾选 Z offset 时使用）。

## pose/ 下应放什么

- 用于 **Pose 循环**（默认 `P` 键）的 CSV，按文件名排序切换。
- 一般为单帧或短序列姿态，用于调试关节映射。

## 配置文件

1. 复制范例：

   ```bash
   cp robot_mmd/train_workflow/dances_config.example.yaml \
      robot_mmd/train_workflow/dances_config.yaml
   ```

2. 在 `dances_config.yaml` 里登记你的 `motion` / `audio` 路径（相对 `media/`）。

`dances_config.yaml` 已被 `.gitignore` 忽略，仅保留在本机。

## 离线准备流程（简要）

```bash
# VMD → CSV
python robot_mmd/train_workflow/scripts/vmd_2_csv.py \
  --input robot_mmd/media/dance/your_motion.vmd \
  --output robot_mmd/media/dance/your_motion.csv

# CSV → H5（训练 / 高精度回放）
python robot_mmd/train_workflow/scripts/csv_2_hdf5.py \
  robot_mmd/media/dance/your_motion.csv \
  robot_mmd/media/dance/your_motion.h5
```

## 版权提示

MMD 模型、动作（VMD）与音乐（WAV）通常受**第三方版权**约束。  
请仅使用你有权使用的数据；本仓库**不提供**任何示例动作或音频。
