"""Package the selected 100STYLE trial checkpoint and reproducibility records."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[2]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
from pathlib import Path
import shutil
from source_bundle import copy_sources

ROOT=PROJECT_ROOT
HERE=SOURCE_DIR
run=ROOT/'logs/rsl_rl/anon_locomotion/v4_100style_clean'
evaluation=ROOT/'output/anon_locomotion/v4_compass_1000'
reference=ROOT/'output/anon_locomotion/100style_reference'
dest=ROOT/'output/anon_locomotion/release_v4_100style'
dest.mkdir(exist_ok=True)
shutil.copyfile(run/'model_1000.pt',dest/'policy.pt')
shutil.copyfile(evaluation/'policy_jit.pt',dest/'residual_actor_jit.pt')
shutil.copyfile(run/'mocap_library.npz',dest/'mocap_library.npz')
config=json.loads((run/'env_config.json').read_text(encoding='utf-8'))
config['mocap_library']=str(dest/'mocap_library.npz')
(dest/'env_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
for name in ['agent_config.json']:
    shutil.copyfile(run/name,dest/name)
copy_sources(dest, ['anon_locomotion_env.py','train_anon_locomotion.py','natural_walk_reference.py','mocap_walk_reference.py','physical_character.py','build_anon_v3.py','retarget_100style.py','download_100style.py'])
for source,target in [(evaluation/'report.json','evaluation.json'),(evaluation/'gait_comparison.json','gait_comparison.json'),(reference/'report.json','reference_report.json'),(reference/'library.json','reference_processing.json'),(ROOT/'media/mocap/100style/SOURCES.json','SOURCES.json')]:
    shutil.copyfile(source,dest/target)
(dest/'ATTRIBUTION.txt').write_text('The 100STYLE Dataset — Ian Mason, Sebastian Starke, Taku Komura\nhttps://www.ianxmason.com/100style/\nhttps://doi.org/10.5281/zenodo.8127870\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/\nChanges: coordinate conversion, retargeting to Anon, joint constraints, neutral calibration, cycle extraction, filtering, mirroring and command blending.\n',encoding='utf-8')
(dest/'release.json').write_text(json.dumps(dict(version='v4_100style_trial',selected_global_iteration=1000,training_parent='release_v3/policy.pt (iteration 800)',selected_additional_updates=200,training_run=str(run),evaluation=str(evaluation),reference=str(reference),actions=25,observations=89,reference_kind='100STYLE Neutral mocap style, not an AMP discriminator',commands=dict(max_translation_m_s=.4,max_yaw_rad_s=.6)),indent=2),encoding='utf-8')
(dest/'README.md').write_text('''# 爱音 100STYLE 试验版

最终选择 checkpoint 1000，从 v3 的 800 续训 200 个更新。训练共运行至 1030，原始 run 均保留。
25 维全身残差动作、89 维观测。上身使用真人动捕姿态/速度参考奖励；腿部使用低权重动捕姿态奖励与原有足端 IK。
参考来源与修改方式见 ATTRIBUTION.txt、SOURCES.json、reference_processing.json。

77 秒：16 方向 0.4 m/s、双向 ±0.6 rad/s 转向和多次停步，0 次跌倒。
平均平移误差 0.0211 m/s，最差分段 0.0354 m/s，最大转向分段误差 0.0748 rad/s，停步最高分段平均速度 0.0136 m/s。
相对 v3，双脚几何滑动估计平均减少约 13%，头部角速度 RMS 减少约 9%；腿部加速度 RMS 增加约 4%，头部倾斜 P95 从 8.19° 增至 8.84°。
这些指标不等同于主观自然度评分。当前仍有轻微滑步，颈部左右转动在后退/侧移时较明显。

policy.pt 必须配合 env_config.json、mocap_library.npz 与当前项目环境使用；JIT 仅导出残差 actor，需外部保持相同观测和参考/PD 控制。
参考视频是运动学重定向；策略视频是没有根辅助力的实际 PhysX 闭环轨迹回放。

```powershell
python source/pmx_prototype/train_anon_locomotion.py --mode eval --num-envs 1 --checkpoint output/anon_locomotion/release_v4_100style/policy.pt --suite compass --linear-speed 0.4 --yaw-speed 0.6 --output output/anon_locomotion/recheck_v4_100style --headless
```
''',encoding='utf-8')
print(dest)
