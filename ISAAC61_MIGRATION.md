# Windows 11 / Isaac Sim 6.1 migration

## Installation layout

- Simulator: `I:\isaac61\IsaacSim` (official Windows 6.1.0 binary).
- Lab: `I:\isaac61\IsaacLab`, tag `v3.0.0-EA`, commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`.
- Interpreter: simulator's bundled Python 3.12. Do not combine this binary with Conda/venv.
- Downloads: `I:\isaac61\downloads`.
- Dependency, Warp, CUDA, Torch and Kit caches: `I:\isaac61\cache`.
- Temporary files: `I:\isaac61\tmp`; Kit data/config/logs also use `I:\isaac61`.
- Old 5.1 installation and `isaac_workspace` junction are preserved. The new junction is `isaac_workspace_61`.

Use `setup_env.bat` to install and `run_isaac61.bat` to run project scripts. The launcher clears active Conda/venv variables for its child process only. It does not alter the global Windows environment.

## Changes

- Lab numeric articulation data explicitly uses `ProxyArray.torch`.
- Root/joint writers use the new keyword-only `_index` methods; joint commands use `robot.actuators.target_command`.
- Native Lab rotations are XYZW. CSV, H5, scalar retargeting and IK math retain WXYZ; read/write boundaries convert once.
- Physics uses `PhysxCfg` on `sim.physics`, and keeps the legacy Lab actuator path (`use_newton_actuators=False`) to preserve the original PD controller behavior.
- Physics/control remain 60 Hz (`dt=1/60`, `decimation=1`). Rendering uses a valid integer interval of 1. Existing comments claiming 30 Hz were inaccurate.
- Viewer settings use `sim.default_visualizer_cfg`; interactive entrypoints default to `--viz kit`. The old project `--headless` flag remains an alias for `--viz none`.
- Task loading uses Lab 3's composed `parse_env_cfg` boundary.
- PPO has separate actor/critic models and explicit observation groups for RSL-RL 5.4.1. Training video uses `VideoRecorderCfg`.
- Project metadata requires Python 3.12 and records motion/audio dependencies.

## Minimal example

```powershell
cd I:\robot_isaac
.\run_isaac61.bat
```

This runs `source\train_workflow\smoke_isaac61.py`: local G1 robot, procedural lighting, zero actions, 300 physics steps, automatic exit. The standard ground asset may be downloaded into the I-drive cache on first use. For an interactive window that stays open:

```powershell
.\run_isaac61.bat source\train_workflow\smoke_isaac61.py --viz kit --steps 0
```

For an unattended check:

```powershell
.\run_isaac61.bat source\train_workflow\smoke_isaac61.py --viz none --steps 60
```

Successful runs print `ISAAC61_SMOKE_PASS` and save `output\isaac61\smoke_replay.json`. Add `--screenshot` with `--viz kit` to save `output\isaac61\smoke.png`. The minimal replay example disables gravity and checks simulation/API operation; it is not a balancing controller.

## Compatibility limits

Isaac Lab 3 is Early Access. Old RSL-RL checkpoints have a different model/state layout and are not assumed to work. Short smoke tests verify API operation, not long-run training convergence or identical numerical trajectories.

## Validation

Completed on 2026-10-01 on this machine: Windows 11 Pro 25H2, RTX 5090 D v2, NVIDIA driver 617.14. No driver change was needed.

- Official Sim 6.1.0 archive: MD5 `a07968e980072c9ca27b2166443e2d89`, verified. Its actual `VERSION` is `6.1.0-rc.26+release.49347.2d230af4.gl`. Archive extraction verified with zero missing or incorrectly sized files; report: `I:\isaac61\logs\archive-verification.json`.
- Bundled Python 3.12.13, PyTorch 2.11.0+cu128, RSL-RL 5.4.1. Full installed-package inventory: `I:\isaac61\logs\installed-packages.txt`. `pip check` passed with no broken requirements.
- Headless minimal scene: 60 steps, visualizer `[]`, CUDA `cuda:0`, 51 joints. Result: `output\isaac61\smoke_replay_headless.json`; log: `smoke-headless-final.log` in the same directory.
- Kit GUI minimal scene: 300 steps, all joint/root states finite, non-identity WXYZ-to-XYZW write checked, viewport screenshot saved. Result: `output\isaac61\smoke_replay.json`; log: `smoke-kit-final.log`; image: `smoke.png`.
- Actual PPO trainer: 4 environments, 480 samples, one update, finite losses, checkpoint saved. Log: `output\isaac61\train-smoke-final.log`.
- Actual evaluation entrypoint: loaded that new checkpoint and completed a 10-step IRIS_OUT rollout, final joint RMS 0.0876 rad. Log: `output\isaac61\eval-smoke.log`.
- Original replay entrypoint: Kit window, task and mapping/retargeting UI initialized; 60 loops completed. Log: `output\isaac61\replay-smoke.log`.
- Bundled-Python compilation, quaternion boundary checks, installer syntax and `git diff --check` passed.

The short training/evaluation above only establishes that the migrated API and model format work. It does not establish policy quality. Full dance replay/recording, video export, old checkpoint migration and long training convergence were not validated.

### Reproduce the short training and evaluation

```powershell
.\run_isaac61.bat source\train_workflow\g1_vmd_1_train.py --dance IRIS_OUT --window_frames 30 --num_envs 4 --max_iterations 1 --headless --episode_seconds 1 --experiment_suffix isaac61_smoke
```

The verified checkpoint is:

```text
I:\robot_isaac\logs\rsl_rl\g1_dance_track_c1_residual_isaac61_smoke\IRIS_OUT\2026-10-01_14-32-04\model_0.pt
```

```powershell
.\run_isaac61.bat source\train_workflow\g1_vmd_2_eval.py --dance IRIS_OUT --window_frames 10 --num_envs 1 --checkpoint logs\rsl_rl\g1_dance_track_c1_residual_isaac61_smoke\IRIS_OUT\2026-10-01_14-32-04\model_0.pt --auto_start --num_episodes 1 --no_stage --headless --play --post_play_seconds 0
.\run_isaac61.bat source\train_workflow\g1_vmd_0_replay.py --max_steps 60
```

### Official references

- [Isaac Sim 6.1 downloads](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/download.html)
- [Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
- [Isaac Lab v3.0.0-EA release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
