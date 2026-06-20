# isaac-RMD

**Unitree G1** motion retargeting and dance-tracking reinforcement learning on
**Isaac Lab** — MMD/VMD pipelines, simulation playback, and RSL-RL training tasks.

**Architecture and workflows (Chinese):**
[robot_mmd/OVERVIEW.md](robot_mmd/OVERVIEW.md)

---

## License

Original code in this repository is licensed under the
**[Apache License, Version 2.0](LICENSE)**.

Some files derived from Isaac Lab retain **BSD-3-Clause** headers; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## What is not included

This repository ships **source code only**. You must obtain the following
yourself:

| Item | Notes |
|------|--------|
| **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** | Required simulator; subject to NVIDIA’s license. Not redistributed here. |
| **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** | Clone and install per [setup_env.sh](setup_env.sh) / [setup_env.bat](setup_env.bat). |
| **MMD motion & audio** | VMD/CSV/H5/WAV under `robot_mmd/media/` are local-only (gitignored). Provide your own data; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). |

---

## Quick start (outline)

1. Install Isaac Sim and Isaac Lab (see setup scripts for pinned versions).
2. From the repo root, in the Isaac Lab conda env: `pip install -e .`
3. Add your motion files under `robot_mmd/media/` and edit
   `robot_mmd/train_workflow/dances_config.yaml`.
4. Run playback or training entry points under `robot_mmd/train_workflow/`.

Details: [robot_mmd/OVERVIEW.md](robot_mmd/OVERVIEW.md).

### IDE type checking (optional)

Copy [pyrightconfig.example.json](pyrightconfig.example.json) to `pyrightconfig.json`
(local file, gitignored) and set `venvPath` to your Conda envs directory.

---

## Disclaimer

This is an independent project. It is **not** affiliated with or endorsed by
NVIDIA, Unitree, or MMD/PMD content rights holders.
