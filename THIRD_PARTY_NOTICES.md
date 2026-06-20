# Third-Party Notices

This file lists third-party software and materials used by **robot_isaac**
(`isaac-RMD`). It supplements the [Apache License 2.0](LICENSE) that applies to
original code in this repository.

---

## Original code in this repository

Unless a file header states otherwise, source code authored in this repository
is licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE).

---

## Isaac Lab (BSD-3-Clause)

Parts of this project are derived from or follow patterns of
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (NVIDIA).

Files that retain Isaac Lab copyright headers are licensed under the
**BSD-3-Clause** license. Those headers must be preserved in copies and
derivatives of the corresponding files.

---

## NVIDIA Isaac Sim (not included — obtain separately)

**Isaac Sim is not distributed with this repository.**

You must download and install Isaac Sim yourself from NVIDIA, accept NVIDIA’s
license terms, and meet the stated hardware and OS requirements.

Typical setup (see [setup_env.sh](setup_env.sh) / [setup_env.bat](setup_env.bat)
for version pins used by this project):

- [Isaac Sim downloads](https://developer.nvidia.com/isaac-sim)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — clone and link to your
  Isaac Sim installation as described in the project setup scripts.

This project’s Apache 2.0 license does **not** grant any rights to Isaac Sim or
other NVIDIA Omniverse software.

---

## MMD motion and audio data (not included — obtain separately)

**MMD/VMD motion captures, derived CSV/H5 trajectories, and accompanying audio
are not distributed with this repository.**

The directory `robot_mmd/media/` is ignored by Git. Example paths in
`robot_mmd/train_workflow/dances_config.yaml` are placeholders for your local
layout only.

To use the MMD pipeline you must:

1. Obtain motion and audio from lawful sources (e.g. your own recordings or
   content you are permitted to use).
2. Place files under `robot_mmd/media/` following the layout described in
   [robot_mmd/OVERVIEW.md](robot_mmd/OVERVIEW.md).
3. Convert VMD to CSV/H5 with the provided scripts (e.g.
   `robot_mmd/train_workflow/scripts/vmd_2_csv.py`) where applicable.

Respect the copyright and license terms of MMD models, motions, and music.
This project’s license covers **code only**, not third-party dance or audio
assets.

---

## Other dependencies

Runtime Python dependencies are primarily provided by your **Isaac Lab**
environment (Isaac Sim, PyTorch, RSL-RL, etc.). Install and version them per
Isaac Lab’s documentation and this repo’s setup scripts.
