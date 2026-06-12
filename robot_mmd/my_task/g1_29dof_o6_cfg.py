from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab_assets import G1_29DOF_CFG


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
G1_29DOF_O6_USD_PATH = os.path.join(_REPO_ROOT, "assets", "g1_29dof_o6_hand_V3.usd")


def _build_g1_29dof_o6_cfg() -> ArticulationCfg:
    if not os.path.isfile(G1_29DOF_O6_USD_PATH):
        raise FileNotFoundError(f"O6 robot USD file not found: {G1_29DOF_O6_USD_PATH}")
    spawn_cfg = G1_29DOF_CFG.spawn
    if spawn_cfg is None or not hasattr(spawn_cfg, "usd_path"):
        raise RuntimeError("G1_29DOF_CFG.spawn does not expose usd_path.")
    return G1_29DOF_CFG.replace(
        spawn=spawn_cfg.replace(usd_path=G1_29DOF_O6_USD_PATH),
    )


G1_29DOF_O6_CFG = _build_g1_29dof_o6_cfg()
