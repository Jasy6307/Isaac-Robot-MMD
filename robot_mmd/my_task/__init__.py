# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""宇树 G1 站立任务 - 场景正中，默认姿态，不执行动作。"""

import gymnasium as gym

from . import agents
from .g1_stand_env_cfg import G1StandEnvCfg

gym.register(
    id="Isaac-G1-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_stand_env_cfg:G1StandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Dance-Track-C0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dance_track_env_cfg:G1DanceTrackC0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DanceTrackC0PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Dance-Track-C1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dance_track_env_cfg:G1DanceTrackC1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DanceTrackC1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Dance-Track-C2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dance_track_env_cfg:G1DanceTrackC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DanceTrackC2PPORunnerCfg",
    },
)
