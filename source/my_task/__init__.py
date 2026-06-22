# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 VMD replay / train Gym task registration."""

import gymnasium as gym

from . import agents
from .g1_replay_env_cfg import G1VmdReplayEnvCfg

gym.register(
    id="Isaac-G1-Vmd-Replay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_replay_env_cfg:G1VmdReplayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VmdReplayPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Vmd-Train-C1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_train_env_cfg:G1VmdTrainC1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VmdTrainC1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Vmd-Train-C2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_train_env_cfg:G1VmdTrainC2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VmdTrainC2PPORunnerCfg",
    },
)
