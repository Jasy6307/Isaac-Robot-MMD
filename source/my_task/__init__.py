# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Register the public G1 dance replay and training tasks."""

import gymnasium as gym

from source.train_workflow.task_ids import G1_TRAIN_TASK_ID

from . import agents
from .g1_replay_env_cfg import G1VmdReplayEnvCfg

gym.register(
    id=G1_TRAIN_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_auto_jump_env_cfg:G1VmdAutoJumpConditionalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VmdAutoJumpConditionalPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Vmd-Replay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_replay_env_cfg:G1VmdReplayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VmdReplayPPORunnerCfg",
    },
)
