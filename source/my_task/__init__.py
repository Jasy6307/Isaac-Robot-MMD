# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 VMD replay / train Gym task registration."""

import gymnasium as gym

gym.register(id="Isaac-G1-Vmd-Auto-Jump-Conditional-v0",entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,kwargs={
        "env_cfg_entry_point":"source.my_task.g1_auto_jump_env_cfg:G1VmdAutoJumpConditionalEnvCfg",
        "rsl_rl_cfg_entry_point":"source.my_task.agents.rsl_rl_ppo_cfg:G1VmdAutoJumpConditionalPPORunnerCfg"})

gym.register(id="Isaac-G1-Vmd-Auto-Jump-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True, kwargs={
        "env_cfg_entry_point": "source.my_task.g1_auto_jump_env_cfg:G1VmdAutoJumpEnvCfg",
        "rsl_rl_cfg_entry_point": "source.my_task.agents.rsl_rl_ppo_cfg:G1VmdAutoJumpPPORunnerCfg"})

for name, config, agent in (
    ("Multi", "G1VmdJumpMultiEnvCfg", "G1VmdJumpMultiPPORunnerCfg"),
    ("Full", "G1VmdJumpFullEnvCfg", "G1VmdJumpFullPPORunnerCfg"),
):
    gym.register(id=f"Isaac-G1-Vmd-Jump-{name}-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True, kwargs={
            "env_cfg_entry_point": f"source.my_task.g1_jump_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"source.my_task.agents.rsl_rl_ppo_cfg:{agent}"})

gym.register(
    id="Isaac-G1-Vmd-Jump-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "source.my_task.g1_jump_env_cfg:G1VmdJumpV2EnvCfg",
        "rsl_rl_cfg_entry_point": "source.my_task.agents.rsl_rl_ppo_cfg:G1VmdJumpV2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Vmd-Jump-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "source.my_task.g1_jump_env_cfg:G1VmdJumpEnvCfg",
        "rsl_rl_cfg_entry_point": "source.my_task.agents.rsl_rl_ppo_cfg:G1VmdJumpPPORunnerCfg",
    },
)

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
