# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for G1 stand task (placeholder for zero-agent compatibility)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1StandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """占位配置 - 零动作模式下不使用。"""

    num_steps_per_env = 24
    max_iterations = 1
    experiment_name = "g1_stand"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg()
