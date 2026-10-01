# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configs for G1 VMD replay / train tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class G1VmdReplayPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Placeholder PPO config for ``Isaac-G1-Vmd-Replay-v0``."""

    num_steps_per_env = 24
    max_iterations = 1
    experiment_name = "g1_stand"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 32],
        obs_normalization=False,
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[64, 32], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class G1VmdTrainPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared PPO config for G1 VMD train (C1/C2)."""

    num_steps_per_env = 120
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_dance_track"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        obs_normalization=True,
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.25, #原0.2
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3, #原1.0e-3
        schedule="adaptive", #原adaptive
        gamma=0.99, #原0.99
        lam=0.95, #原0.95
        desired_kl=0.015, #原0.01
        max_grad_norm=1.0, #原1.0
    )


@configclass
class G1VmdTrainC1PPORunnerCfg(G1VmdTrainPPORunnerCfg):
    """PPO config for ``Isaac-G1-Vmd-Train-C1-v0``."""

    def __post_init__(self) -> None:
        self.experiment_name = "g1_dance_track_c1_residual"
        self.max_iterations = 10000
        self.save_interval = 500


@configclass
class G1VmdTrainC2PPORunnerCfg(G1VmdTrainC1PPORunnerCfg):
    """PPO config for ``Isaac-G1-Vmd-Train-C2-v0``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "g1_dance_track_c2_residual"
