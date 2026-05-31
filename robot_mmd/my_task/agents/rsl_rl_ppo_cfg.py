# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configs for G1 tasks (stand placeholder + dance tracking)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1StandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Placeholder config used by the zero-action stand env."""

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


@configclass
class G1DanceTrackC0PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the C0 (fixed-root) dance tracking smoke env."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_dance_track_c0"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1DanceTrackC1PPORunnerCfg(G1DanceTrackC0PPORunnerCfg):
    """PPO config for the C1 (floating-root) dance tracking env."""

    def __post_init__(self) -> None:
        self.experiment_name = "g1_dance_track_c1"
        self.max_iterations = 10000
        self.save_interval = 500


@configclass
class G1DanceTrackC1ResidualPPORunnerCfg(G1DanceTrackC1PPORunnerCfg):
    """PPO config for C1 residual-control dance tracking."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "g1_dance_track_c1_residual"
