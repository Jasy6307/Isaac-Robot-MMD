"""Isolated phase-aware 17s jump specialist, without changing C1/C2 defaults."""
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from source.my_task.g1_train_env_cfg import G1VmdTrainC1EnvCfg, DEFAULT_DANCE_H5, DEFAULT_WINDOW_SECONDS
from source.my_task.mdp import jump


@configclass
class G1VmdJumpEnvCfg(G1VmdTrainC1EnvCfg):
    jump_specialist: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.6
        self.scene.robot.spawn.activate_contact_sensors = True
        for side in ("left", "right"):
            setattr(self.scene, f"jump_{side}_ground", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link",
                filter_prim_paths_expr=["/World/ground/terrain/CollisionPlane"],
                update_period=0., history_length=4,
            ))
        reset = self.events.reset_robot_joints
        reset.params.update(motion_start_seconds=15.7, segment_seconds=2.6,
                            random_start=False, joint_pos_noise=.01)
        self.observations.policy.jump_state = ObsTerm(
            func=jump.jump_observation,
            params={"h5_path": DEFAULT_DANCE_H5, "window_seconds": DEFAULT_WINDOW_SECONDS},
        )
        self.rewards.joint_pos_tracking.func = jump.phase_joint_tracking
        self.rewards.jump_upward_velocity = RewTerm(func=jump.upward_velocity, weight=15.)
        self.rewards.jump_airborne_close = RewTerm(func=jump.airborne_close, weight=20.)
        self.rewards.jump_ground_slip = RewTerm(func=jump.ground_slip, weight=-3.)
        self.rewards.jump_landing = RewTerm(func=jump.landing_stability, weight=5.)


@configclass
class G1VmdJumpV2EnvCfg(G1VmdJumpEnvCfg):
    jump_reference_shaping: bool = True
    jump_full_step_contact: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_pos_tracking.func = jump.phase_joint_tracking_v2
        self.rewards.joint_pos_tracking.weight = 20.
        self.rewards.root_xy_tracking.weight = 15.
        self.rewards.jump_upward_velocity.weight = 25.
        self.rewards.jump_airborne_close = RewTerm(func=jump.sustained_flight, weight=120.)
        self.rewards.jump_release_ground = RewTerm(func=jump.release_ground_load, weight=30.)
        self.rewards.jump_missed_flight = RewTerm(func=jump.missed_flight, weight=-20.)
        self.rewards.jump_ground_slip.weight = -6.
        self.rewards.jump_landing.weight = 10.


@configclass
class G1VmdJumpMultiEnvCfg(G1VmdJumpV2EnvCfg):
    jump_multi: bool = True
    jump_training_episode_seconds: float = 2.6
    jump_phase_schedules: tuple = (
        jump.PHASES,
        ((20.48,20.60),(20.60,20.75),(20.75,20.95),(20.95,21.15)),
        ((47.88,48.02),(48.02,48.18),(48.18,48.38),(48.38,48.58)),
    )
    jump_shaping_events: tuple = (
        (16.86,17.00,17.08,17.22,16.90),
        (20.53,20.67,20.75,20.89,20.57),
        (47.96,48.10,48.18,48.32,48.00),
    )

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints.params.update(
            motion_start_seconds=None,
            motion_start_choices=(15.7,19.4,46.9,3.,27.,55.,70.),
            motion_start_probabilities=(.25,.25,.25,.0625,.0625,.0625,.0625),
        )


@configclass
class G1VmdJumpFullEnvCfg(G1VmdJumpMultiEnvCfg):
    jump_training_episode_seconds: float = 2500/30.

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = self.jump_training_episode_seconds
        self.events.reset_robot_joints.params.update(motion_start_choices=None,
            motion_start_seconds=0., segment_seconds=self.episode_length_s)
