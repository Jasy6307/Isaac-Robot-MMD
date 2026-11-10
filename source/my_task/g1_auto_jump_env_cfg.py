"""Reference-conditioned G1 dance objectives with early jump exploration."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from source.my_task.g1_jump_env_cfg import G1VmdJumpV2EnvCfg
from source.my_task.mdp import auto_jump, reference_rewards
from source.train_workflow.utils.reference_objectives import REWARD_VERSION


@configclass
class G1VmdAutoJumpEnvCfg(G1VmdJumpV2EnvCfg):
    jump_specialist: bool = False
    auto_jump: bool = True
    jump_reference_shaping: bool = False
    jump_phase_schedules: tuple = ()
    auto_jump_report: dict = {}
    jump_gate_pre_seconds: float = 0.0
    jump_gate_post_seconds: float = 0.25
    jump_gate_ramp_seconds: float = 0.15
    reference_reward_version: str = REWARD_VERSION
    # Robot-level tolerances, shared by all dances and reference times.
    reference_reward_scales: dict = {
        "joint_rad": .12, "root_position_m": (.08, .08, .10),
        "yaw_rad": .15, "tilt_rad": .35, "root_velocity_m_s": .5,
        "foot_position_m": .05, "foot_orientation_rad": .35, "foot_velocity_m_s": .5,
    }

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_robot_joints.params.update(
            motion_start_seconds=None, random_start=False, joint_pos_noise=.01)
        self.observations.policy.jump_state.func = auto_jump.observation
        # Keep action/observation layouts and legacy functions for old bundles.
        for name in (
            "root_xy_tracking", "root_z_tracking", "root_yaw_tracking",
            "jump_upward_velocity", "jump_airborne_close", "jump_release_ground",
            "jump_missed_flight", "jump_ground_slip", "jump_landing",
        ):
            setattr(self.rewards, name, None)
        self.rewards.joint_pos_tracking.func = reference_rewards.joint_tracking
        self.rewards.joint_pos_tracking.weight = 20.
        self.rewards.joint_pos_tracking.params = dict(self.observations.policy.jump_state.params)
        self.rewards.root_position_tracking = RewTerm(func=reference_rewards.root_position, weight=20.)
        self.rewards.root_orientation_tracking = RewTerm(func=reference_rewards.root_orientation, weight=20.)
        self.rewards.root_velocity_tracking = RewTerm(func=reference_rewards.root_velocity, weight=4.)
        self.rewards.foot_position_tracking = RewTerm(func=reference_rewards.foot_position, weight=18.)
        self.rewards.foot_orientation_tracking = RewTerm(func=reference_rewards.foot_orientation, weight=4.)
        self.rewards.foot_velocity_tracking = RewTerm(func=reference_rewards.foot_velocity, weight=4.)
        self.rewards.contact_tracking = RewTerm(func=reference_rewards.contact_tracking, weight=12.)
        self.rewards.jump_exploration = RewTerm(func=reference_rewards.jump_exploration, weight=4.)
        self.rewards.terminated_penalty.weight = -150.
        self.rewards.alive.weight = 5.
        # action_rate (-.03) and action_l2 (-.0001) are inherited unchanged.


@configclass
class G1VmdAutoJumpConditionalEnvCfg(G1VmdAutoJumpEnvCfg):
    auto_jump_exact_residual: bool = True
