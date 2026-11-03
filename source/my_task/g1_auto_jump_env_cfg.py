"""Single-run learning of an unseen G1 reference with inferred jump intent."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from source.my_task.g1_jump_env_cfg import G1VmdJumpV2EnvCfg
from source.my_task.mdp import auto_jump


@configclass
class G1VmdAutoJumpEnvCfg(G1VmdJumpV2EnvCfg):
    jump_specialist: bool = False
    auto_jump: bool = True
    jump_reference_shaping: bool = False
    jump_phase_schedules: tuple = ()
    auto_jump_report: dict = {}

    def __post_init__(self):
        super().__post_init__()
        # Full-window defaults for deployment; the training controller samples
        # short/long/full episodes at runtime without changing the network.
        self.events.reset_robot_joints.params.update(
            motion_start_seconds=None, random_start=False, joint_pos_noise=.01)
        self.observations.policy.jump_state.func = auto_jump.observation
        self.rewards.joint_pos_tracking.func = auto_jump.tracking
        self.rewards.jump_upward_velocity.func = auto_jump.upward
        self.rewards.jump_airborne_close.func = auto_jump.flight
        self.rewards.jump_release_ground.func = auto_jump.release
        self.rewards.jump_missed_flight.func = auto_jump.missed
        self.rewards.jump_ground_slip.func = auto_jump.slip
        self.rewards.jump_landing.func = auto_jump.landing
        self.rewards.support_tracking = RewTerm(func=auto_jump.support_reward, weight=60.,
            params=dict(self.observations.policy.jump_state.params))
        self.rewards.terminated_penalty.weight = -150.
        self.rewards.alive.weight = 5.
        self.rewards.root_z_tracking.weight = 5.
        self.rewards.tracking_tail = RewTerm(func=auto_jump.tracking_tail,weight=-8.,
            params=dict(self.observations.policy.jump_state.params))


@configclass
class G1VmdAutoJumpConditionalEnvCfg(G1VmdAutoJumpEnvCfg):
    auto_jump_exact_residual: bool = True
