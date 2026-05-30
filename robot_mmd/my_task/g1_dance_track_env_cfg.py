# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 dance tracking environment configs.

* ``G1DanceTrackC0EnvCfg`` - C0 fixed-root smoke env to validate the H5
  reference + observation + reward + PPO pipeline on the first 10 seconds of
  ``you_are_important.h5``.
* ``G1DanceTrackC1EnvCfg`` - C1 floating-root env with alive/fall terminations
  and root yaw/XY tracking rewards.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from robot_mmd.my_task.terrain import LoweredGroundTerrainImporter, LoweredGroundTerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import G1_29DOF_CFG  # isort: skip

from robot_mmd.my_task import mdp
from robot_mmd.my_task.g1_stand_env_cfg import G1_TPOSE_INIT_STATE
from robot_mmd.my_task.mdp.actions import (
    ReferenceFrozenJointPositionAction,
    ReferenceFrozenJointPositionActionCfg,
)
from robot_mmd.my_task.mdp.joint_groups import (
    C1_JOINT_POS_OBS_NOISE,
    C1_JOINT_VEL_OBS_NOISE,
    C1_OBS_NOISE_SCALE_BY_EXPR,
    C1_RESET_NOISE_SCALE_BY_EXPR,
    G1_ARM_JOINT_EXPR,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_DANCE_H5 = os.path.join(
    _REPO_ROOT, "robot_mmd", "media", "dance", "you_are_important.h5"
)
DEFAULT_WINDOW_SECONDS = 10.0

# C0: lower ground so leg swings do not collide with the plane (fixed root only).
C0_GROUND_Z_OFFSET = -0.5

# C1-only tuning (floating root): smaller action scale + stronger smoothness penalties.
C1_ACTION_SCALE = 0.5 # 0.5
C1_ACTION_RATE_L2_WEIGHT = -0.05 # -0.01
C1_ACTION_L2_WEIGHT = -1.0e-4 # -1.0e-4
C1_ALIVE_WEIGHT = 0.8
C1_TERMINATED_PENALTY_WEIGHT = -0.4
C1_ROOT_YAW_TRACK_WEIGHT = 2.0
C1_ROOT_YAW_TRACK_SIGMA = 0.10
C1_ROOT_XY_TRACK_WEIGHT = 5.0
C1_ROOT_XY_TRACK_SIGMA = 0.05
# C1 joint tracking group weights (lower body): ankles are down-weighted.
C1_TRACKING_LOWER_BODY_WEIGHT = 1.5
C1_TRACKING_ANKLE_WEIGHT = 0.2
# C1 terminations: relaxed for dance (squat / lean / low CoM).
C1_FALL_MINIMUM_HEIGHT = 0.3
C1_BAD_ORIENTATION_LIMIT_ANGLE = 1.3
# C1 random-segment training defaults.
C1_RANDOM_MOTION_START = True
C1_TRAIN_SEGMENT_SECONDS = 2.0
C1_RANDOM_EPISODE_LENGTH = True
C1_EPISODE_MIN_SECONDS = 2.0
C1_EPISODE_MAX_SECONDS = 4.0
C1_EPISODE_LENGTH_CURRICULUM_SPEC = "0:2:4,3000:3:5,6000:4:7"
# C1: arms track H5 open-loop; waist uses 20% of C0 reset/obs noise; legs unchanged.
C1_RESET_JOINT_POS_NOISE = 0.05


##
# Scene definition
##


def _dance_track_terrain_cfg(*, ground_z_offset: float) -> LoweredGroundTerrainImporterCfg:
    return LoweredGroundTerrainImporterCfg(
        class_type=LoweredGroundTerrainImporter,
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        ground_z_offset=ground_z_offset,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


@configclass
class G1DanceTrackSceneCfg(InteractiveSceneCfg):
    """G1 dance scene: flat ground + single G1 with the T-pose init state."""

    terrain = _dance_track_terrain_cfg(ground_z_offset=0.0)

    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=G1_TPOSE_INIT_STATE,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1DanceTrackC0SceneCfg(G1DanceTrackSceneCfg):
    """C0 scene: ground lowered so fixed-root leg tracking is not blocked by contact."""

    terrain = _dance_track_terrain_cfg(ground_z_offset=C0_GROUND_Z_OFFSET)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint position control. Absolute action = default + scale * raw_action."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Policy/critic observations.

    Policy observation layout (concatenated, ``concatenate_terms=True``):
        joint_pos_rel, joint_vel_rel, projected_gravity, last_action,
        ref_joint_pos_rel (current step),
        ref_joint_pos_rel_next (current step + 1),
        motion_phase.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        actions = ObsTerm(func=mdp.last_action)
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos_rel,
            params={"h5_path": DEFAULT_DANCE_H5, "window_seconds": DEFAULT_WINDOW_SECONDS},
        )
        ref_joint_pos_next = ObsTerm(
            func=mdp.ref_joint_pos_rel_next,
            params={
                "h5_path": DEFAULT_DANCE_H5,
                "window_seconds": DEFAULT_WINDOW_SECONDS,
                "lookahead": 1,
            },
        )
        phase = ObsTerm(
            func=mdp.motion_phase,
            params={"h5_path": DEFAULT_DANCE_H5, "window_seconds": DEFAULT_WINDOW_SECONDS},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """C0 reward set: joint tracking + small action regularizers."""

    joint_pos_tracking = RewTerm(
        func=mdp.joint_pos_tracking_exp,
        weight=1.0,
        params={
            "h5_path": DEFAULT_DANCE_H5,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "sigma": 0.25,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-1.0e-4)


@configclass
class TerminationsCfg:
    """C0 terminations: only timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Reset event: set joints to motion's first frame plus small noise."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_to_motion_start,
        mode="reset",
        params={
            "h5_path": DEFAULT_DANCE_H5,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "joint_pos_noise": 0.05,
        },
    )


def _c1_joint_pos_action_cfg() -> ReferenceFrozenJointPositionActionCfg:
    return ReferenceFrozenJointPositionActionCfg(
        class_type=ReferenceFrozenJointPositionAction,
        asset_name="robot",
        joint_names=[".*"],
        scale=C1_ACTION_SCALE,
        use_default_offset=True,
        frozen_joint_name_expr=G1_ARM_JOINT_EXPR,
        motion_h5_path=DEFAULT_DANCE_H5,
        motion_window_seconds=DEFAULT_WINDOW_SECONDS,
    )


@configclass
class C1ActionsCfg(ActionsCfg):
    """C1: policy controls all joints for checkpoint compatibility; arms overwritten by reference."""

    joint_pos = _c1_joint_pos_action_cfg()


@configclass
class C1EventCfg(EventCfg):
    """C1 reset: arms exact reference; waist 20% noise; legs full noise."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_to_motion_start,
        mode="reset",
        params={
            "h5_path": DEFAULT_DANCE_H5,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "joint_pos_noise": C1_RESET_JOINT_POS_NOISE,
            "reset_root_to_motion_quat": True,
            "joint_noise_scale_by_expr": C1_RESET_NOISE_SCALE_BY_EXPR,
            "random_start": C1_RANDOM_MOTION_START,
            "segment_seconds": C1_TRAIN_SEGMENT_SECONDS,
            "random_episode_length": C1_RANDOM_EPISODE_LENGTH,
            "episode_min_seconds": C1_EPISODE_MIN_SECONDS,
            "episode_max_seconds": C1_EPISODE_MAX_SECONDS,
        },
    )


##
# Environment configurations
##


@configclass
class G1DanceTrackC0EnvCfg(ManagerBasedRLEnvCfg):
    """C0 (fixed root) dance tracking env.

    The root link is welded to the world so the policy only has to learn the
    joint-space imitation problem. Episode = 10 seconds at 50Hz control.
    """

    scene: G1DanceTrackC0SceneCfg = G1DanceTrackC0SceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 3.0, 1.0),
        lookat=(0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Control loop: dt=1/300  physics, decimation=10 -> 30Hz control.
        self.sim.dt = 1.0 / 300.0   # 物理 300 Hz
        self.decimation = 10         # 控制 300/10 = 30 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Window length must match the reference buffer window seconds.
        self.episode_length_s = DEFAULT_WINDOW_SECONDS
        # Fix the root link (C0 only). Gravity stays enabled so joints feel
        # realistic inertia, but the robot cannot fall.
        if self.scene.robot.spawn is not None and self.scene.robot.spawn.articulation_props is not None:
            self.scene.robot.spawn.articulation_props.fix_root_link = True


@configclass
class G1DanceTrackC1EnvCfg(G1DanceTrackC0EnvCfg):
    """C1 (floating root) dance tracking env.

    Root link is freed; we add an ``alive`` reward, fall-height and bad-orientation
    terminations so the policy learns balance while tracking the motion window.

    C1 overrides C0 action scale and smoothness reward weights (see module
    constants ``C1_ACTION_*``) to reduce jitter / launch under gravity.
    Arms are frozen to the H5 reference; waist/upper-body reset and obs noise
    are scaled down (see ``C1_*_NOISE_*`` in ``joint_groups``).
    Ground is at the default height (z=0), not the lowered C0 plane.
    """

    scene: G1DanceTrackSceneCfg = G1DanceTrackSceneCfg(num_envs=512, env_spacing=2.5)
    actions: C1ActionsCfg = C1ActionsCfg()
    events: C1EventCfg = C1EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = C1_EPISODE_MAX_SECONDS
        # Waist / legs keep scaled obs noise; arms get none (see joint_groups).
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_group_noise,
            noise=None,
            params={
                "pos_noise": C1_JOINT_POS_OBS_NOISE,
                "joint_noise_scale_by_expr": C1_OBS_NOISE_SCALE_BY_EXPR,
            },
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_group_noise,
            noise=None,
            params={
                "vel_noise": C1_JOINT_VEL_OBS_NOISE,
                "joint_noise_scale_by_expr": C1_OBS_NOISE_SCALE_BY_EXPR,
            },
        )
        # Free the root link.
        if self.scene.robot.spawn is not None and self.scene.robot.spawn.articulation_props is not None:
            self.scene.robot.spawn.articulation_props.fix_root_link = False

        # C1 shaping: lower alive bonus + explicit non-timeout termination penalty.


        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=C1_ALIVE_WEIGHT)
        self.rewards.terminated_penalty = RewTerm(
            func=mdp.is_terminated_term, weight=C1_TERMINATED_PENALTY_WEIGHT
        )
        self.rewards.root_yaw_tracking = RewTerm(
            func=mdp.root_yaw_tracking_exp,
            weight=C1_ROOT_YAW_TRACK_WEIGHT,
            params={
                "h5_path": DEFAULT_DANCE_H5,
                "window_seconds": DEFAULT_WINDOW_SECONDS,
                "sigma": C1_ROOT_YAW_TRACK_SIGMA,
            },
        )
        self.rewards.root_xy_tracking = RewTerm(
            func=mdp.root_xy_tracking_exp,
            weight=C1_ROOT_XY_TRACK_WEIGHT,
            params={
                "h5_path": DEFAULT_DANCE_H5,
                "window_seconds": DEFAULT_WINDOW_SECONDS,
                "sigma": C1_ROOT_XY_TRACK_SIGMA,
            },
        )

        # Fall + bad orientation terminations (relaxed for exaggerated dance poses).
        self.terminations.time_out = DoneTerm(func=mdp.random_episode_time_out, time_out=True)
        self.terminations.fall_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": C1_FALL_MINIMUM_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": C1_BAD_ORIENTATION_LIMIT_ANGLE, "asset_cfg": SceneEntityCfg("robot")},
        )

        # C1 smoothness reward overrides (action scale lives in C1ActionsCfg).
        self.rewards.action_rate.weight = C1_ACTION_RATE_L2_WEIGHT
        self.rewards.action_l2.weight = C1_ACTION_L2_WEIGHT

        self.rewards.joint_pos_tracking.weight = 5.0
        self.rewards.joint_pos_tracking.params["sigma"] = 0.08
        self.rewards.joint_pos_tracking.params["joint_weight_default"] = 0.0
        self.rewards.joint_pos_tracking.params["joint_weight_by_expr"] = {
            "waist_.*_joint": C1_TRACKING_LOWER_BODY_WEIGHT,
            ".*_hip_.*_joint": C1_TRACKING_LOWER_BODY_WEIGHT,
            ".*_knee_joint": C1_TRACKING_LOWER_BODY_WEIGHT,
            ".*_ankle_.*_joint": C1_TRACKING_ANKLE_WEIGHT,
        }
