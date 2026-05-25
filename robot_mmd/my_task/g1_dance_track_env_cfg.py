# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 dance tracking environment configs.

* ``G1DanceTrackC0EnvCfg`` - C0 fixed-root smoke env to validate the H5
  reference + observation + reward + PPO pipeline on the first 10 seconds of
  ``you_are_important.h5``.
* ``G1DanceTrackC1EnvCfg`` - C1 floating-root env with alive/fall terminations
  (still 10s window, no root tracking reward yet).
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


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_DANCE_H5 = os.path.join(
    _REPO_ROOT, "robot_mmd", "media", "dance", "you_are_important.h5"
)
DEFAULT_WINDOW_SECONDS = 10.0

# C0: lower ground so leg swings do not collide with the plane (fixed root only).
C0_GROUND_Z_OFFSET = -0.5

# C1-only tuning (floating root): smaller action scale + stronger smoothness penalties.
C1_ACTION_SCALE = 0.35 # 0.5
C1_ACTION_RATE_L2_WEIGHT = -0.05 # -0.01
C1_ACTION_L2_WEIGHT = -1.0e-3 # -1.0e-4


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
        # Control loop: dt=0.005s physics, decimation=4 -> 50Hz control.
        # self.decimation = 4
        # self.sim.dt = 0.005
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

    Root link is freed; we add an ``alive`` reward, a fall-height termination
    and a bad-orientation termination so the policy is forced to learn balance
    while still tracking the same 10s motion window.

    C1 overrides C0 action scale and smoothness reward weights (see module
    constants ``C1_ACTION_*``) to reduce jitter / launch under gravity.
    Ground is at the default height (z=0), not the lowered C0 plane.
    """

    scene: G1DanceTrackSceneCfg = G1DanceTrackSceneCfg(num_envs=512, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Free the root link.
        if self.scene.robot.spawn is not None and self.scene.robot.spawn.articulation_props is not None:
            self.scene.robot.spawn.articulation_props.fix_root_link = False

        # Add alive reward (computed lazily so we don't shadow C0 cfg).
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.5)

        # Fall + bad orientation terminations.
        self.terminations.fall_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        # Stop episodes early when physics diverges (launch / drift).
        # Measured relative to per-episode spawn (cloner grid + default root).
        self.terminations.fly_height = DoneTerm(
            func=mdp.root_height_above_spawn,
            params={"max_height_above_spawn": 0.55, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.drift_xy = DoneTerm(
            func=mdp.root_xy_drift_from_spawn,
            params={"max_distance": 2.0, "asset_cfg": SceneEntityCfg("robot")},
        )

        # C1 action / reward overrides (C0 keeps ActionsCfg + RewardsCfg defaults).
        self.actions.joint_pos.scale = C1_ACTION_SCALE
        self.rewards.action_rate.weight = C1_ACTION_RATE_L2_WEIGHT
        self.rewards.action_l2.weight = C1_ACTION_L2_WEIGHT
