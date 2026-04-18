# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""宇树 G1 站立任务 - 机器人在场景正中间以默认姿态站立，不执行任何动作。"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import G1_29DOF_CFG, G1_INSPIRE_FTP_CFG  # isort: skip

# T-pose 初始姿态：手臂水平外展 90°，腿部直立，与 MMD 基准姿态对齐
G1_TPOSE_INIT_STATE = ArticulationCfg.InitialStateCfg(
    # pos=(0.0, 0.0, 0.76),
    pos=(0.0, 0.0, 1.0),
    rot=(0.7071, 0, 0, 0.7071),
    joint_pos={
        # 腿部：直立
        ".*_hip_pitch_joint": 0.0,
        ".*_knee_joint": 0.0,
        ".*_ankle_pitch_joint": 0.0,
        ".*_ankle_roll_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        # 躯干
        "waist_.*_joint": 0.0,
        # 手臂：T-pose，手臂水平外展 90°
        "left_shoulder_pitch_joint": 0,
        "left_shoulder_roll_joint": 1.047,
        "left_shoulder_yaw_joint": 0,
        "left_elbow_joint": 1.5708,
        "left_wrist_.*_joint": 0.0,
        # 手臂：T-pose，手臂水平外展 90°
        "right_shoulder_pitch_joint": 0,
        "right_shoulder_roll_joint": -1.047,
        "right_shoulder_yaw_joint": 0,
        "right_elbow_joint": 1.5708,
        "right_wrist_.*_joint": 0.0,
        # 手部：保持默认（G1 仅有 index/middle/thumb，无 ring/pinky）
        ".*_index_.*": 0.0,
        ".*_middle_.*": 0.0,
        ".*_thumb_.*": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_robot_cfg_for_motion_playback() -> ArticulationCfg:
    """获取适用于动作回放的机器人配置：不固定根链接、禁用重力、增加阻尼，便于观察动作细节且不摔倒。"""
    cfg = G1_29DOF_CFG.copy()
    cfg.init_state = G1_TPOSE_INIT_STATE
    new_spawn = cfg.spawn.replace(
        articulation_props=cfg.spawn.articulation_props.replace(fix_root_link=True),
        rigid_props=cfg.spawn.rigid_props.replace(
            disable_gravity=False,
            linear_damping=2.0,
            angular_damping=2.0,
        ),
        
    )
    return cfg.replace(spawn=new_spawn)


##
# Scene definition
##


@configclass
class G1StandSceneCfg(InteractiveSceneCfg):
    """G1 站立场景 - 平地，单机器人位于原点。"""

    # 平地
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
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
    # G1 机器人 - 位于场景正中间，T-pose 初始姿态
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=G1_TPOSE_INIT_STATE,
    )
    # 光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings - 最小化，仅维持站立
##


@configclass
class ActionsCfg:
    """关节位置控制 - 零动作即默认姿态。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """最小观测 - 仅满足 env 接口。"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """无奖励 - 仅站立，使用零权重占位。"""

    dummy = RewTerm(func=mdp.is_terminated, weight=0.0)


@configclass
class TerminationsCfg:
    """仅超时。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class G1StandEnvCfg(ManagerBasedRLEnvCfg):
    """宇树 G1 站立环境 - 场景正中，默认姿态，不执行动作。"""

    scene: G1StandSceneCfg = G1StandSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # 初始默认视角：从斜上方看向机器人站立位置
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 3.0, 1.0),
        lookat=(0.0, 0.0, 1.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # 单环境，位于原点
        self.scene.num_envs = 1
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
