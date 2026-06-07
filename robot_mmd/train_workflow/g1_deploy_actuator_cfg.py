"""G1 actuator PD profiles for MMD playback.

Profiles
--------
``deploy``
    Pure Unitree ``unitree_rl_lab`` FixStand Kp/Kd (sim-to-real for legs/waist).

``deploy_playback`` (recommended for ``run_g1_mmd_playback.py``)
    Lower body = deploy FixStand; upper body uses higher implicit Kp for MMD tracking.

Reference:
  unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml  (FixStand kp/kd)
"""

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import G1_29DOF_CFG


def build_g1_deploy_actuators() -> dict[str, DCMotorCfg | ImplicitActuatorCfg]:
    """Build G1_29DOF actuators with Unitree deploy FixStand Kp/Kd."""
    base = G1_29DOF_CFG.actuators
    legs_base: DCMotorCfg = base["legs"]
    feet_base: DCMotorCfg = base["feet"]
    waist_base: ImplicitActuatorCfg = base["waist"]
    arms_base: ImplicitActuatorCfg = base["arms"]
    hands_base: ImplicitActuatorCfg = base["hands"]

    return {
        "legs": legs_base.replace(
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
        ),
        "feet": feet_base.replace(
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
        ),
        "waist": waist_base.replace(
            stiffness={
                "waist_yaw_joint": 200.0,
                "waist_roll_joint": 200.0,
                "waist_pitch_joint": 200.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
        ),
        "arms": arms_base.replace(
            stiffness=40.0,
            damping=10.0,
        ),
        "hands": hands_base.replace(
            stiffness=40.0,
            damping=10.0,
        ),
    }


def build_g1_deploy_playback_actuators() -> dict[str, DCMotorCfg | ImplicitActuatorCfg]:
    """Deploy legs/feet + stiffer implicit upper body for per-frame MMD tracking."""
    base = G1_29DOF_CFG.actuators
    legs_base: DCMotorCfg = base["legs"]
    feet_base: DCMotorCfg = base["feet"]
    waist_base: ImplicitActuatorCfg = base["waist"]
    arms_base: ImplicitActuatorCfg = base["arms"]
    hands_base: ImplicitActuatorCfg = base["hands"]

    return {
        "legs": legs_base.replace(
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
        ),
        "feet": feet_base.replace(
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
        ),
        "waist": waist_base.replace(
            stiffness={
                "waist_yaw_joint": 600.0,
                "waist_roll_joint": 600.0,
                "waist_pitch_joint": 600.0,
            },
            damping={
                "waist_yaw_joint": 30.0,
                "waist_roll_joint": 30.0,
                "waist_pitch_joint": 30.0,
            },
        ),
        "arms": arms_base.replace(
            stiffness=500.0,
            damping=50.0,
        ),
        "hands": hands_base.replace(
            stiffness=500.0,
            damping=50.0,
        ),
    }


def apply_robot_pd_profile(robot_cfg: ArticulationCfg, profile: str) -> ArticulationCfg:
    """Return robot cfg with the selected actuator PD profile."""
    key = str(profile or "deploy_playback").strip().lower()
    if key in ("isaaclab", "default"):
        return robot_cfg
    if key in ("deploy", "unitree", "real", "hardware"):
        return robot_cfg.replace(actuators=build_g1_deploy_actuators())
    if key in ("deploy_playback", "playback", "mmd"):
        return robot_cfg.replace(actuators=build_g1_deploy_playback_actuators())
    raise ValueError(
        f"Unknown pd_profile '{profile}'. Expected: isaaclab, deploy, deploy_playback"
    )


def apply_pd_profile_to_scene_robot(robot_cfg: ArticulationCfg, profile: str) -> ArticulationCfg:
    """Re-base on G1_29DOF spawn/init, then apply PD (train/play entry points)."""
    base = G1_29DOF_CFG.replace(
        prim_path=robot_cfg.prim_path,
        init_state=robot_cfg.init_state,
    )
    return apply_robot_pd_profile(base, profile)


def log_pd_profile_summary(profile: str) -> None:
    """Print a short summary of the active PD profile."""
    key = str(profile or "deploy_playback").strip().lower()
    if key in ("isaaclab", "default"):
        print("[INFO] Actuator PD profile: isaaclab (G1_29DOF_CFG defaults)")
        print("[INFO]   legs/feet=DCMotor (hip~100-200, ankle~20), arms/waist implicit (~3000/5000)")
        return
    if key in ("deploy", "unitree", "real", "hardware"):
        print("[INFO] Actuator PD profile: deploy (Unitree rl_lab FixStand, all joints)")
        print("[INFO]   legs: hip Kp/Kd=100/2, knee 150/4; ankle 40/2")
        print("[INFO]   waist Kp/Kd=200/5; arms/hands Kp/Kd=40/10")
        print(
            "[WARN]   Pure deploy arms (Kp=40) lag on fast MMD playback; "
            "use --pd_profile deploy_playback for dance preview."
        )
        return
    print("[INFO] Actuator PD profile: deploy_playback (deploy legs + fast upper body)")
    print("[INFO]   legs: hip Kp/Kd=100/2, knee 150/4; ankle 40/2")
    print("[INFO]   waist Kp/Kd=600/30; arms/hands Kp/Kd=500/50")
    print("[INFO]   Toggle Mapping UI 'PD Drive' to drive joints with these sim actuators.")
