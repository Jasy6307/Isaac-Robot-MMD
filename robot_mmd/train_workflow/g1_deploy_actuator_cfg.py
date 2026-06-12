"""G1 actuator PD profiles used by playback/training scripts.

Profiles
--------
``deploy`` (default)
    Deploy legs/feet gains + stiffer upper body for better dance tracking.

``isaaclab``
    Isaac Lab `G1_29DOF_CFG` default gains.
"""

from __future__ import annotations

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import G1_29DOF_CFG

# Isaac Lab requires every regex in joint_names_expr to match at least one joint,
# and each joint must match exactly one regex (no overlap).
_G1_HAND_JOINT_EXPR = [
    # Keep a single pattern so robots without ring/pinky (e.g. only index/middle/thumb)
    # still satisfy Isaac Lab's "every regex must match at least one joint" rule.
    ".*_(index|middle|thumb|ring|pinky)_.*",
]
_O6_HAND_JOINT_EXPR = [
    "lh_.*",
    "rh_.*",
]
_HAND_DEPLOY_STIFFNESS = 500.0
_HAND_DEPLOY_DAMPING = 50.0


def build_g1_deploy_actuators(*, o6_hands: bool = False) -> dict[str, DCMotorCfg | ImplicitActuatorCfg]:
    """Build merged deploy profile (fast upper-body tracking)."""
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
            joint_names_expr=_O6_HAND_JOINT_EXPR if o6_hands else _G1_HAND_JOINT_EXPR,
            stiffness=_HAND_DEPLOY_STIFFNESS,
            damping=_HAND_DEPLOY_DAMPING,
        ),
    }


def build_g1_deploy_playback_actuators(*, o6_hands: bool = False) -> dict[str, DCMotorCfg | ImplicitActuatorCfg]:
    """Backward-compat alias. Kept for old imports/configs."""
    return build_g1_deploy_actuators(o6_hands=o6_hands)


def _normalize_pd_profile_key(profile: str) -> str:
    key = str(profile or "deploy").strip().lower()
    if key in ("deploy_playback", "playback", "mmd"):
        print(
            "[WARN] pd_profile 'deploy_playback' 已合并到 'deploy'；"
            "请后续改用 --pd_profile deploy"
        )
        return "deploy"
    return key


def _is_deploy_profile(key: str) -> bool:
    return key in ("deploy", "unitree", "real", "hardware")


def apply_robot_pd_profile(
    robot_cfg: ArticulationCfg,
    profile: str,
    *,
    o6_hands: bool = False,
) -> ArticulationCfg:
    """Return robot cfg with the selected actuator PD profile."""
    key = _normalize_pd_profile_key(profile)
    if key in ("isaaclab", "default"):
        return robot_cfg
    if _is_deploy_profile(key):
        return robot_cfg.replace(actuators=build_g1_deploy_actuators(o6_hands=o6_hands))
    raise ValueError(
        f"Unknown pd_profile '{profile}'. Expected: isaaclab, deploy"
    )


def apply_pd_profile_to_scene_robot(
    robot_cfg: ArticulationCfg,
    profile: str,
    *,
    o6_hands: bool = False,
) -> ArticulationCfg:
    """Apply PD profile while preserving scene-specific robot settings."""
    # Important: do not rebuild from G1_29DOF_CFG here.
    # C0/C1 env configs may already customize spawn.articulation_props
    # (e.g. C0 fix_root_link=True). We only want to swap actuator gains.
    return apply_robot_pd_profile(robot_cfg, profile, o6_hands=o6_hands)


def log_pd_profile_summary(profile: str, *, o6_hands: bool = False) -> None:
    """Print a short summary of the active PD profile."""
    key = _normalize_pd_profile_key(profile)
    if key in ("isaaclab", "default"):
        print("[INFO] Actuator PD profile: isaaclab (G1_29DOF_CFG defaults)")
        print("[INFO]   legs/feet=DCMotor (hip~100-200, ankle~20), arms/waist implicit (~3000/5000)")
        return
    if _is_deploy_profile(key):
        print("[INFO] Actuator PD profile: deploy (merged fast-tracking profile)")
        print("[INFO]   legs: hip Kp/Kd=100/2, knee 150/4; ankle 40/2")
        hand_label = "lh_/rh_" if o6_hands else "index/middle/thumb/ring/pinky"
        print(f"[INFO]   waist Kp/Kd=600/30; arms/hands Kp/Kd=500/50 ({hand_label})")
        return
    print("[WARN] Unknown pd_profile, fallback summary unavailable.")
