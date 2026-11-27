"""Body selection for the dance policy, keeping the 29 policy actions stable."""
from pathlib import Path

from source.my_task.robots.g1_29dof_o6_cfg import G1_29DOF_O6_USD_PATH
from source.train_workflow.utils.motion.neck_reference import NECK_JOINTS


G1_PLUS_DANCE_USD = Path(__file__).resolve().parents[3] / "assets/g1_plus_o6/g1_plus_o6_dance.usda"


def configure_dance_body(cfg, choice):
    """Apply after the requested body PD profile has been installed."""
    from isaaclab.actuators import ImplicitActuatorCfg

    robot = cfg.scene.robot
    if choice == "g1":
        robot.spawn.usd_path = G1_29DOF_O6_USD_PATH
    else:
        if not G1_PLUS_DANCE_USD.is_file():
            raise FileNotFoundError(f"G1+ dance USD missing: {G1_PLUS_DANCE_USD}")
        robot.spawn.usd_path = str(G1_PLUS_DANCE_USD)
        robot.init_state.joint_pos.update(dict.fromkeys(NECK_JOINTS, 0.0))
        robot.actuators["neck"] = ImplicitActuatorCfg(
            joint_names_expr=list(NECK_JOINTS),
            effort_limit_sim=dict(zip(NECK_JOINTS, (41., 13.4))),
            velocity_limit_sim=dict(zip(NECK_JOINTS, (34.1, 41.5))),
            stiffness=dict(zip(NECK_JOINTS, (40., 30.))),
            damping=dict(zip(NECK_JOINTS, (3., 2.))),
        )
        action = cfg.actions.joint_pos
        action.reference_only_joint_name_expr = [*action.reference_only_joint_name_expr, *NECK_JOINTS]
        cfg.events.reset_robot_joints.params["joint_noise_scale_by_expr"].update(dict.fromkeys(NECK_JOINTS, 0.0))
        for side in ("left", "right"):
            sensor = getattr(cfg.scene, f"jump_{side}_ground")
            sensor.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{side}_ankle_roll_link"
    print(f"[INFO] Dance robot: {choice} ({robot.spawn.usd_path})", flush=True)
