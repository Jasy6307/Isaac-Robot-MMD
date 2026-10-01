"""Isaac Lab 3 / Sim 6.1 boundaries. Motion files keep their WXYZ schema."""

from __future__ import annotations

import os


def as_torch(value):
    """Explicitly unwrap a Lab ProxyArray; tolerate missing optional properties."""
    return getattr(value, "torch", value)


def wxyz_to_xyzw(q):
    if isinstance(q, (list, tuple)):
        return [q[1], q[2], q[3], q[0]]
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    if isinstance(q, (list, tuple)):
        return [q[3], q[0], q[1], q[2]]
    return q[..., [3, 0, 1, 2]]


def write_joint_state(asset, position, velocity, *, joint_ids=None, env_ids=None):
    asset.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)


def write_root_state(asset, state, *, env_ids=None):
    """State is already in Lab's native XYZW order."""
    asset.write_root_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=state[..., 7:13], env_ids=env_ids)


def write_root_pose(asset, pose, *, env_ids=None):
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def write_root_velocity(asset, velocity, *, env_ids=None):
    asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def set_joint_position_target(asset, target, *, joint_ids=None, env_ids=None):
    asset.actuators.target_command.set_position_index(value=target, joint_ids=joint_ids, env_ids=env_ids)


def add_legacy_headless_arg(parser):
    parser.add_argument("--headless", action="store_true", help="Alias for --viz none (legacy project CLI).")


def prepare_rsl_rl_cfg(cfg):
    """Use Lab's official version adapter before passing configs to RSL-RL."""
    from importlib.metadata import version
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    return handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))


def prepare_launcher_args(args, *, default_visualizer="kit"):
    """Preserve the project's GUI default and old --headless command lines."""
    if getattr(args, "headless", False):
        args.visualizer = "none"
    elif getattr(args, "visualizer", None) is None:
        # Lab parses an explicit --viz none as None and records its intent
        # separately; distinguish it from an omitted visualizer option.
        args.visualizer = "none" if getattr(args, "visualizer_explicit", False) else default_visualizer
    selected = args.visualizer.split(",") if isinstance(args.visualizer, str) else args.visualizer
    args.headless = "kit" not in selected
    args.visualizer_explicit = True
    # Kit tokens also cover logs/user data that otherwise default to the Windows profile.
    runtime = os.environ.get("ISAAC61_ROOT", "I:/isaac61").replace("\\", "/")
    flags = [f"--/app/tokens/{name}={runtime}/{folder}" for name, folder in (
        ("cache", "cache/kit"), ("data", "data"), ("logs", "logs/kit"), ("documents", "documents"),
    )]
    flags.append(f"--/app/userConfigPath={runtime}/config/user.config.json")
    args.kit_args = (str(getattr(args, "kit_args", "") or "") + " " + " ".join(flags)).strip()

