"""Optional physics-step jump telemetry; independent of H5 and policy formats."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np


def tensor(value):
    return getattr(value, "torch", value)


def configure_jump_sensors(env_cfg):
    from isaaclab.sensors import ContactSensorCfg
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    for side in ("left", "right"):
        ground_sensor = getattr(env_cfg.scene, f"jump_{side}_ground", None)
        link_path = (ground_sensor.prim_path if ground_sensor is not None else
                     f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_roll_link")
        setattr(env_cfg.scene, f"jump_{side}_contact", ContactSensorCfg(
            prim_path=link_path,
            update_period=0.0, history_length=max(1, int(env_cfg.decimation)),
        ))


def write_report(directory, rows, metadata):
    """Export raw physics samples plus readable overview and leg tracking plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rows:
        return
    names = list(rows[0])
    values = np.asarray([[row[key] for key in names] for row in rows], dtype=np.float64)
    with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(values)
    np.savez_compressed(directory / "samples.npz", columns=np.asarray(names), values=values)
    data = {key: values[:, i] for i, key in enumerate(names)}
    from source.train_workflow.utils.jump_analysis import export_analysis
    export_analysis(data, directory)
    t = data["time_s"]
    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    for key in ("root_z_m", "ref_root_z_m"):
        axes[0].plot(t, data[key], label=key)
    for key in ("root_vz_m_s", "ref_root_vz_m_s"):
        axes[1].plot(t, data[key], label=key)
    for side in ("left", "right"):
        axes[2].plot(t, data[f"{side}_contact_N"], label=side)
        axes[3].plot(t, data[f"{side}_foot_origin_height_m"], label=side)
    axes[4].plot(t, data["both_feet_no_contact"], label="both feet below force threshold")
    for ax, label in zip(axes, ("root Z (m)", "vertical velocity (m/s)", "foot normal force (N)",
                               "foot link origin above plane (m)", "no-contact indicator")):
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(alpha=.25)
    axes[-1].set_xlabel("rollout time (s)")
    fig.tight_layout()
    fig.savefig(directory / "overview.png", dpi=150)
    plt.close(fig)
    joints = metadata["leg_joint_names"]
    fig, axes = plt.subplots(len(joints), 2, figsize=(17, max(5, len(joints) * 2)), sharex=True, squeeze=False)
    for index, name in enumerate(joints):
        for kind in ("ref", "target", "actual"):
            axes[index, 0].plot(t, data[f"{name}_{kind}_rad"], label=kind)
        for kind in ("applied", "computed", "limit"):
            axes[index, 1].plot(t, data[f"{name}_{kind}_Nm"], label=kind)
        axes[index, 1].plot(t, -data[f"{name}_limit_Nm"], color="gray", linestyle="--")
        axes[index, 0].set_ylabel(name + "\nrad")
        axes[index, 1].set_ylabel("Nm")
        for ax in axes[index]:
            ax.grid(alpha=.25)
            ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("rollout time (s)")
    axes[-1, 1].set_xlabel("rollout time (s)")
    fig.tight_layout()
    fig.savefig(directory / "legs.png", dpi=120)
    plt.close(fig)


class JumpDiagnostics:
    """Hook scene updates inside step, before termination-triggered automatic reset.

    Physics samples pair a held control target with the reference that generated
    that target. ref_time_s makes this zero-order-hold timing explicit.
    """
    def __init__(self, env, output, env_index=0, threshold=5.0, start=0., end=None, extended=False):
        if getattr(env, "_physics_handles_decimation", False):
            raise ValueError("Jump diagnostics requires per-physics-step scene updates; this backend batches decimation")
        self.env = env
        self.output = Path(output)
        self.env_index = env_index
        self.threshold = threshold
        self.start, self.end = start, end
        self.extended = extended
        self.asset = env.scene["robot"]
        self.joint_ids = [i for i, name in enumerate(self.asset.joint_names)
                          if re.fullmatch(r"(left|right)_(hip_.*|knee|ankle_.*)_joint", name)]
        self.joint_names = [self.asset.joint_names[i] for i in self.joint_ids]
        if not self.joint_ids:
            raise ValueError("Jump diagnostics found no leg joints")
        self.foot_ids = [list(self.asset.body_names).index(f"{side}_ankle_roll_link")
                         for side in ("left", "right")]
        self.rows = []
        self.active = False
        self.pending = False
        self.original_update = env.scene.update
        def update(*args, **kwargs):
            result = self.original_update(*args, **kwargs)
            if self.active and self.pending:
                dt = kwargs.get("dt", args[0] if args else env.physics_dt)
                self.sample(float(dt))
            return result
        env.scene.update = update

    def begin(self, episode, buf, metadata):
        self.flush("restarted")
        self.episode = episode
        self.buf = buf
        self.metadata = dict(metadata, env_index=self.env_index, leg_joint_names=self.joint_names,
                             contact_threshold_N=self.threshold, control_hz=buf.control_hz,
                             physics_dt=float(self.env.physics_dt),
                             contact_source="ankle-roll link aggregate normal contact force; includes self contacts",
                             foot_height="link origin relative to terrain plane, not sole clearance",
                             torque_limits="configured effort limits; speed-dependent actuator clipping may be lower",
                             reference_timing="held current control-step reference; compare using ref_time_s")
        self.metadata["extended_turn_channels"]=self.extended
        if "jump_left_ground" in getattr(self.env.scene, "sensors", {}):
            self.metadata["contact_source"] = "ground-only filtered ankle normal forces"
        self.elapsed = 0.
        self.active = True

    def begin_control(self, steps):
        self.ref_step = int(steps[self.env_index].item())
        self.pending = True

    def sample(self, dt):
        import torch
        self.elapsed += dt
        if self.elapsed < self.start or (self.end is not None and self.elapsed > self.end):
            return
        e = self.env_index
        data = self.asset.data
        step = torch.tensor([self.ref_step], device=self.env.device)
        root_delta = self.buf.root_pos_delta(step)[0, 2]
        next_delta = self.buf.root_pos_delta(step + 1)[0, 2]
        origins = tensor(self.env.scene.sim.get_clone_plan().positions)
        root_z = tensor(data.default_root_state)[e, 2] + origins[e, 2] + root_delta
        ref_q = self.buf.q_ref_abs(step)[0]
        ground = float(getattr(self.env.cfg.scene.terrain, "ground_z_offset", 0.))
        row = dict(root_z_m=tensor(data.root_state_w)[e, 2], ref_root_z_m=root_z,
                   root_vz_m_s=tensor(data.root_lin_vel_w)[e, 2],
                   ref_root_vz_m_s=(next_delta - root_delta) * self.buf.control_hz)
        # Native Isaac Lab root quaternions are xyzw in this project/runtime.
        quat = tensor(data.root_state_w)[e, 3:7]
        quat = quat / quat.norm().clamp_min(1.e-8)
        row["root_tilt_rad"] = torch.acos((1.-2.*quat[:2].square().sum()).clamp(-1.,1.))
        if self.extended:
            x,y,z,w=quat.unbind()
            row["root_yaw_rad"]=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            w,x,y,z=self.buf.root_quat_wxyz(step)[0].unbind()
            row["ref_root_delta_yaw_rad"]=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        for axis, index in zip("xy", (0, 1)):
            row[f"root_{axis}_m"] = tensor(data.root_state_w)[e,index]-origins[e,index]
            row[f"ref_root_{axis}_m"] = tensor(data.default_root_state)[e,index]+self.buf.root_pos_delta(step)[0,index]
            row[f"root_v{axis}_m_s"] = tensor(data.root_lin_vel_w)[e,index]
        if hasattr(data, "root_ang_vel_w"):
            for index, axis in enumerate("xyz"):
                row[f"root_w{axis}_rad_s"] = tensor(data.root_ang_vel_w)[e,index]
        contacts = []
        for side, body in zip(("left", "right"), self.foot_ids):
            if f"jump_{side}_ground" in getattr(self.env.scene, "sensors", {}):
                sensor_data = self.env.scene[f"jump_{side}_ground"].data
                force = tensor(sensor_data.normal_force_matrix_w).sum(dim=2)
            else:
                sensor_data = self.env.scene[f"jump_{side}_contact"].data
                force = getattr(sensor_data, "net_normal_forces_w", None)
                if force is None:
                    force = sensor_data.net_forces_w
            magnitude = torch.linalg.vector_norm(tensor(force)[e, 0])
            contacts.append(magnitude >= self.threshold)
            row[f"{side}_contact_N"] = magnitude
            row[f"{side}_contact"] = contacts[-1].float()
            row[f"{side}_foot_origin_height_m"] = tensor(data.body_link_pose_w)[e, body, 2] - ground
            if self.extended:
                pose=tensor(data.body_link_pose_w)[e,body]
                velocity=tensor(data.body_link_lin_vel_w)[e,body]
                angular=tensor(data.body_link_ang_vel_w)[e,body]
                for k,axis in enumerate("xyz"):
                    row[f"{side}_foot_{axis}_m"]=pose[k]-origins[e,k]
                    row[f"{side}_foot_v{axis}_m_s"]=velocity[k]
                    row[f"{side}_foot_w{axis}_rad_s"]=angular[k]
                x,y,z,w=pose[3:7].unbind()
                row[f"{side}_foot_yaw_rad"]=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        row["both_feet_no_contact"] = (~torch.stack(contacts).any()).float()
        channels = {"target_rad": tensor(data.joint_pos_target), "actual_rad": tensor(data.joint_pos),
                    "applied_Nm": tensor(data.applied_torque), "computed_Nm": tensor(data.computed_torque),
                    "limit_Nm": tensor(data.joint_effort_limits)}
        for name, joint in zip(self.joint_names, self.joint_ids):
            row[f"{name}_ref_rad"] = ref_q[joint]
            for suffix, values in channels.items():
                row[f"{name}_{suffix}"] = values[e, joint]
            if self.extended:row[f"{name}_velocity_rad_s"]=tensor(data.joint_vel)[e,joint]
        # One GPU->CPU synchronization per physics sample, not one per scalar.
        values = torch.stack(list(row.values())).detach().cpu().tolist()
        self.rows.append(dict(time_s=self.elapsed, sample_dt_s=dt,
                              ref_time_s=self.ref_step / self.buf.control_hz, **dict(zip(row, values))))

    def end_control(self):
        self.pending = False

    def flush(self, reason):
        if not self.active:
            return
        self.active = self.pending = False
        self.metadata["end_reason"] = reason
        self.metadata["sample_count"] = len(self.rows)
        from datetime import datetime
        directory = self.output / f"episode_{self.episode:04d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        write_report(directory, self.rows, self.metadata)
        print(f"[DIAGNOSTICS] {len(self.rows)} physics samples: {directory.resolve()}")
        self.rows = []

    def close(self):
        self.env.scene.update = self.original_update
        self.flush("shutdown")
