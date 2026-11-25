"""Read-only snapshots of articulation drives, in PhysX DOF order."""
import numpy as np


class JointTelemetry:
    def __init__(self, env):
        self.names = np.asarray(env.names, dtype=str)
        self.limits = env.limits.detach().cpu().numpy().copy()
        self.action_ids = (list(range(env.num_dofs)) if env.num_actions == env.num_dofs
                           else env.leg_ids.detach().cpu().tolist())
        if len(self.action_ids) != env.num_actions:
            raise ValueError('Cannot map policy actions to articulation DOFs')
        self.action_mask = np.zeros(env.num_dofs, dtype=bool)
        self.action_mask[self.action_ids] = True

    def read(self, env, *, reset=False, index=0):
        import torch
        # Stack before transfer: no extra CPU/CUDA synchronization per column.
        action = torch.zeros_like(env.q[index])
        action[self.action_ids] = env.actions[index]
        values = torch.stack((env.q[index], env.qd[index],
                              env.robot.get_dof_position_targets()[index],
                              env.robot.get_dof_velocity_targets()[index], action))
        arrays = values.detach().cpu().numpy().copy()
        return dict(joint_state_valid=True, joint_names=self.names, joint_limits=self.limits,
                    joint_action_mask=self.action_mask, joint_positions=arrays[0],
                    joint_velocities=arrays[1], joint_position_targets=arrays[2],
                    joint_velocity_targets=arrays[3], joint_actions=arrays[4],
                    joint_sample_time=float(env.physics_time), joint_state_reset=bool(reset))
