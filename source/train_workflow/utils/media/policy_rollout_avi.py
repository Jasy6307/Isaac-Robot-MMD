# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause

"""Log policy rollout poses and export viewport AVI without disturbing the live policy loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from source.train_workflow.utils.media.viewport_avi_recorder import ViewportAviRecorder
from source.train_workflow.utils.playback.sim_robot import apply_root_pos_instant


@dataclass
class _PoseSnapshot:
    joint_pos: np.ndarray
    root_state_w: np.ndarray


class PolicyRolloutPoseLog:
    """CPU snapshots of robot articulation state after each policy control step."""

    def __init__(self) -> None:
        self._frames: list[_PoseSnapshot] = []

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def append_from_env(self, env: Any) -> None:
        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=True)
        root_state_w = (
            robot.data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float32, copy=True)
        )
        self._frames.append(_PoseSnapshot(joint_pos=joint_pos, root_state_w=root_state_w))


def export_pose_log_to_avi(
    env: Any,
    pose_log: PolicyRolloutPoseLog,
    recorder: ViewportAviRecorder,
    *,
    export_fps: float,
    control_hz: float,
    pre_render: Callable[[], None] | None = None,
) -> int:
    """Replay logged poses and capture ``export_fps`` AVI frames (decoupled from policy stepping)."""
    n_poses = len(pose_log)
    if n_poses <= 0:
        return 0

    export_fps = max(1.0, float(export_fps))
    control_hz = max(1.0, float(control_hz))
    n_video = max(1, int(round((n_poses - 1) * export_fps / control_hz)) + 1)

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = int(robot.data.joint_pos.shape[0])
    num_joints = int(robot.data.joint_pos.shape[1])

    written = 0
    for vi in range(n_video):
        pose_i = min(int(round(vi * control_hz / export_fps)), n_poses - 1)
        snap = pose_log._frames[pose_i]
        if snap.joint_pos.shape[0] != num_joints:
            continue
        q = (
            torch.tensor(snap.joint_pos, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )
        qd = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q, qd)
        rs = snap.root_state_w
        apply_root_pos_instant(
            env,
            (float(rs[0]), float(rs[1]), float(rs[2])),
            [float(rs[3]), float(rs[4]), float(rs[5]), float(rs[6])],
        )
        if pre_render is not None:
            try:
                pre_render()
            except Exception:
                pass
        if recorder.capture_frame():
            written += 1
    return written
