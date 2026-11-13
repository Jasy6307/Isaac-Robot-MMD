"""Record Isaac Sim CSV playback targets into ``g1_mmd_motion_v1`` HDF5."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from source.train_workflow.utils.motion.sync import get_default_compile_joint_names
from source.train_workflow.utils.format.hdf5 import (
    HDF5_DEFAULT_FPS,
    Hdf5Motion,
    write_hdf5_motion,
)
from source.train_workflow.utils.math.trans_util import quat_inv, quat_mul, quat_normalize


def h5_sibling_path_for_csv(csv_path: str) -> str:
    """``foo.csv`` -> ``foo.h5`` (same directory)."""
    stem, _ext = os.path.splitext(os.path.abspath(str(csv_path)))
    return stem + ".h5"


@dataclass
class PlaybackH5Recorder:
    """Accumulate per-frame playback targets and emit ``Hdf5Motion``."""

    source_csv: str
    compile_joint_names: list[str]
    runtime_joint_names: list[str]
    baseline_joint_pos: np.ndarray
    root_anchor_pos: tuple[float, float, float]
    root_anchor_quat_wxyz: list[float]
    max_frame: int
    has_hand_data: bool = False
    fps: float = HDF5_DEFAULT_FPS
    knee_hinge_projection: bool = True
    root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, -1.0)
    root_quat_rpy_axis_idx: tuple[int, int, int] = (0, 1, 2)
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    groove_pos_to_world: float = 0.1
    _runtime_to_compile: np.ndarray = field(init=False, repr=False)
    _frames: np.ndarray = field(init=False, repr=False)
    _joint_pos_delta: np.ndarray = field(init=False, repr=False)
    _root_pos_delta: np.ndarray = field(init=False, repr=False)
    _root_quat_delta_wxyz: np.ndarray = field(init=False, repr=False)
    _root_valid: np.ndarray = field(init=False, repr=False)
    _root_rot_bone: list[str] = field(init=False, repr=False)
    _root_rpy_deg: np.ndarray = field(init=False, repr=False)
    _recorded: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        compile_index = {name: i for i, name in enumerate(self.compile_joint_names)}
        runtime_to_compile = np.full((len(self.runtime_joint_names),), -1, dtype=np.int64)
        for i, name in enumerate(self.runtime_joint_names):
            if name in compile_index:
                runtime_to_compile[i] = int(compile_index[name])
        self._runtime_to_compile = runtime_to_compile

        n_frames = int(self.max_frame) + 1
        self._frames = np.arange(n_frames, dtype=np.int32)
        n_j = len(self.compile_joint_names)
        self._joint_pos_delta = np.zeros((n_frames, n_j), dtype=np.float32)
        self._root_pos_delta = np.zeros((n_frames, 3), dtype=np.float32)
        self._root_quat_delta_wxyz = np.zeros((n_frames, 4), dtype=np.float32)
        self._root_quat_delta_wxyz[:, 0] = 1.0
        self._root_valid = np.zeros((n_frames,), dtype=np.bool_)
        self._root_rot_bone = [""] * n_frames
        self._root_rpy_deg = np.zeros((n_frames, 3), dtype=np.float32)

        baseline = np.asarray(self.baseline_joint_pos, dtype=np.float32).reshape(-1)
        if baseline.shape[0] != len(self.runtime_joint_names):
            raise ValueError("baseline_joint_pos length must match runtime_joint_names")

    @classmethod
    def begin(
        cls,
        *,
        source_csv: str,
        runtime_joint_names: list[str],
        baseline_joint_pos: Any,
        root_anchor_pos: tuple[float, float, float],
        root_anchor_quat_wxyz: list[float],
        max_frame: int,
        has_hand_data: bool,
        compile_joint_names: list[str] | None = None,
        fps: float = HDF5_DEFAULT_FPS,
        knee_hinge_projection: bool = True,
        root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, -1.0),
        root_quat_rpy_axis_idx: tuple[int, int, int] = (0, 1, 2),
        mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        groove_pos_to_world: float = 0.1,
    ) -> PlaybackH5Recorder:
        names = list(compile_joint_names or get_default_compile_joint_names())
        return cls(
            source_csv=os.path.abspath(str(source_csv)),
            compile_joint_names=names,
            runtime_joint_names=list(runtime_joint_names),
            baseline_joint_pos=np.asarray(baseline_joint_pos, dtype=np.float32),
            root_anchor_pos=(
                float(root_anchor_pos[0]),
                float(root_anchor_pos[1]),
                float(root_anchor_pos[2]),
            ),
            root_anchor_quat_wxyz=quat_normalize([float(v) for v in root_anchor_quat_wxyz]),
            max_frame=int(max_frame),
            has_hand_data=bool(has_hand_data),
            fps=float(fps),
            knee_hinge_projection=bool(knee_hinge_projection),
            root_quat_rpy_scale=(
                float(root_quat_rpy_scale[0]),
                float(root_quat_rpy_scale[1]),
                float(root_quat_rpy_scale[2]),
            ),
            root_quat_rpy_axis_idx=(
                int(root_quat_rpy_axis_idx[0]),
                int(root_quat_rpy_axis_idx[1]),
                int(root_quat_rpy_axis_idx[2]),
            ),
            mmd_center_to_root_offset_local_xyz=(
                float(mmd_center_to_root_offset_local_xyz[0]),
                float(mmd_center_to_root_offset_local_xyz[1]),
                float(mmd_center_to_root_offset_local_xyz[2]),
            ),
            groove_pos_to_world=float(groove_pos_to_world),
        )

    def record_frame(
        self,
        frame_idx: int,
        joint_pos_cmd: Any,
        *,
        root_pos: tuple[float, float, float] | None,
        root_quat_wxyz: list[float] | None,
        root_rot_bone: str | None = None,
        root_rpy_deg: tuple[float, float, float] | None = None,
    ) -> None:
        f = int(frame_idx)
        if f < 0 or f > int(self.max_frame):
            return
        if f in self._recorded:
            return
        self._recorded.add(f)

        runtime_cmd = np.asarray(joint_pos_cmd, dtype=np.float32).reshape(-1)
        baseline = np.asarray(self.baseline_joint_pos, dtype=np.float32).reshape(-1)
        for i, compile_j in enumerate(self._runtime_to_compile):
            if compile_j < 0:
                continue
            self._joint_pos_delta[f, int(compile_j)] = float(runtime_cmd[i] - baseline[i])

        self._root_rot_bone[f] = str(root_rot_bone or "")
        if root_rpy_deg is not None:
            self._root_rpy_deg[f, :] = np.asarray(root_rpy_deg, dtype=np.float32)

        if root_pos is not None and root_quat_wxyz is not None:
            ax, ay, az = self.root_anchor_pos
            # root_pos is the post-compress playback target (see compute_targets_for_motion_frame).
            self._root_pos_delta[f, :] = np.asarray(
                [
                    float(root_pos[0]) - ax,
                    float(root_pos[1]) - ay,
                    float(root_pos[2]) - az,
                ],
                dtype=np.float32,
            )
            dq = quat_normalize(
                quat_mul(
                    quat_normalize([float(v) for v in root_quat_wxyz]),
                    quat_inv(self.root_anchor_quat_wxyz),
                )
            )
            self._root_quat_delta_wxyz[f, :] = np.asarray(dq, dtype=np.float32)
            self._root_valid[f] = True

    def missing_frames(self) -> list[int]:
        expected = set(range(int(self.max_frame) + 1))
        return sorted(expected - self._recorded)

    def build_motion(self) -> Hdf5Motion:
        missing = self.missing_frames()
        if missing:
            preview = ", ".join(str(v) for v in missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise ValueError(f"H5 recording incomplete; missing frames: {preview}{suffix}")
        return Hdf5Motion(
            frames=self._frames.copy(),
            joint_names=[str(n) for n in self.compile_joint_names],
            joint_pos_delta=self._joint_pos_delta.copy(),
            root_pos_delta=self._root_pos_delta.copy(),
            root_quat_delta_wxyz=self._root_quat_delta_wxyz.copy(),
            root_valid=self._root_valid.copy(),
            root_rot_bone=list(self._root_rot_bone),
            root_rpy_deg=self._root_rpy_deg.copy(),
            source_csv=str(self.source_csv),
            has_hand_data=bool(self.has_hand_data),
            fps=float(self.fps),
            knee_hinge_projection=bool(self.knee_hinge_projection),
            root_quat_rpy_scale=self.root_quat_rpy_scale,
            root_quat_rpy_axis_idx=self.root_quat_rpy_axis_idx,
            mmd_center_to_root_offset_local_xyz=self.mmd_center_to_root_offset_local_xyz,
            groove_pos_to_world=float(self.groove_pos_to_world),
        )

    def write(self, path: str | None = None, *, upper_body_source_animation: str | None = None,
              lower_body_source_animation: str | None = None, retarget_mode: str = 'new',
              source_vmd: str | None = None, source_leg_mode: str = 'fk') -> str:
        if retarget_mode not in ('new','legacy'):
            raise ValueError('retarget_mode must be new or legacy')
        if retarget_mode == 'new' and not (upper_body_source_animation and lower_body_source_animation):
            from source.train_workflow.utils.retarget.source_animation import prepare_source_animation
            source = prepare_source_animation(self.source_csv,vmd=source_vmd,leg_mode=source_leg_mode)
            upper_body_source_animation = upper_body_source_animation or str(source)
            lower_body_source_animation = lower_body_source_animation or str(source)
        if retarget_mode == 'legacy' and (upper_body_source_animation or lower_body_source_animation):
            raise ValueError('Legacy mode cannot use new limb solvers')
        out = h5_sibling_path_for_csv(self.source_csv) if path is None else os.path.abspath(str(path))
        motion = self.build_motion()
        baseline = np.zeros(len(self.compile_joint_names), dtype=np.float32)
        for runtime_i, compile_i in enumerate(self._runtime_to_compile):
            if compile_i >= 0:
                baseline[compile_i] = self.baseline_joint_pos[runtime_i]
        report = None
        lower_report = None
        joint_report = None
        if upper_body_source_animation:
            from pathlib import Path
            from source.train_workflow.utils.retarget.upper_body_export import solve_upper_body_arrays
            robot_usd = Path(__file__).resolve().parents[4] / 'assets/g1_29dof_o6_hand_V3.usd'
            motion.joint_pos_delta, report = solve_upper_body_arrays(
                motion.joint_pos_delta, motion.joint_names, baseline, motion.fps,
                upper_body_source_animation, robot_usd)
            report['source_csv'] = self.source_csv
            report['export_mode'] = 'direct-in-memory'
        if lower_body_source_animation:
            from pathlib import Path
            from scipy.spatial.transform import Rotation
            from source.train_workflow.utils.retarget.lower_body_ik import solve_lower_body_arrays
            robot_usd = Path(__file__).resolve().parents[4] / 'assets/g1_29dof_o6_hand_V3.usd'
            root_pos = motion.root_pos_delta + np.asarray(self.root_anchor_pos)
            dq = motion.root_quat_delta_wxyz[:, [1,2,3,0]]
            aq = np.asarray(self.root_anchor_quat_wxyz)[[1,2,3,0]]
            root_q = (Rotation.from_quat(dq)*Rotation.from_quat(aq)).as_quat()[:,[3,0,1,2]]
            motion.joint_pos_delta, lower_report = solve_lower_body_arrays(
                motion.joint_pos_delta, motion.joint_names, baseline, motion.fps,
                lower_body_source_animation, robot_usd, root_pos, root_q)
            lower_report['export_mode'] = 'direct-in-memory'
        if retarget_mode == 'new':
            from pathlib import Path
            from source.train_workflow.utils.motion.joint_constraints import constrain_joint_positions
            robot_usd = Path(__file__).resolve().parents[4] / 'assets/g1_29dof_o6_hand_V3.usd'
            absolute,joint_report = constrain_joint_positions(
                motion.joint_pos_delta+baseline,motion.joint_names,motion.fps,robot_usd)
            motion.joint_pos_delta = (absolute-baseline).astype(np.float32)
        # No H5 exists until the complete upper-body solve has succeeded.
        out = write_hdf5_motion(out, motion)
        import h5py
        with h5py.File(out, "r+") as stream:
            stream.attrs['retarget_mode'] = retarget_mode
            stream.attrs["record_baseline_joint_pos"] = baseline
            stream.attrs["record_root_anchor_quat_wxyz"] = self.root_anchor_quat_wxyz
            stream.attrs["record_root_anchor_pos"] = self.root_anchor_pos
            if report:
                stream.attrs['upper_body_retarget_method'] = report['method']
                stream.attrs['upper_body_source_animation'] = str(upper_body_source_animation)
                stream.attrs['upper_body_export_mode'] = 'direct-in-memory'
            if lower_report:
                stream.attrs['lower_body_retarget_method'] = lower_report['method']
                stream.attrs['lower_body_source_animation'] = str(lower_body_source_animation)
                stream.attrs['lower_body_export_mode'] = 'direct-in-memory'
                stream.attrs['source_leg_mode'] = lower_report.get('source_leg_mode',source_leg_mode)
            if joint_report:
                stream.attrs['all_joint_constraints_validated'] = joint_report['after']['passed']
                stream.attrs['mimic_constraints_validated'] = True
        if report:
            import json
            from pathlib import Path
            Path(out).with_suffix('.upper_body.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if lower_report:
            import json
            from pathlib import Path
            Path(out).with_suffix('.lower_body.json').write_text(
                json.dumps(lower_report,ensure_ascii=False,indent=2),encoding='utf-8')
        if joint_report:
            import json
            from pathlib import Path
            Path(out).with_suffix('.joint_limits.json').write_text(
                json.dumps(joint_report,ensure_ascii=False,indent=2),encoding='utf-8')
        return out
