"""HDF5 robot-motion cache for G1 playback.

Stores precompiled per-frame joint deltas and root deltas so runtime playback
can skip CSV bone interpolation and retarget mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from robot_mmd.train_workflow.utils.csv_motion_loader import (
    FootIkConfig,
    FootIkState,
    build_joint_positions_from_frame,
    frames_have_hand_data,
    get_bone_frame_lists,
    get_frame_indices,
    is_hand_joint_name,
    interpolate_bone,
    load_csv_motion,
)
from robot_mmd.train_workflow.utils.trans_util import (
    mmd_root_offset_quat_to_world,
    quat_from_waist_extrinsic_xyz,
    quat_mul,
    quat_normalize,
    remap_root_csv_euler_xyz,
    rotate_vec_by_quat_wxyz,
)
from robot_mmd.train_workflow.retarget_unitreeG1 import euler_xyz_rad_waist_extrinsic

HDF5_SCHEMA_VERSION = "g1_mmd_motion_v1"
HDF5_DEFAULT_FPS = 30.0


@dataclass
class Hdf5Motion:
    frames: np.ndarray  # int32 [F]
    joint_names: list[str]  # len J
    joint_pos_delta: np.ndarray  # float32 [F, J]
    root_pos_delta: np.ndarray  # float32 [F, 3]
    root_quat_delta_wxyz: np.ndarray  # float32 [F, 4]
    root_valid: np.ndarray  # bool [F]
    root_rot_bone: list[str]  # len F
    root_rpy_deg: np.ndarray  # float32 [F, 3]
    source_csv: str = ""
    has_hand_data: bool = False
    fps: float = HDF5_DEFAULT_FPS
    knee_hinge_projection: bool = True
    root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, -1.0)
    root_quat_rpy_axis_idx: tuple[int, int, int] = (0, 1, 2)
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    groove_pos_to_world: float = 0.1
    _runtime_joint_index_cache: dict[tuple[str, ...], np.ndarray] | None = None
    _runtime_missing_warned: set[tuple[str, ...]] | None = None

    def build_runtime_joint_index(self, runtime_joint_names: list[str]) -> np.ndarray:
        key = tuple(runtime_joint_names)
        cache = self._runtime_joint_index_cache
        if cache is None:
            cache = {}
            self._runtime_joint_index_cache = cache
        if key in cache:
            return cache[key]
        src_index = {name: i for i, name in enumerate(self.joint_names)}
        idx = np.full((len(runtime_joint_names),), -1, dtype=np.int64)
        missing: list[str] = []
        for i, name in enumerate(runtime_joint_names):
            if name not in src_index:
                missing.append(name)
                continue
            idx[i] = int(src_index[name])
        if missing:
            warned = self._runtime_missing_warned
            if warned is None:
                warned = set()
                self._runtime_missing_warned = warned
            mk = tuple(missing)
            if mk not in warned:
                warned.add(mk)
                print(
                    "[WARN] HDF5 缺少部分运行时关节，将使用 default_joint_pos 回退: "
                    + ", ".join(missing)
                )
        cache[key] = idx
        return idx


def infer_hdf5_has_hand_data(motion: Hdf5Motion) -> bool:
    """True when HDF5 metadata or compiled hand-joint deltas indicate finger motion."""
    if bool(getattr(motion, "has_hand_data", False)):
        return True
    try:
        arr = np.asarray(motion.joint_pos_delta)
        for i, jn in enumerate(motion.joint_names):
            if not is_hand_joint_name(str(jn)):
                continue
            if float(np.max(np.abs(arr[:, i]))) > 1e-6:
                return True
    except Exception:
        pass
    return False


def _ensure_h5py():
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("使用 HDF5 需要安装 h5py（例如 pip install h5py）") from exc
    return h5py


def _as_str_list(arr: Any) -> list[str]:
    out: list[str] = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def write_hdf5_motion(path: str, motion: Hdf5Motion) -> str:
    h5py = _ensure_h5py()
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with h5py.File(out_path, "w") as f:
        utf8 = h5py.string_dtype(encoding="utf-8")
        f.attrs["schema_version"] = HDF5_SCHEMA_VERSION
        f.attrs["fps"] = float(motion.fps)
        f.attrs["source_csv"] = str(motion.source_csv)
        f.attrs["has_hand_data"] = bool(motion.has_hand_data)
        f.attrs["knee_hinge_projection"] = bool(motion.knee_hinge_projection)
        f.attrs["root_quat_rpy_scale"] = np.asarray(motion.root_quat_rpy_scale, dtype=np.float32)
        f.attrs["root_quat_rpy_axis_idx"] = np.asarray(motion.root_quat_rpy_axis_idx, dtype=np.int32)
        f.attrs["mmd_center_to_root_offset_local_xyz"] = np.asarray(
            motion.mmd_center_to_root_offset_local_xyz, dtype=np.float32
        )
        f.attrs["groove_pos_to_world"] = float(motion.groove_pos_to_world)

        f.create_dataset("frames", data=np.asarray(motion.frames, dtype=np.int32))
        f.create_dataset("joint_names", data=np.asarray(motion.joint_names, dtype=utf8))
        f.create_dataset("joint_pos_delta", data=np.asarray(motion.joint_pos_delta, dtype=np.float32))
        f.create_dataset("root_pos_delta", data=np.asarray(motion.root_pos_delta, dtype=np.float32))
        f.create_dataset("root_quat_delta_wxyz", data=np.asarray(motion.root_quat_delta_wxyz, dtype=np.float32))
        f.create_dataset("root_valid", data=np.asarray(motion.root_valid, dtype=np.bool_))

        debug = f.create_group("debug")
        debug.create_dataset("root_rot_bone", data=np.asarray(motion.root_rot_bone, dtype=utf8))
        debug.create_dataset("root_rpy_deg", data=np.asarray(motion.root_rpy_deg, dtype=np.float32))
    return out_path


def load_hdf5_motion(path: str) -> Hdf5Motion:
    h5py = _ensure_h5py()
    abs_path = os.path.abspath(path)
    with h5py.File(abs_path, "r") as f:
        schema = str(f.attrs.get("schema_version", ""))
        if schema != HDF5_SCHEMA_VERSION:
            raise ValueError(f"HDF5 schema 不匹配: {schema} != {HDF5_SCHEMA_VERSION}")

        frames = np.asarray(f.get("frames"), dtype=np.int32)
        joint_names = _as_str_list(np.asarray(f.get("joint_names")))
        joint_pos_delta = np.asarray(f.get("joint_pos_delta"), dtype=np.float32)
        root_pos_delta = np.asarray(f.get("root_pos_delta"), dtype=np.float32)
        root_quat_delta_wxyz = np.asarray(f.get("root_quat_delta_wxyz"), dtype=np.float32)
        root_valid = np.asarray(f.get("root_valid"), dtype=np.bool_)
        root_rot_bone = _as_str_list(np.asarray(f.get("debug/root_rot_bone"))) if "debug/root_rot_bone" in f else [""] * len(frames)
        if "debug/root_rpy_deg" in f:
            root_rpy_deg = np.asarray(f.get("debug/root_rpy_deg"), dtype=np.float32)
        else:
            root_rpy_deg = np.zeros((len(frames), 3), dtype=np.float32)

        if joint_pos_delta.ndim != 2 or joint_pos_delta.shape[0] != len(frames):
            raise ValueError("joint_pos_delta 维度非法")
        if root_pos_delta.shape != (len(frames), 3):
            raise ValueError("root_pos_delta 维度非法")
        if root_quat_delta_wxyz.shape != (len(frames), 4):
            raise ValueError("root_quat_delta_wxyz 维度非法")
        if root_valid.shape != (len(frames),):
            raise ValueError("root_valid 维度非法")
        if root_rpy_deg.shape != (len(frames), 3):
            raise ValueError("debug/root_rpy_deg 维度非法")

        rq_scale = tuple(float(v) for v in np.asarray(f.attrs.get("root_quat_rpy_scale", [1.0, 1.0, -1.0])).tolist())
        rq_idx = tuple(int(v) for v in np.asarray(f.attrs.get("root_quat_rpy_axis_idx", [0, 1, 2])).tolist())
        center_off = tuple(
            float(v)
            for v in np.asarray(f.attrs.get("mmd_center_to_root_offset_local_xyz", [0.0, 0.0, 0.0])).tolist()
        )

        return Hdf5Motion(
            frames=frames,
            joint_names=joint_names,
            joint_pos_delta=joint_pos_delta,
            root_pos_delta=root_pos_delta,
            root_quat_delta_wxyz=root_quat_delta_wxyz,
            root_valid=root_valid,
            root_rot_bone=root_rot_bone,
            root_rpy_deg=root_rpy_deg,
            source_csv=str(f.attrs.get("source_csv", "")),
            has_hand_data=bool(f.attrs.get("has_hand_data", False)),
            fps=float(f.attrs.get("fps", HDF5_DEFAULT_FPS)),
            knee_hinge_projection=bool(f.attrs.get("knee_hinge_projection", True)),
            root_quat_rpy_scale=(rq_scale[0], rq_scale[1], rq_scale[2]),
            root_quat_rpy_axis_idx=(rq_idx[0], rq_idx[1], rq_idx[2]),
            mmd_center_to_root_offset_local_xyz=(center_off[0], center_off[1], center_off[2]),
            groove_pos_to_world=float(f.attrs.get("groove_pos_to_world", 0.1)),
        )


def _read_bone_quat_wxyz(frame_data: dict[str, dict], bone_name: str) -> list[float] | None:
    d = frame_data.get(bone_name)
    if d is None:
        return None
    q = d.get("quat_wxyz")
    if q is None or len(q) != 4:
        return None
    try:
        return [float(v) for v in q]
    except Exception:
        return None


def _get_csv_root_quat_with_bone_from_frame(
    frame_data: dict[str, dict],
    bone_frame_lists: dict[str, list[int]],
) -> tuple[str | None, list[float] | None]:
    candidates = ("下半身", "グルーブ", "センター親", "腰", "センター")
    for require_dynamic in (True, False):
        for bone in candidates:
            keyframes = bone_frame_lists.get(bone) or []
            if require_dynamic and len(keyframes) <= 1:
                continue
            q = _read_bone_quat_wxyz(frame_data, bone)
            if q is None:
                continue
            return bone, quat_normalize(q)
    return None, None


def _interpolate_mmd_root_translation_bone_cfg(
    frame_data: dict[str, dict],
    bone_frame_lists: dict[str, list[int]],
) -> tuple[str | None, tuple[float, float, float] | None]:
    g_list = bone_frame_lists.get("グルーブ") or []
    c_list = bone_frame_lists.get("センター") or []
    order: tuple[str, ...]
    if c_list and len(c_list) > len(g_list):
        order = ("センター", "グルーブ")
    else:
        order = ("グルーブ", "センター")
    for bone in order:
        d = frame_data.get(bone)
        if d is None or "pos" not in d:
            continue
        try:
            gx, gy, gz = d["pos"]
            return bone, (float(gx), float(gy), float(gz))
        except Exception:
            continue
    return None, None


def compile_csv_motion_to_hdf5_motion(
    csv_path: str,
    joint_names: list[str],
    *,
    fps: float = HDF5_DEFAULT_FPS,
    knee_hinge_projection: bool = True,
    groove_pos_to_world: float = 0.1,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_quat_rpy_scale: tuple[float, float, float] = (1.0, 1.0, -1.0),
    root_quat_rpy_axis_idx: tuple[int, int, int] = (0, 1, 2),
    foot_ik_cfg: FootIkConfig | None = None,
) -> Hdf5Motion:
    frames_map = load_csv_motion(csv_path)
    has_hand_data = bool(frames_have_hand_data(frames_map))
    frame_list = get_frame_indices(frames_map)
    if not frame_list:
        raise ValueError(f"CSV 无有效帧: {csv_path}")

    all_bones = set()
    for f in frames_map.values():
        all_bones.update(f.keys())
    bone_frame_lists = get_bone_frame_lists(frames_map, frame_list, all_bones)

    max_frame = int(frame_list[-1])
    frames = np.arange(max_frame + 1, dtype=np.int32)

    # 离线预编译用零基准；回放时以运行时 default_joint_pos + delta 重构绝对 joint_pos。
    zero_default = np.zeros((len(joint_names),), dtype=np.float32)
    joint_pos_delta = np.zeros((len(frames), len(joint_names)), dtype=np.float32)
    root_pos_delta = np.zeros((len(frames), 3), dtype=np.float32)
    root_quat_delta = np.zeros((len(frames), 4), dtype=np.float32)
    root_valid = np.zeros((len(frames),), dtype=np.bool_)
    root_rot_bone: list[str] = [""] * len(frames)
    root_rpy_deg = np.zeros((len(frames), 3), dtype=np.float32)

    is_pose = len(frame_list) <= 1
    cfg = foot_ik_cfg
    if cfg is not None:
        cfg = FootIkConfig(**vars(cfg))
        cfg.is_static_pose = bool(is_pose)
        cfg.groove_pos_to_world = float(groove_pos_to_world)
    foot_state = FootIkState()
    for frame in frames:
        frame_data: dict[str, dict] = {}
        for bone in all_bones:
            d = interpolate_bone(int(frame), bone, frames_map, bone_frame_lists.get(bone))
            if d is not None:
                frame_data[bone] = d

        foot_ik_root_pos: tuple[float, float, float] | None = None
        foot_ik_root_quat: list[float] | None = None
        bone_name, mmd_pos = _interpolate_mmd_root_translation_bone_cfg(frame_data, bone_frame_lists)
        _, csv_root_quat_wxyz = _get_csv_root_quat_with_bone_from_frame(frame_data, bone_frame_lists)
        root_rot_bone[int(frame)] = bone_name or ""
        if mmd_pos is not None and csv_root_quat_wxyz is not None:
            dx = float(mmd_pos[0]) * float(groove_pos_to_world)
            dy = float(mmd_pos[1]) * float(groove_pos_to_world)
            dz = float(mmd_pos[2]) * float(groove_pos_to_world)
            if is_pose:
                pos = np.array([-dx, +dz, +dy], dtype=np.float32)
            else:
                pos = np.array([-dx, -dz, +dy], dtype=np.float32)

            q_w = mmd_root_offset_quat_to_world(csv_root_quat_wxyz)
            qx, qy, qz, qw = q_w[1], q_w[2], q_w[3], q_w[0]
            rr, rp, ry = euler_xyz_rad_waist_extrinsic((qx, qy, qz, qw))
            out_r, out_p, out_y = remap_root_csv_euler_xyz(rr, rp, ry, root_quat_rpy_axis_idx, root_quat_rpy_scale)
            q_delta = quat_normalize(quat_from_waist_extrinsic_xyz(out_r, out_p, out_y))
            if (
                abs(mmd_center_to_root_offset_local_xyz[0]) > 1e-12
                or abs(mmd_center_to_root_offset_local_xyz[1]) > 1e-12
                or abs(mmd_center_to_root_offset_local_xyz[2]) > 1e-12
            ):
                dv = rotate_vec_by_quat_wxyz(q_delta, mmd_center_to_root_offset_local_xyz)
                pos = pos + np.asarray(dv, dtype=np.float32)

            root_pos_delta[int(frame), :] = pos
            root_quat_delta[int(frame), :] = np.asarray(q_delta, dtype=np.float32)
            root_valid[int(frame)] = True
            root_rpy_deg[int(frame), :] = np.array(
                [np.degrees(out_r), np.degrees(out_p), np.degrees(out_y)],
                dtype=np.float32,
            )
            foot_ik_root_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            foot_ik_root_quat = [float(v) for v in q_delta.tolist()]

        if frame_data:
            target_pos = build_joint_positions_from_frame(
                frame_data,
                joint_names,
                zero_default,
                knee_hinge_projection=knee_hinge_projection,
                enable_hand=has_hand_data,
                foot_ik_cfg=cfg,
                foot_ik_state=foot_state,
                foot_ik_frame_idx=int(frame),
                foot_ik_root_pos_world=foot_ik_root_pos,
                foot_ik_root_quat_wxyz=foot_ik_root_quat,
            )
            joint_pos_delta[int(frame), :] = target_pos.astype(np.float32, copy=False)

    return Hdf5Motion(
        frames=frames,
        joint_names=[str(n) for n in joint_names],
        joint_pos_delta=joint_pos_delta,
        root_pos_delta=root_pos_delta,
        root_quat_delta_wxyz=root_quat_delta,
        root_valid=root_valid,
        root_rot_bone=root_rot_bone,
        root_rpy_deg=root_rpy_deg,
        source_csv=os.path.abspath(csv_path),
        has_hand_data=has_hand_data,
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


def sample_hdf5_frame(
    motion: Hdf5Motion,
    frame_idx: int,
    runtime_joint_names: list[str],
    default_joint_pos: np.ndarray,
    root_anchor_pos: tuple[float, float, float] | None,
    root_anchor_quat_wxyz: list[float] | None,
) -> tuple[np.ndarray, tuple[float, float, float] | None, list[float] | None, dict[str, Any]]:
    if motion.frames.size <= 0:
        raise ValueError("空 HDF5 轨迹")
    f = int(max(0, min(frame_idx, int(motion.frames[-1]))))

    idx = motion.build_runtime_joint_index(runtime_joint_names)
    delta = np.zeros((len(runtime_joint_names),), dtype=np.float32)
    valid = idx >= 0
    if np.any(valid):
        delta[valid] = motion.joint_pos_delta[f, idx[valid]]
    if not bool(getattr(motion, "has_hand_data", False)):
        for i, jn in enumerate(runtime_joint_names):
            if is_hand_joint_name(jn):
                delta[i] = 0.0
    joint_pos_cmd = np.asarray(default_joint_pos, dtype=np.float32) + np.asarray(delta, dtype=np.float32)

    root_pos: tuple[float, float, float] | None = None
    root_quat: list[float] | None = None
    root_ok = bool(motion.root_valid[f])
    if root_ok and root_anchor_pos is not None and root_anchor_quat_wxyz is not None:
        dpos = motion.root_pos_delta[f]
        root_pos = (
            float(root_anchor_pos[0] + float(dpos[0])),
            float(root_anchor_pos[1] + float(dpos[1])),
            float(root_anchor_pos[2] + float(dpos[2])),
        )
        dq = [float(v) for v in motion.root_quat_delta_wxyz[f].tolist()]
        root_quat = quat_normalize(quat_mul(dq, list(root_anchor_quat_wxyz)))

    debug = {
        "root_rpy_deg": (
            float(motion.root_rpy_deg[f, 0]),
            float(motion.root_rpy_deg[f, 1]),
            float(motion.root_rpy_deg[f, 2]),
        ),
        "root_rot_bone": motion.root_rot_bone[f] if f < len(motion.root_rot_bone) else "",
        "root_valid": root_ok,
    }
    return joint_pos_cmd, root_pos, root_quat, debug

