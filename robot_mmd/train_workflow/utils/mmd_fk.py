"""MMD standard bone hierarchy FK for foot IK world positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from robot_mmd.train_workflow.utils.trans_util import (
    mmd_world_pos_to_isaac,
    quat_mul,
    quat_normalize,
    remap_mmd_world_to_isaac,
    rotate_vec_by_quat_wxyz,
)


@dataclass
class FootIkVizConfig:
    """Runtime tuning for red-sphere B-aligned model world -> Isaac-world axis mapping."""

    # B-aligned model world (mmdX, mmdZ, mmdY) -> Isaac; idx 0,1,2 direct.
    axis_idx: tuple[int, int, int] = (0, 1, 2)
    axis_sign: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    axis_sign_pose: tuple[float, float, float] = (-1.0, 1.0, 1.0)
    pos_scale: float = 1.0
    weight: float = 1.0
    # Isaac-world rest origin when foot IK panel offset is zero (meters).
    left_ref_origin_m: tuple[float, float, float] = (-0.15, -0.15, 0.0)
    right_ref_origin_m: tuple[float, float, float] = (0.15, -0.15, 0.0)


def foot_ik_viz_ref_origin_isaac(
    side: str,
    viz_cfg: FootIkVizConfig | None = None,
) -> tuple[float, float, float]:
    if viz_cfg is not None:
        if side == "left":
            lv = viz_cfg.left_ref_origin_m
            return (float(lv[0]), float(lv[1]), float(lv[2]))
        rv = viz_cfg.right_ref_origin_m
        return (float(rv[0]), float(rv[1]), float(rv[2]))
    if side == "left":
        return (-0.15, -0.15, 0.0)
    return (0.15, -0.15, 0.0)

# Standard MMD parent links (足IK親 / センター親 are helper bones, often not keyed in VMD).
MMD_BONE_PARENT: dict[str, str | None] = {
    "グルーブ": None,
    "センター": "グルーブ",
    "センター先": "センター",
    "センター親": "グルーブ",
    "上半身": "センター",
    "上半身2": "上半身",
    "下半身": "センター",
    "下半身先": "下半身",
    "腰": "センター",
    "首": "上半身2",
    "頭": "首",
    "左肩": "上半身2",
    "右肩": "上半身2",
    "左腕": "左肩",
    "右腕": "右肩",
    "左ひじ": "左腕",
    "右ひじ": "右腕",
    "左手首": "左ひじ",
    "右手首": "右ひじ",
    "左足": "下半身",
    "右足": "下半身",
    "左ひざ": "左足",
    "右ひざ": "右足",
    "左足首": "左ひざ",
    "右足首": "右ひざ",
    "左足IK親": "下半身",
    "右足IK親": "下半身",
    "左足ＩＫ": "左足IK親",
    "左足IK": "左足IK親",
    "右足ＩＫ": "右足IK親",
    "右足IK": "右足IK親",
    "左つま先ＩＫ": "左足ＩＫ",
    "左つま先IK": "左足ＩＫ",
    "右つま先ＩＫ": "右足ＩＫ",
    "右つま先IK": "右足ＩＫ",
}

_IDENTITY_BONES: frozenset[str] = frozenset(
    {
        "左足IK親",
        "右足IK親",
        "センター親",
    }
)

_FOOT_IK_BONES: dict[str, tuple[str, ...]] = {
    "left": ("左足ＩＫ", "左足IK"),
    "right": ("右足ＩＫ", "右足IK"),
}

_TOE_IK_BONES: dict[str, tuple[str, ...]] = {
    "left": ("左つま先ＩＫ", "左つま先IK"),
    "right": ("右つま先ＩＫ", "右つま先IK"),
}

_BONE_FRAME_ALIASES: dict[str, tuple[str, ...]] = {
    "左足ＩＫ": ("左足ＩＫ", "左足IK"),
    "右足ＩＫ": ("右足ＩＫ", "右足IK"),
    "左つま先ＩＫ": ("左つま先ＩＫ", "左つま先IK"),
    "右つま先ＩＫ": ("右つま先ＩＫ", "右つま先IK"),
}


def _lookup_bone_data(frame_data: dict[str, dict], bone: str) -> dict | None:
    aliases = _BONE_FRAME_ALIASES.get(bone, (bone,))
    for name in aliases:
        data = frame_data.get(name)
        if data is not None:
            return data
    return None


def _resolve_bone_name(frame_data: dict[str, dict], candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame_data:
            return name
    return None


def _bone_chain_to_root(bone: str) -> list[str]:
    chain: list[str] = []
    current: str | None = bone
    visited: set[str] = set()
    while current is not None:
        if current in visited:
            break
        visited.add(current)
        chain.append(current)
        current = MMD_BONE_PARENT.get(current)
    chain.reverse()
    return chain


def _bone_local_transform_raw(
    frame_data: dict[str, dict],
    bone: str,
) -> tuple[tuple[float, float, float], list[float]]:
    if bone in _IDENTITY_BONES:
        return (0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 0.0]
    data = _lookup_bone_data(frame_data, bone)
    if data is None:
        return (0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 0.0]
    pos = data.get("pos", (0.0, 0.0, 0.0))
    quat = data.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
    return (
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
    ), quat_normalize([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])


def compute_bone_world_pose_mmd_raw(
    frame_data: dict[str, dict],
    bone_name: str,
) -> tuple[tuple[float, float, float], list[float]] | None:
    """FK world position + orientation (wxyz) at ``bone_name`` in raw VMD units."""
    if bone_name not in MMD_BONE_PARENT:
        return None
    chain = _bone_chain_to_root(bone_name)
    if not chain or chain[-1] != bone_name:
        return None

    rot_w = [1.0, 0.0, 0.0, 0.0]
    pos_w = (0.0, 0.0, 0.0)
    for bone in chain:
        t_l, q_l = _bone_local_transform_raw(frame_data, bone)
        dv = rotate_vec_by_quat_wxyz(rot_w, t_l)
        pos_w = (
            float(pos_w[0] + dv[0]),
            float(pos_w[1] + dv[1]),
            float(pos_w[2] + dv[2]),
        )
        rot_w = quat_mul(rot_w, q_l)
    return pos_w, rot_w


def _mmd_pos_to_model_world_aligned(
    pos_mmd: tuple[float, float, float],
) -> tuple[float, float, float]:
    """B @ pos_mmd: same axis swap as mmd_quat_to_world (X->X, Z->Y, Y->Z)."""
    return (float(pos_mmd[0]), float(pos_mmd[2]), float(pos_mmd[1]))


def _model_world_aligned_to_mmd_storage(
    pos_aligned: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Inverse of ``_mmd_pos_to_model_world_aligned`` for legacy mmd_world_pos_to_isaac."""
    return (float(pos_aligned[0]), float(pos_aligned[2]), float(pos_aligned[1]))


def foot_ik_panel_to_mmd_world_raw(
    foot_local_raw: tuple[float, float, float],
    frame_data: dict[str, dict],
    *,
    side: str = "left",
) -> tuple[float, float, float]:
    """Foot IK panel -> B-aligned model world.

    In PMX/VMD practice, foot IK ``pos`` is already in model-space coordinates (not
    LOWER_B local offset). So do not apply LOWER_B translation/rotation again here,
    otherwise targets drift far away when LOWER_B has non-zero R/P/Y.
    """
    del side
    del frame_data
    return _mmd_pos_to_model_world_aligned(foot_local_raw)


def foot_ik_panel_to_isaac_world(
    foot_local_raw: tuple[float, float, float],
    frame_data: dict[str, dict],
    *,
    pos_scale: float,
    is_pose: bool = False,
    side: str = "left",
    viz_cfg: FootIkVizConfig | None = None,
) -> tuple[float, float, float]:
    """Same Isaac-world target as the red debug sphere."""
    world_aligned = foot_ik_panel_to_mmd_world_raw(foot_local_raw, frame_data, side=side)
    origin = foot_ik_viz_ref_origin_isaac(side, viz_cfg)
    extra = float(viz_cfg.pos_scale) if viz_cfg is not None else 1.0
    scale = float(pos_scale) * extra
    world_mmd = _model_world_aligned_to_mmd_storage(world_aligned)
    legacy = mmd_world_pos_to_isaac(world_mmd, origin, scale, is_pose=is_pose)
    if viz_cfg is None:
        return legacy
    sign = viz_cfg.axis_sign_pose if is_pose else viz_cfg.axis_sign
    tuned = remap_mmd_world_to_isaac(world_aligned, origin, scale, viz_cfg.axis_idx, sign)
    w = max(0.0, min(1.0, float(viz_cfg.weight)))
    if w >= 1.0 - 1e-6:
        return tuned
    if w <= 1e-6:
        return legacy
    return (
        legacy[0] * (1.0 - w) + tuned[0] * w,
        legacy[1] * (1.0 - w) + tuned[1] * w,
        legacy[2] * (1.0 - w) + tuned[2] * w,
    )


def compute_mmd_bone_world_pos_mmd(
    frame_data: dict[str, dict],
    bone_name: str,
    *,
    pos_scale: float = 1.0,
) -> tuple[float, float, float] | None:
    """Forward kinematics: bone origin in MMD model/global coordinates."""
    pose = compute_bone_world_pose_mmd_raw(frame_data, bone_name)
    if pose is None:
        return None
    pos_w, _ = pose
    if abs(float(pos_scale) - 1.0) < 1e-12:
        return pos_w
    s = float(pos_scale)
    return (float(pos_w[0]) * s, float(pos_w[1]) * s, float(pos_w[2]) * s)


def _bone_local_transform(
    frame_data: dict[str, dict],
    bone: str,
    pos_scale: float,
) -> tuple[tuple[float, float, float], list[float]]:
    t_l, q_l = _bone_local_transform_raw(frame_data, bone)
    s = float(pos_scale)
    return (float(t_l[0]) * s, float(t_l[1]) * s, float(t_l[2]) * s), q_l


def _raw_pos_to_meters(
    pos_raw: tuple[float, float, float],
    pos_scale: float,
) -> tuple[float, float, float]:
    s = float(pos_scale)
    return (float(pos_raw[0]) * s, float(pos_raw[1]) * s, float(pos_raw[2]) * s)


def compute_mmd_foot_ik_viz_bundle(
    frame_data: dict[str, dict],
    *,
    pos_scale: float,
    is_pose: bool = False,
    viz_cfg: FootIkVizConfig | None = None,
) -> dict[str, dict[str, tuple[float, float, float] | None]]:
    """Foot/toe IK viz: MMD-local panel -> B-aligned world -> Isaac."""
    empty = {
        "local_m": None,
        "fk_world_m": None,
        "isaac_world_m": None,
    }
    out: dict[str, dict[str, tuple[float, float, float] | None]] = {
        "left": dict(empty),
        "right": dict(empty),
        "left_toe": dict(empty),
        "right_toe": dict(empty),
    }

    def _fill(key: str, bone_candidates: tuple[str, ...], side: str) -> None:
        bone = _resolve_bone_name(frame_data, bone_candidates)
        if bone is None:
            return
        data = _lookup_bone_data(frame_data, bone)
        if data is None:
            return
        pos_raw = data.get("pos")
        if pos_raw is None or len(pos_raw) != 3:
            return
        raw = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
        local_m = _raw_pos_to_meters(raw, pos_scale)
        isaac_world = foot_ik_panel_to_isaac_world(
            raw,
            frame_data,
            pos_scale=pos_scale,
            is_pose=is_pose,
            side=side,
            viz_cfg=viz_cfg,
        )
        world_raw = foot_ik_panel_to_mmd_world_raw(raw, frame_data, side=side)
        world_m = _raw_pos_to_meters(world_raw, pos_scale)
        out[key] = {
            "local_m": local_m,
            "fk_world_m": world_m,
            "isaac_world_m": isaac_world,
        }

    _fill("left", _FOOT_IK_BONES["left"], "left")
    _fill("right", _FOOT_IK_BONES["right"], "right")
    _fill("left_toe", _TOE_IK_BONES["left"], "left")
    _fill("right_toe", _TOE_IK_BONES["right"], "right")
    return out


