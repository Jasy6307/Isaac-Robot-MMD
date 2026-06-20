"""MMD standard bone hierarchy FK for foot IK world positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from robot_mmd.train_workflow.utils.trans_util import (
    mmd_storage_delta_to_isaac_world_delta,
    mmd_world_pos_to_isaac,
    quat_mul,
    quat_normalize,
    remap_mmd_world_to_isaac,
    rotate_vec_by_quat_wxyz,
)

# Single source of truth for red-sphere / foot-target map defaults.
FOOT_IK_VIZ_AXIS_IDX: tuple[int, int, int] = (0, 1, 2)
FOOT_IK_VIZ_AXIS_SIGN: tuple[float, float, float] = (-1.0, -1.0, 1.0)
FOOT_IK_VIZ_AXIS_SIGN_POSE: tuple[float, float, float] = (-1.0, 1.0, 1.0)
FOOT_IK_VIZ_POS_SCALE: float = 1.0
FOOT_IK_VIZ_LEFT_REF_ORIGIN_M: tuple[float, float, float] = (-0.10, -0.10, 0.0)
FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M: tuple[float, float, float] = (0.10, -0.10, 0.0)


@dataclass
class FootIkVizConfig:
    """Foot IK panel -> Isaac world map. Single source for red spheres and leg IK targets."""

    # B-aligned model world (mmdX, mmdZ, mmdY) -> Isaac; idx 0,1,2 direct.
    axis_idx: tuple[int, int, int] = FOOT_IK_VIZ_AXIS_IDX
    axis_sign: tuple[float, float, float] = FOOT_IK_VIZ_AXIS_SIGN
    axis_sign_pose: tuple[float, float, float] = FOOT_IK_VIZ_AXIS_SIGN_POSE
    pos_scale: float = FOOT_IK_VIZ_POS_SCALE
    # Isaac-world rest origin when foot IK panel offset is zero (meters).
    left_ref_origin_m: tuple[float, float, float] = FOOT_IK_VIZ_LEFT_REF_ORIGIN_M
    right_ref_origin_m: tuple[float, float, float] = FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M


def default_foot_ik_viz_config() -> FootIkVizConfig:
    """Return a fresh config using module defaults (UI/CLI/batch tools should reference this)."""
    return FootIkVizConfig()


def foot_ik_viz_triplet_cli(values: tuple[float, float, float] | tuple[int, int, int]) -> str:
    """Format xyz defaults for argparse comma-separated CLI options."""
    return f"{values[0]},{values[1]},{values[2]}"


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
        return FOOT_IK_VIZ_LEFT_REF_ORIGIN_M
    return FOOT_IK_VIZ_RIGHT_REF_ORIGIN_M

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

_TOE_IK_KEYFRAME_POS_EPS: float = 1e-6
_TOE_IK_KEYFRAME_QUAT_EPS: float = 1e-6


def _toe_ik_keyframe_is_meaningful(bone_data: dict) -> bool:
    """True when a toe-IK row carries non-default pos or rotation (not a frame-0 stub)."""
    pos = bone_data.get("pos")
    if pos is not None and len(pos) >= 3:
        if abs(float(pos[0])) + abs(float(pos[1])) + abs(float(pos[2])) > _TOE_IK_KEYFRAME_POS_EPS:
            return True
    quat_wxyz = bone_data.get("quat_wxyz")
    if quat_wxyz is not None and len(quat_wxyz) >= 4:
        qw, qx, qy, qz = (
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        )
        if abs(qx) + abs(qy) + abs(qz) > _TOE_IK_KEYFRAME_QUAT_EPS:
            return True
        if abs(abs(qw) - 1.0) > _TOE_IK_KEYFRAME_QUAT_EPS:
            return True
        return False
    quat_xyzw = bone_data.get("quat")
    if quat_xyzw is not None and len(quat_xyzw) >= 4:
        qx, qy, qz, qw = (
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        )
        if abs(qx) + abs(qy) + abs(qz) > _TOE_IK_KEYFRAME_QUAT_EPS:
            return True
        if abs(abs(qw) - 1.0) > _TOE_IK_KEYFRAME_QUAT_EPS:
            return True
    return False


def motion_side_has_valid_toe_ik_keyframes(
    frames: dict[int, dict[str, dict]] | None,
    side: str,
) -> bool:
    """Return True when motion CSV/VMD contains at least one meaningful toe-IK keyframe."""
    if not frames:
        return False
    candidates = _TOE_IK_BONES.get(side)
    if not candidates:
        return False
    for frame_data in frames.values():
        for bone_name in candidates:
            bone_data = frame_data.get(bone_name)
            if bone_data is None:
                continue
            if _toe_ik_keyframe_is_meaningful(bone_data):
                return True
    return False


def motion_side_has_valid_foot_ik_keyframes(
    frames: dict[int, dict[str, dict]] | None,
    side: str,
) -> bool:
    """Return True when motion CSV/VMD contains at least one meaningful foot-IK keyframe."""
    if not frames:
        return False
    candidates = _FOOT_IK_BONES.get(side)
    if not candidates:
        return False
    for frame_data in frames.values():
        for bone_name in candidates:
            bone_data = frame_data.get(bone_name)
            if bone_data is None:
                continue
            if _toe_ik_keyframe_is_meaningful(bone_data):
                return True
    return False


def motion_has_embedded_foot_ik(frames: dict[int, dict[str, dict]] | None) -> bool:
    """True when CSV/VMD embeds MMD foot-IK data (足IK); Z_editted generation is not needed."""
    if not frames:
        return False
    return motion_side_has_valid_foot_ik_keyframes(
        frames, "left"
    ) or motion_side_has_valid_foot_ik_keyframes(frames, "right")


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


def read_mmd_bone_pos_raw(
    frame_data: dict[str, dict],
    bone_candidates: tuple[str, ...],
) -> tuple[float, float, float] | None:
    bone = _resolve_bone_name(frame_data, bone_candidates)
    if bone is None:
        return None
    data = _lookup_bone_data(frame_data, bone)
    if data is None:
        return None
    pos_raw = data.get("pos")
    if pos_raw is None or len(pos_raw) != 3:
        return None
    return (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))


_MMD_ROOT_TRANSLATION_BONES: tuple[tuple[str, ...], ...] = (
    ("センター", "グルーブ"),
    ("グルーブ", "センター"),
)


def resolve_mmd_root_translation_pos(
    frame_data: dict[str, dict],
    *,
    preferred_bone: str | None = None,
) -> tuple[tuple[float, float, float] | None, str | None]:
    """Return (pos_raw, bone_name) for the bone driving root translation."""
    if preferred_bone:
        pos = read_mmd_bone_pos_raw(frame_data, (preferred_bone,))
        if pos is not None:
            return pos, preferred_bone
    for candidates in _MMD_ROOT_TRANSLATION_BONES:
        pos = read_mmd_bone_pos_raw(frame_data, candidates)
        if pos is not None:
            bone = _resolve_bone_name(frame_data, candidates)
            return pos, bone
    return None, None


def _foot_ik_legacy_isaac_world(
    foot_local_raw: tuple[float, float, float],
    frame_data: dict[str, dict],
    *,
    pos_scale: float,
    is_pose: bool = False,
    side: str = "left",
    viz_cfg: FootIkVizConfig | None = None,
) -> tuple[float, float, float]:
    world_aligned = foot_ik_panel_to_mmd_world_raw(foot_local_raw, frame_data, side=side)
    origin = foot_ik_viz_ref_origin_isaac(side, viz_cfg)
    extra = float(viz_cfg.pos_scale) if viz_cfg is not None else 1.0
    scale = float(pos_scale) * extra
    if viz_cfg is None:
        world_mmd = _model_world_aligned_to_mmd_storage(world_aligned)
        return mmd_world_pos_to_isaac(world_mmd, origin, scale, is_pose=is_pose)
    sign = viz_cfg.axis_sign_pose if is_pose else viz_cfg.axis_sign
    return remap_mmd_world_to_isaac(world_aligned, origin, scale, viz_cfg.axis_idx, sign)


def _foot_ik_delta_to_isaac_world(
    delta_storage: tuple[float, float, float],
    *,
    pos_scale: float,
    is_pose: bool = False,
    viz_cfg: FootIkVizConfig | None = None,
) -> tuple[float, float, float]:
    extra = float(viz_cfg.pos_scale) if viz_cfg is not None else 1.0
    scale = float(pos_scale) * extra
    legacy = mmd_storage_delta_to_isaac_world_delta(
        delta_storage[0],
        delta_storage[1],
        delta_storage[2],
        scale,
        is_pose=is_pose,
    )
    if viz_cfg is None:
        return legacy
    world_aligned = (
        float(delta_storage[0]) * scale,
        float(delta_storage[2]) * scale,
        float(delta_storage[1]) * scale,
    )
    sign = viz_cfg.axis_sign_pose if is_pose else viz_cfg.axis_sign
    return remap_mmd_world_to_isaac(world_aligned, (0.0, 0.0, 0.0), 1.0, viz_cfg.axis_idx, sign)


def foot_ik_panel_to_isaac_world(
    foot_local_raw: tuple[float, float, float],
    frame_data: dict[str, dict],
    *,
    pos_scale: float,
    is_pose: bool = False,
    side: str = "left",
    viz_cfg: FootIkVizConfig | None = None,
    target_root_pos: tuple[float, float, float] | None = None,
    target_root_quat_wxyz: list[float] | None = None,
    center_mmd_pos: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Same Isaac-world target as the red debug sphere."""
    if target_root_pos is not None and center_mmd_pos is not None:
        delta_storage = (
            float(foot_local_raw[0]) - float(center_mmd_pos[0]),
            float(foot_local_raw[1]) - float(center_mmd_pos[1]),
            float(foot_local_raw[2]) - float(center_mmd_pos[2]),
        )
        delta_world = _foot_ik_delta_to_isaac_world(
            delta_storage,
            pos_scale=pos_scale,
            is_pose=is_pose,
            viz_cfg=viz_cfg,
        )
        ref_local = foot_ik_viz_ref_origin_isaac(side, viz_cfg)
        if target_root_quat_wxyz is not None:
            ref_world = rotate_vec_by_quat_wxyz(target_root_quat_wxyz, ref_local)
        else:
            ref_world = ref_local
        return (
            float(target_root_pos[0]) + float(delta_world[0]) + float(ref_world[0]),
            float(target_root_pos[1]) + float(delta_world[1]) + float(ref_world[1]),
            float(target_root_pos[2]) + float(delta_world[2]) + float(ref_world[2]),
        )
    return _foot_ik_legacy_isaac_world(
        foot_local_raw,
        frame_data,
        pos_scale=pos_scale,
        is_pose=is_pose,
        side=side,
        viz_cfg=viz_cfg,
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
    """Foot/toe IK viz: legacy fixed ref-origin map (red debug spheres)."""
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
        isaac_world = _foot_ik_legacy_isaac_world(
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


