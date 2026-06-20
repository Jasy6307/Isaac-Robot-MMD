"""Scan media/dance VMD files; ensure CSV/H5 exist and dances_config.yaml is updated."""

from __future__ import annotations

import os
import re
from typing import Any

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.scripts.vmd_2_csv import read_motion_and_export
from robot_mmd.train_workflow.utils.csv_motion_loader import FootIkConfig
from robot_mmd.train_workflow.utils.hdf5_motion import (
    compile_csv_motion_to_hdf5_motion,
    write_hdf5_motion,
)

# Same default joint order as csv_2_hdf5.py (G1 29DoF + O6 hands).
_DEFAULT_COMPILE_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_middle_mcp_pitch",
    "lh_middle_dip",
    "lh_ring_mcp_pitch",
    "lh_ring_dip",
    "lh_pinky_mcp_pitch",
    "lh_pinky_dip",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
]

_SKIP_VMD_SUFFIXES = ("_z_editted", "_hand")


def get_default_compile_joint_names() -> list[str]:
    """Joint order used when writing ``g1_mmd_motion_v1`` HDF5 files."""
    return list(_DEFAULT_COMPILE_JOINT_NAMES)


def _motion_stem_from_rel(motion_rel: str) -> str:
    base = os.path.basename(str(motion_rel or "").replace("\\", "/"))
    stem, _ext = os.path.splitext(base)
    if stem.endswith("_z_editted"):
        stem = stem[: -len("_z_editted")]
    return stem


def _load_yaml_dances(config_path: str) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        return []
    if not os.path.isfile(config_path):
        return []
    with open(config_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    items = doc.get("dances")
    if not isinstance(items, list):
        return []
    return [e for e in items if isinstance(e, dict)]


def _yaml_covers_stem(entries: list[dict[str, Any]], stem: str) -> bool:
    want = stem.casefold()
    for ent in entries:
        motion_stem = _motion_stem_from_rel(str(ent.get("motion", "")))
        if motion_stem.casefold() == want:
            return True
        ent_id = str(ent.get("id") or "").strip()
        if ent_id.casefold() == want:
            return True
    return False


def _append_dance_yaml_entries(config_path: str, new_entries: list[dict[str, str]]) -> None:
    if not new_entries:
        return
    with open(config_path, "a", encoding="utf-8") as f:
        for ent in new_entries:
            f.write(f"\n  - id: {ent['id']}\n")
            f.write(f"    motion: {ent['motion']}\n")
            if ent.get("audio"):
                f.write(f"    audio: {ent['audio']}\n")


def _list_vmd_files(dance_dir: str) -> list[str]:
    if not os.path.isdir(dance_dir):
        return []
    out: list[str] = []
    for name in os.listdir(dance_dir):
        if not name.lower().endswith(".vmd"):
            continue
        stem, _ = os.path.splitext(name)
        if any(stem.endswith(sfx) for sfx in _SKIP_VMD_SUFFIXES):
            continue
        out.append(os.path.join(dance_dir, name))
    return sorted(out)


def _compile_csv_to_h5(
    csv_path: str,
    h5_path: str,
    *,
    groove_pos_to_world: float,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float],
    knee_hinge_projection: bool,
    foot_ik_cfg: FootIkConfig | None = None,
) -> None:
    motion = compile_csv_motion_to_hdf5_motion(
        csv_path,
        list(_DEFAULT_COMPILE_JOINT_NAMES),
        fps=30.0,
        knee_hinge_projection=bool(knee_hinge_projection),
        groove_pos_to_world=float(groove_pos_to_world),
        mmd_center_to_root_offset_local_xyz=mmd_center_to_root_offset_local_xyz,
        root_quat_rpy_scale=tuple(MMD_ROOT_QUAT_RPY_SCALE_DEFAULT),
        root_quat_rpy_axis_idx=tuple(MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT),
        foot_ik_cfg=foot_ik_cfg,
    )
    write_hdf5_motion(h5_path, motion)


def sync_dance_assets_from_vmd(
    *,
    dance_dir: str,
    dances_config_path: str,
    media_dir: str,
    groove_pos_to_world: float = 0.1,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    knee_hinge_projection: bool = True,
    foot_ik_cfg: FootIkConfig | None = None,
) -> list[str]:
    """Ensure CSV/H5 siblings exist for each VMD; append keyless YAML entries.

    Returns list of dance stems newly registered in ``dances_config.yaml``.
    """
    vmd_paths = _list_vmd_files(dance_dir)
    if not vmd_paths:
        return []

    yaml_entries = _load_yaml_dances(dances_config_path)
    yaml_additions: list[dict[str, str]] = []
    registered: list[str] = []

    for vmd_path in vmd_paths:
        stem = os.path.splitext(os.path.basename(vmd_path))[0]
        csv_path = os.path.join(dance_dir, stem + ".csv")
        h5_path = os.path.join(dance_dir, stem + ".h5")

        if not os.path.isfile(csv_path):
            try:
                print(f"[INFO] VMD -> CSV: {os.path.basename(vmd_path)}")
                read_motion_and_export(vmd_path, csv_path)
            except Exception as exc:
                print(f"[WARN] VMD -> CSV failed ({stem}): {exc}")
                continue

        if os.path.isfile(csv_path) and not os.path.isfile(h5_path):
            try:
                print(f"[INFO] CSV -> H5: {os.path.basename(csv_path)}")
                _compile_csv_to_h5(
                    csv_path,
                    h5_path,
                    groove_pos_to_world=groove_pos_to_world,
                    mmd_center_to_root_offset_local_xyz=mmd_center_to_root_offset_local_xyz,
                    knee_hinge_projection=knee_hinge_projection,
                    foot_ik_cfg=foot_ik_cfg,
                )
            except Exception as exc:
                print(f"[WARN] CSV -> H5 failed ({stem}): {exc}")

        if _yaml_covers_stem(yaml_entries, stem):
            continue

        motion_rel = f"dance/{stem}.csv"
        ent: dict[str, str] = {"id": stem, "motion": motion_rel}
        wav_abs = os.path.join(dance_dir, stem + ".wav")
        if os.path.isfile(wav_abs):
            ent["audio"] = f"dance/{stem}.wav"
        yaml_additions.append(ent)
        yaml_entries.append(ent)
        registered.append(stem)
        print(f"[INFO] Registered dance in YAML (UI only, no hotkey): {stem}")

    if yaml_additions:
        if not os.path.isfile(dances_config_path):
            header = (
                "# Auto-generated dances_config.yaml\n"
                "version: 1\n"
                "dances:\n"
            )
            with open(dances_config_path, "w", encoding="utf-8") as f:
                f.write(header)
        _append_dance_yaml_entries(dances_config_path, yaml_additions)

    return registered


def ui_only_dance_key(dance_id: str, motion_rel: str = "") -> str:
    """Stable lookup key for dropdown-only dances (no keyboard hotkey)."""
    slug = str(dance_id or "").strip()
    if not slug and motion_rel:
        slug = _motion_stem_from_rel(motion_rel)
    slug = re.sub(r"[^\w\-]+", "_", slug).strip("_") or "dance"
    return f"ui:{slug}"
