"""Scan media/dance VMD files; ensure CSV/H5 exist and dances_config.yaml is updated."""

from __future__ import annotations

import os
import re
from typing import Any

from source.paths import iter_dance_assets

from source.train_workflow.utils.retarget.joint_axis_map import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from source.train_workflow.scripts.vmd_2_csv import read_motion_and_export
from source.train_workflow.utils.format.csv_loader import FootIkConfig
from source.train_workflow.utils.format.hdf5 import (
    compile_csv_motion_to_hdf5_motion,
    write_hdf5_motion,
)

# Default compile joint order (G1 29DoF + O6 hands).
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
    for path in iter_dance_assets("*.vmd", dance_dir=dance_dir):
        name = path.name
        stem, _ = os.path.splitext(name)
        if any(stem.endswith(sfx) for sfx in _SKIP_VMD_SUFFIXES):
            continue
        out.append(str(path))
    return sorted(out)


def compile_dance_csv_to_h5(
    csv_path: str,
    h5_path: str,
    *,
    groove_pos_to_world: float = 0.1,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    knee_hinge_projection: bool = True,
    foot_ik_cfg: FootIkConfig | None = None,
    retarget_mode: str = 'new',
    source_leg_mode: str = 'fk',
) -> None:
    if retarget_mode not in ('new','legacy'):
        raise ValueError('Unknown retarget mode')
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
    reports = {}
    if retarget_mode == 'new':
        import ast
        import json
        from pathlib import Path
        import numpy as np
        from scipy.spatial.transform import Rotation
        from source.train_workflow.utils.motion.self_collision import playback_baseline
        from source.train_workflow.utils.retarget.source_animation import prepare_source_animation, ROOT
        from source.train_workflow.utils.retarget.upper_body_export import solve_upper_body_arrays
        from source.train_workflow.utils.retarget.lower_body_ik import solve_lower_body_arrays
        cfg = ROOT/'source/my_task/g1_replay_env_cfg.py'
        baseline = playback_baseline(motion.joint_names,cfg)
        tree = ast.parse(cfg.read_text(encoding='utf-8'))
        call = next(node.value for node in tree.body if isinstance(node,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='G1_TPOSE_INIT_STATE' for t in node.targets))
        anchor = np.array(ast.literal_eval(next(k.value for k in call.keywords if k.arg=='pos')))
        xyzw = np.array(ast.literal_eval(next(k.value for k in call.keywords if k.arg=='rot')))
        source = prepare_source_animation(csv_path,leg_mode=source_leg_mode)
        robot = ROOT/'assets/g1_29dof_o6_hand_V3.usd'
        motion.joint_pos_delta,reports['upper_body'] = solve_upper_body_arrays(
            motion.joint_pos_delta,motion.joint_names,baseline,motion.fps,source,robot)
        root_q = (Rotation.from_quat(motion.root_quat_delta_wxyz[:,[1,2,3,0]])*
                  Rotation.from_quat(xyzw)).as_quat()[:,[3,0,1,2]]
        motion.joint_pos_delta,reports['lower_body'] = solve_lower_body_arrays(
            motion.joint_pos_delta,motion.joint_names,baseline,motion.fps,source,robot,
            motion.root_pos_delta+anchor,root_q)
        from source.train_workflow.utils.motion.joint_constraints import constrain_joint_positions
        absolute,joint_report=constrain_joint_positions(
            motion.joint_pos_delta+baseline,motion.joint_names,motion.fps,robot)
        motion.joint_pos_delta=(absolute-baseline).astype(np.float32)
    write_hdf5_motion(h5_path, motion)
    import h5py
    with h5py.File(h5_path,'r+') as stream:
        stream.attrs['retarget_mode'] = retarget_mode
        if retarget_mode == 'new':
            stream.attrs['all_joint_constraints_validated']=joint_report['after']['passed']
            stream.attrs['mimic_constraints_validated']=True
            Path(h5_path).with_suffix('.joint_limits.json').write_text(json.dumps(joint_report,indent=2),encoding='utf-8')
            stream.attrs['source_leg_mode'] = source_leg_mode
            stream.attrs['record_baseline_joint_pos'] = baseline
            stream.attrs['record_root_anchor_pos'] = anchor
            stream.attrs['record_root_anchor_quat_wxyz'] = xyzw[[3,0,1,2]]
            for limb,report in reports.items():
                stream.attrs[limb+'_retarget_method'] = report['method']
                stream.attrs[limb+'_source_animation'] = str(source)
                stream.attrs[limb+'_export_mode'] = 'direct-in-memory'
                Path(h5_path).with_suffix('.'+limb+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def sync_dance_assets_from_vmd(
    *,
    dance_dir: str,
    dances_config_path: str,
    media_dir: str,
    groove_pos_to_world: float = 0.1,
    mmd_center_to_root_offset_local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    knee_hinge_projection: bool = True,
    foot_ik_cfg: FootIkConfig | None = None,
    retarget_mode: str = 'new',
    source_leg_mode: str = 'fk',
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
        asset_dir = os.path.dirname(vmd_path)
        csv_path = os.path.join(asset_dir, stem + ".csv")
        h5_path = os.path.join(asset_dir, stem + ".h5")

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
                compile_dance_csv_to_h5(
                    csv_path,
                    h5_path,
                    groove_pos_to_world=groove_pos_to_world,
                    mmd_center_to_root_offset_local_xyz=mmd_center_to_root_offset_local_xyz,
                    knee_hinge_projection=knee_hinge_projection,
                    foot_ik_cfg=foot_ik_cfg,
                    retarget_mode=retarget_mode,
                    source_leg_mode=source_leg_mode,
                )
            except Exception as exc:
                print(f"[WARN] CSV -> H5 failed ({stem}): {exc}")

        if _yaml_covers_stem(yaml_entries, stem):
            continue

        motion_rel = os.path.relpath(csv_path, media_dir).replace("\\", "/")
        ent: dict[str, str] = {"id": stem, "motion": motion_rel}
        wav_abs = os.path.join(asset_dir, stem + ".wav")
        if os.path.isfile(wav_abs):
            ent["audio"] = os.path.relpath(wav_abs, media_dir).replace("\\", "/")
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
