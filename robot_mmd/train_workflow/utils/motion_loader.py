"""Load CSV/HDF5 motion bundles and dance registry from YAML."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

from robot_mmd.train_workflow.utils.csv_motion_loader import (
    frames_have_hand_data,
    get_bone_frame_lists,
    get_frame_indices,
    load_csv_motion,
)
from robot_mmd.train_workflow.utils.hdf5_motion import infer_hdf5_has_hand_data, load_hdf5_motion

MotionBundle = dict[str, Any]
MOTION_EXTENSIONS = (".csv", ".h5", ".hdf5")


def format_playback_log_label(label: str) -> str:
    """Format internal motion label for playback logs, e.g. dance [deepbluetown]."""
    m = re.match(r"^(dance|pose)\[([^\]]+)\]\s+(.+)$", label)
    if not m:
        return label
    kind, _key_or_idx, path = m.group(1), m.group(2), m.group(3).strip()
    base = os.path.splitext(os.path.basename(path))[0]
    return f"{kind} [{base}]"


def load_motion(filepath: str) -> MotionBundle | None:
    """Load motion (CSV/HDF5) and return a unified bundle."""
    if not os.path.isfile(filepath):
        return None
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".h5", ".hdf5"):
        motion = load_hdf5_motion(filepath)
        if motion.frames.size <= 0:
            return None
        frame_list = [int(v) for v in motion.frames.tolist()]
        has_hand_data = infer_hdf5_has_hand_data(motion)
        if has_hand_data and not bool(motion.has_hand_data):
            motion = replace(motion, has_hand_data=True)
        return {
            "kind": "hdf5",
            "path": filepath,
            "frame_list": frame_list,
            "hdf5": motion,
            "has_hand_data": has_hand_data,
        }
    if ext == ".csv":
        frames = load_csv_motion(filepath)
        frame_list = get_frame_indices(frames)
        all_bones = set()
        for f in frames.values():
            all_bones.update(f.keys())
        bone_frame_lists = get_bone_frame_lists(frames, frame_list, all_bones)
        csv_has_finger_bones = bool(frames_have_hand_data(frames))
        return {
            "kind": "csv",
            "path": filepath,
            "frame_list": frame_list,
            "frames": frames,
            "bone_frame_lists": bone_frame_lists,
            "all_bones": all_bones,
            "has_hand_data": csv_has_finger_bones,
        }
    print(f"[WARN] 不支持的 motion 扩展名: {filepath}")
    return None


def list_motion_files(dir_path: str, label: str) -> list[str]:
    """List motion files (.csv/.h5/.hdf5) in a directory, sorted by name."""
    if not os.path.isdir(dir_path):
        print(f"[WARN] {label} 目录不存在: {dir_path}")
        return []
    files = sorted(
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        and os.path.splitext(f)[1].lower() in MOTION_EXTENSIONS
    )
    if not files:
        print(f"[WARN] {label} 目录没有可用 motion 文件: {dir_path}")
    return files


def load_pose_motion_dir(pose_dir: str) -> list[tuple[str, str, MotionBundle]]:
    """Load all motion files under pose_dir as [(name, fullpath, bundle)]."""
    motion_files = list_motion_files(pose_dir, "pose")
    out: list[tuple[str, str, MotionBundle]] = []
    for name in motion_files:
        fullpath = os.path.join(pose_dir, name)
        data = load_motion(fullpath)
        if data is None:
            print(f"[WARN] 无法加载 motion: {fullpath}")
            continue
        out.append((name, fullpath, data))
        print(f"[INFO] 已加载 pose: {name}，共 {len(data['frame_list'])} 帧")
    if not out:
        print(f"[WARN] pose 目录没有可用 motion 文件: {pose_dir}")
    return out


def resolve_path_under_media(relative: str, media_dir: str) -> str:
    """Resolve config-relative path under robot_mmd/media/."""
    rel = (relative or "").strip().replace("\\", "/")
    if not rel:
        return ""
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(media_dir, rel))


def load_dances_from_yaml(
    config_path: str,
    *,
    media_dir: str,
    script_dir: str,
) -> tuple[dict[str, tuple[str, MotionBundle]], dict[str, str]]:
    """Load dances from YAML: motion_by_key and wav_by_key."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "读取舞蹈配置需要 PyYAML: pip install pyyaml "
            f"(见 {os.path.join(script_dir, 'dances_requirements.txt')})"
        ) from e

    raw = (config_path or "").strip()
    if not raw:
        print("[WARN] 舞蹈配置文件路径为空，无 dance 键")
        return {}, {}
    if os.path.isfile(raw):
        path = os.path.normpath(os.path.abspath(raw))
    else:
        p1 = os.path.join(script_dir, raw)
        p2 = os.path.abspath(raw)
        if os.path.isfile(p1):
            path = os.path.normpath(p1)
        elif os.path.isfile(p2):
            path = os.path.normpath(p2)
        else:
            print(f"[WARN] 未找到舞蹈配置 YAML: {raw}，无 dance 键")
            return {}, {}

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not doc:
        print(f"[WARN] 舞蹈配置为空: {path}")
        return {}, {}
    items = doc.get("dances")
    if not isinstance(items, list):
        print(f"[WARN] 舞蹈配置缺少 ``dances`` 列表: {path}")
        return {}, {}

    motion_by_key: dict[str, tuple[str, MotionBundle]] = {}
    wav_by_key: dict[str, str] = {}
    for i, ent in enumerate(items):
        if not isinstance(ent, dict):
            print(f"[WARN] dances[{i}] 非映射，已跳过")
            continue
        raw_key = ent.get("key")
        motion_rel = ent.get("motion")
        if not motion_rel or not str(motion_rel).strip():
            print(f"[WARN] dances[{i}] 无 motion，已跳过")
            continue

        if raw_key is None or str(raw_key).strip() == "":
            from robot_mmd.train_workflow.utils.dance_asset_sync import ui_only_dance_key

            label = ent.get("id") or ent.get("label")
            key = ui_only_dance_key(str(label or ""), str(motion_rel).strip())
            hotkey_note = " (UI only)"
        else:
            key = str(raw_key).strip().upper()[:1]
            hotkey_note = ""
        if key in motion_by_key:
            print(f"[WARN] 舞蹈键重复 [{key}]，后项已忽略: {ent.get('id', i)}")
            continue
        motion_p = resolve_path_under_media(str(motion_rel).strip(), media_dir)
        if not os.path.isfile(motion_p):
            print(f"[WARN] 未找到 dance 键 [{key}] 的 motion 文件: {motion_p}")
            continue
        data = load_motion(motion_p)
        if data is None:
            print(f"[WARN] 无法加载 dance 键 [{key}]: {motion_p}")
            continue
        label = ent.get("id") or ent.get("label")
        brief = f" [{label}]" if label else ""
        print(
            f"[INFO] 已绑定 dance [{key}] -> {os.path.basename(motion_p)}"
            f"（{len(data['frame_list'])} 帧）{brief}{hotkey_note}"
            + (" [hand]" if data.get("has_hand_data") else "")
        )
        motion_by_key[key] = (os.path.basename(motion_p), data)
        raw_audio = ent.get("audio", None)
        if raw_audio is None or str(raw_audio).strip() == "":
            continue
        ap = resolve_path_under_media(str(raw_audio).strip(), media_dir)
        if os.path.isfile(ap):
            wav_by_key[key] = ap
        else:
            print(f"[WARN] 舞蹈 [{key}] 的音频不存在，将不播伴音: {ap}")
    return motion_by_key, wav_by_key


def build_dance_hdf5_motion_by_key(
    dance_motion_by_key: dict[str, tuple[str, MotionBundle]],
) -> dict[str, tuple[str, MotionBundle]]:
    """Map each dance key to a sibling .h5/.hdf5 when available."""
    out: dict[str, tuple[str, MotionBundle]] = {}
    for dkey, (_name, data) in dance_motion_by_key.items():
        kind = str(data.get("kind", ""))
        path = str(data.get("path", ""))
        if kind == "hdf5":
            out[dkey] = (_name, data)
            continue
        if kind != "csv" or not path:
            continue
        stem = os.path.splitext(path)[0]
        for ext in (".h5", ".hdf5"):
            alt_path = stem + ext
            if not os.path.isfile(alt_path):
                continue
            alt_data = load_motion(alt_path)
            if alt_data is None or str(alt_data.get("kind", "")) != "hdf5":
                continue
            out[dkey] = (os.path.basename(alt_path), alt_data)
            break
    return out


def _replace_ext(path: str, ext: str) -> str:
    stem = os.path.splitext(path)[0]
    return stem + ext


def z_editted_sibling_path(path: str) -> str:
    """``foo.csv`` -> ``foo_z_editted.csv`` (idempotent if already editted)."""
    stem, ext = os.path.splitext(path)
    if stem.endswith("_z_editted"):
        return path
    return stem + "_z_editted" + (ext or ".csv")


def has_z_editted_sibling(path: str) -> bool:
    """True when a ``*_z_editted.csv`` or ``*_z_editted.h5/.hdf5`` sibling exists."""
    if not path or not str(path).strip():
        return False
    abs_path = os.path.abspath(str(path))
    if os.path.isfile(z_editted_sibling_path(abs_path)):
        return True
    stem = os.path.splitext(abs_path)[0]
    if stem.endswith("_z_editted"):
        return True
    for ext in (".h5", ".hdf5"):
        if os.path.isfile(stem + "_z_editted" + ext):
            return True
    return False


def resolve_playback_motion_entry(
    entry: tuple[str, MotionBundle],
    *,
    prefer_hdf5: bool,
    z_offset_enabled: bool,
) -> tuple[tuple[str, MotionBundle], bool]:
    """Optionally swap to a ``*_z_editted.*`` sibling; warn and keep original if missing."""
    if not z_offset_enabled:
        return entry, False

    name, data = entry
    path = str(data.get("path", ""))
    if not path:
        return entry, False

    candidates: list[str] = []
    if prefer_hdf5 or str(data.get("kind", "")) == "hdf5":
        z_stem = os.path.splitext(z_editted_sibling_path(path))[0]
        for ext in (".h5", ".hdf5"):
            candidates.append(z_stem + ext)
    else:
        candidates.append(z_editted_sibling_path(path))

    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        loaded = load_motion(cand)
        if loaded is None:
            continue
        return (os.path.basename(cand), loaded), True

    print(
        f"[WARN] Z_offset_enable is on but no *_z_editted file for "
        f"'{os.path.basename(path)}'; using original motion."
    )
    return entry, False


def build_dance_hand_motion_by_key(
    dance_motion_by_key: dict[str, tuple[str, MotionBundle]],
) -> dict[str, tuple[str, MotionBundle]]:
    """Map each dance key to a sibling *_hand.csv when available."""
    out: dict[str, tuple[str, MotionBundle]] = {}
    for dkey, (_name, data) in dance_motion_by_key.items():
        kind = str(data.get("kind", ""))
        path = str(data.get("path", ""))
        if kind != "csv" or not path:
            continue
        stem, ext = os.path.splitext(path)
        if stem.endswith("_hand"):
            out[dkey] = (_name, data)
            continue
        hand_csv = stem + "_hand" + ext
        if not os.path.isfile(hand_csv):
            continue
        hand_data = load_motion(hand_csv)
        if hand_data is None or str(hand_data.get("kind", "")) != "csv":
            continue
        out[dkey] = (os.path.basename(hand_csv), hand_data)
    return out


def build_dance_hand_hdf5_motion_by_key(
    dance_hand_motion_by_key: dict[str, tuple[str, MotionBundle]],
) -> dict[str, tuple[str, MotionBundle]]:
    """Map each hand-csv dance key to sibling .h5/.hdf5 when available."""
    out: dict[str, tuple[str, MotionBundle]] = {}
    for dkey, (_name, data) in dance_hand_motion_by_key.items():
        kind = str(data.get("kind", ""))
        path = str(data.get("path", ""))
        if kind == "hdf5":
            out[dkey] = (_name, data)
            continue
        if kind != "csv" or not path:
            continue
        for ext in (".h5", ".hdf5"):
            alt_path = _replace_ext(path, ext)
            if not os.path.isfile(alt_path):
                continue
            alt_data = load_motion(alt_path)
            if alt_data is None or str(alt_data.get("kind", "")) != "hdf5":
                continue
            out[dkey] = (os.path.basename(alt_path), alt_data)
            break
    return out
