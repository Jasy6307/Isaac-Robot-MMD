"""Resolve dance names / motion paths under ``media/dance/``."""

from __future__ import annotations

import os

from source.paths import DANCE_DIR, REPO_ROOT

_Z_EDITTED_SUFFIX = "_z_editted"


def _repo_root() -> str:
    return REPO_ROOT


def default_dance_dir() -> str:
    """Absolute path to ``media/dance``."""
    return DANCE_DIR


def normalize_dance_stem(name: str) -> str:
    """Normalize a dance id / filename stem (strip ext and ``*_z_editted``)."""
    stem = os.path.splitext(str(name or "").strip().replace("\\", "/"))[0]
    base = os.path.basename(stem)
    if base.casefold().endswith(_Z_EDITTED_SUFFIX):
        base = base[: -len(_Z_EDITTED_SUFFIX)]
    return base.strip() or base


def infer_dance_name_from_motion_path(motion_path: str) -> str:
    """Infer canonical dance folder name from an H5/CSV motion path."""
    return normalize_dance_stem(os.path.basename(str(motion_path)))


def _find_dance_csv(canonical: str, root: str) -> str | None:
    for fname in (
        f"{canonical}{_Z_EDITTED_SUFFIX}.csv",
        f"{canonical}.csv",
    ):
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            return path
    return None


def _compile_csv_sibling_to_h5(csv_path: str) -> str:
    root = os.path.dirname(csv_path)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    h5_path = os.path.join(root, stem + ".h5")
    from source.train_workflow.utils.motion.sync import compile_dance_csv_to_h5

    print(f"[INFO] CSV -> H5: {os.path.basename(csv_path)}")
    compile_dance_csv_to_h5(csv_path, h5_path)
    return os.path.abspath(h5_path)


def resolve_dance_h5_by_name(
    dance_name: str,
    *,
    dance_dir: str | None = None,
    compile_from_csv: bool = True,
) -> tuple[str, str]:
    """Resolve ``dance_name`` to an absolute HDF5 path and canonical dance id.

    Search order under ``media/dance/`` (first existing file wins):

    1. ``{name}_z_editted.h5``
    2. ``{name}_z_editted.hdf5``
    3. ``{name}.h5``
    4. ``{name}.hdf5``

    Returns ``(abs_h5_path, canonical_dance_name)`` where ``canonical_dance_name``
    never includes the ``_z_editted`` suffix (e.g. ``IRIS_OUT``).
    """
    canonical = normalize_dance_stem(dance_name)
    if not canonical:
        raise ValueError("dance name must be non-empty")

    root = os.path.abspath(dance_dir or default_dance_dir())
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dance media directory not found: {root}")

    candidates = [
        f"{canonical}{_Z_EDITTED_SUFFIX}.h5",
        f"{canonical}{_Z_EDITTED_SUFFIX}.hdf5",
        f"{canonical}.h5",
        f"{canonical}.hdf5",
    ]
    for fname in candidates:
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            abs_path = os.path.abspath(path)
            variant = "z_editted" if _Z_EDITTED_SUFFIX in fname else "base"
            print(
                f"[INFO] Resolved dance '{canonical}' -> {abs_path} ({variant})"
            )
            return abs_path, canonical

    if compile_from_csv:
        csv_path = _find_dance_csv(canonical, root)
        if csv_path is not None:
            abs_path = _compile_csv_sibling_to_h5(csv_path)
            print(f"[INFO] Resolved dance '{canonical}' -> {abs_path} (compiled from CSV)")
            return abs_path, canonical

    tried = ", ".join(candidates)
    csv_hint = ""
    csv_path = _find_dance_csv(canonical, root)
    if csv_path is None:
        csv_hint = (
            f" No CSV sibling under {root} either; run g1_vmd_0_replay.py to retarget VMD first."
        )
    raise FileNotFoundError(
        f"No HDF5 found for dance '{canonical}' under {root}. Tried: {tried}.{csv_hint}"
    )


def resolve_training_log_root(experiment_name: str, dance_name: str | None = None) -> str:
    """Checkpoint search root: ``logs/rsl_rl/<experiment>[/<dance>]``."""
    root = os.path.abspath(os.path.join(_repo_root(), "logs", "rsl_rl", str(experiment_name)))
    if dance_name:
        canonical = normalize_dance_stem(dance_name)
        if canonical:
            return os.path.join(root, canonical)
    return root
