"""Single-file Isaac Lab evaluation bundles (load only trusted local bundles).

Configurations may contain Python classes/functions, just like the
Isaac Lab configuration itself; this format requires the matching project/runtime.
"""
from __future__ import annotations

import os
import shutil
import json
import tempfile
from pathlib import Path

import torch

FORMAT = "g1-dance-deployment"
VERSION = 1


def atomic_copy_file(source, target):
    """Publish a readable checkpoint alias without exposing a partial copy."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=target.parent,suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(source,temporary)
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def atomic_write_json(target, value):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=target.parent,suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def save_bundle(path, *, checkpoint_path, reference, env_cfg, agent_cfg, task, dance):
    target = Path(path).resolve()
    if target == Path(checkpoint_path).resolve():
        raise ValueError("Deployment output must not overwrite the training checkpoint")
    payload = {
        "format": FORMAT, "version": VERSION, "task": task, "dance": dance,
        "checkpoint": Path(checkpoint_path).read_bytes(),
        "reference": reference, "env_cfg": env_cfg,
        "agent_cfg": agent_cfg,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_bundle(path):
    # Config dictionaries include trusted project classes/functions.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("Unsupported G1 deployment bundle format/version")
    for field in ("task", "dance", "checkpoint", "reference", "env_cfg", "agent_cfg"):
        if field not in payload:
            raise ValueError(f"Deployment bundle missing {field}")
    return payload
