"""Canonical workspace paths shared across training and playback scripts."""

from __future__ import annotations

import os

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SOURCE_DIR, ".."))

MEDIA_DIR = os.path.join(REPO_ROOT, "media")
DANCE_DIR = os.path.join(MEDIA_DIR, "dance")
POSE_DIR = os.path.join(MEDIA_DIR, "pose")
