"""Regression checks for publication safety and reference data preservation.

Run with the project's Python environment (no Isaac app launch required).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
import unittest
import uuid

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.motion.collision_reference import repair_reference_h5
from source.train_workflow.utils.motion.self_collision import playback_baseline
from source.train_workflow.utils.motion.sync import get_default_compile_joint_names
from source.train_workflow.utils.format.hdf5 import Hdf5Motion, write_hdf5_motion
from source.train_workflow.utils.playback.collision_h5_job import CollisionH5Job


class ExportTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.path = ROOT / "output" / ("collision-test-"+uuid.uuid4().hex)
        self.path.mkdir()
        self.raw = self.path / "raw.h5"
        # A static outward-arm, separated-leg pose tests publication without
        # depending on a previous manually repaired dance file.
        names = get_default_compile_joint_names()
        baseline = playback_baseline(names, ROOT / "source/my_task/g1_replay_env_cfg.py")
        delta = np.zeros((3, len(names)), dtype=np.float32)
        delta[:, names.index("left_hip_roll_joint")] = .125
        delta[:, names.index("right_hip_roll_joint")] = -.125
        write_hdf5_motion(str(self.raw), Hdf5Motion(
            frames=np.arange(3, dtype=np.int32), joint_names=names,
            joint_pos_delta=delta, root_pos_delta=np.zeros((3, 3), dtype=np.float32),
            root_quat_delta_wxyz=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
            root_valid=np.ones(3, dtype=bool), root_rot_bone=["center"]*3,
            root_rpy_deg=np.zeros((3, 3), dtype=np.float32),
        ))
        with h5py.File(self.raw, "r+") as stream:
            stream.attrs["record_baseline_joint_pos"] = baseline
        self.usd = ROOT / "assets/g1_29dof_o6_hand_V3.usd"

    def tearDown(self):
        assert self.path.resolve().parent == (ROOT / "output").resolve()
        for file in self.path.iterdir():
            file.unlink()
        self.path.rmdir()

    def test_no_op_preserves_source_and_all_channels(self):
        before = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        out = self.path / "safe.h5"
        report = repair_reference_h5(self.raw, out, self.usd)
        self.assertIn("skipped", report["initial_repair"])
        self.assertEqual(before, hashlib.sha256(self.raw.read_bytes()).hexdigest())
        with h5py.File(self.raw) as src, h5py.File(out) as dst:
            self.assertTrue(dst.attrs["without_collision_validated"])
            def compare(key, item):
                if isinstance(item, h5py.Dataset):
                    np.testing.assert_array_equal(item[:], dst[key][:])
            src.visititems(compare)
        content = out.read_bytes()
        with self.assertRaises(FileExistsError):
            repair_reference_h5(self.raw, out, self.usd)
        self.assertEqual(content, out.read_bytes())

    def test_invalid_reference_is_not_published(self):
        with h5py.File(self.raw, "r+") as stream:
            stream["joint_pos_delta"][1, 0] = np.nan
        out = self.path / "invalid.h5"
        with self.assertRaises(ValueError):
            repair_reference_h5(self.raw, out, self.usd)
        self.assertFalse(out.exists())

    def test_single_frame_static_pose_is_supported(self):
        one = self.path / "one.h5"
        with h5py.File(self.raw) as src, h5py.File(one, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key, value in src.items():
                if isinstance(value, h5py.Group):
                    src.copy(value, dst, name=key)
                else:
                    dst.create_dataset(key, data=value[:] if key == "joint_names" else value[:1], dtype=value.dtype)
        out = self.path / "one-safe.h5"
        report = repair_reference_h5(one, out, self.usd)
        self.assertEqual(report["frames"], 1)
        self.assertNotIn("midpoints", report["after"])

    def test_worker_publishes_and_versions_independent_file(self):
        job = CollisionH5Job()
        csv = self.path / "dance.csv"
        csv.write_text("fixture", encoding="utf-8")
        try:
            raw_path = job.prepare(csv)
            raw_path.write_bytes(self.raw.read_bytes())
            job.start(raw_path, self.usd)
            deadline = time.monotonic()+60
            while job.process is not None and time.monotonic() < deadline:
                time.sleep(.25)
                job.poll()
            self.assertEqual(job.state["state"], "done", job.state)
            first = job.latest_output(csv)
            self.assertTrue(first)
            second_raw = job.prepare(csv)
            self.assertEqual(Path(job.state["output"]).name, "dance_without_collision_001.h5")
            second_raw.write_bytes(self.raw.read_bytes())
            job.start(second_raw, self.usd)
            job.cancel()
            self.assertEqual(job.state["state"], "cancelled")
            self.assertIsNone(job.process)
            self.assertTrue(Path(first).exists())
            self.assertEqual(job.latest_output(csv), first)
        finally:
            job.close()


if __name__ == "__main__":
    unittest.main()
