"""CPU regression checks for H5-free reference restoration.

Run: python source/train_workflow/scripts/test_deployment_bundle.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.format.hdf5 import Hdf5Motion, write_hdf5_motion
from source.train_workflow.utils.deployment import load_bundle, save_bundle

# Load the pure tensor module without importing my_task's Isaac/Gym registration.
spec = importlib.util.spec_from_file_location(
    "deployment_test_motion_buffer", ROOT / "source/my_task/motion_reference/motion_buffer.py"
)
motion_buffer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion_buffer)
Buffer = motion_buffer.DanceMotionReferenceBuffer


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "output", prefix="deployment-test-")
        self.path = Path(self.temp.name)
        self.h5 = self.path / "source.h5"
        self.names = ["arm", "leg", "finger"]
        self.defaults = np.array([.1, -.2, .3], dtype=np.float32)
        rng = np.random.default_rng(42)
        write_hdf5_motion(str(self.h5), Hdf5Motion(
            frames=np.arange(7, dtype=np.int32), joint_names=self.names,
            joint_pos_delta=rng.normal(size=(7, 3)).astype(np.float32),
            root_pos_delta=rng.normal(size=(7, 3)).astype(np.float32),
            root_quat_delta_wxyz=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (7, 1)),
            root_valid=np.ones(7, dtype=bool), root_rot_bone=["center"] * 7,
            root_rpy_deg=np.zeros((7, 3), dtype=np.float32), fps=30.,
        ))
        self.kwargs = dict(runtime_joint_names=self.names, default_joint_pos=self.defaults,
                           control_hz=60., window_seconds=.4, device="cpu")
        self.original = Buffer(h5_path=str(self.h5), **self.kwargs)
        self.reference = self.original.export_reference()

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_without_h5_and_without_checkpoint(self):
        checkpoint = self.path / "model.pt"
        torch.save({"model_state_dict": {"weight": torch.tensor([1., 2.])}}, checkpoint)
        bundle_path = self.path / "deployment.pt"
        save_bundle(bundle_path, checkpoint_path=checkpoint, reference=self.reference,
                    env_cfg=SimpleNamespace(test="env"), agent_cfg=SimpleNamespace(test="agent"),
                    task="task", dance="dance")
        expected_checkpoint = checkpoint.read_bytes()
        self.h5.unlink()
        checkpoint.unlink()
        bundle = load_bundle(bundle_path)
        self.assertEqual(bundle["checkpoint"], expected_checkpoint)
        self.assertEqual(bundle["env_cfg"].test, "env")
        # Any accidental attempt to resolve/open the source H5 must fail this test.
        with patch.object(motion_buffer, "_resolve_h5_path", side_effect=AssertionError("H5 resolve")), \
             patch.object(motion_buffer, "load_hdf5_motion", side_effect=AssertionError("H5 read")):
            restored = Buffer.from_reference(bundle["reference"], path=str(bundle_path), **self.kwargs)
            indices = torch.tensor([0, 1, 12, 23, 24, 100])
            for method in ("q_ref_abs", "q_ref_rel", "root_pos_delta", "root_quat_wxyz", "motion_phase"):
                self.assertTrue(torch.equal(getattr(self.original, method)(indices),
                                            getattr(restored, method)(indices)), method)
            self.assertTrue(torch.equal(self.original.q_ref_rel(indices, offset=1),
                                        restored.q_ref_rel(indices, offset=1)))

    def test_incompatible_runtime_is_rejected(self):
        for changes in ({"control_hz": 30.}, {"window_seconds": .5},
                        {"runtime_joint_names": list(reversed(self.names))},
                        {"default_joint_pos": self.defaults + 1.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Buffer.from_reference(self.reference, path="unused", **(self.kwargs | changes))

    def test_shared_environment_buffer_uses_bundle_without_h5(self):
        bundle_path = str(self.path / "deployment.pt")
        motion_buffer.register_packaged_reference(bundle_path, self.reference)
        asset = SimpleNamespace(joint_names=self.names, data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.tensor(self.defaults).unsqueeze(0))))
        env = SimpleNamespace(scene={"robot": asset}, step_dt=1 / 60., device="cpu")
        self.h5.unlink()
        with patch.dict(sys.modules, {"isaaclab.assets": SimpleNamespace(Articulation=object)}), \
             patch.object(motion_buffer, "_resolve_h5_path", side_effect=AssertionError("H5 resolve")), \
             patch.object(motion_buffer, "load_hdf5_motion", side_effect=AssertionError("H5 read")):
            restored = motion_buffer.get_or_create_motion_buffer(env, bundle_path, .4)
            self.assertIs(restored, motion_buffer.get_or_create_motion_buffer(env, bundle_path, .4))
            self.assertTrue(torch.equal(restored.q_ref_abs_first(), self.original.q_ref_abs_first()))

    def test_malformed_reference_is_rejected(self):
        self.reference["tables"]["_phase"][0] = float("nan")
        with self.assertRaises(ValueError):
            Buffer.from_reference(self.reference, path="unused", **self.kwargs)


if __name__ == "__main__":
    unittest.main()
