"""CPU integration checks for pre-reset sampling and report export."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.jump_diagnostics import JumpDiagnostics
from source.train_workflow.utils.jump_analysis import analyze, window_metrics
from source.train_workflow.scripts.analyze_turn import spectral_rms,control_rows


class Scene(dict):
    def update(self, dt):
        pass


class Buffer:
    control_hz = 30.

    def root_pos_delta(self, steps):
        return torch.stack((steps * 0., steps * 0., steps * .01), dim=-1)

    def q_ref_abs(self, steps):
        return torch.full((len(steps), 1), .25)

    def root_quat_wxyz(self, steps):
        return torch.tensor([[1.,0.,0.,0.]]).repeat(len(steps),1)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "output", prefix="jump-test-")
        data = NS(default_root_state=torch.zeros(1, 13), root_state_w=torch.zeros(1, 13),
                  root_lin_vel_w=torch.tensor([[0., 0., 1.]]),
                  body_link_pose_w=torch.zeros(1, 2, 7), joint_pos_target=torch.tensor([[.3]]),
                  joint_pos=torch.tensor([[.2]]), applied_torque=torch.tensor([[10.]]),
                  computed_torque=torch.tensor([[11.]]), joint_effort_limits=torch.tensor([[20.]]))
        data.root_state_w[0, 2] = .8
        self.scene = Scene(robot=NS(data=data, joint_names=["left_knee_joint"],
                                   body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                           jump_left_contact=NS(data=NS(net_normal_forces_w=torch.tensor([[[0., 0., 100.]]]))),
                           jump_right_contact=NS(data=NS(net_normal_forces_w=torch.zeros(1, 1, 3))))
        self.scene.sim = NS(get_clone_plan=lambda: NS(positions=torch.zeros(1, 3)))
        self.env = NS(scene=self.scene, device="cpu", physics_dt=1 / 120.,
                      cfg=NS(scene=NS(terrain=NS(ground_z_offset=0.))))

    def tearDown(self):
        self.temp.cleanup()

    def test_physics_hook_preserves_pre_reset_state_and_restores_update(self):
        original = self.scene.update
        recorder = JumpDiagnostics(self.env, self.temp.name)
        recorder.begin(0, Buffer(), {"seed": 42})
        recorder.begin_control(torch.tensor([3]))
        for _ in range(4):
            self.scene.update(dt=1 / 120.)
        recorder.end_control()
        # Simulate the RL environment's automatic reset after physics stepping.
        self.scene["robot"].data.root_state_w[0, 2] = .5
        self.scene.update(dt=1 / 120.)
        self.assertEqual(len(recorder.rows), 4)
        self.assertAlmostEqual(recorder.rows[-1]["root_z_m"], .8)
        self.assertEqual(recorder.rows[-1]["both_feet_no_contact"], 0)
        self.assertAlmostEqual(recorder.rows[-1]["ref_root_vz_m_s"], .3)
        self.assertAlmostEqual(recorder.rows[-1]["time_s"], 1 / 30.)
        recorder.close()
        self.assertEqual(self.scene.update, original)
        report = next(Path(self.temp.name).glob("episode_*"))
        for name in ("samples.csv", "samples.npz", "metadata.json", "overview.png", "legs.png"):
            self.assertGreater((report / name).stat().st_size, 0)
        arrays = np.load(report / "samples.npz")
        self.assertEqual(arrays["values"].shape[0], 4)

    def test_time_window_excludes_idle_and_outside_samples(self):
        recorder = JumpDiagnostics(self.env, self.temp.name, start=.015, end=.026)
        self.scene.update(dt=1 / 120.)
        self.assertEqual(recorder.rows, [])
        recorder.begin(0, Buffer(), {})
        recorder.begin_control(torch.tensor([0]))
        for _ in range(4):
            self.scene.update(dt=1 / 120.)
        self.assertEqual(len(recorder.rows), 2)
        self.scene["jump_left_contact"].data.net_normal_forces_w.zero_()
        recorder.end = None
        self.scene.update(dt=1 / 120.)
        self.assertEqual(recorder.rows[-1]["both_feet_no_contact"], 1)
        recorder.close()

    def test_extended_turn_channels_preserve_target_reference_alignment(self):
        data=self.scene["robot"].data
        data.body_link_lin_vel_w=torch.tensor([[[.1,.2,.3],[.4,.5,.6]]])
        data.body_link_ang_vel_w=torch.tensor([[[0.,0.,2.],[0.,0.,-3.]]])
        data.joint_vel=torch.tensor([[.7]])
        data.root_state_w[0,6]=1.;data.body_link_pose_w[:,:,6]=1.
        recorder=JumpDiagnostics(self.env,self.temp.name,extended=True)
        recorder.begin(0,Buffer(),{})
        recorder.begin_control(torch.tensor([390]))
        self.scene.update(dt=1/120.);recorder.end_control()
        row=recorder.rows[0]
        self.assertAlmostEqual(row["ref_time_s"],13.)
        self.assertEqual(row["left_foot_wz_rad_s"],2.)
        self.assertEqual(row["right_foot_wz_rad_s"],-3.)
        self.assertAlmostEqual(row["left_knee_joint_target_rad"],.3)
        self.assertAlmostEqual(row["left_knee_joint_velocity_rad_s"],.7)
        self.assertEqual(row["root_yaw_rad"],0.)
        recorder.close()


class AnalysisTests(unittest.TestCase):
    def test_hf_comparison_separates_smooth_reference_and_oscillating_residual(self):
        t=np.arange(60)/30.
        ref=.3*np.sin(2*np.pi*t);residual=.1*np.sin(2*np.pi*6*t)
        self.assertLess(spectral_rms(ref,30),.001)
        self.assertAlmostEqual(spectral_rms(residual,30),.1/np.sqrt(2),places=3)
        self.assertGreater(spectral_rms(ref+residual,30),.06)

    def test_partial_control_intervals_are_removed_before_differentiating(self):
        steps=np.array([299,299,299,300,300,300,300,301,301,301,301,302])
        data=dict(ref_time_s=steps/30.,sample_dt_s=np.full(len(steps),1/120.))
        self.assertEqual(control_rows(data,30.).tolist(),[6,10])

    def data(self, t, ref, actual):
        return dict(time_s=t, root_z_m=actual, ref_root_z_m=ref,
                    both_feet_no_contact=np.zeros(len(t)))

    def test_high_reference_peak_with_exact_tracking_is_not_candidate(self):
        t = np.arange(0, 6, 1 / 120.)
        ref = .75 + .3 * np.exp(-((t - 3.) / .2) ** 2)
        self.assertEqual(analyze(self.data(t, ref, ref))["review_candidates"], [])

    def test_lag_sign_and_fixed_overlap(self):
        t = np.arange(0, 4, 1 / 120.)
        ref = .75 + .1 * np.sin(2 * np.pi * t)
        actual = .75 + .1 * np.sin(2 * np.pi * (t - .1))
        row = window_metrics(self.data(t, ref, actual), .1, 3.9)
        self.assertAlmostEqual(row["lag_candidate_s"], .1, places=5)
        self.assertLess(row["aligned_height_rmse_m"], row["same_overlap_unshifted_rmse_m"])

    def test_flat_motion_has_no_claimed_lag(self):
        t = np.arange(0, 4, 1 / 120.)
        row = window_metrics(self.data(t, np.ones(len(t)), np.ones(len(t))), 0, 3)
        self.assertIsNone(row["lag_candidate_s"])


if __name__ == "__main__":
    unittest.main()
