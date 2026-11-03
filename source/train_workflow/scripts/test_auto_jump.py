"""CPU tests for intent evidence and adaptive sampling safeguards."""
import sys
import unittest
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion.auto_jump import analyze_signals, event_schedules, apply_review
from source.train_workflow.utils.auto_jump_training import AutoJumpController, curriculum_stage, install_checkpoint_metadata
from source.train_workflow.utils.phase_gate import jump_teacher_gate


class AutoJumpTests(unittest.TestCase):
    def signals(self):
        fps = 60
        t = np.arange(240) / fps
        pulse = np.maximum(0., 1 - ((t - 2) / .22) ** 2) * .12
        feet = np.zeros((len(t), 2, 3))
        feet[:, :, 2] = pulse[:, None]
        feet[:, 0, 1], feet[:, 1, 1] = .2, -.2
        return pulse, feet, fps

    def test_stationary_has_no_events(self):
        z, feet, fps = self.signals()
        self.assertEqual(analyze_signals(z * 0, feet * 0, fps)["events"], [])

    def test_root_bobbing_with_grounded_feet_is_not_jump(self):
        z, feet, fps = self.signals()
        feet[:, :, 2] = 0
        self.assertEqual(analyze_signals(z, feet, fps)["events"], [])

    def test_one_foot_lift_is_not_bilateral_flight(self):
        z, feet, fps = self.signals()
        feet[:, 0, 2] = 0
        self.assertEqual(analyze_signals(z, feet, fps)["events"], [])

    def test_bilateral_reference_jump_and_time_shift(self):
        z, feet, fps = self.signals()
        report = analyze_signals(z, feet, fps)
        self.assertEqual(len(event_schedules(report)), 1)
        shifted = analyze_signals(np.r_[np.zeros(60), z],
                                 np.concatenate((np.repeat(feet[:1], 60, axis=0), feet)), fps)
        a, b = event_schedules(report)[0][2][0], event_schedules(shifted)[0][2][0]
        self.assertAlmostEqual(b - a, 1.)

    def test_uncertain_intent_cannot_enable_flight_reward(self):
        z, feet, fps = self.signals()
        report = analyze_signals(z, feet, fps, np.zeros(len(z), bool))
        self.assertEqual(event_schedules(report), ())

    def test_small_bilateral_bob_is_candidate_not_mandatory_flight(self):
        z, feet, fps = self.signals()
        z *= .5
        feet[:, :, 2] *= .5
        report = analyze_signals(z, feet, fps)
        self.assertTrue(report["events"])
        self.assertEqual(event_schedules(report), ())
        peak = int(np.argmax(z))
        self.assertFalse(any(report["support_feet"][peak]))

    def test_large_slower_reference_flight_is_not_rejected_only_for_speed(self):
        _, feet, fps = self.signals()
        t = np.arange(len(feet)) / fps
        z = np.clip((t - 1.4) * .375, 0., .15)
        z[t > 2.2] = np.maximum(0., .15 - (t[t > 2.2] - 2.2) * .375)
        report = analyze_signals(z, feet, fps)
        self.assertTrue(event_schedules(report))
        self.assertLess(report["events"][0]["evidence"]["upward_mps"], .45)

    def test_support_reward_pays_for_contact_without_rewarding_unknown_or_flight(self):
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse((root / "source/my_task/mdp/auto_jump.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "support_reward")
        metrics = dict(required=torch.tensor([True,True,False]), expected=torch.zeros((3,2), dtype=torch.bool),
                       contacts=torch.tensor([[False,False],[True,False],[False,False]]))
        ns = dict(torch=torch, _support_details=lambda *args: (metrics, torch.zeros(3)))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "positive_support_test", "exec"), ns)
        self.assertEqual(ns["support_reward"](None, None, None).tolist(), [0.,1.,0.])

    def test_support_review_is_reference_bound_and_removes_flight_request(self):
        z, feet, fps = self.signals()
        report = analyze_signals(z, feet, fps)
        report["source_sha256"] = "test"
        review = dict(source_sha256="wrong", support_windows=[])
        with self.assertRaises(ValueError):
            apply_review(report, review)
        review = dict(source_sha256="test", support_windows=[dict(start_s=1.5, end_s=2.5, reason="review")])
        apply_review(report, review)
        self.assertEqual(event_schedules(report), ())
        self.assertEqual(report["events"][0]["confidence"], "reviewed_support")

    def test_repair_sampler_rehearses_support_flight_and_full_motion(self):
        c = self.controller()
        c.env.episode_length_buf = torch.zeros(2000, dtype=torch.long)
        c.env.termination_manager = SimpleNamespace(terminated=torch.zeros(2000, dtype=torch.bool))
        c.report.update(repair_training=True, support_windows=[dict(start_s=27., end_s=28.)],
                        events=[dict(flight_reward=True, phases=[[16.,16.2],[16.2,16.4],[16.4,16.7],[16.7,17.]])])
        starts, targets = c.reset(torch.arange(2000), torch.zeros(2000, dtype=torch.long))
        self.assertTrue(bool((targets == 3000).any()))
        self.assertTrue(bool((starts == 792).any()))
        self.assertTrue(bool((starts == 462).any()))

    def support_case(self, report, protected=False):
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse((root / "source/my_task/mdp/auto_jump.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "support_metrics")
        contacts = torch.tensor([[False,False],[True,False],[False,True],[True,True]])
        phase = torch.zeros((4,4), dtype=torch.bool)
        phase[0,2] = protected
        ns = dict(torch=torch, motion_steps=lambda env: torch.zeros(4),
                  jump=SimpleNamespace(phases=lambda env: phase, full_step_ground_contacts=lambda env: contacts))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "support_test", "exec"), ns)
        env = SimpleNamespace(num_envs=4, device="cpu", step_dt=1., cfg=SimpleNamespace(auto_jump_report=report))
        return ns["support_metrics"](env)

    def test_reviewed_support_allows_single_foot_support(self):
        metrics = self.support_case(dict(support_windows=[dict(start_s=0.,end_s=1.)]))
        self.assertEqual(metrics["unexpected_air"].tolist(), [True,False,False,False])
        self.assertFalse(bool(metrics["missing_feet"].any()))

    def test_unknown_regions_are_not_hard_ground_constraints(self):
        metrics = self.support_case({})
        self.assertFalse(bool(metrics["required"].any()))
        self.assertFalse(bool(metrics["unexpected_air"].any()))

    def test_true_flight_is_protected_and_known_foot_ownership_respected(self):
        metrics = self.support_case(dict(fps=30., support_feet=[[1,0]]), protected=True)
        self.assertFalse(metrics["required"][0])
        self.assertEqual(metrics["missing_feet"][:,0].tolist(), [False,False,True,False])
        self.assertFalse(bool(metrics["missing_feet"][:,1].any()))

    def test_invalid_input_fails(self):
        z, feet, fps = self.signals()
        for bad in (0, float("nan")):
            with self.assertRaises(ValueError):
                analyze_signals(z, feet, bad)
        feet[10, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            analyze_signals(z, feet, fps)

    def controller(self):
        env = SimpleNamespace(device="cpu", step_dt=1/30.)
        report = dict(duration_s=100., source_sha256="test", events=[])
        # Existing test output root; no simulator, no policy or data writes.
        return AutoJumpController(env, report, Path(__file__).resolve().parents[3] / "output/auto-jump/tests")

    def test_sampler_preserves_uniform_coverage(self):
        c = self.controller()
        c.prior[20] = 3
        c.difficulty[20] = 4
        p = c.probabilities()
        self.assertAlmostEqual(float(p.sum()), 1., places=6)
        self.assertTrue(bool((p >= .4 / c.bins).all()))
        self.assertGreater(p[20], p[0])

    def test_unvisited_bins_are_not_marked_successful(self):
        c = self.controller()
        c.sums[10], c.counts[10] = 0., 10.
        c.update()
        self.assertLess(c.difficulty[10], 1.)
        self.assertEqual(c.difficulty[90], 1.)

    def test_difficulty_increases_and_then_recovers(self):
        c = self.controller()
        c.sums[10], c.counts[10], c.failures[10] = 30., 10., 2.
        c.update()
        high = c.difficulty[10].item()
        self.assertGreater(high, 1.)
        c.sums[10], c.counts[10] = 0., 10.
        c.update()
        self.assertLess(c.difficulty[10], high)

    def test_failure_is_assigned_to_original_reference_time(self):
        c = self.controller()
        c.env.episode_length_buf = torch.tensor([30])
        c.env.termination_manager = SimpleNamespace(terminated=torch.tensor([True]), time_outs=torch.tensor([False]))
        c.reset(torch.tensor([0]), torch.tensor([630]))
        self.assertEqual(c.failures[21], 1.)
        self.assertEqual(c.failures[1], 0.)

    def test_probe_keeps_first_failure_and_original_time(self):
        c = self.controller()
        c.env.episode_length_buf = torch.tensor([30])
        c.env.termination_manager = SimpleNamespace(terminated=torch.tensor([True]), time_outs=torch.tensor([False]))
        c.probe = dict(active=torch.tensor([True]), completed=torch.tensor([False]),
                       steps=torch.tensor([0]), failures=torch.zeros(100),
                       starts=torch.tensor([600]), targets=torch.tensor([90]))
        c.reset(torch.tensor([0]), torch.tensor([630]))
        c.reset(torch.tensor([0]), torch.tensor([631]))
        self.assertEqual(c.probe["failures"][21], 1.)
        self.assertEqual(c.probe["steps"][0], 30)
        self.assertFalse(c.probe["completed"][0])

    def test_resume_rejects_wrong_reference(self):
        c = self.controller()
        state = c.state_dict()
        state["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            c.load_state_dict(state)

    def test_curriculum_boundaries(self):
        self.assertEqual([curriculum_stage(x) for x in (0., .55, .85, 1.)],
                         ["short", "long", "full", "full"])

    def test_teacher_gate_is_bounded_smooth_and_absent_from_normal_segments(self):
        schedules = (((2.,2.1),(2.1,2.2),(2.2,2.4),(2.4,2.6)),)
        times = torch.tensor([0.,1.4,1.55,1.7,2.3,3.05,3.2,5.])
        gate = jump_teacher_gate(times,schedules)
        torch.testing.assert_close(gate,torch.tensor([0.,0.,.5,1.,1.,.5,0.,0.]),atol=1e-5,rtol=1e-5)
        self.assertFalse(bool(jump_teacher_gate(times,()).any()))

    def test_intermediate_checkpoints_preserve_curriculum_and_progress(self):
        saved = []
        runner = SimpleNamespace(current_learning_iteration=39,
                                 save=lambda path, infos: saved.append(infos))
        c = self.controller()
        c.counts[10] = 5
        install_checkpoint_metadata(runner, c, start=20, total=100, initial_progress=.5)
        runner.save("test.pt")
        self.assertAlmostEqual(saved[-1]["auto_curriculum"]["progress"], .6)
        self.assertEqual(saved[-1]["auto_curriculum"]["source_sha256"], "test")
        self.assertEqual(saved[-1]["auto_curriculum"]["observed_steps"][10], 5)
        explicit = {"auto_curriculum": {"progress": .7}}
        runner.save("test.pt", infos=explicit)
        self.assertEqual(saved[-1], explicit)

    def test_resume_rejects_different_window_or_frequency(self):
        c = self.controller()
        for key, value in (("duration_s", 99.), ("control_hz", 120.)):
            state = c.state_dict()
            state[key] = value
            with self.assertRaises(ValueError):
                c.load_state_dict(state)

    def test_vectorized_phases_preserve_boundaries_and_empty_schedule(self):
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse((root / "source/my_task/mdp/jump.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "phases")
        ns = dict(torch=torch, motion_steps=lambda env: env.steps, PHASES=())
        exec(compile(ast.Module(body=[node], type_ignores=[]), "phase_test", "exec"), ns)
        schedules = (((1., 2.), (2., 3.), (3., 4.), (4., 5.)),
                     ((2., 3.), (3., 4.), (4., 5.), (5., 6.)))
        steps = torch.tensor([0., 1., 2., 2.99, 3., 4., 5., 6.])
        env = SimpleNamespace(steps=steps, step_dt=1., cfg=SimpleNamespace(jump_phase_schedules=schedules))
        actual = ns["phases"](env)
        expected = torch.tensor([[any(s[i][0] <= t < s[i][1] for s in schedules)
                                  for i in range(4)] for t in steps])
        self.assertTrue(torch.equal(actual, expected))
        env.cfg.jump_phase_schedules = ()
        self.assertFalse(bool(ns["phases"](env).any()))


if __name__ == "__main__":
    unittest.main()
