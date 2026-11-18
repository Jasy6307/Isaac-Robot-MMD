"""CPU tests against the installed RSL-RL PPO implementation (Isaac Python)."""
import copy
import sys
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.bounded_ppo import BoundedAdaptivePPO, STAGE_LR_BOUNDS


class BoundedPPOTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(42)
        obs = TensorDict({"policy": torch.randn(8, 4)}, batch_size=[8])
        groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[8],
            distribution_cfg=dict(class_name="GaussianDistribution", init_std=.6, std_type="scalar"))
        critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8])
        storage = RolloutStorage("rl", 8, 2, obs, [2], "cpu")
        alg = BoundedAdaptivePPO(actor, critic, storage, num_learning_epochs=6,
            num_mini_batches=2, learning_rate=1.e-3, desired_kl=.015)
        return alg, obs

    def update_with_kl(self, alg, obs, kl):
        # Force the measured KL so tests exercise native RSL scheduler branches.
        alg.actor.get_kl_divergence = lambda old, new: torch.full((old[0].shape[0],), kl)
        with torch.inference_mode():
            for _ in range(2):
                alg.act(obs)
                alg.process_env_step(obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {})
            alg.compute_returns(obs)
        rates = []
        handle = alg.optimizer.register_step_pre_hook(
            lambda optimizer, args, kwargs: rates.append(optimizer.param_groups[0]["lr"]))
        try:
            losses = alg.update()
        finally:
            handle.remove()
        self.assertEqual(len(rates), 12)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        return rates

    def test_high_kl_lowers_rate_and_respects_floor_on_every_step(self):
        for stage in ("long", "full"):
            with self.subTest(stage=stage):
                alg, obs = self.make()
                alg.configure_learning_rate(stage)
                alg.learning_rate = alg.lr_max
                rates = self.update_with_kl(alg, obs, .1)
                self.assertLess(rates[0], alg.lr_max)
                self.assertEqual(rates[-1], alg.lr_min)
                self.assertTrue(all(alg.lr_min <= rate <= alg.lr_max for rate in rates))

    def test_low_kl_recovers_rate_and_respects_ceiling_on_every_step(self):
        for stage in ("long", "full"):
            with self.subTest(stage=stage):
                alg, obs = self.make()
                alg.configure_learning_rate(stage)
                alg.learning_rate = alg.lr_min
                rates = self.update_with_kl(alg, obs, .001)
                self.assertGreater(rates[0], alg.lr_min)
                self.assertEqual(rates[-1], alg.lr_max)
                self.assertTrue(all(alg.lr_min <= rate <= alg.lr_max for rate in rates))

    def test_near_target_kl_keeps_rate(self):
        alg, obs = self.make()
        alg.configure_learning_rate("long")
        alg.learning_rate = 8.e-5
        self.assertEqual(self.update_with_kl(alg, obs, .015), [8.e-5] * 12)

    def test_stage_changes_and_chunks_preserve_rate_within_new_bounds(self):
        alg, _ = self.make()
        alg.configure_learning_rate("short")
        alg.learning_rate = 1.e-5
        alg.configure_learning_rate("long")
        self.assertEqual(alg.learning_rate, 3.e-5)
        alg.learning_rate = 8.e-5
        alg.configure_learning_rate("long")
        self.assertEqual(alg.learning_rate, 8.e-5)
        alg.configure_learning_rate("full")
        self.assertEqual(alg.learning_rate, 5.e-5)
        self.assertEqual(alg.optimizer.param_groups[0]["lr"], 5.e-5)
        self.assertEqual(alg.schedule, "adaptive")

    def test_short_stage_enforces_true_ceiling_during_each_minibatch(self):
        alg, obs = self.make()
        alg.configure_learning_rate("short")
        self.assertEqual((alg.lr_min, alg.lr_max), (1.e-5, 1.e-3))
        alg.learning_rate = .008
        rates = self.update_with_kl(alg, obs, .001)
        self.assertTrue(all(rate <= .001 for rate in rates))
        self.assertEqual(rates[-1], .001)
        alg.configure_learning_rate("short")
        self.assertEqual(alg.learning_rate, .001)

    def test_resume_restores_optimizer_rate_and_policy(self):
        alg, _ = self.make()
        alg.configure_learning_rate("full")
        alg.learning_rate = 2.e-5
        checkpoint = copy.deepcopy(alg.save())
        resumed, _ = self.make()
        self.assertTrue(resumed.load(checkpoint))
        self.assertEqual(resumed.learning_rate, 2.e-5)
        self.assertEqual(resumed.learning_rate_policy(), alg.learning_rate_policy())
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"], 2.e-5)
        resumed.configure_learning_rate("full")
        self.assertEqual(resumed.learning_rate, 2.e-5)

    def test_legacy_checkpoint_and_weights_only_loading(self):
        alg, _ = self.make()
        alg.configure_learning_rate("long")
        alg.learning_rate = 7.e-5
        checkpoint = copy.deepcopy(alg.save())
        del checkpoint["bounded_learning_rate"]
        resumed, _ = self.make()
        resumed.load(checkpoint)
        resumed.configure_learning_rate("long")
        self.assertEqual(resumed.learning_rate, 7.e-5)
        fresh, _ = self.make()
        fresh.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False})
        self.assertEqual(fresh.learning_rate, .001)
        self.assertEqual(fresh.optimizer.param_groups[0]["lr"], .001)

    def test_repair_retains_fixed_small_rate(self):
        alg, obs = self.make()
        alg.configure_learning_rate("full", repair=True)
        self.assertEqual(alg.schedule, "fixed")
        self.assertEqual(self.update_with_kl(alg, obs, .001), [1.e-5] * 12)

    def test_invalid_target_or_rate_is_rejected(self):
        alg, _ = self.make()
        for value in (None, 0., float("nan")):
            alg.desired_kl = value
            with self.assertRaises(ValueError):
                alg.configure_learning_rate("long")
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                alg.learning_rate = value


if __name__ == "__main__":
    unittest.main()
